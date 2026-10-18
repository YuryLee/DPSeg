"""Disjoint GPU prediction shards, one ordered full-split COCO evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .course_archive import CourseArchive
from .course_depth_data import load_inference_sample
from .course_dafusion_eval import score_predictions
from .course_submission import _Spec, _predictions, _sha256


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def merge_shards(paths, image_ids, output):
    """Reject missing/duplicate/foreign IDs before scoring; preserve official order."""
    records = {}
    expected = set(image_ids)
    for path in paths:
        with Path(path).open() as source:
            for line in source:
                record = json.loads(line)
                key = record['image_id']
                if key not in expected or key in records:
                    raise ValueError('Duplicate or foreign evaluation image')
                records[key] = record
    if set(records) != expected:
        raise ValueError('Incomplete evaluation shards')
    spec = _Spec(image_ids, (1,), (1024, 1024), 100)
    count = 0
    with Path(output).open('x') as stream:
        for key in image_ids:
            record = records[key]
            record['predictions'], _ = _predictions(record['predictions'], spec, decode=False)
            count += len(record['predictions'])
            stream.write(json.dumps(record, allow_nan=False)+'\n')
    return count


def evaluate(adapter, archive, output, *, rank, world_size, guard, smoke=False):
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.monotonic()
    was_training = adapter.training
    adapter.eval()
    try:
        with CourseArchive(archive, split='val') as dataset:
            if len(dataset) != 2584:
                raise ValueError('Full validation split must contain 2584 images')
            ids = dataset.image_ids[:2*world_size] if smoke else dataset.image_ids
            rank_ids = ids[rank::world_size]
            if not rank_ids:
                raise ValueError('Empty evaluation rank')
            by_id = {record['id']: record for record in dataset.images}
            times = []
            with torch.inference_mode(), (output/f'predictions_rank{rank}.jsonl').open('x') as stream:
                first = load_inference_sample(dataset, rank_ids[0])
                for _ in range(3):
                    guard()
                    adapter.predict_sample(first, score_threshold=.01, max_detections=100)
                torch.cuda.synchronize(adapter.device)
                for index, image_id in enumerate(rank_ids):
                    guard()
                    sample = load_inference_sample(dataset, image_id)
                    torch.cuda.synchronize(adapter.device)
                    begin = time.perf_counter()
                    values = adapter.predict_sample(sample, score_threshold=.01, max_detections=100)
                    torch.cuda.synchronize(adapter.device)
                    elapsed = time.perf_counter()-begin
                    times.append(elapsed)
                    stream.write(json.dumps({'image_id': image_id,
                        'file_name': by_id[image_id]['file_name'], 'predictions': values,
                        'inference_seconds': elapsed}, allow_nan=False)+'\n')
                    if index == 0 or (index+1) % 25 == 0 or index+1 == len(rank_ids):
                        stream.flush()
                        write_json(output/f'progress_rank{rank}.json', {
                            'phase': 'inferring', 'rank': rank, 'images': index+1,
                            'expected': len(rank_ids), 'elapsed_seconds': time.monotonic()-started})
            write_json(output/f'rank{rank}.json', {'status': 'completed', 'rank': rank,
                'images': len(rank_ids), 'ordered_ids': rank_ids,
                'mean_model_ms': 1000*sum(times)/len(times), 'warmup': 3,
                'prediction_sha256': _sha256(output/f'predictions_rank{rank}.jsonl')})
            dist.barrier()
            result = None
            if rank == 0:
                paths = [output/f'predictions_rank{i}.jsonl' for i in range(world_size)]
                count = merge_shards(paths, ids, output/'predictions.jsonl')
                write_json(output/'progress.json', {'phase': 'scoring', 'images': len(ids)})
                result = score_predictions(dataset, output/'predictions.jsonl', output,
                    image_ids=ids if smoke else None)
                result.update(scope='engineering_smoke' if smoke else 'full_split',
                              full_split_images=2584)
                write_json(output/'evaluation.json', result)
                write_json(output/'evaluation_run.json', {
                    'status': 'completed', 'split': 'val', 'images': len(ids),
                    'scope': result['scope'], 'predicted_instances': count,
                    'predictions_sha256': _sha256(output/'predictions.jsonl'),
                    'evaluation_sha256': _sha256(output/'evaluation.json'),
                    'world_size': world_size, 'inference_precision': 'fp32',
                    'depth_used': True, 'ground_truth_in_inference': False,
                    'elapsed_seconds': time.monotonic()-started})
            payload = [result]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
    finally:
        adapter.train(was_training)
