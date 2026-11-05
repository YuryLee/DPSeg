"""Disjoint GPU prediction shards, one ordered full-split COCO evaluation."""
from __future__ import annotations

import json
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .course_archive import CourseArchive
from .course_depth_backbone_eval import merge_shards
from .course_submission import SubmissionWriter, validate_csv
from .course_dafusion_eval import score_predictions
from .course_submission import _sha256


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def evaluate(adapter, archive, output, *, rank, world_size, guard, smoke=False, split="val"):
    if split not in {"val", "test"}:
        raise ValueError("Expected val or test")
    expected_count = {"val": 2584, "test": 2112}[split]
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.monotonic()
    was_training = adapter.training
    adapter.eval()
    try:
        with CourseArchive(archive, split=split) as dataset:
            if len(dataset) != expected_count:
                raise ValueError('Unexpected complete split size')
            ids = dataset.image_ids[:2*world_size] if smoke else dataset.image_ids
            rank_ids = ids[rank::world_size]
            if not rank_ids:
                raise ValueError('Empty evaluation rank')
            by_id = {record['id']: record for record in dataset.images}
            times = []
            with torch.inference_mode(), (output/f'predictions_rank{rank}.jsonl').open('x') as stream:
                first = dataset.load_rgb(dataset.image_ids.index(rank_ids[0]))
                for _ in range(3):
                    guard()
                    adapter.predict(first, score_threshold=.01, max_detections=100)
                torch.cuda.synchronize(adapter.device)
                for index, image_id in enumerate(rank_ids):
                    guard()
                    sample = dataset.load_rgb(dataset.image_ids.index(image_id))
                    torch.cuda.synchronize(adapter.device)
                    begin = time.perf_counter()
                    values = adapter.predict(sample, score_threshold=.01, max_detections=100)
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
                if split == 'test' and not smoke:
                    with SubmissionWriter(output/'submission.csv', ids) as writer:
                        with (output/'predictions.jsonl').open() as predictions:
                            for line in predictions:
                                record = json.loads(line)
                                writer.write(record['image_id'], record['predictions'])
                    write_json(output/'submission_validation.json', validate_csv(output/'submission.csv', ids))
                result.update(input_modality='RGB', depth_used=False, camera_intrinsics_used=False, scope='engineering_smoke' if smoke else 'full_split',
                              full_split_images=expected_count)
                write_json(output/'evaluation.json', result)
                write_json(output/'evaluation_run.json', {
                    'status': 'completed', 'split': split, 'images': len(ids),
                    'scope': result['scope'], 'predicted_instances': count,
                    'predictions_sha256': _sha256(output/'predictions.jsonl'),
                    'evaluation_sha256': _sha256(output/'evaluation.json'),
                    'world_size': world_size, 'inference_precision': 'fp32',
                    'depth_used': False, 'camera_intrinsics_used': False, 'ground_truth_in_inference': False,
                    'elapsed_seconds': time.monotonic()-started})
            payload = [result]
            dist.broadcast_object_list(payload, src=0)
            return payload[0]
    finally:
        adapter.train(was_training)
