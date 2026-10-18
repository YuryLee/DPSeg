#!/usr/bin/env python3
"""Keep each saved instance's largest 8-connected mask component; score unchanged."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '4'

import argparse
from concurrent.futures import ThreadPoolExecutor
import contextlib
from datetime import datetime, timezone
import itertools
import json
from pathlib import Path
import sys
import time

import numpy as np
from pycocotools import mask as masks
from scipy import ndimage
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg import course_submission as submission
from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
from tools.summarize_course_finetune import read, sha256, check_hash, require


def largest_component(binary):
    """Retain the first scan-order component on equal largest areas."""
    ys, xs = np.where(binary)
    require(len(xs) > 0, 'Empty instance mask')
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    labels, count = ndimage.label(binary[y0:y1, x0:x1], structure=np.ones((3, 3), np.uint8))
    if count == 1:
        return binary, count, False
    sizes = np.bincount(labels.ravel())[1:]
    winner = int(sizes.argmax()) + 1
    retained = np.zeros_like(binary)
    retained[y0:y1, x0:x1] = labels == winner
    return retained, count, bool(np.count_nonzero(sizes == sizes[winner - 1]) > 1)


def transform(record):
    started = time.monotonic()
    statistics = dict(instances=0, changed_instances=0, discarded_components=0,
                      largest_ties=0, input_area=0, output_area=0)
    values = []
    for value in record['predictions']:
        rle = value['segmentation']
        binary = masks.decode({'size': rle['size'], 'counts': rle['counts'].encode('ascii')})
        require(binary.shape == (1024, 1024) and binary.dtype == np.uint8, 'Invalid native mask')
        retained, components, tie = largest_component(binary)
        statistics['instances'] += 1
        statistics['input_area'] += int(binary.sum())
        statistics['output_area'] += int(retained.sum())
        if components > 1:
            encoded = masks.encode(np.asfortranarray(retained))
            value = {**value, 'segmentation': {'size': encoded['size'], 'counts': encoded['counts'].decode('ascii')}}
            statistics['changed_instances'] += 1
            statistics['discarded_components'] += components - 1
            statistics['largest_ties'] += int(tie)
        values.append(value)
    require([p['score'] for p in values] == [p['score'] for p in record['predictions']], 'Confidence changed')
    require([p['category_id'] for p in values] == [p['category_id'] for p in record['predictions']], 'Category changed')
    return {'image_id': record['image_id'], 'file_name': record['file_name'],
            'predictions': values, 'postprocess_seconds': time.monotonic() - started}, statistics


def self_check():
    diagonal = np.eye(3, dtype=np.uint8)
    kept, count, tie = largest_component(diagonal)
    require(count == 1 and np.array_equal(kept, diagonal), '8-connectivity failed')
    binary = np.zeros((8, 8), np.uint8)
    binary[1:3, 1:3] = 1
    binary[6, 6] = 1
    kept, count, tie = largest_component(binary)
    require(count == 2 and kept.sum() == 4 and kept[6, 6] == 0 and binary.sum() == 5,
            'Largest region preservation or input immutability failed')
    binary = np.zeros((8, 8), np.uint8)
    binary[1, 1] = binary[6, 6] = 1
    kept, count, tie = largest_component(binary)
    require(tie and kept[1, 1] == 1 and kept[6, 6] == 0, 'Deterministic tie rule failed')
    native = np.zeros((1024, 1024), np.uint8)
    native[10:14, 20:24] = 1
    native[800, 900] = 1
    rle = masks.encode(np.asfortranarray(native))
    prediction = {'category_id': 1, 'score': .37,
                  'segmentation': {'size': rle['size'], 'counts': rle['counts'].decode()}}
    result, statistics = transform({'image_id': 7, 'file_name': 'fixture.png', 'predictions': [prediction, prediction]})
    require(statistics['instances'] == statistics['changed_instances'] == 2, 'Independent instances merged')
    require(statistics['output_area'] == 32 and all(p['score'] == .37 for p in result['predictions']), 'RLE or score changed incorrectly')
    require(masks.decode(result['predictions'][0]['segmentation']).sum() == 16, 'Native RLE round trip failed')
    print(json.dumps({'status': 'pass', 'checks': ['8-connectivity', 'largest-region preservation',
        'input immutability', 'scan-order tie', 'independent overlapping instances', 'native RLE and unchanged scores']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=ROOT / 'reports/course_component_cleanup/protocol.json')
    parser.add_argument('--candidate', choices=('original', 'sampling'), default='original')
    parser.add_argument('--split', choices=('val', 'test'), default='val')
    parser.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        self_check()
        return
    protocol = read(args.protocol)
    require(protocol['status'] == 'frozen', 'Frozen protocol required')
    for name, digest in protocol['source_sha256'].items():
        check_hash(ROOT / name, digest)
    candidate = protocol['candidates'][args.candidate]
    if args.split == 'test':
        decision = read(args.protocol.parent / 'decision.json')
        require(decision['protocol_sha256'] == sha256(args.protocol)
                and decision['adoption_gate_passed'] is True
                and decision['selected_candidate'] == args.candidate, 'Test requires validation-selected passing candidate')
        folder = Path(candidate['formal_run'])
        run = read(folder / 'run.json')
        require(run['status'] == 'completed' and run['best_epoch'] == candidate['selected_epoch'], 'Source training incomplete')
        require(run['best_checkpoint_sha256'] == candidate['checkpoint_sha256'], 'Test weight differs from frozen validation weight')
        source = Path(run['test_output'])
        receipt = read(source / 'evaluation_run.json')
        require(receipt['selected_epoch'] == candidate['selected_epoch'] and
                receipt['evaluated_checkpoint_sha256'] == candidate['checkpoint_sha256'], 'Test checkpoint receipt differs')
    else:
        source = Path(candidate['validation_path'])
        receipt = read(source / 'evaluation_run.json')
        require(receipt['predictions_sha256'] == candidate['validation_predictions_sha256'], 'Validation predictions changed')
    require(receipt['status'] == 'completed' and receipt['split'] == args.split
            and receipt['scope'] == 'full_split', 'Only completed full-split predictions allowed')
    source_path = source / 'predictions.jsonl'
    check_hash(source_path, receipt['predictions_sha256'])
    output = args.protocol.parent / args.candidate / args.split
    require(not output.exists(), 'Refusing to overwrite a cleanup run')
    output.mkdir(parents=True)
    torch.set_num_threads(4)
    state = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
             'candidate': args.candidate, 'split': args.split, 'selected_epoch': candidate['selected_epoch'],
             'protocol_sha256': sha256(args.protocol), 'script_sha256': sha256(__file__),
             'source_predictions_sha256': receipt['predictions_sha256'], 'new_inference': False,
             'confidence_unchanged': True, 'gpu_used': False}
    started = time.monotonic()
    try:
        with CourseArchive(protocol['archive'], args.split) as dataset:
            expected = 2584 if args.split == 'val' else 2112
            require(len(dataset) == expected, 'Complete split required')
            spec = submission._Spec(dataset.image_ids, (1,), (1024, 1024), 100)
            total, count = {}, 0
            writer = (submission.SubmissionWriter(output / 'submission.csv', dataset.image_ids)
                      if args.split == 'test' else contextlib.nullcontext())
            with source_path.open() as stream, (output / 'predictions.jsonl').open('x') as saved, writer as csv_writer, ThreadPoolExecutor(max_workers=4) as pool:
                while batch := list(itertools.islice(stream, 16)):
                    records = [json.loads(line, object_pairs_hook=submission._unique_object) for line in batch]
                    for offset, record in enumerate(records):
                        submission._check_id(record['image_id'], count + offset, spec)
                        record['predictions'], _ = submission._predictions(record['predictions'], spec, decode=False)
                    for result, stats in pool.map(transform, records):
                        saved.write(json.dumps(result, allow_nan=False) + '\n')
                        if csv_writer is not None:
                            csv_writer.write(result['image_id'], result['predictions'])
                        count += 1
                        for key, value in stats.items():
                            total[key] = total.get(key, 0) + value
                        if count % 100 == 0 or count == expected:
                            saved.flush()
                            print(json.dumps({'images': count, 'expected': expected, 'statistics': total}), flush=True)
            require(count == expected, 'Missing source images')
            check_hash(source_path, receipt['predictions_sha256'])
            state.update(images=count, statistics=total)
            evaluation = score_predictions(dataset, output / 'predictions.jsonl', output)
            if args.split == 'test':
                validation = validate_csv_fast(output / 'submission.csv', dataset.image_ids)
                (output / 'submission_validation.json').write_text(json.dumps(validation, indent=2) + '\n')
                state['submission_sha256'] = validation['sha256']
            state.update(status='completed', metrics=evaluation['metrics'],
                         predictions_sha256=sha256(output / 'predictions.jsonl'),
                         evaluation_sha256=sha256(output / 'evaluation.json'),
                         elapsed_seconds=time.monotonic() - started)
            require(not torch.cuda.is_initialized(), 'Unexpected GPU initialization')
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error))
        raise
    finally:
        (output / 'run.json').write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(state, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
