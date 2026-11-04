#!/usr/bin/env python3
"""Check full UOAIS validation coverage and recompute saved seven-metric means.

This verifies an existing validation artifact. It does not select a model or
produce a final OSD/OCID/OCBD test result.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

METRICS = (
    'Objects Precision', 'Objects Recall', 'Objects F-measure',
    'Boundary Precision', 'Boundary Recall', 'Boundary F-measure',
    'obj_detected_075_percentage',
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.epoch < 1 or args.output.exists():
        parser.error('Use a positive epoch and a new output file')
    run = json.loads((args.run/'run.json').read_text())
    protocol_raw = (args.run/'protocol.json').read_bytes()
    protocol = json.loads(protocol_raw)
    protocol_hash = hashlib.sha256(protocol_raw).hexdigest()
    if (run['scope'] != 'full_train_val' or protocol_hash != run['protocol_sha256']
            or protocol['status'] != 'frozen' or run['source_sha256'] != protocol['source_sha256']):
        raise ValueError('Expected an identified full UOAIS run and its frozen protocol')
    if args.epoch > protocol['training']['epochs']:
        raise ValueError('Epoch exceeds the declared training budget')
    summary_path = args.run/f'validation_epoch{args.epoch:02d}.json'
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    if (summary['epoch'] != args.epoch or summary['scope'] != 'full_validation'
            or summary['images'] != 5000 or summary['test_used'] is not False
            or summary['score_threshold'] != .5 or set(summary['metrics']) != set(METRICS)):
        raise ValueError('Expected complete validation with the fixed seven-metric protocol')
    annotation = protocol['annotations']['val']
    raw = Path(annotation['path']).read_bytes()
    if hashlib.sha256(raw).hexdigest() != annotation['sha256']:
        raise ValueError('Original validation annotation changed')
    data = json.loads(raw)
    expected = {row['id'] for row in data['images']}
    if len(expected) != 5000:
        raise ValueError('Original validation split does not contain 5,000 unique images')
    del raw, data

    world = len(run['hardware'])
    if not world or {worker['rank'] for worker in run['hardware']} != set(range(world)):
        raise ValueError('Training worker ranks are incomplete')
    expected_files = {f'val_epoch{args.epoch:02d}_rank{rank}.jsonl' for rank in range(world)}
    actual_files = {path.name for path in args.run.glob(f'val_epoch{args.epoch:02d}_rank*.jsonl')}
    if actual_files != expected_files:
        raise ValueError('Missing or extra validation shard')
    seen, samples, files = set(), set(), []
    values = {key: [] for key in METRICS}
    for rank in range(world):
        path = args.run/f'val_epoch{args.epoch:02d}_rank{rank}.jsonl'
        checksum, count = hashlib.sha256(), 0
        with path.open('rb') as stream:
            for raw in stream:
                checksum.update(raw)
                row = json.loads(raw)
                if row['image_id'] not in expected or row['image_id'] in seen or row['sample_id'] in samples:
                    raise ValueError('Unexpected or duplicate validation image')
                seen.add(row['image_id']); samples.add(row['sample_id']); count += 1
                for key in METRICS:
                    value = row['metrics'][key]
                    if not math.isfinite(value) or not -1e-12 <= value <= 1+1e-12:
                        raise ValueError(f'Invalid validation metric: {row["sample_id"]}, {key}')
                    values[key].append(value)
        if count != len(range(rank, 5000, world)):
            raise ValueError('Shard does not cover its expected number of images')
        files.append({'rank': rank, 'path': str(path.resolve()), 'rows': count, 'sha256': checksum.hexdigest()})
    if seen != expected:
        raise ValueError('Validation does not cover the exact original image set')
    means = {key: math.fsum(items)/len(items) for key, items in values.items()}
    difference = max(abs(means[key]-summary['metrics'][key]) for key in METRICS)
    if difference >= 2e-12:
        raise ValueError('Saved summary differs from arithmetic per-image means')
    receipt = {'status': 'passed', 'scope': 'full_validation_integrity_only_not_final_test_comparison',
               'created_utc': datetime.now(timezone.utc).isoformat(), 'family': run['family'],
               'epoch': args.epoch, 'images': len(seen), 'expected_unique_images': len(expected),
               'duplicates': 0, 'missing': 0, 'extra': 0, 'files': files,
               'annotation_sha256': annotation['sha256'], 'protocol_sha256': protocol_hash,
               'summary_sha256': hashlib.sha256(summary_raw).hexdigest(), 'recomputed_metrics': means,
               'maximum_summary_difference': difference, 'test_used': False,
               'audit_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({'status': 'passed', 'family': run['family'], 'epoch': args.epoch,
                      'images': len(seen), 'max_difference': difference,
                      'metrics_percent': {key: round(value*100, 4) for key, value in means.items()},
                      'receipt': str(args.output.resolve())}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
