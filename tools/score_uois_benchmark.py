#!/usr/bin/env python3
"""Score saved native-resolution RLE predictions with pinned UOIS metrics.

Each JSONL row has {"image_id": <manifest sample id>, "predictions": [...]}.
An empty prediction is an explicit [] row. Missing/repeated/extra images fail.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.uois_benchmark_data import BenchmarkDataset, file_sha256


def read_predictions(path, image_ids):
    records = {}
    with Path(path).open() as stream:
        for line_number, line in enumerate(stream, 1):
            value = json.loads(line)
            if set(value) != {'image_id', 'predictions'} or not isinstance(value['predictions'], list):
                raise ValueError(f'Invalid prediction schema on line {line_number}')
            key = value['image_id']
            if not isinstance(key, str) or key in records:
                raise ValueError(f'Invalid or duplicate image ID on line {line_number}')
            records[key] = value['predictions']
    if set(records) != set(image_ids):
        missing, extra = set(image_ids) - set(records), set(records) - set(image_ids)
        raise ValueError(f'Prediction coverage mismatch: missing={len(missing)}, extra={len(extra)}')
    return records


def json_safe(value):
    """Preserve reference undefined metrics as null; never coerce them to zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(json_safe(value), stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--score-threshold', type=float, default=.5)
    parser.add_argument('--model-description', required=True,
                        help='State model, training dataset and whether this is a reproduction or transfer test')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output already exists; use a new directory')
    if not 0 <= args.score_threshold <= 1:
        parser.error('score-threshold must be in [0,1]')
    from rgbd_seg.uois_benchmark_metrics import (
        evaluate_label_maps, predictions_to_label_map, macro_average, reference_provenance,
    )
    data = BenchmarkDataset(args.manifest)
    predictions = read_predictions(args.predictions, [r['id'] for r in data.records])
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'status': 'scoring', 'dataset': data.manifest['dataset'], 'images': len(data),
        'model_description': args.model_description,
        'manifest': str(args.manifest.resolve()), 'manifest_sha256': file_sha256(args.manifest),
        'predictions': str(args.predictions.resolve()), 'predictions_sha256': file_sha256(args.predictions),
        'score_threshold': args.score_threshold, 'score_comparison': 'strictly greater than',
        'prediction_overlap': 'input list order, later retained masks overwrite earlier masks',
        'annotation': data.manifest['annotation_policy'],
        'evaluation_resolution': 'native annotation resolution, no GT resizing',
        'aggregation': 'arithmetic per-image mean including empty scenes',
        'units': 'fractions 0–1; display as percent by multiplying by 100',
        'calibration': data.calibration,
        'reference': reference_provenance(),
        'undefined_metrics': 'original undefined values saved as null; no nanmean or zero substitution',
        'sources_sha256': {str(p.relative_to(ROOT)): file_sha256(p) for p in (
            Path(__file__), ROOT / 'rgbd_seg/uois_benchmark_data.py',
            ROOT / 'rgbd_seg/uois_benchmark_metrics.py',
            ROOT / 'third_party/DA-Fusion/dafusion/dafusion/eval/compute_PRF.py')},
    }
    write_json(args.output / 'protocol.json', protocol)
    rows = []
    with (args.output / 'per_image.jsonl').open('x') as stream:
        for index, record in enumerate(data.records):
            truth = data.ground_truth(index)
            prediction = predictions_to_label_map(predictions[record['id']], truth.shape,
                                                  score_threshold=args.score_threshold)
            result = evaluate_label_maps(prediction, truth)
            result['image_id'] = record['id']
            rows.append(result)
            stream.write(json.dumps(json_safe(result), ensure_ascii=False, allow_nan=False) + '\n')
    summary = macro_average(rows)
    write_json(args.output / 'evaluation.json', {
        'status': 'completed', 'images': len(rows), 'summary': summary,
        'protocol_sha256': file_sha256(args.output / 'protocol.json'),
        'per_image_sha256': file_sha256(args.output / 'per_image.jsonl'),
    })
    print(json.dumps(json_safe(summary), ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
