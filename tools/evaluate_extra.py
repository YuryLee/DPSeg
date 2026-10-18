#!/usr/bin/env python3
"""Evaluate class-agnostic visible-instance metrics from a checkpoint or cached predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rgbd_seg.data import RGBDDataset, dataset_fingerprint
from rgbd_seg.instance_metrics import evaluate_extra_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    origin = parser.add_mutually_exclusive_group(required=True)
    origin.add_argument('--checkpoint')
    origin.add_argument('--predictions', help='Local torch payload with predictions and image_ids, loaded weights_only')
    parser.add_argument('--save-predictions', help='Optional local tensor cache with image IDs and dataset fingerprint')
    parser.add_argument('--split', choices=('train', 'val', 'test'), default='val')
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--mask-threshold', type=float, default=0.5)
    parser.add_argument('--boundary-tolerance', type=float, default=0.003)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    ds = RGBDDataset(args.data, args.split)
    if not len(ds):
        parser.error('Evaluation split is empty')
    image_ids = [record['image_id'] for record in ds.samples]
    fingerprint = dataset_fingerprint(args.data)
    source = Path(args.checkpoint or args.predictions)
    metadata = {'input_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'input_type': 'checkpoint' if args.checkpoint else 'cached_predictions',
                'dataset_sha256': fingerprint, 'split': args.split, 'samples': len(ds),
                'provenance': ds.manifest['provenance'], 'device': args.device}
    if args.predictions:
        cached = torch.load(source, map_location='cpu', weights_only=True)
        if cached.get('image_ids') != image_ids or cached.get('dataset_sha256') != fingerprint:
            raise ValueError('Cached predictions must match dataset fingerprint and ordered image IDs')
        predictions = cached['predictions']
    else:
        from rgbd_seg.model import load_model
        model, checkpoint = load_model(source, args.device)
        if checkpoint['classes'] != ds.classes:
            raise ValueError('Checkpoint and dataset classes differ')
        metadata['model_config'] = checkpoint['config']
        predictions = []
        with torch.inference_mode():
            for index in range(len(ds)):
                sample, _ = ds[index]
                pred = model([{key: value.to(args.device) for key, value in sample.items()}])[0]
                predictions.append({key: value.detach().cpu() for key, value in pred.items()})
    if args.save_predictions:
        target = Path(args.save_predictions)
        if target.exists():
            raise FileExistsError('Refusing to overwrite cached predictions')
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'predictions': predictions, 'image_ids': image_ids, 'dataset_sha256': fingerprint}, target)
    result = evaluate_extra_predictions(ds, predictions, args.score_threshold, args.mask_threshold, args.boundary_tolerance)
    result['run'] = metadata
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps({'output': str(output.resolve()), 'metrics_macro': result['metrics_macro'], 'totals': result['totals']}, indent=2))


if __name__ == '__main__':
    main()
