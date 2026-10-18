#!/usr/bin/env python3
"""Bounded development-only diagnostic of removing depth from fixed trained models."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from rgbd_seg.data import RGBDDataset, dataset_fingerprint
from rgbd_seg.evaluation import evaluate_predictions
from rgbd_seg.model import load_model


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default='data/graspnet_expanded_v2')
    parser.add_argument('--models', default='outputs/graspnet_v2')
    parser.add_argument('--output', default='reports/v2/depth_diagnostic.json')
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    torch.set_num_threads(2)
    dataset = RGBDDataset(args.data, 'val')
    if len(dataset) != 4:
        raise ValueError('This bounded diagnostic requires exactly four development frames')
    result = {
        'scope': 'Fixed final weights; four development frames only; no retraining, method selection or latency measurement.',
        'intervention': 'Set both depth and depth_valid tensors to zero; retain identical RGB, weights and evaluation settings.',
        'interpretation': 'Tests prediction dependence on depth input and a missing-depth failure mode. It does not isolate the causal benefit of depth training, initialization or architecture; missing depth can be out of distribution.',
        'dataset_sha256': dataset_fingerprint(args.data),
        'split': 'val',
        'samples': [sample['id'] for sample in dataset.samples],
        'device': args.device,
        'score_difference_definition': 'Per-frame mean absolute difference between descending score vectors padded with zeros to their common maximum length. Scores are rank-aligned, not object-matched; this statistic measures output distribution change only.',
        'methods': {},
    }
    for mode in ('E2', 'E3'):
        checkpoint_path = Path(args.models) / mode / 'model.pt'
        model, checkpoint = load_model(checkpoint_path, args.device)
        if checkpoint['classes'] != dataset.classes or checkpoint['config']['mode'] != mode:
            raise ValueError('Unexpected checkpoint classes or mode')
        model.eval()
        outputs = {'normal': [], 'missing_depth': []}
        frame_diagnostics = []
        with torch.inference_mode():
            for index in range(len(dataset)):
                sample, _ = dataset[index]
                for condition in outputs:
                    inputs = {key: value.to(args.device) for key, value in sample.items()}
                    if condition == 'missing_depth':
                        inputs['depth'] = torch.zeros_like(inputs['depth'])
                        inputs['depth_valid'] = torch.zeros_like(inputs['depth_valid'])
                    prediction = model([inputs])[0]
                    # COCO uses mask>=0.5. Persisting only binary masks reduces CPU memory
                    # while preserving exactly the mask supplied to COCO evaluation.
                    outputs[condition].append({key: ((value >= 0.5).cpu() if key == 'masks' else value.cpu()) for key, value in prediction.items()})
                    del inputs, prediction
                normal = outputs['normal'][-1]['scores'].sort(descending=True).values
                missing = outputs['missing_depth'][-1]['scores'].sort(descending=True).values
                count = max(len(normal), len(missing))
                padded_normal, padded_missing = torch.zeros(count), torch.zeros(count)
                padded_normal[:len(normal)] = normal
                padded_missing[:len(missing)] = missing
                frame_diagnostics.append({
                    'sample_id': dataset.samples[index]['id'],
                    'normal_candidates': len(normal), 'missing_depth_candidates': len(missing),
                    'normal_candidates_score_ge05': int((normal >= 0.5).sum()),
                    'missing_depth_candidates_score_ge05': int((missing >= 0.5).sum()),
                    'rank_aligned_score_mae': float((padded_normal-padded_missing).abs().mean()) if count else 0.0,
                })
        metrics = {condition: evaluate_predictions(dataset, predictions) for condition, predictions in outputs.items()}
        reference_path = Path(args.models) / mode / 'evaluation.json'
        reference = json.loads(reference_path.read_text())
        reference_delta = {key: metrics['normal'][key]-value for key, value in reference['metrics'].items() if value is not None}
        result['methods'][mode] = {
            'checkpoint_sha256': file_sha256(checkpoint_path),
            'config': checkpoint['config'],
            'metrics': metrics,
            'missing_minus_normal_pp': {key: metrics['missing_depth'][key]-value for key, value in metrics['normal'].items() if value is not None},
            'normal_vs_recorded_metrics_max_abs_delta_pp': max(abs(value) for value in reference_delta.values()),
            'recorded_checkpoint_matches': reference['checkpoint_sha256'] == file_sha256(checkpoint_path),
            'per_frame': frame_diagnostics,
            'mean_frame_rank_aligned_score_mae': sum(frame['rank_aligned_score_mae'] for frame in frame_diagnostics)/len(frame_diagnostics),
        }
        print(json.dumps({'mode': mode, 'normal': metrics['normal']['mask_ap'], 'missing': metrics['missing_depth']['mask_ap']}), flush=True)
        del outputs, model, checkpoint
        gc.collect()
        if args.device.startswith('cuda'):
            torch.cuda.empty_cache()
    target = Path(args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(str(target.resolve()))


if __name__ == '__main__':
    main()
