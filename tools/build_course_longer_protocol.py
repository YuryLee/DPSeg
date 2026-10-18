#!/usr/bin/env python3
"""Freeze two validation-selected learning-rate continuation arms."""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_course_longer_training import sources
from tools.run_course_dafusion import now, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--engineering-receipt', type=Path)
    args = parser.parse_args()
    parent_path = ROOT/'reports/course_resolution_ddp/protocol.json'
    parent = json.loads(parent_path.read_text())
    last = ROOT/'reports/course_resolution_ddp/formal_v1/1024/last.pt'
    base = copy.deepcopy(parent)
    for key in ('execution_order', 'comparison', 'adoption_gate', 'next_stages',
                'engineering_validation', 'final_candidate_comparison'):
        base.pop(key, None)
    base.update(schema='course_longer_training_v1', created_at_utc=now(), frozen_at_utc=now(),
                status='frozen', source_sha256=sources(), selected_resolution=1024)
    base['material_passport'] = {'origin_skill': 'academic-research-suite/experiment-agent',
        'mode': 'run', 'scope': 'User-authorized longer training and best validation metric search',
        'verification_status': 'FORMAL_PENDING' if args.engineering_receipt else 'ENGINEERING_PENDING'}
    base['hardware'] = {'physical_gpus': ['2','3','5','6','7'], 'memory_fraction': .9,
                       'excluded_gpus': [0,1,4], 'reason': 'Other jobs use GPUs0/1; GPU4 serves the demo'}
    base['continuation'] = {'parent_epoch': 5, 'parent_optimizer_steps': 2190,
        'weights': {'path': str(last), 'sha256': sha256(last)},
        'protocol': {'path': str(parent_path), 'sha256': sha256(parent_path)},
        'optimizer': 'Restore all AdamW moments and per-parameter step counters; reset only LR schedule',
        'rng': 'Reseed each of the five ranks with seed+rank; equal between arms, not exact seven-rank RNG continuation',
        'data_order': 'Continue original deterministic epoch order and hflip at epoch index 5',
        'initial_candidate': 'Retain original e05 if all continuation epochs score worse'}
    base['training'].update(epochs=10, per_rank_batch_sizes=[3,2,2,2,2], accumulation=3,
        effective_batch_size=33, updates_per_epoch=438, total_updates=4380,
        total_image_exposures=144320, save_every=200,
        tail_batch='1312 microbatches of 11 images, grouped by three; final update has 11 images; no duplicates/drop')
    base['evaluation']['test_policy'] = 'No test/Kaggle selection; lock best full-validation checkpoint before test'
    base['limits'] = {'per_trial_seconds': 86400, 'external_trial_seconds': 87000}
    base['limitations'] = ['Single seed; no statistical significance claim',
        'Five GPUs with gradient accumulation change microbatch mask normalization and RNG from the seven-GPU parent',
        'Both LR arms have identical batch layout, initialization, optimizer state, data order and budget',
        'Historical local test/public scores are known; all new checkpoint selection uses full validation',
        'Best denotes the best evaluated candidate; no claim of a global optimum']
    base['next_stages'] = ['Select the best native full-validation APs checkpoint across both arms and e05',
        'Inspect learning curves; extend if the best is at the end and still improving',
        'Train point refinement against the selected base and compare full-validation tiled combinations',
        'Lock configuration, run local test once, validate and name submission CSV']
    if args.engineering_receipt:
        receipt = json.loads(args.engineering_receipt.read_text())
        if receipt.get('status') != 'passed' or receipt.get('source_sha256') != base['source_sha256']:
            raise ValueError('Engineering gate is missing or bound to different source')
        base['engineering_validation'] = {'path': str(args.engineering_receipt.resolve()),
                                         'sha256': sha256(args.engineering_receipt)}
    args.output.mkdir(parents=True, exist_ok=False)
    for arm, rate, warmup in [('lr5e5', 5e-5, 100), ('lr2e5', 2e-5, 0)]:
        protocol = copy.deepcopy(base)
        protocol['arm'] = arm
        protocol['training']['lr'] = rate
        protocol['training']['schedule'] = {'start_lr': 2e-5, 'warmup_steps': warmup,
            'minimum_ratio': .1, 'kind': 'linear warmup from parent terminal LR, then cosine'}
        path = args.output/f'{arm}.json'
        write_json(path, protocol)
        print(path.resolve())


if __name__ == '__main__':
    main()
