#!/usr/bin/env python3
"""Bind one completed continuation epoch to full train/validation artifacts.

The full-run auditor remains required after all epochs complete. This read-only
check does not infer live frozen-model integrity from a checkpoint receipt.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha, tensor_digest
from tools.audit_course_point_refinement import inspect_rows
from tools.run_course_point_continuation import parent_inputs
from tools.run_course_point_refinement import VARIANTS, checked, now, read, require


def audit(protocol, root, epoch):
    require(protocol['scope'] == 'full_experiment'
            and protocol['evaluation_policy'] == 'validation_only', 'Expected full validation-only protocol')
    parent_epoch = protocol['initial_head']['checkpoint']['epoch']
    require(parent_epoch < epoch <= parent_epoch + protocol['epochs'], 'Epoch outside continuation')
    require(not (root / 'test').exists(), 'Unexpected test output')
    folder = root / f'val_epoch_{epoch:02d}'
    require((folder / 'comparison.json').exists(), 'Requested epoch validation is incomplete')
    report = read(folder / 'comparison.json')
    history = read(root / 'validation_history.json')
    require(epoch - parent_epoch <= len(history)
            and history[epoch - parent_epoch - 1] == report, 'Validation history binding differs')
    require(report['epoch'] == epoch and report['split'] == 'val'
            and report['images'] == 2584, 'Validation scope differs')
    _, parent_validation = parent_inputs(protocol)
    initial = read(root / 'initial_head_reload.json')
    require(initial['checkpoint_sha256'] == protocol['initial_head']['checkpoint']['sha256']
            and initial['head_tensor_sha256'] == parent_validation['checkpoint']['head_tensor_sha256'],
            'Initial heads differ from validated parent')
    world = len(protocol['gpus'])
    require(protocol['global_batch_size'] == world * protocol['batch_size'], 'Batch layout differs')
    require(protocol['training_images'] == 14432
            and 14432 % protocol['global_batch_size'] == 0, 'Training coverage policy differs')
    steps = 14432 // protocol['global_batch_size']
    with CourseArchive(protocol['archive'], 'train') as data:
        train_ids = data.image_ids
    with CourseArchive(protocol['archive'], 'val') as data:
        val_ids = data.image_ids
    require(len(train_ids) == 14432 and len(val_ids) == 2584, 'Official split sizes differ')
    artifacts = {protocol['_path']: protocol['_sha256']}
    consumed = []
    for rank in range(world):
        coverage = root / f'coverage_epoch{epoch:02d}_rank{rank}.json'
        consumed.extend(read(coverage))
        path = root / f'train_epoch{epoch:02d}_rank{rank}.jsonl'
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        require([row['step'] for row in rows] == list(range(1, steps + 1)), 'Update coverage differs')
        for step, row in enumerate(rows, 1):
            require(row['epoch'] == epoch and row['global_step'] == (epoch - parent_epoch - 1) * steps + step,
                    'Training counters differ')
            require(row['global_instances'] > 0 and all(math.isfinite(row[key])
                    for key in ('point_loss', 'depth_point_loss', 'lr', 'seconds')), 'Invalid training log')
        artifacts[str(path)] = file_sha(path)
        artifacts[str(coverage)] = file_sha(coverage)
    require(len(consumed) == len(set(consumed)) == 14432 and set(consumed) == set(train_ids),
            'Training images missing or duplicated')
    checkpoint = read(root / f'checkpoint_epoch_{epoch:02d}.json')
    require(checkpoint['epoch'] == epoch and file_sha(checkpoint['path']) == checkpoint['sha256'],
            'Checkpoint identity differs')
    require(report['checkpoint'] == checkpoint == read(folder / 'head_checkpoint.json'),
            'Validation checkpoint binding differs')
    values = torch.load(checkpoint['path'], map_location='cpu', weights_only=True)
    require(values['protocol_sha256'] == protocol['_sha256']
            and values['source_sha256'] == protocol['source_sha256']
            and values['point_policy'] == protocol['point_policy']
            and values['base_checkpoint_sha256'] == protocol['weights']['sha256'], 'Head provenance differs')
    digest = tensor_digest(values['heads'])
    require(digest == checkpoint['head_tensor_sha256'] and digest != initial['head_tensor_sha256'],
            'Head digest differs or weights did not change')
    require(all(torch.isfinite(v).all() for v in values['heads'].values()), 'Non-finite head weights')
    require(read(folder / 'image_ids.json') == val_ids, 'Validation image IDs differ')
    inspect_rows(folder, val_ids, artifacts)
    for variant in VARIANTS:
        require(report['metrics'][variant] == read(folder / variant / 'evaluation.json')['metrics'],
                'Reported metrics differ from evaluator')
    artifacts[checkpoint['path']] = checkpoint['sha256']
    artifacts[str(folder / 'comparison.json')] = file_sha(folder / 'comparison.json')
    comparison = {}
    for variant in VARIANTS:
        before, after = parent_validation['metrics'][variant], report['metrics'][variant]
        comparison[variant] = {key: {'parent': before[key], 'continued': after[key],
                                    'delta': after[key] - before[key]}
                               for key in ('mask_ap', 'mask_ap_small')}
    return {'status': 'passed', 'scope': 'completed_epoch_only', 'at_utc': now(),
            'epoch': epoch, 'parent_head_epoch': parent_epoch, 'updates': steps,
            'training_images': len(train_ids), 'validation_images': len(val_ids),
            'protocol_sha256': protocol['_sha256'], 'checkpoint': checkpoint,
            'comparison_to_parent': comparison, 'test_used': False,
            'artifact_sha256': artifacts,
            'limitations': ['This does not prove full-run completion or final checkpoint selection',
                           'Live base-weight and gradient integrity require the final rank receipts',
                           'Whole-image validation does not establish tiled-system performance']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--folder', type=Path, required=True)
    parser.add_argument('--epoch', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    result = audit(checked(args.protocol), args.folder.resolve(), args.epoch)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'scope', 'epoch', 'comparison_to_parent')}))


if __name__ == '__main__':
    main()
