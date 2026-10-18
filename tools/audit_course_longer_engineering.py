#!/usr/bin/env python3
"""Audit the actual two-update engineering run before a base continuation.

Supports the original continuation and its versioned extension. A successful
receipt proves training/resumption mechanics only, never full-validation quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_continuation_state import validated_parent_state
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_longer_training import continuation_lr
from rgbd_seg.course_optimized import file_sha, load_for_inference, tensor_digest
from rgbd_seg.course_submission import _Spec, _predictions


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def check_exit(control, run, protocol, protocol_sha):
    require(run['status'] == control['status'] == 'completed'
            and run['scope'] == control['scope'] == 'engineering_smoke'
            and control['exit_code'] == 0 and not control['active_process_alive'],
            'Engineering requires a completed run and observed successful process exit')
    trials = [v for v in control['trials'] if v['arm'] == protocol['arm']]
    require(len(trials) == 1 and trials[0]['status'] == 'completed'
            and trials[0]['exit_code'] == 0 and trials[0]['protocol_sha256'] == protocol_sha,
            'Engineering supervisor trial differs')
    args = run['arguments']
    require(args['smoke_steps'] == 2 and args['smoke_tail_images'] == 11
            and args['smoke_dense'] and not args['resume'] and args['resolution'] == 1024,
            'Engineering must exercise two dense updates including the 11-image tail')


def check_optimizer(before, after, parent_steps):
    require(before['state'].keys() == after['state'].keys(), 'Optimizer state coverage differs')
    require([v['params'] for v in before['param_groups']] ==
            [v['params'] for v in after['param_groups']], 'Optimizer parameter order differs')
    changed = False
    for key, old in before['state'].items():
        new = after['state'][key]
        require(float(old['step']) == parent_steps and float(new['step']) == parent_steps + 2,
                'AdamW state did not advance from the cumulative parent step by two')
        for name in ('exp_avg', 'exp_avg_sq'):
            require(new[name].shape == old[name].shape and torch.isfinite(new[name]).all(),
                    'AdamW moments are invalid')
            changed |= not torch.equal(new[name], old[name])
    require(changed, 'Optimizer moments did not change during engineering')
    return len(after['state'])


def audit(protocol_path, folder, control):
    p, run, process = read(protocol_path), read(folder / 'run.json'), read(control / 'process.json')
    digest = file_sha(protocol_path)
    require(p['status'] == 'frozen' and p['schema'] in
            ('course_longer_training_v1', 'course_longer_extension_v1'), 'Unsupported engineering protocol')
    check_exit(process, run, p, digest)
    require(run['protocol_sha256'] == digest and run['source_sha256'] == p['source_sha256']
            and not run['test_used'] and run['resolution'] == 1024, 'Engineering provenance differs')
    require(Path(run['arguments']['output']).resolve() == folder.resolve()
            and Path(run['arguments']['protocol']).resolve() == protocol_path.resolve(),
            'Engineering output or protocol path differs')
    for name, expected in p['source_sha256'].items():
        require(file_sha(ROOT / name) == expected, f'Frozen source changed: {name}')
    if p['schema'] == 'course_longer_extension_v1':
        require(bool(p.get('extension_evidence')), 'Extension review evidence is missing')
        for item in p['extension_evidence']:
            require(file_sha(item['path']) == item['sha256'], 'Extension review evidence changed')
    inputs = {'archive': file_sha(p['archive']),
              'pretrained': file_sha(Path(p['checkpoint']) / 'model.safetensors'),
              'data_audit': file_sha(p['data_audit']['path'])}
    require(inputs == p['input_sha256'], 'Engineering input identity differs')
    parent_spec = p['continuation']
    for name in ('weights', 'protocol'):
        item = parent_spec[name]
        require(file_sha(item['path']) == item['sha256'], 'Engineering parent identity differs')
    parent_protocol = read(parent_spec['protocol']['path'])
    parent = torch.load(parent_spec['weights']['path'], map_location='cpu', weights_only=False)
    state = validated_parent_state(parent, parent_protocol, parent_spec['protocol']['sha256'],
                                   run['model_metadata'], parent_spec['parent_epoch'])
    require(state['parent_optimizer_steps'] == parent_spec['parent_optimizer_steps'] == run['parent_global_step']
            and state['selection'] == run['parent_validation'], 'Engineering parent counters differ')
    initial = tensor_digest(parent['model'])
    require(initial == state['selection']['model_sha256'] == run['initial_model_sha256'],
            'Engineering did not restore the validated parent model')

    train, world = p['training'], len(p['hardware']['physical_gpus'])
    require(world == run['world_size'] == 5 and train['effective_batch_size'] == run['effective_batch_size'] == 33
            and train['accumulation'] == 3 and train['per_rank_batch_sizes'] == [3, 2, 2, 2, 2],
            'Engineering hardware or batch contract differs')
    require([v['rank'] for v in run['ranks']] == list(range(world))
            and [str(v['physical_gpu']) for v in run['ranks']] == list(map(str, p['hardware']['physical_gpus'])),
            'Engineering rank devices differ')
    require(run['optimizer_steps'] == run['expected_optimizer_steps'] == 2
            and run['training_image_exposures'] == run['expected_training_image_exposures'] == 44,
            'Engineering training counts differ')
    epoch = parent_spec['parent_epoch'] + 1
    selection = read(folder / 'selection.json')
    require(not selection['test_used'] and len(selection['epochs']) == 1
            and selection['parent_candidate'] == state['selection'], 'Engineering selection scope differs')
    entry = selection['epochs'][0]
    require(entry['epoch'] == epoch and entry['global_step'] == 2 and entry['training_images'] == 44,
            'Engineering epoch counts differ')
    with CourseArchive(p['archive'], 'train') as dataset:
        train_ids = dataset.image_ids
        counts = Counter(v['image_id'] for v in dataset.coco_ground_truth()['annotations'])
    dense_ids = sorted(train_ids, key=lambda i: (-counts[i], i))[:44]
    order = read(folder / f'train_order_epoch_{epoch:02d}.json')
    require(order['epoch'] == epoch and order['image_ids'] == dense_ids,
            'Engineering did not use the declared dense-image sample')
    partitions = order['rank_batches']
    require(len(partitions) == world and all(len(v) == 4 for v in partitions), 'Engineering microbatch coverage differs')
    recovered = []
    for micro in range(4):
        for rank in range(world):
            indices = partitions[rank][micro]
            require(len(indices) == train['per_rank_batch_sizes'][rank], 'Engineering rank batch size differs')
            recovered.extend(train_ids[i] for i in indices)
    require(recovered == dense_ids and len(set(recovered)) == 44, 'Engineering training images repeated or omitted')
    reference = None
    for rank in range(world):
        log = rows(folder / f'train_rank{rank}.jsonl')
        require(len(log) == 2, 'Engineering rank update count differs')
        for step, row in enumerate(log, 1):
            require(row['epoch'] == epoch and row['global_step'] == step and row['total_steps'] == 2
                    and row['total_global_step'] == state['parent_optimizer_steps'] + step
                    and row['images_in_epoch'] == [33, 44][step-1]
                    and row['global_batch_images'] == [33, 11][step-1] and row['expected_train_images'] == 44,
                    'Engineering rank counters differ')
            require(all(math.isfinite(row[k]) for k in ('loss', 'local_loss', 'grad_norm_before_clip', 'head_lr')),
                    'Non-finite engineering update')
            schedule = train['schedule']
            lr = continuation_lr(step-1, train['total_updates'], peak_lr=train['lr'],
                                 start_lr=schedule['start_lr'], warmup_steps=schedule['warmup_steps'],
                                 minimum_ratio=schedule['minimum_ratio'])
            require(math.isclose(row['head_lr'], lr, rel_tol=1e-12), 'Engineering learning rate differs')
        shared = [(v['loss'], v['head_lr']) for v in log]
        if reference is None:
            reference = shared
            require(math.isclose(sum(v['loss'] * v['global_batch_images'] for v in log) / 44,
                                 entry['training_loss_mean'], rel_tol=1e-12), 'Engineering loss aggregation differs')
        else:
            require(shared == reference, 'Engineering global loss disagrees across ranks')
    last = torch.load(folder / 'last.pt', map_location='cpu', weights_only=False)
    require(last['protocol_sha256'] == digest and last['source_sha256'] == p['source_sha256']
            and last['model_metadata'] == run['model_metadata'], 'Engineering last checkpoint provenance differs')
    saved = last['state']
    require(saved['epoch'] == 1 and saved['global_step'] == 2 and saved['images_in_epoch'] == 0
            and saved['stage'] == 'training' and saved['val_results'] == [entry]
            and len(last['rank_rng']) == world, 'Engineering resumable state differs')
    trained = tensor_digest(last['model'])
    require(trained == entry['model_sha256'] and trained != initial, 'Engineering model did not update')
    require(read(folder / f'model_sync_epoch_{epoch:02d}.json') ==
            {'status': 'pass', 'world_size': world, 'model_sha256': trained}, 'Engineering model synchronization differs')
    optimizer_count = check_optimizer(parent['optimizer'], last['optimizer'], state['parent_optimizer_steps'])
    del parent, last
    adapter, reload_receipt = load_for_inference(folder / f'epoch_{epoch:02d}.pt', protocol_path)
    require(reload_receipt['model_sha256'] == trained and reload_receipt['epoch'] == epoch,
            'Scored checkpoint cannot be reconstructed for inference')
    del adapter

    val = Path(entry['output'])
    require(val.resolve().parent == folder.resolve(), 'Engineering validation is outside the run folder')
    evaluation, receipt = read(val / 'evaluation.json'), read(val / 'evaluation_run.json')
    require(evaluation['status'] == receipt['status'] == 'completed'
            and evaluation['scope'] == receipt['scope'] == 'engineering_smoke'
            and evaluation['split'] == receipt['split'] == 'val'
            and evaluation['images'] == receipt['images'] == 10 and evaluation['full_split_images'] == 2584
            and not receipt['ground_truth_in_inference'] and receipt['world_size'] == world
            and receipt['inference_precision'] == 'fp32' and evaluation['metrics'] == entry['metrics'],
            'Engineering validation scope or scores differ')
    with CourseArchive(p['archive'], 'val') as dataset:
        ids = dataset.image_ids[:10]
    gathered = {}
    for rank in range(world):
        path = val / f'predictions_rank{rank}.jsonl'
        rank_receipt, records = read(val / f'rank{rank}.json'), rows(path)
        require(rank_receipt['status'] == 'completed' and rank_receipt['rank'] == rank
                and rank_receipt['images'] == len(records) == 2
                and rank_receipt['ordered_ids'] == [v['image_id'] for v in records] == ids[rank::world]
                and rank_receipt['prediction_sha256'] == file_sha(path), 'Engineering validation shard differs')
        gathered.update({v['image_id']: v for v in records})
    merged = rows(val / 'predictions.jsonl')
    require(merged == [gathered[i] for i in ids], 'Engineering merged predictions differ from rank output')
    spec, instances = _Spec(ids, (1,), (1024, 1024), 100), 0
    for row in merged:
        _predictions(row['predictions'], spec, decode=True)
        instances += len(row['predictions'])
    require(instances == evaluation['predicted_instances'] == receipt['predicted_instances']
            and file_sha(val / 'predictions.jsonl') == evaluation['predictions_sha256'] == receipt['predictions_sha256']
            and file_sha(val / 'evaluation.json') == receipt['evaluation_sha256'], 'Engineering scored predictions differ')
    return {'status': 'passed', 'scope': 'engineering_only_not_model_quality',
            'at_utc': datetime.now(timezone.utc).isoformat(), 'auditor_sha256': file_sha(__file__),
            'source_sha256': p['source_sha256'], 'protocol_sha256': digest,
            'parent_checkpoint_sha256': parent_spec['weights']['sha256'],
            'parent_protocol_sha256': parent_spec['protocol']['sha256'],
            'run_sha256': file_sha(folder / 'run.json'), 'exit_sha256': file_sha(control / 'process.json'),
            'input_sha256': inputs, 'training_contract': train, 'hardware_contract': p['hardware'],
            'parent_tensor_identity': initial, 'trained_tensor_identity': trained,
            'parent_optimizer_steps': state['parent_optimizer_steps'], 'optimizer_states_checked': optimizer_count,
            'adamw_step_increment': 2, 'train_images': 44, 'effective_batches': [33, 11],
            'gpus_synchronized': world, 'validation_images': 10, 'validated_rle_instances': instances,
            'inference_reload': reload_receipt, 'test_used': False,
            'limitations': ['Ten validation images verify execution only; their scores cannot select a model',
                            'An extension requires its own exact-parent engineering run and receipt']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('protocol', 'folder', 'control', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Engineering audit output already exists')
    torch.set_num_threads(4)
    result = audit(args.protocol, args.folder, args.control)
    write_json(args.output, result)
    print(json.dumps({k: result[k] for k in ('status', 'scope', 'optimizer_states_checked',
                     'train_images', 'validation_images', 'validated_rle_instances')}))


if __name__ == '__main__':
    main()
