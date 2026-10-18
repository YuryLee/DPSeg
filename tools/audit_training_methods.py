#!/usr/bin/env python3
"""CPU-only integrity audit of the frozen 12-train/28-evaluation v5 matrix.

This checks provenance and trace consistency, not performance gates or significance.
Run after every formal artifact is complete. No model forward or GPU is used.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

SEEDS = (17, 23, 41)
VARIANTS = ('baseline', 'bce2', 'bace', 'freqfusion')
PROTOCOL_SHA = 'f13bde8a0764dd747de51aa28cb12e91079355d6b97e0fec7925659cfc4fd8a8'
FROZEN_SOURCES = {
    'tools/run_v5.py': 'f4b39b470ee66d02c7e277b056bb759f124939f84adbd939bec507d4749dba35',
    'tools/run.py': '4e56847b62c469aa57894629b92bdd62ff55614bd7f6484b75a01b9b48620eec',
    'tools/evaluate_multiscale.py': '255083bf2632394f922d00ad8335bf15ff3c92175f8edea542acb1b0768a41e0',
    'rgbd_seg/v5_model.py': '5e9c4ed7048eee0ae36fdecd82dd610ef32cd1c146ce76dc0a54eb2fc2e4f8ca',
    'rgbd_seg/model.py': '7e5161ef723a0d2e61384dd8a1ec1d7b40b1e4385b553c5454b73d7b5260c4ff',
    'rgbd_seg/pretrained.py': '98798e7066a21ec68b0f6e334184cb003fdc3976286ab37f236385989ac86f60',
    'rgbd_seg/boundary_loss.py': '26777d6e2e1aa74cb485e3f9cce8c72987b17e1e25159ae385b4253dc2e252a4',
    'rgbd_seg/freqfusion_fpn.py': 'b0fa7f15877e7ad3e5e5d18212f743c7828c74cfc304eb97f33897e03e295a16',
    'rgbd_seg/augment.py': '63849b5082d72b47642351e17753d9fd964576aaec665bf5d42b5db707639401',
    'rgbd_seg/data.py': '839f427b5652da4e5fe24f2bd793aa6c89a27146c3a7922be198444210ded1a4',
    'rgbd_seg/evaluation.py': 'd6687b4229f5404ca858818be3f28e21575f17941243a7646316a7f40b56c9bc',
    'rgbd_seg/multiscale_inference.py': '91a677aae5b1a7dda54e1fc84ef97daf35b23499c29552d7c0e39f3bcf428409',
}
DATA = {
    'graspnet': {'folder': 'data/graspnet_expanded_v2', 'depth': .001,
                 'sha256': 'eba3403a12e724c19fd21124d2bd06156eadf1dadb5ca84d03ea579392d5fa77',
                 'manifest': '0bcbcc606823781304a7c20f45e18461c36d586d5ba7416e76905ff0a55e78ea',
                 'provenance': 'public_graspnet_subset', 'splits': {'train': 64, 'val': 4, 'test': 24}},
    'tless': {'folder': 'data/tless_confirmation', 'depth': .0001,
              'sha256': '760f165383f3f5d763a83c3dc8fb73d52ba44b18b3ddc4bdf1e33b4f56f83df5',
              'manifest': '9c9f7725c12b8df9d39362ed62878586ce35ee4b30802cab313eff56f363ef42',
              'provenance': 'public_tless_subset', 'splits': {'test': 100}},
}
FIXED_TRAINING = {'batch_size': 2, 'lr': .0002, 'backbone_lr_factor': .1,
                  'weight_decay': .0001, 'warmup_steps': 100, 'cosine_decay': True,
                  'crop_probability': .35, 'gradient_clip_norm': 5.,
                  'foreground_mean_init': False, 'cache_data': True}
LOSS_KEYS = {'loss_classifier', 'loss_box_reg', 'loss_mask', 'loss_objectness', 'loss_rpn_box_reg'}
METRIC_KEYS = {'mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ap_small', 'mask_ap_medium', 'mask_ap_large',
               'mask_ar_1', 'mask_ar_10', 'mask_ar_100', 'mask_ar_small', 'mask_ar_medium', 'mask_ar_large'}
TRACE_KEYS = ('batch_indices', 'sample_ids', 'augmentation_rng_before_sha256', 'augmentation_rng_after_sha256')
_HASHES = {}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sha(path):
    path = Path(path).resolve()
    st = path.stat()
    stamp = (st.st_size, st.st_mtime_ns, st.st_ino)
    if path in _HASHES:
        require(_HASHES[path][0] == stamp, f'File changed during audit: {path}')
        return _HASHES[path][1]
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    result = digest.hexdigest()
    _HASHES[path] = (stamp, result)
    return result


def read_json(path):
    return json.loads(Path(path).read_text(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def read_lines(path):
    return [json.loads(line, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
            for line in Path(path).read_text().splitlines()]


def state_digest(state):
    return hashlib.sha256(repr(state).encode('ascii')).hexdigest()


def expected_schedule(step):
    return min(1., (step + 1) / 100) * (.1 + .9 * (1 + math.cos(math.pi * step / 800)) / 2)


def replay_sampling(seed, records, target_sizes, steps=800):
    """Replay RNG calls from actual GT counts; do not construct augmented images.

    target_sizes contains (instance count, count with area<96²). The frozen
    augmentation's branch/choice counts depend on these, not model predictions.
    The source hash protects this explicitly bounded replay contract.
    """
    sampling, augmentation = random.Random(seed), random.Random(seed + 1000)
    rows = []
    for step in range(steps):
        indices = [sampling.randrange(len(records)) for _ in range(2)]
        before = state_digest(augmentation.getstate())
        for index in indices:
            objects, small_medium = target_sizes[index]
            if objects and augmentation.random() < .35:
                augmentation.choice(range(small_medium or objects))
                augmentation.uniform(.5, .8)
                augmentation.uniform(.5, .8)
                augmentation.uniform(-.1, .1)
                augmentation.uniform(-.1, .1)
            augmentation.random()  # horizontal flip
            for _ in range(3):
                augmentation.uniform(.85, 1.15)
        rows.append({'step': step + 1, 'batch_indices': indices,
                     'sample_ids': [records[i]['id'] for i in indices],
                     'augmentation_rng_before_sha256': before,
                     'augmentation_rng_after_sha256': state_digest(augmentation.getstate())})
    return rows, sampling.getstate(), augmentation.getstate()


def audit_loss_trace(rows, sampling_rows, expected_rows, variant, steps=800):
    require(len(rows) == len(sampling_rows) == len(expected_rows) == steps, 'Incomplete/extra step rows')
    keys = LOSS_KEYS | ({'loss_mask_aux'} if variant in ('bce2', 'bace') else set())
    previous_time = -1.
    for index, (row, trace, expected) in enumerate(zip(rows, sampling_rows, expected_rows)):
        step = index + 1
        require(row['step'] == step and trace == expected, f'Sampling/augmentation mismatch at {step}')
        require({key: row[key] for key in ('step', *TRACE_KEYS)} == trace, f'Loss/trace mismatch at {step}')
        require({key for key in row if key.startswith('loss_')} == keys, f'Loss keys differ at {step}')
        require(all(finite(row[k]) for k in ('loss', 'grad_norm', 'elapsed_s', *keys)), f'Nonfinite scalar at {step}')
        require(row['grad_norm'] >= 0 and row['elapsed_s'] >= 0, f'Invalid norm/time at {step}')
        require(math.isclose(row['loss'], math.fsum(row[k] for k in keys), rel_tol=2e-6, abs_tol=1e-6), f'Total loss mismatch at {step}')
        require(row['elapsed_s'] >= previous_time, 'Nonmonotonic elapsed time')
        previous_time = row['elapsed_s']
        if variant == 'bce2':
            require(row['loss_mask_aux'] == row['loss_mask'], f'BCE2 auxiliary differs at {step}')
        for field, position in (('lr_used', index), ('lr_next', step)):
            require(len(row[field]) == 2, 'Expected exactly two optimizer groups')
            for actual, base in zip(row[field], (.00002, .0002)):
                require(finite(actual) and math.isclose(actual, base * expected_schedule(position), rel_tol=1e-12, abs_tol=1e-16), f'{field} mismatch at {step}')
    return {'steps': steps, 'all_losses_and_grad_norm_finite': True, 'schedule_verified': True,
            'sampling_and_augmentation_rng_replayed': True,
            'bce2_auxiliary_exactly_equals_mask_every_step': True if variant == 'bce2' else None}


def shared_initialization(model):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith('core.backbone.rgb.fpn.fusions.'):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(value.dtype), list(value.shape)], separators=(',', ':')).encode())
        digest.update(value.numpy().tobytes())
        count += 1
    return digest.hexdigest(), count


def audit_runtime(receipt):
    require(receipt['cuda_visible_devices'] == '5' and receipt['device'] == 'cuda:0', 'Expected physical GPU5 as logical cuda:0')
    require('A100' in receipt['device_name'] and '40GB' in receipt['device_name'], 'Unexpected GPU model')
    require(receipt['threads'] == 4 and receipt['cuda_memory_fraction'] == .55, 'Runtime resource settings differ')


def audit_supervisor(path, expected_train, expected_eval):
    """Check actual process exits/logs in addition to runner-owned completed flags."""
    ledger = read_json(path)
    require(ledger['status'] == 'completed', 'Supervisor matrix is not completed')
    require(ledger['source_sha256'] == FROZEN_SOURCES and ledger['protocol_sha256'] == PROTOCOL_SHA, 'Supervisor freeze differs')
    require(ledger['physical_gpu'] == 5 and ledger['expected_trainings'] == 12 and ledger['expected_evaluations'] == 28, 'Supervisor matrix specification differs')
    expected = {output.resolve(): 'train' for _, _, output in expected_train}
    expected.update({output.resolve(): 'eval' for *_, output in expected_eval})
    jobs = ledger['jobs']
    require(len(jobs) == 40 and len({job['id'] for job in jobs}) == 40, 'Missing/duplicate supervisor jobs')
    seen, checked = set(), []
    for job in jobs:
        output = Path(job['output'])
        output = (ROOT / output).resolve() if not output.is_absolute() else output.resolve()
        require(output in expected and output not in seen and expected[output] == job['kind'], 'Unexpected/duplicate job output')
        seen.add(output)
        require(job['status'] == 'completed' and job['returncode'] == 0, f"Unsuccessful process: {job['id']}")
        timeout = 1800 if job['kind'] == 'train' else 900
        require(job['timeout_s'] == timeout and finite(job['duration_s']) and 0 < job['duration_s'] <= timeout + 10, 'Invalid timeout/duration receipt')
        require(isinstance(job['pid'], int) and job['pid'] > 0, 'Invalid process identifier')
        start, end, alive = (datetime.fromisoformat(job[key]) for key in ('started_at_utc', 'ended_at_utc', 'last_alive_at_utc'))
        require(start <= alive <= end, 'Supervisor process timestamps out of order')
        command = job['command']
        require(Path(command[1]).name == 'run_v5.py' and command[2] == job['kind'], 'Wrong supervisor entrypoint')
        receipt_path = output / ('run.json' if job['kind'] == 'train' else 'evaluation.json')
        receipt = read_json(receipt_path)
        require(command[1:] == receipt['command'], 'Supervisor command differs from executed command')
        log = Path(job['log'])
        # log_bytes is sampled while the process is alive. Quiet evaluations
        # often print their first line only when finished, after the last poll.
        require(log.stat().st_size > 0 and log.stat().st_size >= job['log_bytes'] >= 0, 'Supervisor log bytes missing')
        structured = []
        for line in log.read_text().splitlines():
            if line.startswith('{'):
                try:
                    structured.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        require(structured and Path(structured[-1].get('output', '')).resolve() == output, 'Final completion record missing from process log')
        final = structured[-1]
        if job['kind'] == 'train':
            require(final.get('exit_status') == 'completed' and final.get('phase') == 'formal' and final.get('steps') == 800 and final.get('variant') == receipt['variant'], 'Training completion log mismatch')
            losses = read_lines(output / 'loss.jsonl')
            displayed = [row for row in structured if 'step' in row]
            require([row['step'] for row in displayed] == [1, *range(25, 801, 25)], 'Missing progress records')
            require(all(row == losses[row['step'] - 1] for row in displayed), 'Process log loss differs from saved trace')
        else:
            require(final.get('metrics') == receipt['metrics'], 'Process log metrics differ from evaluation artifact')
        checked.append({'id': job['id'], 'returncode': 0, 'timeout_s': timeout,
                        'duration_s': job['duration_s'], 'log_sha256': sha(log)})
    require(seen == set(expected), 'Supervisor omitted matrix entries')
    return {'status': 'pass', 'ledger_sha256': sha(path), 'jobs': checked,
            'duration_allowance_note': 'At most ten seconds allowed for supervisor polling/reporting delay; all exits must be successful.'}


def audit_train(path, seed, variant, expected_rows, rng_states):
    import torch
    from rgbd_seg.v5_model import build_v5_model, optimizer_groups
    run = read_json(path / 'run.json')
    require(run['exit_status'] == 'completed' and run['phase'] == 'formal', 'Run not formally completed')
    require(run['seed'] == seed and run['variant'] == variant, 'Seed/variant/path mismatch')
    require(run['steps'] == run['completed_steps'] == 800 and not run['resume_supported'], 'Expected complete non-resumed 800 steps')
    require(run['fixed_training'] == FIXED_TRAINING, 'Training constants differ')
    require(run['source_sha256'] == FROZEN_SOURCES and run['protocol']['sha256'] == PROTOCOL_SHA, 'Unfrozen training source/protocol')
    require(sha(run['protocol']['path']) == PROTOCOL_SHA, 'Protocol bytes changed')
    spec = DATA['graspnet']
    require(Path(run['data_root']).resolve() == (ROOT / spec['folder']).resolve(), 'Wrong training root')
    require(run['dataset_sha256'] == spec['sha256'] and run['manifest_sha256'] == spec['manifest'], 'Training dataset receipt mismatch')
    require(run['depth_scale'] == .001 and run['training_samples'] == 64 and run['classes'] == ['object'], 'Training samples/classes/depth differ')
    require(run['provenance'] == spec['provenance'], 'Unexpected training provenance')
    audit_runtime(run)
    artifacts = {'checkpoint_sha256': 'model.pt', 'weights_sha256': 'weights.pt',
                 'training_state_sha256': 'training_state.pt', 'loss_trace_sha256': 'loss.jsonl',
                 'sampling_trace_sha256': 'sampling.jsonl'}
    actual_hashes = {name: sha(path / name) for name in artifacts.values()}
    for key, name in artifacts.items():
        require(run[key] == actual_hashes[name], f'Artifact SHA mismatch: {name}')
    require(run['weight_bytes'] == (path / 'weights.pt').stat().st_size, 'Weights byte count differs')
    for key in ('config_file', 'init_checkpoint'):
        require(sha(run[key]['path']) == run[key]['sha256'], f'{key} bytes differ')
    config = read_json(run['config_file']['path'])
    require(config['seed'] == seed, 'Config file seed differs')
    expected_config = {'mode': 'E2', 'arch': 'r50_v2', 'size': 720,
                       'seed': seed, 'max_depth_m': 5., **config, 'num_classes': 2}
    require(run['base_config'] == expected_config, 'Normalized config differs')
    expected_settings = {'mode': 'E2', 'arch': 'r50_v2', 'size': 720, 'max_size': 1280,
                         'train_scales': [480, 640, 720], 'max_depth_m': 5., 'frozen_batch_norm': True,
                         'trainable_backbone_layers': 3, 'zero_init_depth': True,
                         'anchor_sizes': [16, 32, 64, 128, 256], 'score_threshold': .01, 'max_detections': 100}
    require(all(run['base_config'].get(k) == v for k, v in expected_settings.items()), 'Nonprotocol architecture settings')
    rows, trace = read_lines(path / 'loss.jsonl'), read_lines(path / 'sampling.jsonl')
    trace_audit = audit_loss_trace(rows, trace, expected_rows, variant)

    model, initialization = build_v5_model(config, variant, init_rgb=run['init_checkpoint']['path'])
    require(initialization == run['rgb_initialization'] and not initialization['adapted_keys'], 'RGB initialization receipt differs')
    initial_sha, initial_count = shared_initialization(model)
    require(run['shared_initialization']['sha256'] == initial_sha and run['shared_initialization']['state_tensors'] == initial_count, 'Reconstructed shared initialization differs')
    require(hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest() == run['initial_torch_cpu_rng_sha256'], 'Reconstructed CPU RNG differs')
    groups, group_metadata = optimizer_groups(model)
    require(group_metadata == run['optimizer']['groups'], 'Optimizer groups differ from actual trainable parameters')
    actual_names = [name for group in group_metadata for name in group['parameter_names']]
    trainable_names = [name for name, value in model.named_parameters() if value.requires_grad]
    require(len(actual_names) == len(set(actual_names)) and set(actual_names) == set(trainable_names), 'Optimizer coverage missing/duplicated')
    require(run['parameters'] == sum(p.numel() for p in model.parameters()), 'Parameter count differs')
    require(run['trainable_parameters'] == sum(p.numel() for p in model.parameters() if p.requires_grad), 'Trainable parameter count differs')
    require(run['optimizer']['type'] == 'AdamW' and run['optimizer']['weight_decay'] == .0001 and run['optimizer']['warmup_steps'] == 100 and run['optimizer']['cosine_decay'] is True, 'Optimizer policy differs')

    checkpoint = torch.load(path / 'model.pt', map_location='cpu', weights_only=True, mmap=True)
    require(checkpoint['format'] == 'rgbd_v5_v1', 'Wrong checkpoint format')
    for key in ('base_config', 'config', 'variant', 'provenance', 'classes', 'depth_scale', 'manifest_sha256', 'dataset_sha256', 'protocol', 'source_sha256', 'phase', 'seed', 'steps'):
        require(checkpoint[key] == run[key], f'Checkpoint/run mismatch: {key}')
    require(model.config == run['config'], 'Variant config differs from implementation')
    final_state, initial_state = checkpoint['state_dict'], model.state_dict()
    weights = torch.load(path / 'weights.pt', map_location='cpu', weights_only=True, mmap=True)
    require(final_state.keys() == initial_state.keys() == weights.keys(), 'State keys mismatch')
    for name, value in final_state.items():
        require(value.shape == initial_state[name].shape and value.dtype == initial_state[name].dtype, f'State shape/dtype mismatch: {name}')
        require(bool(torch.isfinite(value).all()) and torch.equal(value, weights[name]), f'Nonfinite/unequal checkpoint weights: {name}')
    training_state = torch.load(path / 'training_state.pt', map_location='cpu', weights_only=True, mmap=True)
    require(training_state['completed_steps'] == 800 and not training_state['resume_supported'], 'Final training state mismatch')
    require(training_state['sampling_rng'] == rng_states[0] and training_state['augmentation_rng'] == rng_states[1], 'Final independent RNG states differ')
    require(training_state['scheduler']['last_epoch'] == 800, 'Scheduler not at step800')
    require(training_state['scheduler']['_last_lr'] == rows[-1]['lr_next'], 'Saved scheduler rates differ')
    opt_groups = training_state['optimizer']['param_groups']
    require(len(opt_groups) == 2, 'Optimizer state group count differs')
    all_ids = []
    for saved, meta, lr_final in zip(opt_groups, group_metadata, rows[-1]['lr_next']):
        require(len(saved['params']) == len(meta['parameter_names']), 'Optimizer state parameter count differs')
        require(saved['lr'] == lr_final and saved['initial_lr'] == meta['lr'], 'Optimizer state learning rates differ')
        require(saved['weight_decay'] == .0001 and tuple(saved['betas']) == (.9, .999) and saved['eps'] == 1e-8, 'AdamW state hyperparameters differ')
        all_ids.extend(saved['params'])
    require(len(all_ids) == len(set(all_ids)), 'Duplicate optimizer state IDs')
    require(set(training_state['optimizer']['state']).issubset(all_ids), 'Unknown optimizer state IDs')
    for state in training_state['optimizer']['state'].values():
        for value in state.values():
            if torch.is_tensor(value):
                require(bool(torch.isfinite(value).all()), 'Nonfinite optimizer state')
        require(0 < float(state['step']) <= 800, 'Invalid per-parameter update count')
    require(finite(run['training_duration_s']) and 0 < run['training_duration_s'] <= run['duration_s'] <= 1800, 'Training duration violates timeout budget')
    result = {'status': 'pass', 'seed': seed, 'variant': variant, 'path': str(path),
              'run_sha256': sha(path / 'run.json'), 'artifacts_sha256': actual_hashes,
              'shared_initialization_sha256': initial_sha, 'parameter_groups_verified': True,
              'parameter_count': run['parameters'], 'trace_checks': trace_audit,
              'cpu_rng_sha256': run['initial_torch_cpu_rng_sha256'],
              'cuda_rng_sha256': run['initial_torch_cuda_rng_sha256'],
              'init_checkpoint_sha256': run['init_checkpoint']['sha256']}
    del groups, model, checkpoint, weights, final_state, initial_state, training_state
    gc.collect()
    return result, run


def audit_evaluation(path, dataset, dataset_name, seed, variant, split, run, run_path):
    import numpy as np
    from PIL import Image
    from pycocotools import mask as mask_utils
    from tools.evaluate_multiscale import area_counts
    result, metadata = read_json(path / 'evaluation.json'), read_json(path / 'metadata.json')
    require({k: v for k, v in result.items() if k != 'metrics'} == metadata, 'Evaluation/metadata mismatch')
    require(result['exit_status'] == 'completed' and result['phase'] == 'formal', 'Evaluation not formally completed')
    require(result['variant'] == variant and result['training_seed'] == seed and result['split'] == split, 'Evaluation path/identity mismatch')
    require(result['training_steps'] == 800 and result['method'] == 'full', 'Wrong checkpoint step/inference method')
    require(result['source_sha256'] == FROZEN_SOURCES and result['protocol']['sha256'] == PROTOCOL_SHA, 'Unfrozen evaluation source/protocol')
    require(result['checkpoint_sha256'] == sha(run_path / 'model.pt'), 'Evaluated wrong checkpoint bytes')
    require(Path(result['checkpoint']).resolve() == (run_path / 'model.pt').resolve(), 'Wrong checkpoint path')
    require(result['training_run_sha256'] == sha(run_path / 'run.json'), 'Training receipt changed after evaluation')
    require(result['model'] == run['config'] and result['parameters'] == run['parameters'], 'Evaluation model config/count mismatch')
    require(result['classes'] == ['object'] and result['score_threshold'] == .01 and result['mask_threshold'] == .5, 'Evaluation labels/threshold mismatch')
    spec = DATA[dataset_name]
    require(result['dataset_sha256'] == spec['sha256'] and result['manifest_sha256'] == spec['manifest'], 'Evaluation dataset receipt mismatch')
    require(Path(result['data_root']).resolve() == dataset.root, 'Wrong evaluation data root')
    require(result['samples'] == len(dataset) and result['evaluation_depth_scale'] == spec['depth'], 'Evaluation count/depth mismatch')
    require(result['training_depth_scale'] == .001 and result['cross_dataset_evaluation'] == (dataset_name == 'tless'), 'Cross-dataset units/provenance mismatch')
    require(result['training_dataset_sha256'] == DATA['graspnet']['sha256'], 'Original training dataset mismatch')
    require(result['provenance'] == spec['provenance'], 'Wrong evaluation provenance')
    audit_runtime(result)
    timing = result['timing']
    require(timing['warmup'] == 3 and timing['batch'] == 1 and timing['precision'] == 'float32' and timing['threads'] == 4, 'Timing protocol differs')
    frames = result['per_frame']
    require(len(frames) == len(dataset), 'Missing frame timings')
    times = [frame['time_ms'] for frame in frames]
    require(all(finite(t) and t > 0 for t in times), 'Invalid elapsed times')
    for key, value in {'mean_ms': np.mean(times), 'median_ms': np.median(times), 'p95_ms': np.percentile(times, 95), 'fps': 1000 * len(times) / sum(times)}.items():
        require(math.isclose(timing[key], float(value), rel_tol=1e-12, abs_tol=1e-9), f'Timing statistic differs: {key}')
    metrics = result['metrics']
    require(set(metrics) == METRIC_KEYS and all(v is None or (finite(v) and 0 <= v <= 100) for v in metrics.values()), 'Invalid/incomplete COCO metric dictionary')
    predictions_path = path / 'predictions.jsonl'
    require(sha(predictions_path) == result['prediction_sha256'], 'Prediction bytes mismatch')
    predictions = read_lines(predictions_path)
    require(len(predictions) == len(dataset), 'Incomplete prediction rows')
    prediction_count = 0
    for row, record, frame in zip(predictions, dataset.samples, frames):
        require(row['sample_id'] == frame['sample_id'] == record['id'] and row['image_id'] == frame['image_id'] == record['image_id'], 'Prediction sample order/IDs mismatch')
        with Image.open(record['rgb_path']) as image:
            width, height = image.size
        require((row['height'], row['width']) == (height, width), 'Predictions not in original image coordinates')
        n = len(row['scores'])
        require(n <= 100 and len(row['boxes']) == len(row['labels']) == len(row['masks_rle']) == n, 'Candidate count mismatch')
        require(all(finite(s) and .01 - 1e-7 <= s <= 1 for s in row['scores']) and all(label == 1 for label in row['labels']), 'Invalid scores/classes')
        for box, rle in zip(row['boxes'], row['masks_rle']):
            require(len(box) == 4 and all(finite(v) for v in box), 'Invalid box')
            require(0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height, 'Box outside original image')
            require(rle['size'] == [height, width], 'RLE shape mismatch')
            mask = mask_utils.decode(rle)
            require(mask.shape == (height, width) and np.isin(mask, [0, 1]).all() and mask.any(), 'Empty/nonbinary/invalid decoded mask')
        counts = frame['counts']
        require(counts['method'] == 'full' and counts['views'] == 1 and counts['output_candidates'] == n, 'Inference counts differ')
        require(all(counts[k] == 0 for k in ('crop_area_removed', 'crop_border_removed', 'nms_removed', 'topk_removed')), 'Extra crop/fusion filtering in full inference')
        prediction_count += n
    require(result['original_gt_areas'] == area_counts(dataset), 'Ground-truth area receipt differs')
    return {'status': 'pass', 'dataset': dataset_name, 'seed': seed, 'variant': variant, 'split': split,
            'evaluation_sha256': sha(path / 'evaluation.json'), 'metadata_sha256': sha(path / 'metadata.json'),
            'predictions_sha256': sha(predictions_path), 'images': len(dataset), 'predictions': prediction_count,
            'ordered_ids_geometry_rle_verified': True, 'depth_units_verified': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training-root', type=Path, default=ROOT / 'outputs/paper_v5')
    parser.add_argument('--evaluation-root', type=Path, default=ROOT / 'reports/v5/evaluation')
    parser.add_argument('--ledger', type=Path, default=ROOT / 'reports/v5/experiment_ledger.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'reports/v5/training_methods_audit.json')
    args = parser.parse_args()
    require(not args.output.exists(), 'Audit output already exists; choose a new --output to preserve it')
    # Force CPU before importing torch; model construction below never forwards.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    from rgbd_seg.data import RGBDDataset, dataset_fingerprint
    torch.set_num_threads(2)
    expected_train = [(seed, variant, args.training_root / f'seed{seed}' / variant) for seed in SEEDS for variant in VARIANTS]
    expected_eval = [('graspnet', seed, variant, split, args.evaluation_root / 'graspnet' / f'seed{seed}' / variant / split)
                     for seed in SEEDS for variant in VARIANTS for split in ('val', 'test')]
    expected_eval += [('tless', 17, variant, 'test', args.evaluation_root / 'tless/seed17' / variant / 'test') for variant in VARIANTS]
    missing = [str(path / 'run.json') for _, _, path in expected_train if not (path / 'run.json').is_file()]
    missing += [str(path / 'evaluation.json') for *_, path in expected_eval if not (path / 'evaluation.json').is_file()]
    report = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'incomplete' if missing else 'running',
              'expected_training_runs': 12, 'expected_evaluations': 28, 'missing': missing,
              'audit_source_sha256': sha(__file__), 'protocol_sha256': PROTOCOL_SHA,
              'source_sha256': FROZEN_SOURCES, 'attempt': 2, 'training': [], 'evaluation': [], 'errors': [],
              'limitations': ['Checks consistency of persisted evidence; does not establish undisclosed runtime behavior.',
                             'Augmentation RNG is replayed, but augmented image pixels were not saved and cannot be byte-compared.',
                             'GPU initial RNG is compared across methods, not regenerated on CPU.',
                             'Equal seeds/shared initialization do not guarantee bitwise-identical final CUDA weights; that is not required here.',
                             'No performance aggregation, gate, significance test, or model inference is performed.']}
    if not missing:
        try:
            require(all(sha(ROOT / name) == digest for name, digest in FROZEN_SOURCES.items()), 'Current source differs from reviewed frozen source')
            require(sha(ROOT / 'docs/training_methods_protocol.md') == PROTOCOL_SHA, 'Current protocol differs from freeze')
            report['supervisor'] = audit_supervisor(args.ledger, expected_train, expected_eval)
            datasets = {}
            for name, spec in DATA.items():
                root = ROOT / spec['folder']
                require(sha(root / 'manifest.json') == spec['manifest'] and dataset_fingerprint(root) == spec['sha256'], f'Dataset bytes changed: {name}')
                for split, count in spec['splits'].items():
                    dataset = RGBDDataset(root, split)
                    require(len(dataset) == count and dataset.classes == ['object'] and dataset.depth_scale == spec['depth'], f'Dataset specification mismatch: {name}/{split}')
                    datasets[name, split] = dataset
            training = datasets['graspnet', 'train']
            target_sizes = []
            for index in range(len(training)):
                _, target = training[index]
                target_sizes.append((len(target['masks']), int((target['area'] < 96 ** 2).sum())))
            replays = {seed: replay_sampling(seed, training.samples, target_sizes) for seed in SEEDS}
            runs = {}
            for seed, variant, path in expected_train:
                try:
                    rows, sampling_state, augmentation_state = replays[seed]
                    checked, run = audit_train(path, seed, variant, rows, (sampling_state, augmentation_state))
                    report['training'].append(checked)
                    runs[seed, variant] = (run, path)
                    print(f'Verified training seed{seed}/{variant}', flush=True)
                except Exception as error:
                    report['errors'].append({'scope': str(path), 'error': f'{type(error).__name__}: {error}'})
            for seed in SEEDS:
                checked = [row for row in report['training'] if row['seed'] == seed]
                if len(checked) == 4:
                    for key in ('shared_initialization_sha256', 'cpu_rng_sha256', 'cuda_rng_sha256', 'init_checkpoint_sha256'):
                        require(len({row[key] for row in checked}) == 1, f'Paired initialization/RNG differs: seed{seed}/{key}')
            for dataset_name, seed, variant, split, path in expected_eval:
                try:
                    require((seed, variant) in runs, 'Original training audit did not pass')
                    run, run_path = runs[seed, variant]
                    checked = audit_evaluation(path, datasets[dataset_name, split], dataset_name, seed, variant, split, run, run_path)
                    report['evaluation'].append(checked)
                    print(f'Verified evaluation {dataset_name}/seed{seed}/{variant}/{split}', flush=True)
                except Exception as error:
                    report['errors'].append({'scope': str(path), 'error': f'{type(error).__name__}: {error}'})
            # Detect files changed during the audit, including reused SHA cache entries.
            for path in list(_HASHES):
                sha(path)
        except Exception as error:
            report['errors'].append({'scope': 'global', 'error': f'{type(error).__name__}: {error}'})
        report['status'] = 'pass' if not report['errors'] and len(report['training']) == 12 and len(report['evaluation']) == 28 else 'fail'
    report['actual_file_sha256'] = {str(path): digest for path, (_, digest) in _HASHES.items()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'status': report['status'], 'training_passed': len(report['training']),
                      'evaluation_passed': len(report['evaluation']), 'errors': len(report['errors']),
                      'output': str(args.output)}, ensure_ascii=False))
    return 0 if report['status'] == 'pass' else 2


if __name__ == '__main__':
    raise SystemExit(main())
