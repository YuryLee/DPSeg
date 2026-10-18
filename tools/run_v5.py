#!/usr/bin/env python3
"""Isolated, auditable v5 training and full-image COCO evaluation (no resume)."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from PIL import Image
from rgbd_seg.augment import augment
from rgbd_seg.data import RGBDDataset, dataset_fingerprint, load_rgbd
from rgbd_seg.evaluation import evaluate_predictions
from rgbd_seg.multiscale_inference import run_inference
from tools.evaluate_multiscale import RLEPredictions, area_counts, encode_prediction, ensure_empty, sha256

VARIANTS = ('baseline', 'bce2', 'bace', 'freqfusion')
SOURCES = ('tools/run_v5.py', 'tools/run.py', 'tools/evaluate_multiscale.py',
           'rgbd_seg/v5_model.py', 'rgbd_seg/model.py', 'rgbd_seg/pretrained.py',
           'rgbd_seg/boundary_loss.py', 'rgbd_seg/freqfusion_fpn.py',
           'rgbd_seg/augment.py', 'rgbd_seg/data.py', 'rgbd_seg/evaluation.py',
           'rgbd_seg/multiscale_inference.py')
FIXED_TRAINING = {'batch_size': 2, 'lr': 0.0002, 'backbone_lr_factor': 0.1,
                 'weight_decay': 0.0001, 'warmup_steps': 100, 'cosine_decay': True,
                 'crop_probability': 0.35, 'gradient_clip_norm': 5.0,
                 'foreground_mean_init': False, 'cache_data': True}


def write(path, value):
    """Replace this run's own receipt atomically; output ownership is checked first."""
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def digest_state(state):
    return hashlib.sha256(repr(state).encode('ascii')).hexdigest()


def shared_initialization(model):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(model.state_dict().items()):
        if name.startswith('core.backbone.rgb.fpn.fusions.'):
            continue
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)], separators=(',', ':')).encode())
        digest.update(tensor.numpy().tobytes())
        count += 1
    return {'sha256': digest.hexdigest(), 'state_tensors': count,
            'scope': 'Sorted state_dict names, dtypes, shapes and C-order raw bytes; excludes only core.backbone.rgb.fpn.fusions.* added parameters/buffers'}


def schedule_multiplier(step, steps):
    # Exactly the old run.py warmup+cosine formula and step convention.
    warmup = min(1., (step + 1) / max(1, FIXED_TRAINING['warmup_steps']))
    decay = .1 + .9 * (1 + math.cos(math.pi * step / max(1, steps))) / 2
    return warmup * decay


def next_batch(cached, records, sampling_rng, augmentation_rng):
    indices = [sampling_rng.randrange(len(cached)) for _ in range(2)]
    before = digest_state(augmentation_rng.getstate())
    batch = [augment(*cached[index], augmentation_rng, .35) for index in indices]
    trace = {'batch_indices': indices, 'sample_ids': [records[index]['id'] for index in indices],
             'augmentation_rng_before_sha256': before,
             'augmentation_rng_after_sha256': digest_state(augmentation_rng.getstate())}
    return batch, trace


def validate_groups(model, groups, metadata):
    """Verify helper metadata against actual optimizer objects, not name guesses."""
    names = {id(parameter): name for name, parameter in model.named_parameters() if parameter.requires_grad}
    if len(groups) != len(metadata):
        raise ValueError('Optimizer groups and metadata lengths differ')
    seen = []
    for group, receipt in zip(groups, metadata):
        ids = [id(parameter) for parameter in group['params']]
        if any(item not in names for item in ids):
            raise ValueError('Optimizer includes a frozen or unknown parameter')
        if receipt['parameter_names'] != [names[item] for item in ids]:
            raise ValueError('Optimizer parameter names differ from metadata')
        if receipt['lr'] != group['lr'] or receipt['numel'] != sum(p.numel() for p in group['params']):
            raise ValueError('Optimizer learning rate/size differs from metadata')
        seen.extend(ids)
    if len(seen) != len(set(seen)) or set(seen) != set(names):
        raise ValueError('Optimizer must contain each trainable parameter exactly once')


def source_snapshot():
    return {name: sha256(ROOT / name) for name in SOURCES}


def sync(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def configure_runtime(args):
    if args.threads <= 0 or not 0 < args.cuda_memory_fraction <= 1:
        raise ValueError('threads must be positive; CUDA memory fraction must be in (0,1]')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; use cpu or an allocated CUDA device')
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction, device)
    elif device.type != 'cpu':
        raise ValueError('Only CPU and CUDA are supported')
    libraries = {name: version(name) for name in ('torch', 'torchvision', 'numpy', 'Pillow', 'pycocotools')}
    return {'python': platform.python_version(), 'libraries': libraries,
            'device': str(device), 'device_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else platform.processor() or 'CPU',
            'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
            'cuda_version': torch.version.cuda, 'cudnn_version': torch.backends.cudnn.version(),
            'threads': torch.get_num_threads(), 'cuda_memory_fraction': args.cuda_memory_fraction,
            'cudnn_benchmark': torch.backends.cudnn.benchmark,
            'cudnn_deterministic': torch.backends.cudnn.deterministic,
            'deterministic_algorithms': torch.are_deterministic_algorithms_enabled()}


def memory_receipt(device):
    if torch.device(device).type != 'cuda':
        return {'peak_cuda_allocated_bytes': None, 'peak_cuda_reserved_bytes': None}
    return {'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device),
            'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(device)}


def input_receipt(dataset, protocol):
    return {'data_root': str(dataset.root), 'manifest_sha256': sha256(dataset.root / 'manifest.json'),
            'dataset_sha256': dataset_fingerprint(dataset.root),
            'protocol': {'path': str(Path(protocol).resolve()), 'sha256': sha256(protocol)},
            'source_sha256': source_snapshot()}


def verify_inputs(receipt, files=()):
    if source_snapshot() != receipt['source_sha256']:
        raise RuntimeError('Source changed during execution; result cannot be marked completed')
    root = Path(receipt['data_root'])
    if sha256(root / 'manifest.json') != receipt['manifest_sha256'] or dataset_fingerprint(root) != receipt['dataset_sha256']:
        raise RuntimeError('Dataset changed during execution')
    if sha256(receipt['protocol']['path']) != receipt['protocol']['sha256']:
        raise RuntimeError('Protocol changed during execution')
    for path, expected in files:
        if sha256(path) != expected:
            raise RuntimeError(f'Input file changed during execution: {path}')


def move(mapping, device):
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in mapping.items()}


def train(args):
    from rgbd_seg.v5_model import build_v5_model, optimizer_groups
    if args.steps <= 0:
        raise ValueError('steps must be positive')
    runtime = configure_runtime(args)
    dataset = RGBDDataset(args.data, 'train')
    if not len(dataset):
        raise ValueError('Training split is empty')
    receipt = input_receipt(dataset, args.protocol)
    config_path, init_path = Path(args.config).resolve(), Path(args.init_rgb).resolve()
    config_sha, init_sha = sha256(config_path), sha256(init_path)
    base_config = json.loads(config_path.read_text())
    base_config['num_classes'] = len(dataset.classes) + 1
    seed = int(base_config.get('seed', 17))
    output = ensure_empty(args.output)
    run = {**receipt, **runtime, 'run_id': output.name, 'command': sys.argv,
           'created_at_utc': datetime.now(timezone.utc).isoformat(), 'exit_status': 'running',
           'phase': 'formal' if args.steps == 800 else 'preflight',
           'variant': args.variant, 'seed': seed, 'steps': args.steps, 'completed_steps': 0,
           'provenance': dataset.manifest['provenance'], 'classes': dataset.classes,
           'depth_scale': dataset.depth_scale, 'training_samples': len(dataset),
           'config_file': {'path': str(config_path), 'sha256': config_sha},
           'init_checkpoint': {'path': str(init_path), 'sha256': init_sha},
           'fixed_training': FIXED_TRAINING,
           'selection': 'Fixed final step; no validation checkpoint selection; every variant starts from COCO, not a previous run',
           'resume_supported': False}
    write(output / 'run.json', run)
    try:
        # Preserve legacy seeding, initialization, caching and optimizer order.
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        model, init_info = build_v5_model(base_config, args.variant, init_rgb=init_path)
        run['shared_initialization'] = shared_initialization(model)
        run['initial_torch_cpu_rng_sha256'] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()
        if torch.device(args.device).type == 'cuda':
            run['initial_torch_cuda_rng_sha256'] = hashlib.sha256(torch.cuda.get_rng_state(args.device).cpu().numpy().tobytes()).hexdigest()
        model = model.to(args.device)
        cached = [dataset[index] for index in range(len(dataset))]
        groups, group_metadata = optimizer_groups(model, .0002, .1)
        validate_groups(model, groups, group_metadata)
        optimizer = torch.optim.AdamW(groups, lr=.0002, weight_decay=.0001)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_multiplier(step, args.steps))
        run.update({'base_config': model.base_config, 'config': model.config,
                    'rgb_initialization': init_info, 'optimizer': {'type': 'AdamW', 'groups': group_metadata,
                    'weight_decay': .0001, 'warmup_steps': 100, 'cosine_decay': True},
                    'parameters': sum(p.numel() for p in model.parameters()),
                    'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)})
        write(output / 'run.json', run)
        sampling_rng, augmentation_rng = random.Random(seed), random.Random(seed + 1000)
        model.train()
        sync(args.device)
        if torch.device(args.device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)
        start = time.perf_counter()
        with (output / 'loss.jsonl').open('x') as log, (output / 'sampling.jsonl').open('x') as sampling_log:
            for step in range(args.steps):
                batch, trace = next_batch(cached, dataset.samples, sampling_rng, augmentation_rng)
                samples = [move(sample, args.device) for sample, _ in batch]
                targets = [move(target, args.device) for _, target in batch]
                optimizer.zero_grad(set_to_none=True)
                losses = model(samples, targets)
                total = sum(losses.values())
                if not torch.isfinite(total):
                    raise FloatingPointError(f'Nonfinite loss at step {step + 1}')
                total.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                lr_used = [group['lr'] for group in optimizer.param_groups]
                optimizer.step()
                scheduler.step()
                row = {'step': step + 1, **trace, 'loss': float(total.detach()),
                       **{key: float(value.detach()) for key, value in losses.items()},
                       'grad_norm': float(norm), 'lr_used': lr_used,
                       'lr_next': [group['lr'] for group in optimizer.param_groups],
                       'elapsed_s': time.perf_counter() - start}
                log.write(json.dumps(row, allow_nan=False) + '\n')
                log.flush()
                sampling_log.write(json.dumps({'step': step + 1, **trace}, sort_keys=True) + '\n')
                sampling_log.flush()
                run['completed_steps'] = step + 1
                if step == 0 or (step + 1) % 25 == 0:
                    print(json.dumps(row), flush=True)
        sync(args.device)
        run['training_duration_s'] = time.perf_counter() - start
        run.update(memory_receipt(args.device))
        state = {key: value.detach().cpu() for key, value in model.state_dict().items()}
        checkpoint = {'format': 'rgbd_v5_v1', 'base_config': model.base_config,
                      'variant': args.variant, 'config': model.config, 'state_dict': state,
                      'provenance': run['provenance'], 'classes': dataset.classes,
                      'depth_scale': dataset.depth_scale,
                      'manifest_sha256': run['manifest_sha256'], 'dataset_sha256': run['dataset_sha256'],
                      'protocol': run['protocol'], 'source_sha256': run['source_sha256'],
                      'phase': run['phase'], 'seed': seed, 'steps': args.steps}
        torch.save(checkpoint, output / 'model.pt')
        torch.save(state, output / 'weights.pt')
        # This receipt supports audit/future work only. There is deliberately no resume path.
        numpy_state = np.random.get_state()
        training_state = {'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
                          'completed_steps': args.steps, 'sampling_rng': sampling_rng.getstate(),
                          'augmentation_rng': augmentation_rng.getstate(), 'python_rng': random.getstate(),
                          'numpy_rng': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
                          'torch_cpu_rng': torch.get_rng_state(),
                          'torch_cuda_rng': torch.cuda.get_rng_state(args.device) if torch.device(args.device).type == 'cuda' else None,
                          'resume_supported': False}
        torch.save(training_state, output / 'training_state.pt')
        verify_inputs(run, ((config_path, config_sha), (init_path, init_sha)))
        run.update({'duration_s': time.perf_counter() - start, 'exit_status': 'completed',
                    'checkpoint_sha256': sha256(output / 'model.pt'),
                    'weights_sha256': sha256(output / 'weights.pt'),
                    'training_state_sha256': sha256(output / 'training_state.pt'),
                    'loss_trace_sha256': sha256(output / 'loss.jsonl'),
                    'sampling_trace_sha256': sha256(output / 'sampling.jsonl'),
                    'weight_bytes': (output / 'weights.pt').stat().st_size})
        write(output / 'run.json', run)
        print(json.dumps({'output': str(output.resolve()), 'exit_status': 'completed',
                          'variant': args.variant, 'phase': run['phase'], 'steps': args.steps}), flush=True)
    except BaseException as error:
        run.update({'exit_status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        write(output / 'run.json', run)
        raise


def check_training_receipt(path, checkpoint, dataset, fingerprint):
    """Validate original training evidence; evaluation may use different depth units."""
    path = Path(path).resolve()
    run_path = path.parent / 'run.json'
    run = json.loads(run_path.read_text())
    if checkpoint.get('format') != 'rgbd_v5_v1' or run.get('exit_status') != 'completed':
        raise ValueError('A completed v5 training run is required')
    if run.get('checkpoint_sha256') != sha256(path):
        raise ValueError('Checkpoint SHA does not match completed training receipt')
    for key in ('base_config', 'config', 'variant', 'provenance', 'classes', 'depth_scale',
                'manifest_sha256', 'dataset_sha256', 'protocol', 'source_sha256', 'phase', 'seed', 'steps'):
        if key not in checkpoint or checkpoint[key] != run.get(key):
            raise ValueError(f'Checkpoint/run metadata mismatch: {key}')
    if run.get('completed_steps') != run.get('steps'):
        raise ValueError('Training receipt has incomplete steps')
    if checkpoint['classes'] != dataset.classes or checkpoint['config']['num_classes'] != len(dataset.classes) + 1:
        raise ValueError('Checkpoint and evaluation classes differ')
    train_root = Path(run['data_root'])
    manifest = json.loads((train_root / 'manifest.json').read_text())
    train_fingerprint = fingerprint if train_root.resolve() == dataset.root else dataset_fingerprint(train_root)
    if sha256(train_root / 'manifest.json') != run['manifest_sha256'] or train_fingerprint != run['dataset_sha256']:
        raise ValueError('Original training dataset changed since training')
    if manifest['classes'] != checkpoint['classes'] or manifest['depth_scale'] != checkpoint['depth_scale'] or manifest['provenance'] != checkpoint['provenance']:
        raise ValueError('Original training manifest metadata differs from checkpoint')
    if source_snapshot() != run['source_sha256'] or sha256(run['protocol']['path']) != run['protocol']['sha256']:
        raise ValueError('Training implementation or training protocol changed since training')
    return {'training_run_path': str(run_path), 'training_run_sha256': sha256(run_path),
            'training_data_root': str(train_root.resolve()), 'training_dataset_sha256': train_fingerprint,
            'training_manifest_sha256': run['manifest_sha256'], 'training_provenance': run['provenance'],
            'training_depth_scale': checkpoint['depth_scale'], 'training_protocol': run['protocol'],
            'training_steps': run['steps'], 'training_seed': run['seed'], 'phase': run['phase'],
            'cross_dataset_evaluation': fingerprint != train_fingerprint}


def validate_prediction_ids(path, dataset):
    """Check persisted ordering/geometry before COCO consumes its positional API."""
    count = 0
    with Path(path).open() as stream:
        for index, line in enumerate(stream):
            if index >= len(dataset):
                raise ValueError('More predictions than dataset samples')
            prediction, expected = json.loads(line), dataset.samples[index]
            if prediction['sample_id'] != expected['id'] or prediction['image_id'] != expected['image_id']:
                raise ValueError('Prediction IDs/order differ from evaluation split')
            with Image.open(expected['rgb_path']) as image:
                width, height = image.size
            if (prediction['height'], prediction['width']) != (height, width):
                raise ValueError('Prediction geometry differs from original image')
            if any(rle['size'] != [height, width] for rle in prediction['masks_rle']):
                raise ValueError('RLE geometry differs from original image')
            count += 1
    if count != len(dataset):
        raise ValueError('Missing prediction records')


def evaluate(args):
    from rgbd_seg.v5_model import load_v5_model
    runtime = configure_runtime(args)
    dataset = RGBDDataset(args.data, args.split)
    if not len(dataset) or any(r.get('instance_path') is None and r.get('instance_mask_paths') is None for r in dataset.samples):
        raise ValueError('Evaluation requires a nonempty, fully annotated split')
    receipt = input_receipt(dataset, args.protocol)
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint_sha = sha256(checkpoint_path)
    model, checkpoint = load_v5_model(checkpoint_path, args.device)
    integrity = check_training_receipt(checkpoint_path, checkpoint, dataset, receipt['dataset_sha256'])
    output = ensure_empty(args.output)
    metadata = {**receipt, **runtime, **integrity, 'exit_status': 'running',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
                'checkpoint': str(checkpoint_path), 'checkpoint_sha256': checkpoint_sha,
                'checkpoint_bytes': checkpoint_path.stat().st_size, 'variant': checkpoint['variant'],
                'model': checkpoint['config'], 'method': 'full', 'split': args.split,
                'provenance': dataset.manifest['provenance'], 'classes': dataset.classes,
                'evaluation_depth_scale': dataset.depth_scale, 'samples': len(dataset),
                'parameters': sum(parameter.numel() for parameter in model.parameters()),
                'mask_threshold': .5, 'score_threshold': checkpoint['config'].get('score_threshold'),
                'original_gt_areas': area_counts(dataset),
                'timing_scope': 'In-memory RGB-D tensors through device transfer, full model, original-size mask binarization and CPU output. Excludes file IO, RLE encoding, COCO and plotting.'}
    write(output / 'metadata.json', metadata)
    try:
        times, frames = [], []
        model.eval()
        with torch.inference_mode():
            first = dataset.samples[0]
            sample = load_rgbd(first['rgb_path'], first['depth_path'], dataset.depth_scale)
            for _ in range(3):
                run_inference(model, sample, method='full')
            sync(args.device)
            if torch.device(args.device).type == 'cuda':
                torch.cuda.reset_peak_memory_stats(args.device)
            del sample
            with (output / 'predictions.jsonl').open('x') as stream:
                for record in dataset.samples:
                    sample = load_rgbd(record['rgb_path'], record['depth_path'], dataset.depth_scale)
                    sync(args.device)
                    started = time.perf_counter()
                    prediction, counts = run_inference(model, sample, method='full')
                    sync(args.device)
                    elapsed = time.perf_counter() - started
                    times.append(elapsed)
                    height, width = sample['image'].shape[-2:]
                    encoded = encode_prediction(prediction, record['id'], record['image_id'], height, width)
                    stream.write(json.dumps(encoded, allow_nan=False) + '\n')
                    frames.append({'sample_id': record['id'], 'image_id': record['image_id'],
                                   'time_ms': elapsed * 1000, 'counts': counts})
                    del prediction, sample, encoded
        metadata.update(memory_receipt(args.device))
        prediction_path = output / 'predictions.jsonl'
        validate_prediction_ids(prediction_path, dataset)
        metrics = evaluate_predictions(dataset, RLEPredictions(prediction_path))
        if len(metrics) != 12:
            raise RuntimeError('Expected all twelve official COCO segmentation metrics')
        verify_inputs(metadata, ((checkpoint_path, checkpoint_sha),
                                 (integrity['training_run_path'], integrity['training_run_sha256'])))
        # Repeat original training content validation for cross-dataset evaluation too.
        if check_training_receipt(checkpoint_path, checkpoint, dataset, dataset_fingerprint(dataset.root)) != integrity:
            raise RuntimeError('Training provenance changed during evaluation')
        metadata.update({'exit_status': 'completed', 'prediction_sha256': sha256(prediction_path),
                         'prediction_identity_check': 'All sample IDs, global image IDs, order and original dimensions verified',
                         'timing': {'device': args.device, 'threads': args.threads, 'batch': 1,
                                    'warmup': 3, 'precision': 'float32', 'scope': metadata['timing_scope'],
                                    'mean_ms': float(np.mean(times) * 1000), 'median_ms': float(np.median(times) * 1000),
                                    'p95_ms': float(np.percentile(times, 95) * 1000), 'fps': len(times) / sum(times)},
                         'per_frame': frames})
        write(output / 'evaluation.json', {**metadata, 'metrics': metrics})
        write(output / 'metadata.json', metadata)
        print(json.dumps({'output': str(output.resolve()), 'metrics': metrics}, ensure_ascii=False), flush=True)
    except BaseException as error:
        metadata.update({'exit_status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        write(output / 'metadata.json', metadata)
        raise


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    sub = result.add_subparsers(dest='command', required=True)
    train_parser = sub.add_parser('train')
    train_parser.add_argument('--config', required=True)
    train_parser.add_argument('--variant', choices=VARIANTS, required=True)
    train_parser.add_argument('--init-rgb', required=True)
    train_parser.add_argument('--steps', type=int, default=800, help='Non-800 runs are preflight only; no resume support')
    eval_parser = sub.add_parser('eval', aliases=['evaluate'])
    eval_parser.add_argument('--split', choices=('val', 'test'), required=True)
    eval_parser.add_argument('--checkpoint', required=True)
    for child in (train_parser, eval_parser):
        child.add_argument('--data', required=True)
        child.add_argument('--output', required=True, help='New or empty directory; nonempty outputs are never overwritten')
        child.add_argument('--device', default='cpu')
        child.add_argument('--threads', type=int, default=4)
        child.add_argument('--cuda-memory-fraction', type=float, default=.55)
        child.add_argument('--protocol', default=str(ROOT / 'docs/training_methods_protocol.md'))
    return result


def main():
    args = parser().parse_args()
    (train if args.command == 'train' else evaluate)(args)


if __name__ == '__main__':
    main()
