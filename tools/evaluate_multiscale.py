#!/usr/bin/env python3
"""Evaluate fixed-weight inference policies with original-coordinate masks and auditable timing."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from pycocotools import mask as mask_utils
from rgbd_seg.data import RGBDDataset, dataset_fingerprint, load_rgbd
from rgbd_seg.evaluation import evaluate_predictions
from rgbd_seg.model import load_model


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path, value):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def ensure_empty(path):
    path = Path(path)
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise FileExistsError(f'Refusing to overwrite nonempty output: {path}')
    path.mkdir(parents=True, exist_ok=True)
    return path


def check_run(checkpoint_path, checkpoint, dataset, fingerprint):
    run_path = checkpoint_path.parent / 'run.json'
    run = json.loads(run_path.read_text())
    checkpoint_sha = sha256(checkpoint_path)
    if run.get('exit_status') != 'completed' or run.get('checkpoint_sha256') != checkpoint_sha:
        raise ValueError('run.json must attest a completed run with this checkpoint SHA256')
    for key in ('config', 'provenance', 'manifest_sha256'):
        if checkpoint.get(key) != run.get(key):
            raise ValueError(f'Checkpoint/run metadata mismatch: {key}')
    if checkpoint['classes'] != dataset.classes:
        raise ValueError('Checkpoint and evaluation classes differ')
    if checkpoint['config']['num_classes'] != len(dataset.classes) + 1:
        raise ValueError('Checkpoint num_classes does not match foreground classes')
    command = run.get('command', [])
    if command.count('--data') != 1:
        raise ValueError('Training command must contain one --data argument')
    train_root = Path(command[command.index('--data') + 1])
    if not train_root.is_absolute():
        train_root = ROOT / train_root
    train_manifest_path = train_root / 'manifest.json'
    train_manifest = json.loads(train_manifest_path.read_text())
    if sha256(train_manifest_path) != run['manifest_sha256']:
        raise ValueError('Training manifest changed since checkpoint creation')
    training_fingerprint = fingerprint if train_root.resolve() == dataset.root else dataset_fingerprint(train_root)
    if training_fingerprint != run.get('dataset_sha256'):
        raise ValueError('Training dataset content changed since run.json receipt')
    if train_manifest['classes'] != checkpoint['classes'] or train_manifest['depth_scale'] != checkpoint['depth_scale']:
        raise ValueError('Training classes/depth units differ from checkpoint')
    return {'checkpoint_sha256': checkpoint_sha, 'run_sha256': sha256(run_path),
            'run_path': str(run_path.resolve()), 'training_dataset_sha256': training_fingerprint,
            'training_manifest_sha256': run['manifest_sha256'],
            'training_code_sha256': run.get('code_sha256'),
            'training_steps': run['steps'], 'training_seed': run['seed'],
            'training_depth_scale': checkpoint['depth_scale'],
            'evaluation_equals_training_dataset': fingerprint == training_fingerprint,
            'cross_dataset_evaluation': fingerprint != training_fingerprint,
            'checks': 'completed run, checkpoint bytes, configuration, provenance, classes, original training manifest and full dataset content verified'}


def encode_prediction(prediction, sample_id, image_id, height, width):
    masks = prediction['masks'].detach().cpu()
    if masks.shape != (len(prediction['scores']), 1, height, width):
        raise ValueError('Inference must return original-resolution Nx1xHxW masks')
    if masks.dtype != torch.bool:
        raise ValueError('Inference interface must explicitly return binary boolean masks')
    boxes, scores, labels = (prediction[key].detach().cpu() for key in ('boxes', 'scores', 'labels'))
    if boxes.shape != (len(masks), 4) or labels.shape != (len(masks),):
        raise ValueError('Prediction vectors and boxes have inconsistent shapes')
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError('Nonfinite or invalid prediction probabilities/boxes')
    rles = []
    for mask in masks[:, 0]:
        rle = mask_utils.encode(np.asfortranarray(mask.numpy().astype(np.uint8)))
        rles.append({'size': rle['size'], 'counts': rle['counts'].decode('ascii')})
    return {'sample_id': sample_id, 'image_id': image_id, 'height': height, 'width': width,
            'boxes': boxes.tolist(), 'scores': scores.tolist(), 'labels': labels.tolist(), 'masks_rle': rles}


class RLEPredictions:
    """Decode one frame at a time; a 100-frame stress set need not retain dense masks."""
    def __init__(self, path):
        self.path = Path(path)
        self.offsets = []
        with self.path.open('rb') as stream:
            while True:
                offset = stream.tell()
                if not stream.readline():
                    break
                self.offsets.append(offset)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, index):
        with self.path.open('rb') as stream:
            stream.seek(self.offsets[index])
            record = json.loads(stream.readline())
        if record['masks_rle']:
            masks = mask_utils.decode(record['masks_rle']).transpose(2, 0, 1).copy()
        else:
            masks = np.zeros((0, record['height'], record['width']), dtype=np.uint8)
        return {'masks': torch.from_numpy(masks)[:, None],
                'boxes': torch.tensor(record['boxes'], dtype=torch.float32).reshape(-1, 4),
                'scores': torch.tensor(record['scores'], dtype=torch.float32),
                'labels': torch.tensor(record['labels'], dtype=torch.int64)}


def area_counts(dataset):
    frames = []
    for index, record in enumerate(dataset.samples):
        _, target = dataset[index]
        areas = target['masks'].flatten(1).sum(1).tolist()
        frames.append({'sample_id': record['id'], 'image_id': record['image_id'],
                       'instances': len(areas), 'small': sum(a <= 1024 for a in areas),
                       'medium': sum(1024 <= a <= 9216 for a in areas),
                       'large': sum(a >= 9216 for a in areas), 'areas_px': areas})
    return {'definition': 'Original-resolution binary visible-mask pixel areas. Official COCO closed intervals: small [0,1024], medium [1024,9216], large [9216,inf]; exact boundaries can occur in two bins.',
            'totals': {key: sum(frame[key] for frame in frames) for key in ('instances', 'small', 'medium', 'large')},
            'per_frame': frames}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', required=True)
    parser.add_argument('--split', choices=('val', 'test'), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True, help='New or empty directory; one subdirectory per setting')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--protocol', default=str(ROOT / 'docs/paper_trial_protocol.md'))
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--methods', nargs='+', choices=('full', 'highres', 'crop_fuse'), default=['full', 'highres', 'crop_fuse'])
    parser.add_argument('--extra', action='store_true', help='Only GraspNet nonoverlapping-mask metrics at fixed score=.5')
    args = parser.parse_args()
    if args.threads <= 0 or len(args.methods) != len(set(args.methods)):
        parser.error('threads must be positive; methods must be unique')
    torch.set_num_threads(args.threads)
    dataset = RGBDDataset(args.data, args.split)
    if not len(dataset):
        raise ValueError('Evaluation split must be nonempty')
    if any(r.get('instance_path') is None and r.get('instance_mask_paths') is None for r in dataset.samples):
        raise ValueError('Every evaluated image requires annotations')
    if args.extra and (dataset.manifest['provenance'] != 'public_graspnet_subset' or any(r.get('instance_mask_paths') is not None for r in dataset.samples)):
        raise ValueError('Extra UOIS metrics are permitted only for original nonoverlapping GraspNet annotations; TLESS is prohibited')
    output = ensure_empty(args.output)
    checkpoint_path = Path(args.checkpoint).resolve()
    fingerprint = dataset_fingerprint(dataset.root)
    model, checkpoint = load_model(checkpoint_path, args.device)
    integrity = check_run(checkpoint_path, checkpoint, dataset, fingerprint)
    from rgbd_seg.multiscale_inference import run_inference, PROTOCOL
    source_files = ['rgbd_seg/model.py', 'rgbd_seg/data.py', 'rgbd_seg/evaluation.py',
                    'rgbd_seg/multiscale_inference.py', 'tools/evaluate_multiscale.py']
    if args.extra:
        source_files.append('rgbd_seg/instance_metrics.py')
    protocol_path = Path(args.protocol).resolve()
    protocol_sha = sha256(protocol_path)
    source_sha = {name: sha256(ROOT / name) for name in source_files}
    device = torch.device(args.device)
    cuda = device.type == 'cuda'
    def sync():
        if cuda:
            torch.cuda.synchronize(device)
    metadata = {**integrity, 'checkpoint': str(checkpoint_path), 'model': checkpoint['config'],
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'command': sys.argv,
                'checkpoint_bytes': checkpoint_path.stat().st_size,
                'classes': dataset.classes, 'dataset_root': str(dataset.root), 'dataset_sha256': fingerprint,
                'manifest_sha256': sha256(dataset.root / 'manifest.json'), 'split': args.split,
                'provenance': dataset.manifest['provenance'], 'samples': len(dataset),
                'parameters': sum(parameter.numel() for parameter in model.parameters()),
                'input_contract': {'rgb': 'float32 [0,1]', 'depth': 'float32 metres, invalid=0',
                                   'depth_valid': 'float32 binary, aligned with RGB and depth',
                                   'evaluation_raw_to_metres': dataset.depth_scale,
                                   'gt_used_for_crops': False, 'mask_threshold': 0.5},
                'source_sha256': source_sha, 'inference_protocol': PROTOCOL,
                'protocol_document': str(protocol_path), 'protocol_sha256': protocol_sha,
                'software': {'python': platform.python_version(), 'torch': str(torch.__version__),
                             'torchvision': version('torchvision'), 'pycocotools': version('pycocotools'), 'numpy': np.__version__},
                'device': {'requested': args.device, 'name': torch.cuda.get_device_name(device) if cuda else platform.processor(),
                           'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'), 'threads': args.threads},
                'interpretation': 'Fixed-weight inference-policy comparison; previously observed evaluation data; no new independent confirmation or training ablation.'}
    write_new(output / 'metadata.json', metadata)
    def sample_for(record):
        return load_rgbd(record['rgb_path'], record['depth_path'], dataset.depth_scale)
    completed = {}
    for method in args.methods:
        setting = ensure_empty(output / method)
        prediction_path = setting / 'predictions.jsonl'
        times, diagnostics = [], []
        with torch.inference_mode():
            first = sample_for(dataset.samples[0])
            for _ in range(3):
                run_inference(model, first, method=method)
            del first
            sync()
            if cuda:
                torch.cuda.reset_peak_memory_stats(device)
            with prediction_path.open('x', encoding='utf-8') as stream:
                for record in dataset.samples:
                    sample = sample_for(record)
                    height, width = sample['image'].shape[-2:]
                    sync()
                    started = time.perf_counter()
                    prediction, counts = run_inference(model, sample, method=method)
                    sync()
                    elapsed = time.perf_counter() - started
                    times.append(elapsed)
                    encoded = encode_prediction(prediction, record['id'], record['image_id'], height, width)
                    stream.write(json.dumps(encoded, allow_nan=False) + '\n')
                    stream.flush()
                    diagnostics.append({'sample_id': record['id'], 'latency_ms': elapsed * 1000, **counts})
                    del prediction, encoded, sample
            peak = torch.cuda.max_memory_allocated(device) if cuda else None
            peak_reserved = torch.cuda.max_memory_reserved(device) if cuda else None
        predictions = RLEPredictions(prediction_path)
        metrics = evaluate_predictions(dataset, predictions)
        result = {**metadata, 'method': method, 'status': 'completed', 'metrics': metrics,
                  'predictions_sha256': sha256(prediction_path), 'prediction_format': 'JSONL original-coordinate COCO RLE, binary mask threshold 0.5, same ordered dataset IDs',
                  'original_area_counts': area_counts(dataset), 'per_frame': diagnostics,
                  'timing': {'warmup': 3, 'batch': 1, 'precision': 'float32', 'device': args.device,
                             'scope': 'In-memory CPU RGB-D through device transfer, model preprocessing, all views, original-coordinate masks, CPU postprocessing/fusion. CUDA synchronized. Excludes input IO, RLE serialization, GT loading, metrics and rendering.',
                             'mean_ms': float(np.mean(times) * 1000), 'median_ms': float(np.median(times) * 1000),
                             'p95_ms': float(np.percentile(times, 95) * 1000), 'fps': len(times) / sum(times),
                             'peak_cuda_allocated_bytes': peak, 'peak_cuda_reserved_bytes': peak_reserved,
                             'memory_scope': 'Reset after per-setting warmup; includes resident model weights. Reserved memory may retain cache from previous settings.'}}
        if args.extra:
            from rgbd_seg.instance_metrics import evaluate_extra_predictions
            write_new(setting / 'extra.json', evaluate_extra_predictions(dataset, predictions, 0.5, 0.5, 0.003))
        if sha256(protocol_path) != protocol_sha:
            raise RuntimeError('Experiment protocol changed during evaluation')
        if source_sha != {name: sha256(ROOT / name) for name in source_files}:
            raise RuntimeError('Evaluation source changed during execution')
        if sha256(checkpoint_path) != integrity['checkpoint_sha256']:
            raise RuntimeError('Checkpoint changed during evaluation')
        write_new(setting / 'evaluation.json', result)
        completed[method] = {'metrics': metrics, 'timing': result['timing'], 'evaluation_sha256': sha256(setting / 'evaluation.json')}
        print(json.dumps({'method': method, 'output': str(setting), 'metrics': metrics, 'mean_ms': result['timing']['mean_ms']}), flush=True)
    if dataset_fingerprint(dataset.root) != fingerprint:
        raise RuntimeError('Evaluation dataset changed during execution')
    write_new(output / 'summary.json', {'status': 'completed', 'metadata': metadata, 'settings': completed})


if __name__ == '__main__':
    main()
