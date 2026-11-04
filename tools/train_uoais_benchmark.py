#!/usr/bin/env python3
"""Train the two UOAIS bases with shared data/budget and validation-only selection.

Launch with torchrun. --smoke-steps runs real examples but never yields a formal
benchmark or a selected checkpoint. --resume restores optimizer/scaler/RNG and
the exact image position; each epoch covers the official train indices once.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['WANDB_MODE'] = 'disabled'
os.environ['WANDB_DISABLED'] = 'true'

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from rgbd_seg.uoais_training_data import UOAISTrainingDataset
from rgbd_seg.uoais_dpseg_training import collate_training_examples
from rgbd_seg.uoais_benchmark_training import TrainingExamples, build_model
from rgbd_seg.uois_benchmark_geometry import letterbox_sample, restore_predictions
from rgbd_seg.uois_benchmark_metrics import (
    PAPER_METRIC_NAMES, predictions_to_label_map, evaluate_label_maps, reference_provenance)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def save_torch(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix+'.tmp')
    torch.save(value, temp)
    temp.replace(path)


def gather(value):
    results = [None] * dist.get_world_size()
    dist.all_gather_object(results, value)
    return results


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'])
    torch.cuda.set_rng_state(state['cuda'])


def source_hashes():
    names = ['tools/train_uoais_benchmark.py', 'rgbd_seg/uoais_benchmark_training.py',
             'rgbd_seg/uoais_training_data.py', 'rgbd_seg/uoais_dpseg_training.py',
             'rgbd_seg/uois_benchmark_data.py', 'rgbd_seg/uois_benchmark_geometry.py',
             'rgbd_seg/uois_benchmark_metrics.py', 'rgbd_seg/course_depth_data.py',
             'rgbd_seg/course_dafusion.py', 'rgbd_seg/course_resolution.py',
             'rgbd_seg/course_finetune_mask2former.py', 'rgbd_seg/course_mask2former.py']
    hashes = {name: sha(ROOT/name) for name in names}
    vendor = ROOT / 'third_party/DA-Fusion'
    receipt = json.loads((vendor/'source_receipt.json').read_text())
    for name, record in receipt['files'].items():
        actual = sha(vendor/name)
        if actual != record['sha256']:
            raise ValueError('Changed pinned vendor file: '+name)
        hashes['third_party/DA-Fusion/'+name] = actual
    return hashes


def make_optimizer(model, metadata, config):
    prefixes = metadata['backbone_parameter_prefixes']
    matched = {p: 0 for p in prefixes}
    norm_types = (torch.nn.LayerNorm, torch.nn.GroupNorm, torch.nn.BatchNorm2d,
                  torch.nn.SyncBatchNorm, torch.nn.InstanceNorm2d)
    groups, seen = {}, set()
    for module_name, module in model.named_modules():
        for name, parameter in module.named_parameters(recurse=False):
            if not parameter.requires_grad or parameter in seen:
                continue
            seen.add(parameter)
            full = module_name+'.'+name
            backbone = False
            for prefix in prefixes:
                if full.startswith(prefix):
                    matched[prefix] += parameter.numel()
                    backbone = True
            decay = not (isinstance(module, norm_types+(torch.nn.Embedding,))
                         or parameter.ndim <= 1 or 'relative_position_bias_table' in name
                         or 'absolute_pos_embed' in name)
            groups.setdefault((backbone, decay), []).append(parameter)
    if not all(matched.values()):
        raise ValueError('Unmatched pretrained backbone prefix')
    params, summary = [], []
    for (backbone, decay), tensors in groups.items():
        lr = config['lr'] * (.1 if backbone else 1.)
        wd = config['weight_decay'] if decay else 0.
        params.append({'params': tensors, 'lr': lr, 'base_lr': lr, 'weight_decay': wd})
        summary.append({'pretrained_backbone': backbone, 'lr': lr, 'weight_decay': wd,
                        'parameters': sum(t.numel() for t in tensors)})
    return torch.optim.AdamW(params), summary


def learning_rate_factor(step, total, warmup):
    if step < warmup:
        return .1 + .9*(step+1)/warmup
    return .5*(1+math.cos(math.pi * min(1., (step-warmup)/max(1, total-warmup-1))))


@torch.inference_mode()
def validate(model, dataset, family, out, epoch, *, smoke=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    indices = np.linspace(0, len(dataset)-1, 8, dtype=int).tolist() if smoke else list(range(len(dataset)))
    local = indices[rank::world]
    sums = np.zeros(len(PAPER_METRIC_NAMES), dtype=np.float64)
    model.eval()
    path = out/f'val_epoch{epoch:02d}_rank{rank}.jsonl'
    started = time.monotonic()
    with path.open('w') as stream:
        for position, index in enumerate(local):
            sample, target = dataset[index]
            geometry = None
            if family == 'dpseg':
                sample, geometry = letterbox_sample(sample)
            with torch.autocast('cuda', dtype=torch.float16):
                predictions = model.predict_sample(sample, score_threshold=.01, max_detections=100)
            if geometry is not None:
                predictions = restore_predictions(predictions, geometry)
            # Both implementations use the SAME explicit overlap rule: the most
            # confident retained instance wins a pixel. No GT controls ordering.
            predictions = sorted(predictions, key=lambda row: row['score'])
            h, w = target['masks'].shape[-2:]
            truth = np.zeros((h, w), np.int32)
            for instance_id, mask in enumerate(target['masks'].numpy(), 1):
                truth[mask.astype(bool)] = instance_id
            label_map = predictions_to_label_map(predictions, (h, w), score_threshold=.5)
            result = evaluate_label_maps(label_map, truth)
            values = np.asarray([result['metrics'][name] for name in PAPER_METRIC_NAMES])
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite native validation metric on {dataset.records[index]["sample_id"]}')
            sums += values
            stream.write(json.dumps({'sample_id': dataset.records[index]['sample_id'],
                'image_id': dataset.records[index]['id'], 'predictions': predictions, **result}, allow_nan=False)+'\n')
            if (position+1) % 50 == 0 or position+1 == len(local):
                stream.flush()
                progress = {'status': 'validating', 'epoch': epoch, 'rank': rank,
                            'done': position+1, 'total': len(local), 'updated_utc': now()}
                write_json(out/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
    totals = torch.tensor([*sums, len(local)], dtype=torch.float64, device=model.device)
    dist.all_reduce(totals)
    count = int(totals[-1])
    if count != len(indices):
        raise ValueError('Incomplete distributed validation coverage')
    result = {'epoch': epoch, 'images': count, 'scope': 'engineering_smoke' if smoke else 'full_validation',
              'metrics': {name: float(totals[i]/count) for i, name in enumerate(PAPER_METRIC_NAMES)},
              'score_threshold': .5, 'overlap_rule': 'ascending confidence; later overwrites',
              'elapsed_seconds': time.monotonic()-started, 'test_used': False}
    if rank == 0:
        write_json(out/f'validation_epoch{epoch:02d}.json', result)
    model.train()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--family', choices=['dafusion', 'dpseg'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke-steps', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if args.smoke_steps < 0:
        parser.error('smoke-steps must be nonnegative')
    config = json.loads(args.protocol.read_text())
    if not args.smoke_steps and config.get('status') != 'frozen':
        raise ValueError('Formal training requires a frozen protocol')
    train = config['training']
    rank, local_rank, world = [int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    if train['global_batch'] % world:
        raise ValueError('Global batch must be divisible by GPU count')
    batch_size = train['global_batch']//world
    if batch_size < 1:
        raise ValueError('At least one image per rank is required')
    torch.set_num_threads(train['threads'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    random.seed(train['seed']); np.random.seed(train['seed']); torch.manual_seed(train['seed'])
    out = args.output.resolve()
    if rank == 0:
        if not args.resume and out.exists() and any(out.iterdir()):
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
        if args.resume and (out/'run.json').exists():
            attempt = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
            (out/f'run_before_resume_{attempt}.json').write_bytes((out/'run.json').read_bytes())
    dist.barrier()
    hashes = source_hashes()
    if not args.smoke_steps and hashes != config.get('source_sha256'):
        raise ValueError('Frozen training sources changed')
    protocol_hash = sha(args.protocol)
    run = {'status': 'initializing', 'started_utc': now(), 'family': args.family,
           'scope': 'engineering_smoke' if args.smoke_steps else 'full_train_val',
           'protocol_sha256': protocol_hash, 'source_sha256': hashes,
           'hardware': gather({'rank': rank, 'pid': os.getpid(), 'gpu': torch.cuda.get_device_name(),
                               'visibility': os.environ.get('CUDA_VISIBLE_DEVICES')}),
           'versions': {name: importlib.metadata.version(name) for name in
                        ('torch', 'torchvision', 'transformers', 'numpy', 'pycocotools', 'Pillow')},
           'python': sys.executable, 'test_used': False, 'metrics_reference': reference_provenance()}
    if rank == 0:
        write_json(out/'run.json', run)
        write_json(out/'protocol.json', config)
    try:
        dataset = UOAISTrainingDataset(config['dataset_root'], 'train', config['calibration'])
        val = dataset.for_split('val')
        for split, expected in config['annotations'].items():
            receipt = dataset.annotation_receipts[split]
            if any(receipt[k] != expected[k] for k in ('sha256', 'images', 'annotations')):
                raise ValueError('Official annotation identity changed: '+split)
        if len(dataset) % train['global_batch']:
            raise ValueError('Protocol requires exact full batches without repeated/dropped images')
        checkpoint = config['models'][args.family]['pretrained']
        checkpoint_path = Path(checkpoint)
        weight_file = checkpoint_path/'model.safetensors' if checkpoint_path.is_dir() else checkpoint_path
        if sha(weight_file) != config['models'][args.family]['pretrained_sha256']:
            raise ValueError('Pretrained checkpoint identity changed')
        model, metadata = build_model(args.family, checkpoint, train['seed'])
        model.to(device).train()
        optimizer, groups = make_optimizer(model, metadata, train)
        scaler = torch.amp.GradScaler('cuda', init_scale=256.)
        per_epoch = len(dataset)//train['global_batch']
        total_steps = per_epoch*train['epochs']
        state = {'epoch': 0, 'batch_in_epoch': 0, 'global_step': 0, 'optimizer_updates': 0,
                 'amp_skipped_steps': 0, 'best_overlap_f': -1., 'validation': []}
        if args.resume:
            saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            if (saved['protocol_sha256'] != protocol_hash or saved['source_sha256'] != hashes
                    or saved['scope'] != run['scope'] or saved['family'] != args.family
                    or len(saved['rank_rng']) != world):
                raise ValueError('Resume protocol, code, scope, family or world size changed')
            model.load_state_dict(saved['model'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            scaler.load_state_dict(saved['scaler'])
            state = saved['state']
            resume_rng = saved['rank_rng'][rank]
            if state['global_step'] != state['epoch']*per_epoch + state['batch_in_epoch']:
                raise ValueError('Resume image position is inconsistent')
            if args.smoke_steps and state['global_step'] >= args.smoke_steps:
                raise ValueError('Resume smoke target must exceed the saved step')
            if rank == 0 and (out/'training_log.jsonl').exists():
                rows = (out/'training_log.jsonl').read_text().splitlines(keepends=True)
                keep, rolled_back = [], []
                for row in rows:
                    (keep if json.loads(row)['global_step'] <= state['global_step'] else rolled_back).append(row)
                if rolled_back:
                    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S')
                    (out/f'rolled_back_steps_{stamp}.jsonl').write_text(''.join(rolled_back))
                    (out/'training_log.jsonl').write_text(''.join(keep))
            del saved
        ddp = DistributedDataParallel(model, device_ids=[local_rank],
                                      find_unused_parameters=True, broadcast_buffers=False)
        if args.resume:
            restore_rng(resume_rng)
        else:
            # Equal initialization across ranks; independent dropout/point sampling.
            random.seed(train['seed']+rank); np.random.seed(train['seed']+rank)
            torch.manual_seed(train['seed']+rank)
        run.update(status='training', model_metadata=metadata, optimizer_groups=groups,
                   dataset=dataset.summary(), expected_steps=total_steps,
                   started_training_utc=now())
        if rank == 0:
            write_json(out/'run.json', run)
        print(json.dumps({'rank': rank, 'phase': 'model_and_data_ready',
                          'train_images': len(dataset), 'val_images': len(val),
                          'parameters': sum(p.numel() for p in model.parameters())}), flush=True)

        def save_last():
            rank_rng = gather(rng_state())
            if rank == 0:
                save_torch(out/'last.pt', {'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict(), 'state': state, 'rank_rng': rank_rng,
                    'protocol_sha256': protocol_hash, 'source_sha256': hashes,
                    'scope': run['scope'], 'family': args.family, 'model_metadata': metadata})
            dist.barrier()

        examples = TrainingExamples(dataset, args.family, train['seed'])
        overflow_streak = 0
        started = time.monotonic()
        while state['epoch'] < train['epochs']:
            epoch = state['epoch']
            order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(train['seed']+1009*epoch)).tolist()
            batches = [[(index, epoch) for index in order[start+rank*batch_size:start+(rank+1)*batch_size]]
                       for start in range(state['batch_in_epoch']*train['global_batch'], len(order), train['global_batch'])]
            # A dedicated generator prevents worker creation/resume from consuming
            # model dropout and point-sampling RNG state.
            loader = DataLoader(examples, batch_sampler=batches, num_workers=train['workers'],
                                collate_fn=collate_training_examples,
                                generator=torch.Generator().manual_seed(train['seed']+epoch),
                                multiprocessing_context='fork' if train['workers'] else None)
            step_start = time.monotonic()
            for samples, targets, metas in loader:
                factor = learning_rate_factor(state['global_step'], total_steps, train['warmup_steps'])
                for group in optimizer.param_groups:
                    group['lr'] = group['base_lr']*factor
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast('cuda', dtype=torch.float16):
                    losses = ddp(samples, targets)
                    loss = sum(losses.values())
                finite = torch.isfinite(loss).int()
                dist.all_reduce(finite, op=dist.ReduceOp.MIN)
                if not finite.item():
                    raise FloatingPointError('Nonfinite training loss')
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train['gradient_clip'])
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                skipped = scaler.get_scale() < scale_before
                state['global_step'] += 1
                state['batch_in_epoch'] += 1
                state['optimizer_updates'] += int(not skipped)
                state['amp_skipped_steps'] += int(skipped)
                overflow_streak = overflow_streak+1 if skipped else 0
                if overflow_streak >= 16:
                    raise FloatingPointError('16 consecutive AMP skipped updates')
                values = torch.tensor([float(loss.detach()), sum(m['dropped_instances'] for m in metas),
                                       int(skipped), scaler.get_scale()],
                                      dtype=torch.float64, device=device)
                dist.all_reduce(values)
                # Every rank participates even if overflow unexpectedly differs.
                if int(values[2]) not in (0, world) or float(values[3]) != world*scaler.get_scale():
                    raise RuntimeError('AMP status differs across ranks')
                record = {'updated_utc': now(), 'epoch': epoch+1, 'global_step': state['global_step'],
                          'total_steps': total_steps, 'images_in_epoch': state['batch_in_epoch']*train['global_batch'],
                          'loss': float(values[0]/world), 'dropped_by_augmentation': int(values[1]),
                          'gradient_norm': float(grad_norm) if torch.isfinite(grad_norm) else None,
                          'lr': train['lr']*factor, 'amp_scale': scaler.get_scale(),
                          'amp_skipped': skipped, 'optimizer_updates': state['optimizer_updates'],
                          'step_seconds': time.monotonic()-step_start,
                          'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
                if rank == 0:
                    with (out/'training_log.jsonl').open('a') as stream:
                        stream.write(json.dumps(record, allow_nan=False)+'\n')
                if state['global_step'] <= 3 or state['global_step'] % 25 == 0:
                    write_json(out/f'progress_rank{rank}.json', {'status': 'training', 'rank': rank, **record})
                    print(json.dumps({'rank': rank, **record}), flush=True)
                if state['global_step'] % train['save_every'] == 0:
                    save_last()
                if args.smoke_steps and state['global_step'] >= args.smoke_steps:
                    break
                step_start = time.monotonic()
            del loader
            if args.smoke_steps:
                if not state['optimizer_updates']:
                    raise RuntimeError('Real-data check made no optimizer update')
                save_last()
                validate(model, val, args.family, out, epoch+1, smoke=True)
                break
            if state['batch_in_epoch'] != per_epoch:
                raise RuntimeError('Training epoch coverage is incomplete')
            save_last()
            result = validate(model, val, args.family, out, epoch+1)
            state['validation'].append(result)
            overlap_f = result['metrics']['Objects F-measure']
            if overlap_f > state['best_overlap_f']:
                state['best_overlap_f'] = overlap_f
                if rank == 0:
                    save_torch(out/'best.pt', {'model': model.state_dict(), 'model_metadata': metadata,
                        'family': args.family, 'selection': result, 'protocol_sha256': protocol_hash,
                        'source_sha256': hashes, 'scope': run['scope']})
            state['epoch'] += 1
            state['batch_in_epoch'] = 0
            save_last()
        if rank == 0:
            run.update(status='smoke_passed' if args.smoke_steps else 'training_completed',
                       finished_utc=now(), state=state, elapsed_training_seconds=time.monotonic()-started,
                       last_checkpoint_sha256=sha(out/'last.pt'))
            write_json(out/'run.json', run)
        dist.barrier()
    except BaseException:
        failure = {'status': 'failed', 'time': now(), 'rank': rank,
                   'traceback': traceback.format_exc()}
        write_json(out/f'failure_rank{rank}.json', failure)
        write_json(out/f'progress_rank{rank}.json', failure)
        traceback.print_exc()
        sys.stdout.flush(); sys.stderr.flush()
        # Collective NCCL teardown can wait forever when another rank is still
        # in backward. Exit the failed worker so torchrun terminates its peers.
        os._exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
