#!/usr/bin/env python3
"""Continue the RGB control on a declared free GPU set, retaining AdamW state.

The training loop is copied unchanged from run_course_longer_training.py.
Only RGB construction, source receipts and the historical GPU4 exclusion differ.
"""
import argparse
from contextlib import nullcontext
from datetime import timedelta
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from tools.run_course_dafusion import (now, sha256, write_json, save_torch,
    epoch_order, lr_factor, make_optimizer, rng_state, restore_rng,
    runtime_source_sha256 as baseline_sources, archive_rolled_back_log)
from rgbd_seg.course_resolution import rank_batches
from tools.run_course_progressive_rgb import evaluate_rgb as evaluate, sources as progressive_sources
from rgbd_seg.course_longer_training import continuation_lr
from rgbd_seg.course_progressive_rgb import build_trainable, check_parent_checkpoint


def sources():
    result = progressive_sources()
    for name in ('tools/run_course_progressive_rgb_multigpu.py',
                 'tools/supervise_course_progressive_rgb_multigpu.py'):
        result[name] = sha256(ROOT/name)
    return result


def model_digest(model):
    digest = hashlib.sha256()
    for name, tensor in model.state_dict().items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape), value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def all_objects(value, world):
    result = [None] * world
    dist.all_gather_object(result, value)
    return result


def check_model_sync(adapter, world):
    digests = all_objects(model_digest(adapter), world)
    if len(set(digests)) != 1:
        raise RuntimeError(f'Models diverged across GPUs: {digests}')
    return digests[0]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--resolution', type=int, choices=[800, 1024], required=True)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--smoke-tail-images', type=int, default=0)
    p.add_argument('--smoke-dense', action='store_true')
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    rank, local_rank, world = [int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    train = protocol['training']
    parent_epoch = protocol['continuation']['parent_epoch']
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if (world != len(protocol['hardware']['physical_gpus']) or len(visible) != world or len(set(visible)) != world
            or any(not n.isdecimal() for n in visible)
            or visible != protocol['hardware']['physical_gpus']):
        raise ValueError('Use exactly the frozen available GPU set')
    if (args.smoke_steps < 0 or args.smoke_tail_images < 0
            or (not args.smoke_steps and (protocol['status'] != 'frozen' or args.smoke_tail_images or args.smoke_dense))):
        raise ValueError('Formal runs require a frozen protocol')
    if not args.smoke_steps:
        gate = protocol.get('engineering_validation')
        if not gate or sha256(gate['path']) != gate['sha256']:
            raise ValueError('Formal training requires a verified engineering receipt')
        receipt = json.loads(Path(gate['path']).read_text())
        if receipt['status'] != 'passed' or receipt['source_sha256'] != protocol['source_sha256']:
            raise ValueError('Engineering receipt is stale or failed')
    torch.set_num_threads(train['threads'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_per_process_memory_fraction(.90, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    seed = train['seed']
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    out = args.output.resolve()
    if rank == 0:
        if not args.resume and out.exists() and any(out.iterdir()):
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    began = time.monotonic()
    source_hashes = sources()
    if source_hashes != protocol['source_sha256']:
        raise ValueError('Frozen runtime sources changed')
    protocol_sha = sha256(args.protocol)
    run = {'status': 'initializing', 'started_at_utc': now(),
        'resolution': args.resolution, 'world_size': world,
        'scope': 'engineering_smoke' if args.smoke_steps else 'full_train_val',
        'protocol_sha256': protocol_sha, 'source_sha256': source_hashes,
        'test_used': False, 'arguments': {k: str(v) if isinstance(v, Path) else v for k,v in vars(args).items()},
        'ranks': all_objects({'rank': rank, 'pid': os.getpid(), 'local_rank': local_rank,
            'physical_gpu': visible[rank], 'gpu': torch.cuda.get_device_name(device)}, world),
        'versions': {n: importlib.metadata.version(n) for n in ('torch', 'transformers', 'numpy', 'pycocotools')}}
    if rank == 0:
        if args.resume:
            old = json.loads((out/'run.json').read_text())
            if (old['status'] == 'completed' or old['protocol_sha256'] != protocol_sha
                    or old['source_sha256'] != source_hashes or old['resolution'] != args.resolution):
                raise ValueError('Resume provenance/status changed')
            run['previous_run'] = old
        if not args.smoke_steps:
            checks = {'archive': sha256(protocol['archive']),
                'pretrained': sha256(Path(protocol['checkpoint'])/'model.safetensors'),
                'data_audit': sha256(protocol['data_audit']['path'])}
            if checks != protocol['input_sha256']:
                raise ValueError('Frozen data/checkpoint identity changed')
            run['verified_inputs'] = checks
        write_json(out/'run.json', run)
    dist.barrier()
    dataset = None

    def guard():
        if time.monotonic()-began > protocol['limits']['per_trial_seconds']:
            raise TimeoutError('Declared trial time budget exceeded')
        if shutil.disk_usage(out).free < 3 * 1024**3:
            raise OSError('Disk reserve below 3 GiB')

    try:
        from rgbd_seg.course_depth_data import CourseDepthTrainingDataset, collate_fn
        dataset = CourseDepthTrainingDataset(protocol['archive'], split='train', augment=True, seed=seed)
        if len(dataset) != 14432:
            raise ValueError('Unexpected train split')
        data_summary = dataset.summary()
        audit = json.loads(Path(protocol['data_audit']['path']).read_text())
        key = 'annotation_member_name_and_bytes_sha256'
        if data_summary[key] != audit['splits']['train'][key]:
            raise ValueError('Training annotations changed')
        adapter, metadata = build_trainable(protocol['checkpoint'], device=device,
            seed=seed, resolution=args.resolution)
        optimizer, groups = make_optimizer(adapter, metadata, train['lr'])
        parent_spec = protocol['continuation']
        for key in ('weights', 'protocol'):
            item = parent_spec[key]
            if sha256(item['path']) != item['sha256']:
                raise ValueError(f'Parent identity changed: {key}')
        parent_protocol = json.loads(Path(parent_spec['protocol']['path']).read_text())
        for name, expected in parent_protocol['source_sha256'].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            if sha256(path) != expected:
                raise ValueError(f'Historical source changed: {name}')
        parent = torch.load(parent_spec['weights']['path'], map_location='cpu', weights_only=False)
        parent_selection = check_parent_checkpoint(parent, parent_protocol,
            parent_spec['protocol']['sha256'], metadata, parent_epoch)
        adapter.load_state_dict(parent['model'], strict=True)
        optimizer.load_state_dict(parent['optimizer'])
        for group, summary in zip(optimizer.param_groups, groups):
            group['base_lr'] = summary['base_lr']
            group['lr'] = summary['base_lr']
        parent_global_step = parent['state']['global_step']
        del parent
        initial_digest = check_model_sync(adapter, world)
        if initial_digest != parent_selection['model_sha256']:
            raise ValueError('Initial model differs from the validated parent tensors')
        ddp = DDP(adapter, device_ids=[local_rank], output_device=local_rank,
                  broadcast_buffers=False, find_unused_parameters=True,
                  gradient_as_bucket_view=True)
        # Rank layout changed: explicitly reseed RNG identically in both arms.
        # AdamW moments are retained; this is not an exact old RNG continuation.
        random.seed(seed+rank); np.random.seed(seed+rank); torch.manual_seed(seed+rank)
        batches = train['per_rank_batch_sizes']
        if len(batches) != world:
            raise ValueError('Per-rank batch layout differs from world size')
        batch = batches[rank]
        accumulation = train['accumulation']
        micro_global = sum(batches)
        effective = micro_global * accumulation
        if effective != train['effective_batch_size']:
            raise ValueError('Global batch differs from protocol')
        n_train = args.smoke_steps*effective if args.smoke_steps else len(dataset)
        if args.smoke_tail_images:
            if not 0 < args.smoke_tail_images < effective:
                raise ValueError('Invalid engineering tail batch')
            n_train = (args.smoke_steps-1)*effective + args.smoke_tail_images
        if n_train > len(dataset):
            raise ValueError('Smoke exceeds dataset')
        n_epochs = 1 if args.smoke_steps else train['epochs']
        total_steps = math.ceil(n_train/effective)*n_epochs
        state = {'epoch': 0, 'images_in_epoch': 0, 'global_step': 0,
                 'stage': 'training', 'epoch_loss_sum': 0.,
                 'best_key': [parent_selection['metrics']['mask_ap_small'], parent_selection['metrics']['mask_ap']],
                 'best_epoch': parent_epoch, 'val_results': []}
        if rank == 0 and not args.resume:
            save_torch(out/'best.pt', {'model': adapter.state_dict(), 'model_metadata': metadata,
                'epoch': parent_epoch, 'selection': parent_selection,
                'source_sha256': source_hashes, 'protocol_sha256': protocol_sha,
                'parent_initial_candidate': True, 'continuation': parent_spec})
        if args.resume:
            checkpoint = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            if (checkpoint['protocol_sha256'] != protocol_sha or checkpoint['source_sha256'] != source_hashes
                    or checkpoint['model_metadata'] != metadata):
                raise ValueError('Checkpoint provenance changed')
            adapter.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            state = checkpoint['state']
            if ((state['images_in_epoch'] != n_train and state['images_in_epoch'] % effective)
                    or len(state['val_results']) != state['epoch']
                    or state['global_step'] != state['epoch']*math.ceil(n_train/effective)+math.ceil(state['images_in_epoch']/effective)):
                raise ValueError('Checkpoint counters disagree')
            restore_rng(checkpoint['rank_rng'][rank])
            del checkpoint
            check_model_sync(adapter, world)
            if rank == 0:
                archive_rolled_back_log(out/'training_log.jsonl', state['global_step'])
            archive_rolled_back_log(out/f'train_rank{rank}.jsonl', state['global_step'])
        run.update(model_metadata=metadata, training_data=data_summary, optimizer_groups=groups,
            initial_model_sha256=initial_digest, expected_optimizer_steps=total_steps,
            expected_training_image_exposures=n_train*n_epochs, effective_batch_size=effective,
            parent_epoch=parent_epoch, parent_global_step=parent_global_step, continuation=parent_spec,
            parent_validation=parent_selection,
            loss_aggregation='Native local microbatch losses weighted by image count across accumulation and DDP ranks',
            numerical_equivalence_to_historical_single_gpu=False)
        if rank == 0:
            write_json(out/'run.json', run)
            write_json(out/'initialization.json', metadata)

        def save_last():
            guard()
            rank_rng = all_objects(rng_state(), world)
            if rank == 0:
                save_torch(out/'last.pt', {'model': adapter.state_dict(), 'optimizer': optimizer.state_dict(),
                    'state': state, 'rank_rng': rank_rng, 'model_metadata': metadata,
                    'protocol_sha256': protocol_sha, 'source_sha256': source_hashes})
            dist.barrier()

        with (out/f'train_rank{rank}.jsonl').open('a') as rank_log:
            while state['epoch'] < n_epochs:
                epoch = state['epoch']
                absolute_epoch = parent_epoch+epoch+1
                dataset.set_epoch(parent_epoch+epoch)
                order = epoch_order(len(dataset), seed, parent_epoch+epoch)[:n_train]
                if args.smoke_dense:
                    order = sorted(range(len(dataset)), key=lambda i: (
                        -len(dataset.annotations_by_image[dataset.image_ids[i]]), dataset.image_ids[i]))[:n_train]
                local_batches = rank_batches(order, rank=rank, batch_sizes=batches)
                if rank == 0:
                    write_json(out/f'train_order_epoch_{absolute_epoch:02d}.json', {
                        'epoch': absolute_epoch, 'image_ids': [dataset.image_ids[i] for i in order],
                        'rank_batches': [rank_batches(order, rank=r,
                            batch_sizes=batches) for r in range(world)]})
                if state['stage'] == 'training':
                    adapter.train()
                    run.update(status='training', epoch=absolute_epoch)
                    if rank == 0:
                        write_json(out/'run.json', run)
                    dataset.close()
                    consumed_microbatches = math.ceil(state['images_in_epoch']/micro_global)
                    remaining_batches = local_batches[consumed_microbatches:]
                    loader = DataLoader(dataset, batch_sampler=remaining_batches,
                        num_workers=train['workers_per_rank'],
                        collate_fn=collate_fn, pin_memory=True, persistent_workers=False,
                        generator=torch.Generator().manual_seed(seed+7919*epoch+rank),
                        **({'prefetch_factor': 2, 'multiprocessing_context': 'spawn'} if train['workers_per_rank'] else {}))
                    optimizer.zero_grad(set_to_none=True)
                    group_loss = 0.
                    group_empty = 0
                    step_started = time.perf_counter()
                    for micro, (samples, targets) in enumerate(loader):
                        guard()
                        group_target = min(effective, n_train-state['images_in_epoch'])
                        end_group = (micro+1) % accumulation == 0 or micro+1 == len(remaining_batches)
                        schedule = train['schedule']
                        # Smoke uses the initial learning rates of the formal run.
                        schedule_total = train['updates_per_epoch']*train['epochs']
                        factor = continuation_lr(state['global_step'], schedule_total,
                            peak_lr=train['lr'], start_lr=schedule['start_lr'],
                            warmup_steps=schedule['warmup_steps'],
                            minimum_ratio=schedule['minimum_ratio'])/train['lr']
                        for group in optimizer.param_groups:
                            group['lr'] = group['base_lr']*factor
                        with (nullcontext() if end_group else ddp.no_sync()):
                            with torch.autocast('cuda', dtype=torch.bfloat16):
                                losses = ddp(samples, targets)
                                loss = sum(losses.values())
                            if loss.ndim != 0 or not torch.isfinite(loss):
                                raise FloatingPointError('Non-finite native loss')
                            # DDP divides by world size. Correct it so different
                            # local batch sizes contribute in proportion to images.
                            (loss * (world*len(samples)/group_target)).backward()
                        group_loss += float(loss.detach())*len(samples)/group_target
                        group_empty += sum(adapter.last_batch_metadata['resized_empty_instances'])
                        del loss, losses
                        if not end_group:
                            continue
                        norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        state['global_step'] += 1
                        state['images_in_epoch'] += group_target
                        values = torch.tensor([group_loss, group_empty], device=device, dtype=torch.float64)
                        dist.all_reduce(values)
                        mean_loss, empty = float(values[0]), int(values[1])
                        state['epoch_loss_sum'] += mean_loss*group_target
                        record = {'epoch': absolute_epoch, 'global_step': state['global_step'],
                            'total_steps': total_steps, 'total_global_step': parent_global_step+state['global_step'],
                            'images_in_epoch': state['images_in_epoch'],
                            'expected_train_images': n_train, 'loss': mean_loss,
                            'global_batch_images': group_target,
                            'local_loss': group_loss, 'resized_empty_instances': empty,
                            'grad_norm_before_clip': float(norm), 'head_lr': train['lr']*factor,
                            'step_seconds_with_data': time.perf_counter()-step_started,
                            'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(device)}
                        rank_log.write(json.dumps(record, allow_nan=False)+'\n')
                        if rank == 0:
                            with (out/'training_log.jsonl').open('a') as stream:
                                stream.write(json.dumps(record, allow_nan=False)+'\n')
                        if state['global_step'] == 1 or state['global_step'] % 25 == 0 or state['images_in_epoch'] == n_train:
                            rank_log.flush()
                            write_json(out/f'progress_rank{rank}.json', {'status': 'training', **record})
                            print(json.dumps({'rank': rank, **record}), flush=True)
                        if state['global_step'] % train['save_every'] == 0:
                            save_last()
                        group_loss, group_empty = 0., 0
                        step_started = time.perf_counter()
                    del loader
                    if state['images_in_epoch'] != n_train:
                        raise RuntimeError('Incomplete image coverage')
                    state['stage'] = 'validation'
                    save_last()
                digest = check_model_sync(adapter, world)
                if rank == 0:
                    write_json(out/f'model_sync_epoch_{absolute_epoch:02d}.json', {
                        'status': 'pass', 'world_size': world, 'model_sha256': digest})
                    run.update(status='validating', epoch=absolute_epoch)
                    write_json(out/'run.json', run)
                val_dir = out/f'val_epoch_{absolute_epoch:02d}'
                if args.resume:
                    attempt = 1
                    while val_dir.exists():
                        attempt += 1
                        val_dir = out/f'val_epoch_{absolute_epoch:02d}_attempt{attempt}'
                result = evaluate(adapter, protocol['archive'], val_dir, rank=rank,
                    world_size=world, guard=guard, smoke=bool(args.smoke_steps))
                metrics = result['metrics']
                key = (metrics['mask_ap_small'], metrics['mask_ap'])
                if args.smoke_steps:
                    key = tuple(0. if value is None else value for value in key)
                if any(value is None or not np.isfinite(value) for value in key):
                    raise ValueError('Invalid full validation selection metric')
                entry = {'epoch': absolute_epoch, 'metrics': metrics, 'output': str(val_dir),
                    'global_step': state['global_step'], 'training_images': n_train,
                    'training_loss_mean': state['epoch_loss_sum']/n_train, 'model_sha256': digest}
                state['val_results'].append(entry)
                if rank == 0:
                    save_torch(out/f'epoch_{absolute_epoch:02d}.pt', {
                        'model': adapter.state_dict(), 'model_metadata': metadata,
                        'epoch': absolute_epoch, 'selection': entry,
                        'source_sha256': source_hashes, 'protocol_sha256': protocol_sha})
                if key > tuple(state['best_key']):
                    state.update(best_key=list(key), best_epoch=absolute_epoch)
                    if rank == 0:
                        save_torch(out/'best.pt', {'model': adapter.state_dict(), 'model_metadata': metadata,
                            'epoch': absolute_epoch, 'selection': entry, 'source_sha256': source_hashes,
                            'protocol_sha256': protocol_sha})
                if rank == 0:
                    write_json(out/'selection.json', {'best_epoch': state['best_epoch'],
                        'best_key': state['best_key'], 'epochs': state['val_results'],
                        'parent_candidate': parent_selection,
                        'rule': 'Validation APsmall, then AP, then earlier epoch', 'test_used': False})
                state.update(epoch=epoch+1, images_in_epoch=0, epoch_loss_sum=0., stage='training')
                save_last()
        peaks = all_objects(torch.cuda.max_memory_allocated(device), world)
        if rank == 0:
            run.update(status='completed', finished_at_utc=now(), elapsed_seconds=time.monotonic()-began,
                best_epoch=state['best_epoch'], best_validation_key=state['best_key'],
                best_checkpoint_sha256=sha256(out/'best.pt'), optimizer_steps=state['global_step'],
                training_image_exposures=n_train*n_epochs, rank_peak_allocated_bytes=peaks)
            write_json(out/'run.json', run)
            print(json.dumps({'status': 'completed', 'resolution': args.resolution,
                'best_validation_key': state['best_key']}), flush=True)
        dist.barrier()
    except BaseException as error:
        write_json(out/f'failure_rank{rank}.json', {'status': 'failed', 'at_utc': now(),
            'error_type': type(error).__name__, 'error': str(error), 'pid': os.getpid()})
        if rank == 0:
            run.update(status='failed', stopped_at_utc=now(), error=str(error))
            write_json(out/'run.json', run)
        raise
    finally:
        if dataset is not None:
            dataset.close()
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
