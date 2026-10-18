#!/usr/bin/env python3
"""Fine-tune either course RGB baseline, select on full val, then test once."""
import argparse
from datetime import datetime, timezone
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
from torch.utils.data import DataLoader, Subset


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def save_torch(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(value, temporary)
    temporary.replace(path)


def lr_factor(step, total, warmup=250):
    warmup = min(warmup, max(1, total // 10))
    if step < warmup:
        return .1 + .9 * (step + 1) / warmup
    ratio = (step - warmup) / max(1, total - warmup - 1)
    return .1 + .9 * .5 * (1 + math.cos(math.pi * min(1., ratio)))


def selection_key(result):
    metrics = result['metrics']
    values = (metrics['mask_ap_small'], metrics['mask_ap'])
    if any(value is None or not math.isfinite(value) for value in values):
        raise ValueError('Selection requires finite APsmall and AP on the complete validation split')
    return values


def epoch_order(n, seed, epoch):
    return torch.randperm(n, generator=torch.Generator().manual_seed(seed + 1009*epoch)).tolist()


def validate_resume_state(state, n_train, effective_batch, n_epochs):
    epoch, images, step = (state[key] for key in ('epoch', 'images_in_epoch', 'global_step'))
    if any(type(v) is not int for v in (epoch, images, step)):
        raise ValueError('Checkpoint counters must be integers')
    if not 0 <= epoch <= n_epochs or not 0 <= images <= n_train:
        raise ValueError('Checkpoint counters outside this run')
    if epoch == n_epochs and (images or state['stage'] != 'training'):
        raise ValueError('Completed training checkpoint has inconsistent position')
    if images != n_train and images % effective_batch:
        raise ValueError('Checkpoint is not at an optimizer-step boundary')
    if state['stage'] not in ('training', 'validation') or (state['stage'] == 'validation' and images != n_train):
        raise ValueError('Checkpoint stage disagrees with completed images')
    if step != epoch*math.ceil(n_train/effective_batch) + math.ceil(images/effective_batch):
        raise ValueError('Checkpoint optimizer steps disagree with sample coverage')
    if len(state['val_results']) != epoch:
        raise ValueError('Checkpoint validation history disagrees with trained epochs')


def archive_rolled_back_log(path, global_step):
    if not path.exists():
        return None
    lines = path.read_text().splitlines(keepends=True)
    kept, rolled_back = [], []
    for line in lines:
        step = json.loads(line)['global_step']
        (kept if step <= global_step else rolled_back).append(line)
    if not rolled_back:
        return None
    archive = path.with_name(f'training_log_rolled_back_{time.time_ns()}.jsonl')
    archive.write_text(''.join(rolled_back))
    temporary = path.with_suffix('.tmp')
    temporary.write_text(''.join(kept)); temporary.replace(path)
    return {'path': str(archive), 'rows': len(rolled_back), 'reason': 'Steps after the last saved checkpoint; retained as failed-attempt evidence'}


def build_adapter(name, checkpoint, device, seed):
    if name == 'maskrcnn':
        from rgbd_seg.course_finetune_maskrcnn import build_trainable
    else:
        from rgbd_seg.course_finetune_mask2former import build_trainable
    return build_trainable(checkpoint, device=device, seed=seed)


def make_optimizer(adapter, metadata, learning_rate):
    prefix = metadata['backbone_parameter_prefix']
    grouped, summary = {}, []
    for name, parameter in adapter.named_parameters():
        if not parameter.requires_grad:
            continue
        backbone = name.startswith(prefix)
        decay = not (parameter.ndim <= 1 or name.endswith('.bias') or 'embed' in name.lower())
        key = (backbone, decay)
        grouped.setdefault(key, []).append(parameter)
    if not any(backbone for backbone, _ in grouped):
        raise ValueError(f'No backbone matched declared prefix {prefix}')
    groups = []
    for (backbone, decay), parameters in grouped.items():
        rate, wd = learning_rate * (.1 if backbone else 1), .05 if decay else 0.
        groups.append({'params': parameters, 'lr': rate, 'base_lr': rate, 'weight_decay': wd})
        summary.append({'backbone': backbone, 'weight_decay': wd, 'base_lr': rate,
                        'parameters': sum(p.numel() for p in parameters)})
    return torch.optim.AdamW(groups), summary


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch']); torch.cuda.set_rng_state_all(state['cuda'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', choices=['maskrcnn', 'mask2former'], required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--protocol', type=Path)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--accumulation', type=int, default=1)
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--lr', type=float, default=.0002)
    p.add_argument('--precision', choices=['bf16', 'fp32'], default='bf16')
    p.add_argument('--save-every', type=int, default=300)
    p.add_argument('--timeout-seconds', type=int, default=28800)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.batch_size, args.accumulation, args.threads, args.save_every, args.timeout_seconds) <= 0:
        raise ValueError('Training limits must be positive')
    if args.workers < 0 or args.smoke_steps < 0:
        raise ValueError('workers/smoke-steps must be nonnegative')
    if not str(args.device).startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('This full-data run requires CUDA; no silent CPU fallback')
    torch.set_num_threads(args.threads)
    torch.cuda.set_device(args.device)
    torch.cuda.set_per_process_memory_fraction(.85, args.device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    if args.precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise RuntimeError('Requested BF16 is unsupported')
    from rgbd_seg.course_training_data import CourseTrainingDataset, collate_fn
    from rgbd_seg.course_finetune_eval import evaluate
    out = args.output.resolve()
    if not args.resume and out.exists() and any(out.iterdir()):
        raise FileExistsError('New runs require a new or empty output directory')
    out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    source_names = ['tools/run_course_finetune.py', 'rgbd_seg/course_training_data.py',
                    'rgbd_seg/course_finetune_eval.py', f'rgbd_seg/course_finetune_{args.model}.py',
                    'rgbd_seg/course_archive.py', 'rgbd_seg/course_submission.py']
    sources = {name: sha256(ROOT/name) for name in source_names}
    protocol_sha = sha256(args.protocol) if args.protocol else None
    if not args.smoke_steps:
        if args.protocol is None:
            raise ValueError('Formal training requires a frozen protocol')
        protocol = json.loads(args.protocol.read_text())
        for key in ['epochs', 'batch_size', 'accumulation', 'seed', 'lr', 'precision', 'workers', 'threads']:
            if getattr(args, key) != protocol['training'][key]:
                raise ValueError(f'Command differs from frozen training protocol: {key}')
        if protocol['models'][args.model]['gpu'] != os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise ValueError('CUDA device differs from frozen protocol')
        if str(args.archive.resolve()) != protocol['archive']:
            raise ValueError('Archive differs from frozen protocol')
        if str(args.checkpoint.resolve()) != protocol['models'][args.model]['pretrained_checkpoint']:
            raise ValueError('Initialization checkpoint differs from frozen protocol')
    run = {'status': 'initializing', 'started_at_utc': now(), 'pid': os.getpid(),
           'model': args.model, 'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           'scope': 'engineering_smoke' if args.smoke_steps else 'full_train_val_test',
           'source_sha256': sources, 'protocol_sha256': protocol_sha,
           'physical_cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
           'device_name': torch.cuda.get_device_name(args.device),
           'versions': {n: importlib.metadata.version(n) for n in ['torch', 'torchvision', 'transformers', 'pycocotools', 'numpy']},
           'depth_used': False, 'test_used_for_checkpoint_selection': False}
    if args.resume:
        previous = json.loads((out/'run.json').read_text())
        if previous['source_sha256'] != sources or previous['protocol_sha256'] != protocol_sha:
            raise ValueError('Resume source/protocol changed; explicit migration required')
        if previous['status'] == 'completed':
            raise ValueError('Run is already completed')
        for key, value in previous['arguments'].items():
            if key not in ('resume', 'timeout_seconds') and run['arguments'].get(key) != value:
                raise ValueError(f'Resume argument changed: {key}')
        run['previous_run_before_resume'] = previous
    write_json(out/'run.json', run)

    def guard():
        if time.monotonic() - started > args.timeout_seconds:
            raise TimeoutError('Declared per-process hard time budget exceeded')
        if shutil.disk_usage(out).free < 2 * 1024**3:
            raise OSError('Disk reserve below 2 GiB; existing checkpoints and logs retained')

    def eval_progress(value):
        guard()
        record = {'phase': run['status'], 'epoch': run.get('epoch'), **value}
        write_json(out/'progress.json', record)
        print(json.dumps(record), flush=True)

    dataset = None
    try:
        guard()
        dataset = CourseTrainingDataset(args.archive, split='train', augment=True, seed=args.seed)
        if len(dataset) != 14432:
            raise ValueError('Course training split must contain all 14,432 images')
        adapter, metadata = build_adapter(args.model, args.checkpoint, args.device, args.seed)
        optimizer, groups = make_optimizer(adapter, metadata, args.lr)
        run.update(model_metadata=metadata, optimizer_groups=groups, training_data=dataset.summary())
        write_json(out/'run.json', run)
        write_json(out/'initialization.json', metadata)
        effective_batch = args.batch_size * args.accumulation
        n_train = min(len(dataset), args.smoke_steps * effective_batch) if args.smoke_steps else len(dataset)
        n_epochs = 1 if args.smoke_steps else args.epochs
        epoch_steps = math.ceil(n_train/effective_batch)
        total_steps = epoch_steps * n_epochs
        state = {'epoch': 0, 'images_in_epoch': 0, 'global_step': 0,
                 'epoch_loss_sum': 0., 'best_key': [-1., -1.], 'best_epoch': None, 'val_results': [],
                 'stage': 'training'}
        if args.resume:
            checkpoint = torch.load(out/'last.pt', map_location=args.device, weights_only=False)
            if checkpoint['source_sha256'] != sources or checkpoint['protocol_sha256'] != protocol_sha:
                raise ValueError('Checkpoint provenance differs from this process')
            adapter.load_state_dict(checkpoint['model'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer'])
            state = checkpoint['state']
            validate_resume_state(state, n_train, effective_batch, n_epochs)
            rolled_back = archive_rolled_back_log(out/'training_log.jsonl', state['global_step'])
            write_json(out/f'resume_{time.time_ns()}.json', {'at_utc': now(), 'checkpoint_sha256': sha256(out/'last.pt'),
                       'checkpoint_state': state, 'rolled_back_log': rolled_back})
            # CPU RNG tensors must not remain mapped to CUDA by torch.load.
            checkpoint['rng']['torch'] = checkpoint['rng']['torch'].cpu()
            checkpoint['rng']['cuda'] = [v.cpu() for v in checkpoint['rng']['cuda']]
            restore_rng(checkpoint['rng'])
            del checkpoint

        def save_last():
            guard()
            save_torch(out/'last.pt', {'model': adapter.state_dict(), 'optimizer': optimizer.state_dict(),
                       'state': state, 'rng': rng_state(), 'source_sha256': sources,
                       'protocol_sha256': protocol_sha, 'saved_at_utc': now()})

        with (out/'training_log.jsonl').open('a') as train_log:
            while state['epoch'] < n_epochs:
                epoch = state['epoch']
                dataset.set_epoch(epoch)
                order = epoch_order(len(dataset), args.seed, epoch)[:n_train]
                ids_in_order = [dataset.image_ids[index] for index in order]
                manifest = {'epoch': epoch+1, 'count': len(order), 'image_ids': ids_in_order,
                            'ordered_image_ids_sha256': hashlib.sha256(json.dumps(ids_in_order).encode()).hexdigest()}
                manifest_path = out/f'train_order_epoch_{epoch+1:02d}.json'
                if manifest_path.exists():
                    if json.loads(manifest_path.read_text()) != manifest:
                        raise ValueError('Resume shuffled order differs')
                else:
                    write_json(manifest_path, manifest)
                if state['stage'] == 'training':
                    adapter.train()
                    run.update(status='training', epoch=epoch+1)
                    write_json(out/'run.json', run)
                    dataset.close()
                    loader = DataLoader(Subset(dataset, order[state['images_in_epoch']:]),
                                        batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                                        collate_fn=collate_fn, pin_memory=True, persistent_workers=False,
                                        generator=torch.Generator().manual_seed(args.seed+7919*epoch),
                                        **({'prefetch_factor': 2} if args.workers else {}))
                    optimizer.zero_grad(set_to_none=True)
                    group_images, group_loss, group_target = 0, 0., 0
                    step_started = time.perf_counter()
                    for images, targets in loader:
                        guard()
                        if group_images == 0:
                            group_target = min(effective_batch, n_train - state['images_in_epoch'])
                            factor = lr_factor(state['global_step'], total_steps)
                            for group in optimizer.param_groups:
                                group['lr'] = group['base_lr'] * factor
                        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.precision == 'bf16'):
                            loss_dict = adapter(images, targets)
                            loss = sum(loss_dict.values())
                        if loss.ndim != 0 or not torch.isfinite(loss):
                            raise FloatingPointError('Training loss is not a finite scalar')
                        (loss * (len(images)/group_target)).backward()
                        value = float(loss.detach())
                        group_images += len(images)
                        group_loss += value * len(images)
                        state['images_in_epoch'] += len(images)
                        state['epoch_loss_sum'] += value * len(images)
                        if group_images == group_target:
                            norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1., error_if_nonfinite=True)
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            state['global_step'] += 1
                            record = {'epoch': epoch+1, 'global_step': state['global_step'],
                                      'total_steps': total_steps, 'images_in_epoch': state['images_in_epoch'],
                                      'expected_train_images': n_train, 'loss': group_loss/group_images,
                                      'grad_norm_before_clip': float(norm), 'head_lr': args.lr*factor,
                                      'step_seconds_with_data': time.perf_counter()-step_started,
                                      'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(args.device)}
                            train_log.write(json.dumps(record, allow_nan=False)+'\n')
                            if state['global_step'] == 1 or state['global_step'] % 25 == 0 or state['images_in_epoch'] == n_train:
                                train_log.flush()
                                write_json(out/'progress.json', {'status': 'training', **record})
                                print(json.dumps(record), flush=True)
                            if state['global_step'] % args.save_every == 0:
                                save_last()
                            group_images, group_loss = 0, 0.
                            step_started = time.perf_counter()
                    if state['images_in_epoch'] != n_train or group_images:
                        raise RuntimeError('Training did not consume the declared complete image order')
                    del loader
                    state['stage'] = 'validation'
                    save_last()
                run.update(status='validating', epoch=epoch+1)
                write_json(out/'run.json', run)
                val_dir = out/f'val_epoch_{epoch+1:02d}'
                if val_dir.exists() and any(val_dir.iterdir()):
                    # A resumed evaluation is an explicit new attempt, never overwrite old evidence.
                    if not args.resume:
                        raise FileExistsError(val_dir)
                    attempt = 2
                    while (out/f'val_epoch_{epoch+1:02d}_attempt{attempt}').exists():
                        attempt += 1
                    val_dir = out/f'val_epoch_{epoch+1:02d}_attempt{attempt}'
                result = evaluate(adapter, args.archive, 'val', val_dir, progress_callback=eval_progress,
                                  max_images=3 if args.smoke_steps else None)
                entry = {'epoch': epoch+1, 'metrics': result['metrics'], 'output': str(val_dir),
                         'training_loss_mean': state['epoch_loss_sum']/n_train,
                         'training_images': n_train, 'global_step': state['global_step']}
                state['val_results'].append(entry)
                # A three-image engineering smoke may have no small GT. Never
                # use this fallback for the formal full validation split.
                key = ((result['metrics']['mask_ap_small'] or 0., result['metrics']['mask_ap'] or 0.)
                       if args.smoke_steps else selection_key(result))
                if key > tuple(state['best_key']):
                    state['best_key'], state['best_epoch'] = list(key), epoch+1
                    save_torch(out/'best.pt', {'model': adapter.state_dict(), 'model_metadata': metadata,
                               'model_name': args.model, 'epoch': epoch+1, 'selection': entry,
                               'source_sha256': sources, 'protocol_sha256': protocol_sha})
                write_json(out/'selection.json', {'rule': 'Max validation APsmall, then AP; exact ties keep earlier epoch',
                           'validation_scope': 'engineering_smoke' if args.smoke_steps else 'full_2584_images',
                           'best_epoch': state['best_epoch'], 'best_key': state['best_key'],
                           'epochs': state['val_results'], 'test_used_for_selection': False})
                state.update(epoch=epoch+1, images_in_epoch=0, epoch_loss_sum=0., stage='training')
                save_last()
        run.update(status='training_completed', best_epoch=state['best_epoch'],
                   optimizer_steps=state['global_step'], training_images_seen=n_train*n_epochs,
                   best_checkpoint_sha256=sha256(out/'best.pt'))
        write_json(out/'run.json', run)
        if not args.smoke_steps:
            selected = torch.load(out/'best.pt', map_location=args.device, weights_only=True)
            adapter.load_state_dict(selected['model'], strict=True)
            del selected
            run['status'] = 'testing'
            write_json(out/'run.json', run)
            test_dir = out/'test'
            if test_dir.exists() and any(test_dir.iterdir()):
                if not args.resume:
                    raise FileExistsError(test_dir)
                attempt = 2
                while (out/f'test_attempt{attempt}').exists():
                    attempt += 1
                test_dir = out/f'test_attempt{attempt}'
            result = evaluate(adapter, args.archive, 'test', test_dir, export_submission=True,
                              progress_callback=eval_progress)
            test_receipt = json.loads((test_dir/'evaluation_run.json').read_text())
            test_receipt['evaluated_checkpoint_sha256'] = run['best_checkpoint_sha256']
            test_receipt['selected_epoch'] = state['best_epoch']
            write_json(test_dir/'evaluation_run.json', test_receipt)
            run['test_output'] = str(test_dir)
            run['test_evaluation_run_sha256'] = sha256(test_dir/'evaluation_run.json')
            run['test_metrics'] = result['metrics']
        run.update(status='completed', finished_at_utc=now(), elapsed_seconds=time.monotonic()-started)
        write_json(out/'run.json', run)
        print(json.dumps({'status': 'completed', 'model': args.model, 'output': str(out),
                          'best_epoch': state['best_epoch'], 'test_metrics': run.get('test_metrics')}), flush=True)
    except BaseException as error:
        run.update(status='failed', error_type=type(error).__name__, error=str(error), stopped_at_utc=now())
        write_json(out/'run.json', run)
        raise
    finally:
        if dataset is not None:
            dataset.close()


if __name__ == '__main__':
    main()
