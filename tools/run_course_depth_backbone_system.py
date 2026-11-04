#!/usr/bin/env python3
"""Train matching point heads, then evaluate locked course variants and CSVs."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
import torch.distributed as dist
from torch import nn
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers.models.mask2former.modeling_mask2former import Mask2FormerHungarianMatcher

from rgbd_seg.course_depth_backbone import build_trainable, CourseVariantDataset
from rgbd_seg.course_depth_backbone_eval import evaluate
from rgbd_seg.course_depth_backbone_points import PairedPointHeads, training_batch, predict_paired
from rgbd_seg.course_depth_backbone_system import tiled_pair
from rgbd_seg.course_training_data import collate_fn
from rgbd_seg.course_optimized import tensor_digest
from tools.run_course_dafusion import now, sha256, write_json, save_torch, rng_state


def sources():
    from tools.run_course_depth_backbone import sources as base_sources
    result = base_sources()
    for name in ('rgbd_seg/course_depth_backbone_points.py', 'rgbd_seg/course_depth_backbone_system.py',
                 'tools/run_course_depth_backbone_system.py', 'rgbd_seg/course_point_refinement.py',
                 'rgbd_seg/course_tiled_probability.py', 'rgbd_seg/course_depth_refinement.py'):
        result[name] = sha256(ROOT/name)
    return result


class Predictor(nn.Module):
    def __init__(self, adapter, heads, variant):
        super().__init__()
        self.adapter, self.heads, self.variant = adapter, heads, variant
        self.depth_encoding = adapter.depth_encoding

    @property
    def device(self):
        return self.adapter.device

    @torch.inference_mode()
    def predict_sample(self, sample, **kwargs):
        if self.variant == 'native':
            return self.adapter.predict_sample(sample, **kwargs)
        if self.variant == 'point':
            values = predict_paired(self.adapter, self.heads, sample, head_names=('point_depth',))
            return [c.public() for c in values['point_depth']]
        if self.variant == 'full':
            return tiled_pair(self.adapter, self.heads.depth, sample)[1]
        raise ValueError('Unknown inference variant')


def check_sources(protocol):
    if protocol['system_source_sha256'] != sources():
        raise ValueError('Frozen point/system sources changed')


def load_base(protocol, weights, device):
    if sha256(weights) != protocol['base_checkpoint_sha256']:
        raise ValueError('Fixed base checkpoint identity changed')
    model = protocol['model']
    adapter, metadata = build_trainable(protocol['checkpoint'], device='cpu',
        encoding=model['depth_encoding'], backbone=model['backbone'],
        backbone_checkpointing=model['backbone_checkpointing'], seed=protocol['training']['seed'])
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint['model_metadata'] != metadata or checkpoint['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Base checkpoint construction/provenance differs')
    if protocol.get('scope') != 'engineering_smoke' and checkpoint['epoch'] != 15:
        raise ValueError('The fixed-budget base must be epoch15')
    adapter.load_state_dict(checkpoint['model'], strict=True)
    del checkpoint
    adapter.requires_grad_(False).eval().to(device)
    return adapter


def train_points(protocol, adapter, heads, output, rank, world, guard, smoke):
    config = adapter.model.config
    matcher = Mask2FormerHungarianMatcher(cost_class=config.class_weight,
        cost_mask=config.mask_weight, cost_dice=config.dice_weight, num_points=config.train_num_points)
    ddp = DDP(heads, device_ids=[adapter.device.index], broadcast_buffers=False)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=.0003, weight_decay=.01)
    seed = protocol['training']['seed']
    batch_size = 4
    epochs = 1 if smoke else 6
    history, step = [], 0
    base_digest = tensor_digest(adapter.state_dict())
    with CourseVariantDataset(protocol['archive'], 'train', augment=True, seed=seed,
                              encoding=adapter.depth_encoding) as dataset:
        indices = list(range(world*batch_size*2)) if smoke else list(range(len(dataset)))
        if len(indices) % (world*batch_size):
            raise ValueError('Point training must cover all images without padding/drop')
        sampler = DistributedSampler(Subset(dataset, indices), num_replicas=world,
            rank=rank, shuffle=True, seed=seed, drop_last=False)
        steps_per_epoch = len(sampler)//batch_size
        total_steps = steps_per_epoch*epochs
        if rank == 0:
            write_json(output/'training_data.json', dataset.summary())
        for epoch in range(1, epochs+1):
            dataset.set_epoch(epoch-1)
            sampler.set_epoch(epoch-1)
            heads.train()
            loader = DataLoader(Subset(dataset, indices), batch_size=batch_size, sampler=sampler,
                num_workers=2, collate_fn=collate_fn, persistent_workers=False,
                multiprocessing_context='spawn', prefetch_factor=2,
                generator=torch.Generator().manual_seed(seed+7919*epoch+rank))
            coverage = []
            torch.manual_seed(seed+10000*rank+epoch)
            with (output/f'train_epoch{epoch:02d}_rank{rank}.jsonl').open('x') as log:
                for micro, (samples, targets) in enumerate(loader):
                    guard()
                    begin = time.monotonic()
                    coverage.extend(int(t['image_id'][0]) for t in targets)
                    values, counts = training_batch(adapter, samples, targets, matcher)
                    if values is None:
                        raise ValueError('Unexpected batch without matched instances')
                    fine, coarse, depth, truth = values
                    predictions = ddp(fine, coarse, depth)
                    sums = {name: F.binary_cross_entropy_with_logits(coarse+delta, truth,
                        reduction='none').mean((1, 2)).sum() for name, delta in predictions.items()}
                    stats = torch.stack((sums['point'].detach(), sums['point_depth'].detach(),
                                         torch.tensor(float(len(fine)), device=adapter.device)))
                    dist.all_reduce(stats)
                    if not stats[2] > 0 or not torch.isfinite(stats).all():
                        raise FloatingPointError('Invalid point loss/count')
                    loss = sum(sums.values())*world/stats[2]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for head in (heads.plain, heads.depth):
                        nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                    warmup = min(100, max(1, total_steps//10))
                    factor = ((step+1)/warmup if step < warmup else
                        .1+.9*.5*(1+math.cos(math.pi*(step-warmup)/max(1, total_steps-warmup-1))))
                    for group in optimizer.param_groups:
                        group['lr'] = .0003*factor
                    optimizer.step()
                    step += 1
                    torch.cuda.synchronize(adapter.device)
                    row = {'epoch': epoch, 'epochs': epochs, 'step': micro+1,
                        'steps_per_epoch': steps_per_epoch, 'global_step': step,
                        'images': len(coverage), 'expected_images': len(sampler),
                        'point_loss': float(stats[0]/stats[2]),
                        'depth_point_loss': float(stats[1]/stats[2]),
                        'global_instances': int(stats[2]), 'seconds': time.monotonic()-begin,
                        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(adapter.device)}
                    log.write(json.dumps(row, allow_nan=False)+'\n')
                    if micro == 0 or (micro+1) % 10 == 0 or micro+1 == steps_per_epoch:
                        log.flush()
                        write_json(output/f'progress_rank{rank}.json', row)
                        if rank == 0:
                            print(json.dumps(row), flush=True)
                    del samples, targets, values, fine, coarse, depth, truth, predictions, sums, stats, loss
            write_json(output/f'coverage_epoch{epoch:02d}_rank{rank}.json', coverage)
            dist.barrier()
            if rank == 0:
                merged = [iid for r in range(world) for iid in
                    json.loads((output/f'coverage_epoch{epoch:02d}_rank{r}.json').read_text())]
                if len(merged) != len(set(merged)) or set(merged) != {dataset.image_ids[i] for i in indices}:
                    raise ValueError('Point training image coverage differs')
                save_torch(output/f'heads_epoch_{epoch:02d}.pt', {
                    'heads': heads.state_dict(), 'epoch': epoch, 'fine_channels': heads.plain.layers[0].in_channels-1,
                    'base_checkpoint_sha256': protocol['base_checkpoint_sha256'],
                    'source_sha256': protocol['system_source_sha256'],
                    'protocol_sha256': protocol['_sha256']})
            dist.barrier()
            heads.eval()
            predictor = Predictor(adapter, heads, 'point').eval()
            result = evaluate(predictor, protocol['archive'], output/f'val_epoch_{epoch:02d}',
                rank=rank, world_size=world, guard=guard, smoke=smoke)
            history.append({'epoch': epoch, 'metrics': result['metrics'], 'global_step': step,
                            'checkpoint': str((output/f'heads_epoch_{epoch:02d}.pt').resolve())})
            if rank == 0:
                write_json(output/'validation_history.json', history)
                save_torch(output/'last_training.pt', {'heads': heads.state_dict(),
                    'optimizer': optimizer.state_dict(), 'epoch': epoch, 'global_step': step,
                    'validation_history': history, 'protocol_sha256': protocol['_sha256']})
        chosen = max(history, key=lambda e: (e['metrics']['mask_ap_small'] or 0.,
                                             e['metrics']['mask_ap'] or 0., -e['epoch']))
        if tensor_digest(adapter.state_dict()) != base_digest or any(p.grad is not None for p in adapter.parameters()):
            raise ValueError('Frozen base changed or received gradients')
        digest = tensor_digest(heads.state_dict())
        digests = [None]*world
        dist.all_gather_object(digests, digest)
        if len(set(digests)) != 1:
            raise ValueError('Point head replicas diverged')
        if rank == 0:
            write_json(output/'selection.json', {'status': 'locked', 'selected': chosen,
                'selection_rule': 'Full validation point-depth APsmall, then AP, then earlier epoch',
                'test_used': False, 'base_checkpoint_sha256': protocol['base_checkpoint_sha256'],
                'selected_head_sha256': sha256(chosen['checkpoint']), 'epochs': epochs,
                'training_images': len(indices), 'global_steps': step,
                'base_unchanged': True, 'base_gradients_absent': True, 'head_replicas_equal': True})
    dist.barrier()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', type=Path, required=True)
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('points', 'evaluate'), required=True)
    p.add_argument('--heads', type=Path)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text())
    check_sources(protocol)
    protocol['_sha256'] = sha256(args.protocol)
    if args.smoke != (protocol.get('scope') == 'engineering_smoke'):
        raise ValueError('Engineering/formal protocol scope mismatch')
    rank, local_rank, world = (int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE'))
    if (os.environ.get('CUDA_VISIBLE_DEVICES') != ','.join(protocol['hardware']['physical_gpus'])
            or world != len(protocol['hardware']['physical_gpus'])):
        raise ValueError('GPU layout differs')
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.cuda.set_per_process_memory_fraction(protocol['hardware']['memory_fraction'], device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    output = args.output.resolve()
    if rank == 0:
        output.mkdir(parents=True, exist_ok=False)
    dist.barrier()
    started = time.monotonic()

    def guard():
        if time.monotonic()-started > protocol['limits']['per_trial_seconds']:
            raise TimeoutError('Declared trial time exceeded')
        if shutil.disk_usage(output).free < 20*1024**3:
            raise OSError('Disk reserve below20GiB')

    run = {'status': 'initializing', 'mode': args.mode, 'scope': protocol.get('scope'),
           'protocol_sha256': protocol['_sha256'], 'started_at_utc': now(),
           'source_sha256': protocol['system_source_sha256'], 'arguments': vars(args).copy()}
    run['arguments'] = {k: str(v) if isinstance(v, Path) else v for k, v in run['arguments'].items()}
    try:
        if rank == 0:
            write_json(output/'run.json', run)
        adapter = load_base(protocol, args.base, device)
        torch.manual_seed(protocol['training']['seed'])
        fine_channels = adapter.model.model.pixel_level_module.encoder.channels[0] + adapter.model.config.mask_feature_size
        heads = PairedPointHeads(fine_channels=fine_channels).to(device)
        if args.mode == 'points':
            train_points(protocol, adapter, heads, output, rank, world, guard, args.smoke)
        else:
            if args.heads is None:
                raise ValueError('Locked head checkpoint is required')
            checkpoint = torch.load(args.heads, map_location='cpu', weights_only=True)
            if (checkpoint['base_checkpoint_sha256'] != protocol['base_checkpoint_sha256']
                    or checkpoint['source_sha256'] != protocol['system_source_sha256']):
                raise ValueError('Head provenance differs from fixed base/system')
            if not args.smoke:
                selection = json.loads(Path(protocol['head_selection']).read_text())
                if (selection['status'] != 'locked' or selection['test_used']
                        or sha256(args.heads) != selection['selected_head_sha256']):
                    raise ValueError('Only validation-locked head can enter test evaluation')
                metadata = json.loads(Path(protocol['official_test_metadata']).read_text())
                from rgbd_seg.course_archive import CourseArchive
                with CourseArchive(protocol['archive'], 'test') as test:
                    if test.image_ids != [r['id'] for r in metadata['images']]:
                        raise ValueError('Official new test metadata IDs/order differ')
            heads.load_state_dict(checkpoint['heads'], strict=True)
            heads.eval().requires_grad_(False)
            del checkpoint
            results = {}
            # Full validation precedes any test output; hyperparameters remain fixed.
            for split in (('val',) if args.smoke else ('val', 'test')):
                results[split] = {}
                for variant in ('native', 'full'):
                    predictor = Predictor(adapter, heads, variant).eval()
                    result = evaluate(predictor, protocol['archive'], output/f'{split}_{variant}',
                        rank=rank, world_size=world, guard=guard, smoke=args.smoke, split=split)
                    results[split][variant] = result
                if rank == 0:
                    write_json(output/'comparison.json', results)
        if rank == 0:
            run.update(status='completed', finished_at_utc=now(), elapsed_seconds=time.monotonic()-started)
            write_json(output/'run.json', run)
        dist.barrier()
    except BaseException as e:
        write_json(output/f'failure_rank{rank}.json', {'type': type(e).__name__, 'error': str(e), 'at_utc': now()})
        if rank == 0:
            run.update(status='failed', error=str(e), stopped_at_utc=now())
            write_json(output/'run.json', run)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
