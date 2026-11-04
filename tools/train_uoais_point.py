#!/usr/bin/env python3
"""Train paired residual point heads on UOAIS, freezing the UOAIS-trained base.

Formal runs require the completed base and its full-validation-selected weights.
Engineering runs use a completed smoke base and never produce selected weights.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
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
from tools.train_uoais_benchmark import (
    now, sha, write_json, save_torch, gather, rng_state, restore_rng, source_hashes)
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from transformers.models.mask2former.modeling_mask2former import Mask2FormerHungarianMatcher
from rgbd_seg.uoais_point_training import load_uoais_base, point_loss_sums, empty_point_batch
from rgbd_seg.uoais_training_data import UOAISTrainingDataset
from rgbd_seg.uoais_benchmark_training import TrainingExamples
from rgbd_seg.uoais_dpseg_training import collate_training_examples
from rgbd_seg.course_point_refinement import POLICY, PairedPointHeads, training_batch, predict_paired
from rgbd_seg.course_optimized import tensor_digest
from rgbd_seg.uois_benchmark_geometry import letterbox_sample, restore_predictions
from rgbd_seg.uois_benchmark_metrics import (
    PAPER_METRIC_NAMES, predictions_to_label_map, evaluate_label_maps)
from tools.score_uois_benchmark import json_safe

ARMS = ('native', 'point', 'point_depth')


def point_sources():
    result = source_hashes()
    for name in ('tools/train_uoais_point.py', 'rgbd_seg/uoais_point_training.py',
                 'rgbd_seg/course_point_refinement.py', 'rgbd_seg/course_tiled_probability.py',
                 'rgbd_seg/course_depth_refinement.py', 'rgbd_seg/course_optimized.py',
                 'tools/score_uois_benchmark.py'):
        result[name] = sha(ROOT/name)
    return result


@torch.inference_mode()
def validate(adapter, heads, data, out, epoch, *, smoke=False):
    rank, world = dist.get_rank(), dist.get_world_size()
    heads.eval()
    indices = np.linspace(0, len(data)-1, 8, dtype=int).tolist() if smoke else list(range(len(data)))
    sums = np.zeros((len(ARMS), len(PAPER_METRIC_NAMES)), np.float64)
    undefined = np.zeros_like(sums)
    folder = out/f'val_epoch{epoch:02d}'
    if rank == 0:
        folder.mkdir(exist_ok=True)
    dist.barrier()
    start = time.monotonic()
    with (folder/f'predictions_rank{rank}.jsonl').open('w') as stream:
        for position, index in enumerate(indices[rank::world]):
            original, target = data[index]
            sample, geometry = letterbox_sample(original)
            paired = predict_paired(adapter, heads, sample)
            truth = np.zeros(target['masks'].shape[-2:], np.int32)
            for instance_id, mask in enumerate(target['masks'].numpy(), 1):
                truth[mask.astype(bool)] = instance_id
            row = {'sample_id': data.records[index]['sample_id'], 'arms': {}}
            identity = [(v.query_id, v.score) for v in paired['native']]
            for arm_index, arm in enumerate(ARMS):
                if [(v.query_id, v.score) for v in paired[arm]] != identity:
                    raise ValueError('Point refinement changed candidates or scores')
                restored = restore_predictions([v.public() for v in paired[arm]], geometry)
                predictions = sorted(restored, key=lambda value: value['score'])
                labels = predictions_to_label_map(predictions, truth.shape, score_threshold=.5)
                result = evaluate_label_maps(labels, truth)
                values = np.asarray([result['metrics'][name] for name in PAPER_METRIC_NAMES])
                # Preserve undefined boundary means as null, never nanmean/zero.
                undefined[arm_index] += ~np.isfinite(values)
                sums[arm_index] += np.where(np.isfinite(values), values, 0.)
                if not np.isfinite(result['metrics']['Objects F-measure']):
                    raise ValueError('Overlap F is undefined; cannot select a checkpoint')
                row['arms'][arm] = {'predictions': predictions, **result}
            stream.write(json.dumps(json_safe(row), allow_nan=False)+'\n')
            if (position+1) % 25 == 0 or position+1 == len(indices[rank::world]):
                stream.flush()
                progress = {'status': 'validating', 'epoch': epoch, 'rank': rank,
                            'done': position+1, 'total': len(indices[rank::world]), 'updated_utc': now()}
                write_json(out/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
    combined = torch.tensor(np.stack((sums, undefined)), dtype=torch.float64, device=adapter.device)
    dist.all_reduce(combined)
    summary = {'epoch': epoch, 'images': len(indices), 'scope': 'engineering_smoke' if smoke else 'full_validation',
               'metrics': {}, 'undefined_images': {}, 'seconds': time.monotonic()-start,
               'selection_metric': 'Objects F-measure', 'test_used': False, 'precision': 'float32'}
    for i, arm in enumerate(ARMS):
        summary['metrics'][arm] = {key: (None if combined[1, i, j] else float(combined[0, i, j]/len(indices)))
                                   for j, key in enumerate(PAPER_METRIC_NAMES)}
        summary['undefined_images'][arm] = {key: int(combined[1, i, j]) for j, key in enumerate(PAPER_METRIC_NAMES)
                                           if combined[1, i, j]}
    if rank == 0:
        write_json(folder/'summary.json', summary)
    heads.train()
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--global-batch', type=int, default=4)
    p.add_argument('--learning-rate', type=float, default=1e-4)
    p.add_argument('--seed', type=int, default=29)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--save-every', type=int, default=500)
    p.add_argument('--smoke-steps', type=int, default=0)
    p.add_argument('--resume', action='store_true')
    args = p.parse_args()
    if min(args.epochs, args.global_batch, args.learning_rate, args.save_every) <= 0 or args.smoke_steps < 0:
        p.error('Invalid training budget')
    rank, local_rank, world = [int(os.environ[key]) for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    if args.global_batch % world:
        p.error('Batch must be divisible by number of ranks')
    batch = args.global_batch//world
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    if args.smoke_steps:
        # Engineering can share the owned 80-GB GPU with the base job. Bound this
        # process so the ongoing base run keeps its measured memory headroom.
        torch.cuda.set_per_process_memory_fraction(.35, device)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=2))
    out = args.output.resolve()
    if rank == 0:
        if not args.resume and out.exists():
            raise FileExistsError(out)
        out.mkdir(parents=True, exist_ok=True)
    dist.barrier()
    try:
        adapter, base_protocol, base = load_uoais_base(args.base_run, engineering=bool(args.smoke_steps), device=device)
        sources = point_sources()
        data = UOAISTrainingDataset(base_protocol['dataset_root'], 'train', base_protocol['calibration'])
        val = data.for_split('val')
        for split, expected in base_protocol['annotations'].items():
            if data.annotation_receipts[split] != expected:
                raise ValueError('Point training data differs from the base training data')
        if len(data) % args.global_batch:
            raise ValueError('Exact full-batch coverage is required')
        base_digest = tensor_digest(adapter.state_dict())
        protocol = {'schema': 'uoais-point-training-v1', 'scope': 'engineering_smoke' if args.smoke_steps else 'full_train_val',
                    'base': base, 'source_sha256': sources, 'point_policy': POLICY,
                    'training': {'epochs': args.epochs, 'global_batch': args.global_batch,
                                 'learning_rate': args.learning_rate, 'weight_decay': .01,
                                 'seed': args.seed, 'workers': args.workers, 'save_every': args.save_every,
                                 'world_size': world, 'precision': 'float32 base and point heads',
                                 'augmentation': base_protocol['training']['augmentation'],
                                 'warmup_steps': 100, 'cosine_minimum_ratio': .1},
                    'selection': {'split': 'val', 'images': 5000, 'metric': 'Objects F-measure',
                                  'score_threshold': .5, 'tie_break': 'earliest epoch',
                                  'overlap_rule': 'ascending confidence; highest score wins'},
                    'base_tensor_sha256': base_digest, 'test_used': False}
        protocol_hash = __import__('hashlib').sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
        if rank == 0:
            if args.resume:
                previous = json.loads((out/'protocol.json').read_text())
                if previous != protocol:
                    raise ValueError('Point resume protocol changed')
            else:
                write_json(out/'protocol.json', protocol)
        torch.manual_seed(args.seed)
        heads = PairedPointHeads().to(device)
        optimizer = torch.optim.AdamW(heads.parameters(), lr=args.learning_rate, weight_decay=.01)
        state = {'epoch': 0, 'batch_in_epoch': 0, 'step': 0, 'updates': 0, 'empty_steps': 0,
                 'best': {'point': -1., 'point_depth': -1.}, 'validation': []}
        steps_per_epoch = len(data)//args.global_batch
        total_steps = steps_per_epoch*args.epochs
        if args.resume:
            saved = torch.load(out/'last.pt', map_location='cpu', weights_only=False)
            if saved['protocol_sha256'] != protocol_hash or len(saved['rank_rng']) != world:
                raise ValueError('Point resume checkpoint differs')
            heads.load_state_dict(saved['heads'], strict=True)
            optimizer.load_state_dict(saved['optimizer'])
            state = saved['state']
            if state['step'] != state['epoch']*steps_per_epoch+state['batch_in_epoch']:
                raise ValueError('Point resume sample position differs')
            if args.smoke_steps and state['step'] >= args.smoke_steps:
                raise ValueError('Resume target must exceed saved steps')
            resume_rng = saved['rank_rng'][rank]
            del saved
        ddp = DistributedDataParallel(heads, device_ids=[local_rank], broadcast_buffers=False)
        if args.resume:
            restore_rng(resume_rng)
        else:
            random.seed(args.seed+rank); np.random.seed(args.seed+rank); torch.manual_seed(args.seed+rank)
        cfg = adapter.model.config
        matcher = Mask2FormerHungarianMatcher(cost_class=cfg.class_weight, cost_mask=cfg.mask_weight,
                                             cost_dice=cfg.dice_weight, num_points=cfg.train_num_points)
        hardware = gather({'pid': os.getpid(), 'rank': rank, 'gpu': torch.cuda.get_device_name(),
                           'visibility': os.environ.get('CUDA_VISIBLE_DEVICES')})
        run = {'status': 'training', 'scope': protocol['scope'], 'started_utc': now(),
               'protocol_sha256': protocol_hash, 'base': base, 'hardware': hardware,
               'source_sha256': sources, 'expected_steps': total_steps}
        if rank == 0:
            write_json(out/'run.json', run)

        def save_last():
            states = gather(rng_state())
            if rank == 0:
                save_torch(out/'last.pt', {'heads': heads.state_dict(), 'optimizer': optimizer.state_dict(),
                    'state': state, 'rank_rng': states, 'protocol_sha256': protocol_hash,
                    'base_checkpoint_sha256': base['checkpoint_sha256'], 'source_sha256': sources})
            dist.barrier()

        examples = TrainingExamples(data, 'dpseg', args.seed)
        started = time.monotonic()
        while state['epoch'] < args.epochs:
            epoch = state['epoch']
            order = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed+1009*epoch)).tolist()
            batches = [[(index, epoch) for index in order[start+rank*batch:start+(rank+1)*batch]]
                       for start in range(state['batch_in_epoch']*args.global_batch, len(data), args.global_batch)]
            loader = DataLoader(examples, batch_sampler=batches, collate_fn=collate_training_examples,
                                num_workers=args.workers, multiprocessing_context='fork' if args.workers else None,
                                generator=torch.Generator().manual_seed(args.seed+epoch))
            heads.train()
            for samples, targets, metadata in loader:
                step_started = time.monotonic()
                values, counts = training_batch(adapter, samples, targets, matcher)
                local_instances = sum(counts)
                fine, coarse, depth, truth = values if values is not None else empty_point_batch(device)
                losses = point_loss_sums(ddp(fine, coarse, depth), coarse, truth, real_instances=local_instances)
                stats = torch.stack((losses['point'].detach(), losses['point_depth'].detach(),
                                     torch.tensor(float(local_instances), device=device)))
                dist.all_reduce(stats)
                if not torch.isfinite(stats).all():
                    raise FloatingPointError('Nonfinite point loss')
                loss = sum(losses.values()) * world/stats[2].clamp(min=1)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                for head in (heads.plain, heads.depth):
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1., error_if_nonfinite=True)
                step = state['step']
                factor = ((step+1)/100 if step < 100 else
                          .1+.9*.5*(1+math.cos(math.pi*(step-100)/max(1, total_steps-101))))
                for group in optimizer.param_groups:
                    group['lr'] = args.learning_rate*factor
                if stats[2] > 0:
                    optimizer.step(); state['updates'] += 1
                else:
                    state['empty_steps'] += 1
                state['step'] += 1; state['batch_in_epoch'] += 1
                row = {'updated_utc': now(), 'step': state['step'], 'epoch': epoch+1,
                       'images_in_epoch': state['batch_in_epoch']*args.global_batch,
                       'point_loss': float(stats[0]/stats[2].clamp(min=1)),
                       'depth_point_loss': float(stats[1]/stats[2].clamp(min=1)),
                       'instances': int(stats[2]), 'updates': state['updates'],
                       'seconds': time.monotonic()-step_started,
                       'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30}
                if rank == 0:
                    with (out/'training_log.jsonl').open('a') as stream:
                        stream.write(json.dumps(row, allow_nan=False)+'\n')
                if state['step'] <= 3 or state['step'] % 25 == 0:
                    write_json(out/f'progress_rank{rank}.json', {'status': 'training', 'rank': rank, **row})
                    print(json.dumps({'rank': rank, **row}), flush=True)
                del fine, coarse, depth, truth, values, losses, loss
                if state['step'] % args.save_every == 0:
                    save_last()
                if args.smoke_steps and state['step'] >= args.smoke_steps:
                    break
            del loader
            save_last()
            result = validate(adapter, heads, val, out, epoch+1, smoke=bool(args.smoke_steps))
            if args.smoke_steps:
                if not state['updates']:
                    raise RuntimeError('Point engineering run made no update')
                break
            if state['batch_in_epoch'] != steps_per_epoch:
                raise ValueError('Incomplete training epoch')
            state['validation'].append(result)
            for arm in ('point', 'point_depth'):
                score = result['metrics'][arm]['Objects F-measure']
                if score > state['best'][arm]:
                    state['best'][arm] = score
                    if rank == 0:
                        save_torch(out/f'best_{arm}.pt', {'heads': heads.state_dict(), 'selection': result,
                            'selected_arm': arm, 'scope': protocol['scope'], 'point_policy': POLICY,
                            'base_checkpoint_sha256': base['checkpoint_sha256'],
                            'protocol_sha256': protocol_hash, 'source_sha256': sources})
            state['epoch'] += 1; state['batch_in_epoch'] = 0
            save_last()
        unchanged = tensor_digest(adapter.state_dict()) == base_digest
        if not unchanged or any(parameter.grad is not None for parameter in adapter.parameters()):
            raise ValueError('Frozen base changed or received gradients')
        digests = gather(tensor_digest(heads.state_dict()))
        if len(set(digests)) != 1:
            raise ValueError('Point heads diverged across ranks')
        if rank == 0:
            run.update(status='smoke_passed' if args.smoke_steps else 'training_completed',
                       state=state, finished_utc=now(), elapsed_seconds=time.monotonic()-started,
                       base_unchanged=True, base_gradients_absent=True,
                       head_tensor_sha256=digests[0], last_checkpoint_sha256=sha(out/'last.pt'))
            write_json(out/'run.json', run)
        dist.barrier()
    except BaseException:
        failure = {'status': 'failed', 'rank': rank, 'time': now(), 'traceback': traceback.format_exc()}
        write_json(out/f'failure_rank{rank}.json', failure)
        traceback.print_exc(); sys.stdout.flush(); sys.stderr.flush()
        os._exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
