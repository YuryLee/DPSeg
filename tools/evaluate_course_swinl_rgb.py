#!/usr/bin/env python3
"""Evaluate the fixed epoch15 RGB Swin-L checkpoint on the complete local test."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.distributed as dist

from rgbd_seg.course_swinl_rgb import build_trainable
from rgbd_seg.course_swinl_rgb_eval import evaluate, write_json
from tools.train_course_swinl_rgb import sources
from tools.run_course_dafusion import sha256, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    rank, local_rank, world = [int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    if (os.environ['CUDA_VISIBLE_DEVICES'].split(',') != protocol['hardware']['physical_gpus']
            or world != len(protocol['hardware']['physical_gpus'])):
        raise ValueError('Evaluation GPU allocation differs from protocol')
    if protocol['source_sha256'] != sources():
        raise ValueError('Frozen sources changed')
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    torch.set_num_threads(protocol['training']['threads'])
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    started = time.monotonic()

    def guard():
        if time.monotonic() - started > protocol['limits']['per_trial_seconds']:
            raise TimeoutError('Evaluation time budget exceeded')

    run = {'status': 'loading', 'started_at_utc': now(), 'split': 'test',
           'selection': 'Fixed epoch15, declared before training', 'test_used_for_selection': False}
    try:
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=False)
            write_json(args.output / 'run.json', run)
        dist.barrier()
        saved = torch.load(args.weights, map_location='cpu', weights_only=True, mmap=True)
        training_receipt = json.loads((args.weights.parent / 'run.json').read_text())
        if (saved['epoch'] != 15 or saved['source_sha256'] != protocol['source_sha256']
                or saved['protocol_sha256'] != sha256(args.protocol)
                or training_receipt['status'] != 'completed'
                or training_receipt['final_checkpoint_sha256'] != sha256(args.weights)):
            raise ValueError('Fixed epoch15 checkpoint provenance differs')
        adapter, metadata = build_trainable(protocol['checkpoint'], seed=protocol['training']['seed'],
            backbone_checkpointing=protocol['model']['backbone_checkpointing'])
        if metadata != saved['model_metadata']:
            raise ValueError('Checkpoint architecture or initialization changed')
        adapter.load_state_dict(saved['model'], strict=True)
        del saved
        adapter.requires_grad_(False).eval().to(device)
        result = evaluate(adapter, protocol['archive'], args.output / 'test',
            rank=rank, world_size=world, guard=guard, split='test')
        if rank == 0:
            run.update(status='completed', finished_at_utc=now(),
                elapsed_seconds=time.monotonic()-started, metrics=result['metrics'],
                weights_sha256=sha256(args.weights), protocol_sha256=sha256(args.protocol))
            write_json(args.output / 'run.json', run)
    except BaseException as error:
        if rank == 0 and args.output.exists():
            run.update(status='failed', error=str(error), stopped_at_utc=now())
            write_json(args.output / 'run.json', run)
        raise
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
