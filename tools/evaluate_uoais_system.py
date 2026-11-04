#!/usr/bin/env python3
"""Validate/final-test the four DPSeg stages using new UOAIS point weights.

Real test runs require a complete validation result for these exact weights and
sources. Engineering checks cannot write a final method selection.
"""
from __future__ import annotations
import argparse
from datetime import timedelta
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_uoais_benchmark import now, sha, write_json, gather
from tools.train_uoais_point import point_sources
from tools.score_uois_benchmark import json_safe
import numpy as np
import torch
import torch.distributed as dist
from rgbd_seg.uoais_point_training import load_uoais_base
from rgbd_seg.uoais_training_data import UOAISTrainingDataset
from rgbd_seg.uois_benchmark_data import BenchmarkDataset
from rgbd_seg.uois_benchmark_geometry import letterbox_sample, restore_predictions
from rgbd_seg.course_point_refinement import POLICY, PairedPointHeads
from rgbd_seg.course_progressive_ablation import predict_rgbd
from rgbd_seg.course_tiled_probability import TILE_POLICY, PROBABILITY_POLICY
from rgbd_seg.uois_benchmark_metrics import PAPER_METRIC_NAMES, predictions_to_label_map, evaluate_label_maps

STAGES = ('rgbd_fusion', 'depth_point', 'depth_point_tiled', 'depth_point_tiled_probability')


def select_stage(summary):
    if summary.get('scope') != 'full_validation' or summary.get('images') != 5000:
        raise ValueError('Method selection requires all 5,000 validation images')
    def key(stage):
        values = summary['metrics'][stage]
        overlap, boundary = values['Objects F-measure'], values['Boundary F-measure']
        if overlap is None or not np.isfinite(overlap):
            raise ValueError('Cannot select with undefined Overlap F')
        return overlap, (-1. if boundary is None else boundary), -STAGES.index(stage)
    return max(STAGES, key=key)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--point-run', type=Path, required=True)
    p.add_argument('--split', choices=['val', 'osd', 'ocid'], default='val')
    p.add_argument('--manifest', type=Path)
    p.add_argument('--validation-run', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--engineering', action='store_true')
    args = p.parse_args()
    if args.engineering and args.split != 'val':
        p.error('Engineering checks use only UOAIS validation examples')
    if args.split != 'val' and (args.manifest is None or args.validation_run is None):
        p.error('Real test requires the original manifest and a complete validation run')
    if args.output.exists():
        p.error('Use a new output directory')
    rank, local_rank, world = [int(os.environ[k]) for k in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE')]
    torch.set_num_threads(4)
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.engineering:
        torch.cuda.set_per_process_memory_fraction(.35, device)
    dist.init_process_group('nccl', device_id=device, timeout=timedelta(hours=2))
    out = args.output.resolve()
    if rank == 0:
        out.mkdir(parents=True)
    dist.barrier()
    try:
        head_run = json.loads((args.point_run/'run.json').read_text())
        head_protocol = json.loads((args.point_run/'protocol.json').read_text())
        if hashlib.sha256(json.dumps(head_protocol, sort_keys=True).encode()).hexdigest() != head_run['protocol_sha256']:
            raise ValueError('Point protocol snapshot changed')
        expected_status = 'smoke_passed' if args.engineering else 'training_completed'
        expected_scope = 'engineering_smoke' if args.engineering else 'full_train_val'
        if head_run['status'] != expected_status or head_run['scope'] != expected_scope:
            raise ValueError('Point training scope/completion does not match requested evaluation')
        if not args.engineering and (head_run['state']['epoch'] != head_protocol['training']['epochs']
                or len(head_run['state']['validation']) != head_protocol['training']['epochs']):
            raise ValueError('Point training epochs or full validation history are incomplete')
        current = point_sources()
        if head_run['source_sha256'] != current or head_protocol['point_policy'] != POLICY:
            raise ValueError('Point source or policy changed')
        adapter, base_protocol, base = load_uoais_base(head_protocol['base']['base_run'],
                                                     engineering=args.engineering, device=device)
        head_path = args.point_run/('last.pt' if args.engineering else 'best_point_depth.pt')
        saved = torch.load(head_path, map_location='cpu', weights_only=False)
        if (saved['base_checkpoint_sha256'] != base['checkpoint_sha256']
                or saved['protocol_sha256'] != head_run['protocol_sha256']
                or saved['source_sha256'] != current):
            raise ValueError('Point checkpoint does not belong to the selected UOAIS base')
        if not args.engineering and (saved['selection']['images'] != 5000
                                     or saved['selection']['scope'] != 'full_validation'):
            raise ValueError('Point weights lack full validation selection')
        heads = PairedPointHeads().to(device).eval().requires_grad_(False)
        heads.load_state_dict(saved['heads'], strict=True)
        del saved
        sources = {**current, 'tools/evaluate_uoais_system.py': sha(__file__),
                   'rgbd_seg/course_progressive_ablation.py': sha(ROOT/'rgbd_seg/course_progressive_ablation.py')}
        point_hash = sha(head_path)
        fixed_stage = None
        if args.split != 'val':
            selection = json.loads((args.validation_run/'selection.json').read_text())
            if (selection['scope'] != 'full_validation' or selection['images'] != 5000
                    or selection['base_checkpoint_sha256'] != base['checkpoint_sha256']
                    or selection['point_checkpoint_sha256'] != point_hash
                    or selection['source_sha256'] != sources or selection['stage'] not in STAGES):
                raise ValueError('Final test must use the previously validated weights and method')
            fixed_stage = selection['stage']
        if args.split == 'val':
            data = UOAISTrainingDataset(base_protocol['dataset_root'], 'val', base_protocol['calibration'])
            if data.annotation_receipts != base_protocol['annotations']:
                raise ValueError('Validation data changed')
            indices = np.linspace(0, len(data)-1, 8, dtype=int).tolist() if args.engineering else list(range(len(data)))
        else:
            data = BenchmarkDataset(args.manifest)
            if data.manifest['dataset'] != args.split or len(data) != {'osd':111, 'ocid':2390}[args.split]:
                raise ValueError('Incomplete or wrong original test manifest')
            indices = list(range(len(data)))
        scope = 'engineering_smoke' if args.engineering else ('full_validation' if args.split == 'val' else 'final_test')
        protocol = {'scope': scope, 'split': args.split, 'images': len(indices),
                    'base_checkpoint_sha256': base['checkpoint_sha256'], 'point_checkpoint_sha256': point_hash,
                    'source_sha256': sources, 'score_threshold': .5,
                    'overlap_rule': 'ascending confidence; highest score wins', 'precision': 'float32',
                    'tile_policy': TILE_POLICY, 'probability_policy': PROBABILITY_POLICY,
                    'stages': STAGES, 'fixed_final_stage_from_validation': fixed_stage,
                    'hardware': gather({'pid':os.getpid(), 'rank':rank, 'gpu':torch.cuda.get_device_name(),
                                        'visibility':os.environ.get('CUDA_VISIBLE_DEVICES')}),
                    'started_utc': now()}
        if rank == 0:
            write_json(out/'protocol.json', protocol)
        sums = np.zeros((len(STAGES), len(PAPER_METRIC_NAMES)), np.float64)
        missing = np.zeros_like(sums)
        start = time.monotonic()
        with torch.inference_mode(), (out/f'predictions_rank{rank}.jsonl').open('x') as stream:
            for position, index in enumerate(indices[rank::world]):
                original = data.raw_sample(index) if args.split == 'val' else data.inference_sample(index)
                sample, geometry = letterbox_sample(original)
                outputs, merge_info = predict_rgbd(adapter, heads, sample)
                # Ground truth is decoded only after predictions have been made.
                if args.split == 'val':
                    masks = data.raw_target(index)['masks'].numpy()
                    truth = np.zeros(masks.shape[-2:], np.int32)
                    for instance, mask in enumerate(masks, 1):
                        truth[mask.astype(bool)] = instance
                    image_id = data.records[index]['sample_id']
                else:
                    truth, image_id = data.ground_truth(index), data.records[index]['id']
                row = {'image_id': image_id, 'stages': {}, 'merge': merge_info}
                for i, stage in enumerate(STAGES):
                    predictions = sorted(restore_predictions(outputs[stage], geometry), key=lambda value:value['score'])
                    labels = predictions_to_label_map(predictions, truth.shape, score_threshold=.5)
                    result = evaluate_label_maps(labels, truth)
                    values = np.asarray([result['metrics'][name] for name in PAPER_METRIC_NAMES])
                    missing[i] += ~np.isfinite(values)
                    sums[i] += np.where(np.isfinite(values), values, 0.)
                    row['stages'][stage] = {'predictions': predictions, **result}
                stream.write(json.dumps(json_safe(row), allow_nan=False)+'\n')
                if (position+1) % 10 == 0 or position+1 == len(indices[rank::world]):
                    stream.flush()
                    progress = {'status': 'evaluating', 'rank': rank, 'done':position+1,
                                'total': len(indices[rank::world]), 'updated_utc': now()}
                    write_json(out/f'progress_rank{rank}.json', progress)
                    print(json.dumps(progress), flush=True)
        combined = torch.tensor(np.stack((sums, missing)), dtype=torch.float64, device=device)
        dist.all_reduce(combined)
        summary = {**protocol, 'status': 'completed', 'seconds': time.monotonic()-start,
                   'metrics': {}, 'undefined_images': {}}
        for i, stage in enumerate(STAGES):
            summary['metrics'][stage] = {key: None if combined[1,i,j] else float(combined[0,i,j]/len(indices))
                                         for j,key in enumerate(PAPER_METRIC_NAMES)}
            summary['undefined_images'][stage] = {key: int(combined[1,i,j]) for j,key in enumerate(PAPER_METRIC_NAMES)
                                                  if combined[1,i,j]}
        if rank == 0:
            write_json(out/'summary.json', summary)
            if scope == 'full_validation':
                write_json(out/'selection.json', {'scope':scope, 'images':len(indices),
                    'stage':select_stage(summary), 'base_checkpoint_sha256':base['checkpoint_sha256'],
                    'point_checkpoint_sha256':point_hash, 'source_sha256':sources,
                    'rule':'Overlap F, then Boundary F, then simpler stage; validation only'})
        dist.barrier()
    except BaseException:
        write_json(out/f'failure_rank{rank}.json', {'status':'failed', 'traceback':traceback.format_exc(), 'time':now()})
        traceback.print_exc();sys.stdout.flush();sys.stderr.flush();os._exit(1)
    finally:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
