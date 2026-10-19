#!/usr/bin/env python3
"""Evaluate fixed progressive arms, with disjoint shards and full-split COCO AP."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_inference_sample
from rgbd_seg.course_distributed_eval import merge_shards, write_json
from rgbd_seg.course_finetune_eval import score_predictions
from rgbd_seg.course_dafusion_eval import validate_csv_fast
from rgbd_seg.course_optimized import file_sha, tensor_digest
from rgbd_seg.course_portable import load_model, verify_sources
from rgbd_seg.course_progressive_ablation import STAGES, predict_rgbd
from rgbd_seg.course_progressive_rgb import build_trainable
from rgbd_seg.course_submission import SubmissionWriter
from rgbd_seg.course_tiled_probability import TILE_POLICY, PROBABILITY_POLICY
from tools.check_course_prediction_reload import compare_predictions
from tools.audit_course_point_refinement import compare_csv_predictions


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def arms(group):
    return STAGES if group == 'all' else STAGES[:1] if group == 'rgb' else STAGES[1:]


def verify(p):
    if (p['status'] != 'frozen' or p['stages'] != list(STAGES)
            or p['tile_policy'] != TILE_POLICY or p['probability_policy'] != PROBABILITY_POLICY):
        raise ValueError('Experiment policy differs from the predeclared ablation')
    verify_sources(p['source_sha256'])
    for spec in p['inputs'].values():
        if file_sha(spec['path']) != spec['sha256']:
            raise ValueError(f"Input changed: {spec['path']}")


def load_rgb(p):
    spec = p['rgb']
    value = torch.load(spec['weights']['path'], map_location='cpu', weights_only=True)
    protocol = read(spec['protocol']['path'])
    if (file_sha(spec['weights']['path']) != spec['weights']['sha256']
            or file_sha(spec['protocol']['path']) != spec['protocol']['sha256']
            or value['protocol_sha256'] != spec['protocol']['sha256']
            or value['source_sha256'] != protocol['source_sha256'] or value['epoch'] != 15):
        raise ValueError('RGB weights/protocol/epoch differ')
    verify_sources(value['source_sha256'])
    adapter, metadata = build_trainable(p['coco_checkpoint'], device='cuda:0', seed=17)
    if metadata != value['model_metadata'] or metadata['depth_used']:
        raise ValueError('RGB model metadata differs')
    adapter.load_state_dict(value['model'], strict=True)
    digest = tensor_digest(adapter.state_dict())
    if digest != value['selection']['model_sha256'] or digest != spec['tensor_sha256']:
        raise ValueError('RGB tensor identity differs')
    return adapter.eval().requires_grad_(False), dict(epoch=15, tensor_sha256=digest,
        checkpoint_sha256=spec['weights']['sha256'], depth_used=False)


def worker(args, p):
    if os.environ['CUDA_VISIBLE_DEVICES'] != str(p['gpus'][args.rank]) or torch.cuda.device_count() != 1:
        raise ValueError('Unexpected GPU mapping')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = heads = rgb = None
    reloads, before = {}, {}
    if args.group != 'rgb':
        adapter, heads, reloads['rgbd'] = load_model(base_weights=p['inputs']['base']['path'],
            head_weights=p['inputs']['head']['path'], coco_checkpoint=p['coco_checkpoint'],
            manifest=p['inputs']['manifest']['path'], device='cuda:0')
        before['base'] = tensor_digest(adapter.state_dict())
        before['heads'] = tensor_digest(heads.state_dict())
    if args.group != 'rgbd':
        rgb, reloads['rgb'] = load_rgb(p)
        before['rgb'] = tensor_digest(rgb.state_dict())
    ids = read(args.folder/'image_ids.json')[args.rank::len(p['gpus'])]
    selected = arms(args.group)
    began, checks, totals = time.monotonic(), [], dict.fromkeys(selected, 0)
    with CourseArchive(p['archive'], args.split) as data, ExitStack() as stack, torch.inference_mode():
        streams = {a: stack.enter_context((args.folder/a/f'predictions_rank{args.rank}.jsonl').open('x')) for a in selected}
        for index, image_id in enumerate(ids):
            sample = load_inference_sample(data, image_id)
            outputs, diag = ({}, {}) if adapter is None else predict_rgbd(adapter, heads, sample)
            if rgb is not None:
                outputs[STAGES[0]] = rgb.predict_sample(sample)
            if set(outputs) != set(selected):
                raise ValueError('Missing comparison arm')
            if args.smoke:
                check = {'image_id': image_id, 'no_gt_in_inference': True}
                if adapter is not None:
                    from rgbd_seg.course_tiled_point import predict as legacy_tiled
                    from rgbd_seg.course_point_refinement import predict_paired
                    legacy, legacy_diag = legacy_tiled(adapter, heads.depth, sample, zero_check=True)
                    err = compare_predictions(legacy['tiled_point_depth_probability_depth'], outputs[STAGES[4]])
                    paired = predict_paired(adapter, heads, sample)
                    for stage, old in ((STAGES[1], 'native'), (STAGES[2], 'point_depth')):
                        err = max(err, compare_predictions([c.public() for c in paired[old]], outputs[stage]))
                    if diag != legacy_diag['merging']:
                        raise ValueError('Tiled merge changed')
                    check.update(exact_legacy_masks=True, maximum_score_error=err,
                                 zero_views=legacy_diag['zero_check_views'])
                if rgb is not None:
                    changed = dict(sample)
                    changed['depth_m'] = torch.zeros_like(sample['depth_m'])
                    changed['depth_valid'] = torch.zeros_like(sample['depth_valid'])
                    changed['camera_K'] = torch.eye(3)
                    compare_predictions(outputs[STAGES[0]], rgb.predict_sample(changed), score_tolerance=0)
                    check['rgb_ignores_geometry'] = True
                checks.append(check)
            for arm, values in outputs.items():
                totals[arm] += len(values)
                streams[arm].write(json.dumps({'image_id': image_id, 'predictions': values}, allow_nan=False)+'\n')
            if index == 0 or (index+1) % 10 == 0 or index+1 == len(ids):
                for stream in streams.values(): stream.flush()
                write_json(args.folder/f'progress_rank{args.rank}.json', dict(status='inferring',
                    split=args.split, images=index+1, expected=len(ids), elapsed_seconds=time.monotonic()-began))
    after = {}
    if adapter is not None:
        after.update(base=tensor_digest(adapter.state_dict()), heads=tensor_digest(heads.state_dict()))
    if rgb is not None: after['rgb'] = tensor_digest(rgb.state_dict())
    if before != after:
        raise ValueError('Frozen parameters changed')
    write_json(args.folder/f'rank{args.rank}.json', dict(status='completed', ordered_ids=ids,
        images=len(ids), instances=totals, checks=checks, reloads=reloads, frozen_tensor_sha256=after,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(), elapsed_seconds=time.monotonic()-began,
        prediction_sha256={a: file_sha(args.folder/a/f'predictions_rank{args.rank}.jsonl') for a in selected}))


def infer(p, protocol, folder, split, group, smoke=False):
    folder.mkdir()
    selected = arms(group)
    for arm in selected: (folder/arm).mkdir()
    with CourseArchive(p['archive'], split) as data:
        ids = data.image_ids
    if len(ids) != {'val': 2584, 'test': 2112}[split]:
        raise ValueError('Unexpected full split size')
    if smoke: ids = [ids[i] for i in (0, 97, 251, 501, 777, 1024, 1450, 2583)]
    write_json(folder/'image_ids.json', ids)
    processes = []
    with ExitStack() as stack:
        for rank, gpu in enumerate(p['gpus']):
            log = stack.enter_context((folder/f'worker{rank}.log').open('x'))
            command = [sys.executable, '-u', str(Path(__file__).resolve()), '--protocol', str(protocol),
                '--output', str(folder.parent), '--group', group, '--phase', split, '--worker',
                '--split', split, '--folder', str(folder), '--rank', str(rank)]
            if smoke: command.append('--smoke')
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
                       TOKENIZERS_PARALLELISM='false', PYTHONUNBUFFERED='1')
            processes.append(subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        started = time.monotonic()
        write_json(folder/'workers.json', [{'pid':v.pid, 'rank':i, 'gpu':p['gpus'][i]} for i,v in enumerate(processes)])
        while any(v.poll() is None for v in processes):
            write_json(folder/'process_status.json', dict(checked_at_utc=now(),
                elapsed_seconds=time.monotonic()-started,
                workers=[{'pid':v.pid, 'exit_code':v.poll()} for v in processes]))
            if time.monotonic()-started > p['worker_timeout_seconds']:
                for proc in processes:
                    if proc.poll() is None: os.killpg(proc.pid, signal.SIGTERM)
                for proc in processes:
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL); proc.wait()
                break
            time.sleep(10)
        exits = [{'pid':v.pid, 'exit_code':v.wait()} for v in processes]
        write_json(folder/'worker_exits.json', exits)
        if any(v['exit_code'] != 0 for v in exits):
            raise RuntimeError(f'Inference worker failed; inspect {folder}; no automatic retry')
    for arm in selected:
        shards = [folder/arm/f'predictions_rank{i}.jsonl' for i in range(len(processes))]
        for rank, path in enumerate(shards):
            receipt = read(folder/f'rank{rank}.json')
            if receipt['ordered_ids'] != ids[rank::len(processes)] or receipt['prediction_sha256'][arm] != file_sha(path):
                raise ValueError('Shard identity/coverage differs')
        merge_shards(shards, ids, folder/arm/'predictions.jsonl')
    if smoke:
        checks = [c for rank in range(len(processes)) for c in read(folder/f'rank{rank}.json')['checks']]
        if group != 'rgb' and {v for c in checks for v in c['zero_views']} != set(range(5)):
            raise ValueError('Engineering did not cover all five views')
        write_json(folder/'engineering.json', dict(status='passed', images=len(ids), checks=checks,
            protocol_sha256=file_sha(protocol), actual_worker_exit_codes=[v['exit_code'] for v in exits]))
        return None
    metrics = {}
    with CourseArchive(p['archive'], split) as data:
        for arm in selected:
            write_json(folder/'progress.json', dict(status='scoring', arm=arm, images=len(ids)))
            result = score_predictions(data, folder/arm/'predictions.jsonl', folder/arm)
            metrics[arm] = result['metrics']
    write_json(folder/'metrics.json', metrics)
    write_json(folder/'progress.json', dict(status='completed', images=len(ids)))
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--phase', choices=('val', 'test'), required=True)
    parser.add_argument('--group', choices=('rgbd', 'rgb', 'all'), required=True)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--folder', type=Path)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--split', choices=('val', 'test'))
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    p = read(args.protocol); verify(p)
    if args.worker:
        worker(args, p); return
    torch.set_num_threads(4)
    from tools.launch_course_longer_training import check_devices
    check_devices([str(g) for g in p['gpus']])
    args.output.mkdir(parents=True, exist_ok=True)
    run_path = args.output/f'{args.group}_{args.phase}_run.json'
    if run_path.exists(): raise FileExistsError(run_path)
    run = dict(status='running', pid=os.getpid(), phase=args.phase, group=args.group,
               protocol_sha256=file_sha(args.protocol), started_at_utc=now())
    write_json(run_path, run)
    try:
        if args.phase == 'val':
            if args.group == 'all': raise ValueError('Validate the two training arms separately')
            infer(p, args.protocol, args.output/f'{args.group}_smoke', 'val', args.group, True)
            infer(p, args.protocol, args.output/f'{args.group}_val', 'val', args.group)
        else:
            if args.group != 'all': raise ValueError('Test requires all five locked arms')
            lock_path = args.output/'validation_lock.json'
            lock = read(lock_path)
            if (lock['status'] != 'locked' or lock['protocol_sha256'] != file_sha(args.protocol)
                    or lock['test_used'] or lock['stages'] != list(STAGES)):
                raise ValueError('All five arms must be locked after validation and before test')
            for spec in lock['validation_inputs'].values():
                if file_sha(spec['path']) != spec['sha256']: raise ValueError('Validation input changed')
            lock_sha = file_sha(lock_path)
            metrics = infer(p, args.protocol, args.output/'test', 'test', 'all')
            ids = read(args.output/'test/image_ids.json')
            csvs = {}
            for i, arm in enumerate(STAGES):
                folder = args.output/'test'/arm
                path = folder/f'dpseg_s{i}_swin_t_1024_e15_{arm}.csv'
                predictions = folder/'predictions.jsonl'
                with SubmissionWriter(path, ids) as writer, predictions.open() as source:
                    for line in source:
                        row = json.loads(line); writer.write(row['image_id'], row['predictions'])
                receipt = validate_csv_fast(path, ids)
                compare_csv_predictions(path, predictions, ids)
                csvs[arm] = dict(path=str(path), sha256=file_sha(path), validation=receipt,
                                 exact_scored_predictions=True)
            if file_sha(lock_path) != lock_sha: raise ValueError('Validation lock changed during test')
            write_json(args.output/'comparison.json', dict(status='completed', stages=list(STAGES),
                val=lock['metrics'], test=metrics, csvs=csvs,
                protocol_sha256=file_sha(args.protocol), validation_lock_sha256=lock_sha,
                test_labels_used_for_selection=False, kaggle_submitted=False,
                local_test_distinct_from_kaggle_public=True, limitations=p['limitations']))
        run.update(status='completed', completed_at_utc=now())
    except BaseException as error:
        run.update(status='failed', error_type=type(error).__name__, error=str(error), failed_at_utc=now())
        raise
    finally:
        write_json(run_path, run)


if __name__ == '__main__':
    main()
