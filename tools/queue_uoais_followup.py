#!/usr/bin/env python3
"""Queue the already-defined UOAIS follow-up stages after a live base run.

The default only prints/checks the plan. --execute waits for complete training
and free assigned GPUs before running it. Failures stop this branch; existing
outputs are never overwritten and failed jobs are not automatically restarted.
OCBD remains a separate, required benchmark until its original data is obtained.
This module uses only the standard library and never changes the base runners.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/dafusion_benchmark_comparison'
PYTHONS = {
    'dafusion': ROOT.parent / 'dpseg_model_comparison_v4/calibnet_adapter/.venv/bin/python',
    'dpseg': ROOT / '.venv/bin/python',
}
BASES = {'dafusion': 'uoais_dafusion_swinl_20ep', 'dpseg': 'uoais_dpseg_swint_20ep'}
GPUS = {'dafusion': [5, 6], 'dpseg': [3, 7]}


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temp.replace(path)


def process_matches(pid, base):
    try:
        args = (Path('/proc') / str(pid) / 'cmdline').read_bytes().split(b'\0')
        args = [value.decode() for value in args if value]
        status = (Path('/proc') / str(pid) / 'status').read_text()
    except FileNotFoundError:
        return False
    if any(line.startswith('State:') and 'Z' in line for line in status.splitlines()):
        return False
    if not any(value.endswith('tools/train_uoais_benchmark.py') for value in args):
        return False
    try:
        output = Path(args[args.index('--output') + 1])
    except (ValueError, IndexError):
        return False
    return (output if output.is_absolute() else ROOT / output).resolve() == base.resolve()


def base_ready(base, family, live_check=process_matches):
    """A completion file alone cannot authorize a downstream training launch."""
    base = Path(base)
    run, protocol = read(base / 'run.json'), read(base / 'protocol.json')
    if run['family'] != family or run['scope'] != 'full_train_val':
        raise ValueError('Expected the full UOAIS training run for this family')
    if digest(base / 'protocol.json') != run['protocol_sha256']:
        raise ValueError('Base protocol changed')
    if protocol['status'] != 'frozen' or protocol['training']['epochs'] != 20:
        raise ValueError('Expected the frozen twenty-epoch base protocol')
    if list(base.glob('failure*.json')):
        raise RuntimeError('Base failure record needs review before follow-up')
    workers = run['hardware']
    if len(workers) != 2:
        raise ValueError('Expected both base workers in the launch receipt')
    live = [worker['pid'] for worker in workers if live_check(worker['pid'], base)]
    if run['status'] != 'training_completed':
        if run['status'] not in ('initializing', 'training') or len(live) != len(workers):
            raise RuntimeError('Base is incomplete and its full worker group is no longer live')
        return False, {'reason': 'base_training_running', 'worker_pids': live}
    state = run['state']
    history = state['validation']
    if (state['epoch'] != 20 or state['global_step'] != 112500
            or state['batch_in_epoch'] != 0 or len(history) != 20):
        raise ValueError('Base completion lacks the full training/validation budget')
    for epoch, result in enumerate(history, 1):
        if (result['epoch'] != epoch or result['scope'] != 'full_validation'
                or result['images'] != 5000 or result.get('test_used') is not False
                or not math.isfinite(result['metrics']['Objects F-measure'])):
            raise ValueError('Base has an incomplete or invalid validation history')
    if not (base / 'best.pt').is_file():
        raise ValueError('Validation-selected base checkpoint is missing')
    if live:
        return False, {'reason': 'base_finishing_cleanup', 'worker_pids': live}
    return True, {'reason': 'base_completed_and_workers_exited'}


def plan(family):
    base = REPORT / BASES[family]
    python = str(PYTHONS[family])
    gpu = GPUS[family]
    stages = []

    def stage(name, command, output, completion, expected, gpus):
        stages.append({'name': name, 'command': list(map(str, command)), 'output': str(output),
                       'completion': str(output / completion), 'expected': expected, 'gpus': gpus})

    if family == 'dafusion':
        for index, split in enumerate(('osd', 'ocid')):
            output = REPORT / f'uoais_dafusion_{split}_final_v1'
            manifest = ROOT / f'data/uois/{split}_manifest.json'
            count = {'osd': 111, 'ocid': 2390}[split]
            stage(f'predict_{split}', [python, 'tools/predict_uois_trained.py',
                  '--training-run', base, '--manifest', manifest, '--physical-gpu', gpu[index],
                  '--output', output], output, 'completion.json',
                  {'status': 'completed', 'images': count}, [gpu[index]])
            score = output / 'scores'
            stage(f'score_{split}', [python, 'tools/score_uois_benchmark.py',
                  '--manifest', manifest, '--predictions', output / 'predictions.jsonl',
                  '--output', score, '--score-threshold', '.5', '--model-description',
                  'Public DA-Fusion Swin-L reimplementation; trained on UOAIS-Sim; validation-selected; local shared protocol'],
                  score, 'evaluation.json', {'status': 'completed', 'images': count}, [])
    else:
        point = REPORT / 'uoais_dpseg_point_5ep'
        launcher = [python, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2']
        stage('train_point', launcher + ['tools/train_uoais_point.py', '--base-run', base,
              '--output', point, '--epochs', '5', '--global-batch', '4', '--learning-rate', '0.0001',
              '--seed', '29', '--workers', '2', '--save-every', '500'],
              point, 'run.json', {'status': 'training_completed', 'scope': 'full_train_val'}, gpu)
        validation = REPORT / 'uoais_dpseg_system_val_full_v1'
        common = launcher + ['tools/evaluate_uoais_system.py', '--point-run', point]
        stage('validate_system', common + ['--split', 'val', '--output', validation],
              validation, 'summary.json', {'status': 'completed', 'scope': 'full_validation', 'images': 5000}, gpu)
        for split in ('osd', 'ocid'):
            output = REPORT / f'uoais_dpseg_system_{split}_final_v1'
            stage(f'test_{split}', common + ['--split', split, '--manifest', ROOT / f'data/uois/{split}_manifest.json',
                  '--validation-run', validation, '--output', output], output, 'summary.json',
                  {'status': 'completed', 'scope': 'final_test', 'images': {'osd': 111, 'ocid': 2390}[split]}, gpu)
    return {'family': family, 'base_run': str(base), 'stages': stages,
            'remaining_after_queue': 'OCBD acquisition/evaluation and the final cross-model comparison report remain required.'}


def gpu_users(indices):
    if not indices:
        return []
    cards = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    chosen = {uuid.strip() for index, uuid in csv.reader(cards.splitlines()) if int(index) in indices}
    if len(chosen) != len(indices):
        raise RuntimeError('Assigned physical GPU is missing')
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader,nounits'], text=True, timeout=15)
    return [{'gpu_uuid': uuid.strip(), 'pid': int(pid)} for uuid, pid in csv.reader(processes.splitlines()) if uuid.strip() in chosen]


def source_snapshot(family):
    run = read(REPORT / BASES[family] / 'run.json')
    files = set(run['source_sha256']) | {
        'tools/queue_uoais_followup.py', 'tools/train_uoais_point.py', 'rgbd_seg/uoais_point_training.py',
        'tools/evaluate_uoais_system.py', 'tools/predict_uois_trained.py', 'tools/score_uois_benchmark.py',
        'rgbd_seg/course_point_refinement.py', 'rgbd_seg/course_progressive_ablation.py',
        'rgbd_seg/course_tiled_probability.py', 'rgbd_seg/course_depth_refinement.py',
        'rgbd_seg/course_optimized.py',
    }
    result = {name: digest(ROOT / name) for name in sorted(files)}
    if any(result[name] != expected for name, expected in run['source_sha256'].items()):
        raise ValueError('Frozen base source has changed')
    return result


def verify_completion(stage):
    result = read(stage['completion'])
    if any(result.get(key) != value for key, value in stage['expected'].items()):
        raise ValueError(f'Incomplete stage output: {stage["name"]}')
    if stage['name'] == 'train_point':
        state = result['state']
        if state['epoch'] != 5 or state['step'] != 56250 or len(state['validation']) != 5:
            raise ValueError('Point stage did not complete its declared full budget')
        if not result['base_unchanged'] or not result['base_gradients_absent']:
            raise ValueError('Point stage changed the frozen base')
    if stage['name'] == 'validate_system':
        selected = read(Path(stage['output']) / 'selection.json')
        if selected['scope'] != 'full_validation' or selected['images'] != 5000:
            raise ValueError('System stage lacks full validation selection')
    return {'path': stage['completion'], 'sha256': digest(stage['completion'])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--family', choices=list(BASES), required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    planned = plan(args.family)
    if not PYTHONS[args.family].is_file():
        raise FileNotFoundError(PYTHONS[args.family])
    sources = source_snapshot(args.family)
    readiness = base_ready(Path(planned['base_run']), args.family)
    for stage in planned['stages']:
        if Path(stage['output']).exists():
            raise FileExistsError(f'Follow-up output exists; review it before making a new queue: {stage["output"]}')
    if not args.execute:
        print(json.dumps({**planned, 'base_readiness': readiness, 'execute': False, 'source_sha256': sources}, indent=2))
        return
    if args.output is None:
        parser.error('--execute requires a new --output directory')
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'plan.json', {**planned, 'source_sha256': sources, 'created_utc': now()})
    state = {'status': 'waiting_for_base', 'family': args.family, 'pid': os.getpid(), 'completed_stages': [], 'started_utc': now()}
    last_message = None

    def update(status, **extra):
        nonlocal last_message
        state.update(status=status, updated_utc=now(), **extra)
        write(out / 'state.json', state)
        message = (status, state.get('stage'), extra.get('reason'))
        if message != last_message:
            print(json.dumps(state), flush=True)
            last_message = message

    env = os.environ.copy()
    for name in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'GROUP_RANK'):
        env.pop(name, None)
    env.update(OMP_NUM_THREADS='4', OPENBLAS_NUM_THREADS='1', HF_HUB_OFFLINE='1',
               TRANSFORMERS_OFFLINE='1', WANDB_MODE='disabled', PYTHONUNBUFFERED='1')
    if args.family == 'dpseg':
        env['PYTHONPATH'] = str(ROOT.parent / 'dpseg_generalization/.deps')
    try:
        while True:
            ready, details = base_ready(Path(planned['base_run']), args.family)
            if ready:
                break
            update('waiting_for_base', **details)
            time.sleep(30)
        for stage in planned['stages']:
            while True:
                users = gpu_users(stage['gpus'])
                free = shutil.disk_usage(ROOT).free / 2**30
                if not users and free >= 50:
                    break
                update('waiting_for_resources', stage=stage['name'], gpu_users=users, disk_free_gib=free)
                time.sleep(30)
            if source_snapshot(args.family) != sources:
                raise ValueError('Queued source changed; review before launching a later stage')
            if Path(stage['output']).exists():
                raise FileExistsError(stage['output'])
            env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, stage['gpus']))
            log = out / (stage['name'] + '.log')
            with log.open('x') as stream:
                child = subprocess.Popen(stage['command'], cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT)
                update('running_stage', stage=stage['name'], child_pid=child.pid, log=str(log))
                code = child.wait()
            if code:
                raise RuntimeError(f'{stage["name"]} exited {code}; see {log}')
            completion = verify_completion(stage)
            state['completed_stages'].append({'name': stage['name'], 'finished_utc': now(), **completion})
            update('stage_completed', stage=stage['name'], child_pid=None)
        update('available_benchmarks_completed', remaining=planned['remaining_after_queue'])
    except BaseException:
        update('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
