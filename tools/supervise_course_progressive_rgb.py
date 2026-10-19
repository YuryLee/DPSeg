#!/usr/bin/env python3
"""Run the matched RGB baseline, with engineering gates and actual exit records."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def protocol(phase, output, gpus):
    from tools.run_course_progressive_rgb import sources
    template = ('reports/course_resolution_ddp/protocol.json' if phase == 'initial'
                else 'artifacts/protocols/base_epoch15.json')
    p = copy.deepcopy(read(ROOT / template))
    p.update(schema='course_progressive_rgb_control_v1', status='frozen',
             created_at_utc=now(), selected_resolution=1024,
             source_sha256=sources(), input_modality='RGB',
             main_table_epoch=15, test_used_for_selection=False)
    p['hardware'] = {'physical_gpus': [str(n) for n in gpus], 'memory_fraction': .9,
                     'reason': 'Only free GPUs; preserve all unrelated running jobs'}
    p['training'].update(per_rank_batch_sizes=[4, 4, 3], accumulation=3,
        effective_batch_size=33, workers_per_rank=2, threads=4)
    p['limitations'] = [
        'RGB and RGB-D share data, 1024 input, initialization source, 15 epochs, '
        'global batch33, AdamW and the two-stage learning-rate schedule.',
        'Available GPU layout and local microbatch partition differ from the historical '
        'RGB-D run; loss normalization and stochastic trajectories are not bitwise identical.',
        'Single seed; no test/Kaggle tuning. Main table uses fixed epoch15 in both arms.']
    if phase == 'continuation':
        parent = output / 'initial' / 'last.pt'
        parent_protocol = output / 'initial_protocol.json'
        p['continuation'].update(
            weights={'path': str(parent), 'sha256': file_sha(parent)},
            protocol={'path': str(parent_protocol), 'sha256': file_sha(parent_protocol)},
            initial_candidate='Retain completed RGB epoch5; main ablation fixes epoch15')
    return p


def execute(p, path, output, phase, smoke, record_path):
    from tools.launch_course_longer_training import check_devices
    check_devices(p['hardware']['physical_gpus'])
    output.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        f"--nproc_per_node={len(p['hardware']['physical_gpus'])}", '--max_restarts=0',
        str(ROOT / 'tools/run_course_progressive_rgb.py'), '--phase', phase,
        '--protocol', str(path), '--output', str(output), '--resolution', '1024']
    if smoke:
        command += ['--smoke-steps', '2', '--smoke-tail-images', '11', '--smoke-dense']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(p['hardware']['physical_gpus']),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1',
        TOKENIZERS_PARALLELISM='false')
    start = time.monotonic()
    timeout = 3600 if smoke else p['limits']['external_trial_seconds']
    log = output.parent / (output.name + '.log')
    with log.open('x') as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(status='running', phase=phase, smoke=smoke, pid=process.pid,
            supervisor_pid=os.getpid(), command=command, log=str(log),
            protocol=str(path), protocol_sha256=file_sha(path), started_at_utc=now())
        write_json(record_path, record)
        while process.poll() is None:
            record.update(last_process_check_at_utc=now(), elapsed_seconds=time.monotonic()-start)
            write_json(record_path, record)
            if time.monotonic()-start > timeout:
                record['hard_timeout'] = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                break
            time.sleep(10)
        code = process.wait()
    record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                  finished_at_utc=now(), elapsed_seconds=time.monotonic()-start)
    write_json(record_path, record)
    if code:
        raise RuntimeError(f'{phase} child failed with exit {code}; no automatic retry')
    run = read(output / 'run.json')
    if run['status'] != 'completed':
        raise ValueError('Successful child exit lacks a completed training record')
    return record


def audit_smoke(p, output, exit_record):
    import torch
    if exit_record['exit_code'] != 0:
        raise ValueError('Engineering child failed')
    run = read(output / 'run.json')
    checkpoint = torch.load(output / 'last.pt', map_location='cpu', weights_only=False)
    if (run['scope'] != 'engineering_smoke' or run['training_image_exposures'] != 44
            or run['optimizer_steps'] != 2 or len(checkpoint['rank_rng']) != 3):
        raise ValueError('Engineering train/RNG coverage differs')
    validation = next(output.glob('val_epoch_*/evaluation_run.json'))
    ev = read(validation)
    if ev['images'] != 6 or ev['depth_used']:
        raise ValueError('RGB engineering validation coverage/modality differs')
    for state in checkpoint['optimizer']['state'].values():
        for key in ('exp_avg', 'exp_avg_sq'):
            if not torch.isfinite(state[key]).all():
                raise ValueError('Engineering optimizer is non-finite')
    if checkpoint['source_sha256'] != p['source_sha256']:
        raise ValueError('Engineering checkpoint source identity differs')
    return dict(status='passed', at_utc=now(), source_sha256=p['source_sha256'],
        training_images=44, updates=2, validation_images=6, rng_states=3,
        actual_exit_code=0, optimizer_finite=True, depth_used=False,
        checkpoint_sha256=file_sha(output / 'last.pt'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpus', default='2,3,7')
    parser.add_argument('--engineering-only', action='store_true')
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    gpus = [int(x) for x in args.gpus.split(',')]
    if len(gpus) != 3 or len(set(gpus)) != 3:
        raise ValueError('This frozen layout requires exactly three distinct GPUs')
    state = dict(status='running', supervisor_pid=os.getpid(), started_at_utc=now(),
        gpus=gpus, phase='preparing', completed_phases=[])
    write_json(output / 'supervisor.json', state)
    try:
        for phase in ('initial', 'continuation'):
            p = protocol(phase, output, gpus)
            path = output / (phase + '_smoke_protocol.json'); write_json(path, p)
            state.update(phase=phase+'_engineering'); write_json(output / 'supervisor.json', state)
            smoke = output / ('smoke_' + phase)
            rec = execute(p, path, smoke, phase, True, output / (phase+'_engineering_process.json'))
            audit = audit_smoke(p, smoke, rec)
            audit_path = output / (phase + '_engineering_audit.json'); write_json(audit_path, audit)
            if args.engineering_only:
                break
            p['engineering_validation'] = {'path': str(audit_path), 'sha256': file_sha(audit_path)}
            path = output / (phase + '_protocol.json'); write_json(path, p)
            state.update(phase=phase+'_formal'); write_json(output / 'supervisor.json', state)
            execute(p, path, output / phase, phase, False, output / (phase+'_process.json'))
            state['completed_phases'].append(phase)
            write_json(output / 'supervisor.json', state)
        state.update(status='completed', finished_at_utc=now(),
                     scope='engineering_only' if args.engineering_only else 'rgb_15_epochs')
    except BaseException as error:
        state.update(status='failed', finished_at_utc=now(), error=str(error))
        write_json(output / 'supervisor.json', state)
        raise
    write_json(output / 'supervisor.json', state)


if __name__ == '__main__':
    main()
