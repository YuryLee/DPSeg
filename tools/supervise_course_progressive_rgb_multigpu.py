#!/usr/bin/env python3
"""Keep the current initial trial, then continue RGB epoch6--15 on six GPUs."""
from __future__ import annotations

import argparse
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
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha
from tools.supervise_course_progressive_rgb import protocol as old_protocol
from tools.run_course_progressive_rgb_multigpu import sources


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def process_stat(pid):
    """Read Linux state/start identity and the unreaped child's wait status."""
    fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
    return dict(state=fields[0], start_ticks=int(fields[19]), raw_wait_status=int(fields[49]))


def identity(pid, suffix, start_ticks=None):
    command = Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0', b' ').decode()
    stat = process_stat(pid)
    if suffix not in command or (start_ticks is not None and stat['start_ticks'] != start_ticks):
        raise ValueError(f'Unexpected process identity: {pid}')
    return stat


def take_over(output, old_pid, child_pid, state_path):
    old_state = read(state_path)
    record = read(output/'initial_process.json')
    if (old_state['status'] != 'running' or old_state['phase'] != 'initial_formal'
            or old_state['supervisor_pid'] != old_pid or record['pid'] != child_pid
            or record['status'] != 'running'):
        raise ValueError('Takeover requires the live original initial trial')
    if (output/'continuation').exists() or (output/'smoke_continuation').exists():
        raise ValueError('Continuation already exists; inspect before changing it')
    old_stat = identity(old_pid, str(ROOT/'tools/supervise_course_progressive_rgb.py'))
    child_stat = identity(child_pid, str(ROOT/'tools/run_course_progressive_rgb.py'))
    os.kill(old_pid, signal.SIGSTOP)
    try:
        for _ in range(100):
            if process_stat(old_pid)['state'] in ('T', 't'): break
            time.sleep(.05)
        else: raise RuntimeError('Original coordinator did not stop')
        state = dict(status='running', supervisor_pid=os.getpid(), started_at_utc=old_state['started_at_utc'],
            phase='initial_formal', completed_phases=[], active_gpus=[2, 3, 7],
            gpus=[2, 3, 4, 5, 6, 7], continuation_gpus=[2, 3, 4, 5, 6, 7],
            change_requested_at_utc=now(), previous_supervisor_pid=old_pid,
            hardware_change='User requested more GPUs; retain initial5 and continue10 on six GPUs')
        handoff = dict(status='monitoring_original_child', requested_at_utc=now(),
            original_supervisor_pid=old_pid, original_supervisor_start_ticks=old_stat['start_ticks'],
            initial_child_pid=child_pid, initial_child_start_ticks=child_stat['start_ticks'],
            original_state=old_state, new_supervisor_pid=os.getpid(),
            training_process_interrupted=False, effective_batch_unchanged=33,
            observer='Read unreaped child wait status from Linux /proc/PID/stat before reaping')
        write_json(output/'six_gpu_handoff.json', handoff)
        write_json(state_path, state)
    except BaseException:
        os.kill(old_pid, signal.SIGCONT)
        raise
    return state, record, handoff


def finish_initial(output, state, record, handoff, state_path):
    child, old = handoff['initial_child_pid'], handoff['original_supervisor_pid']
    protocol = read(output/'initial_protocol.json')
    began = datetime.fromisoformat(record['started_at_utc'])
    while True:
        stat = process_stat(child)
        if stat['start_ticks'] != handoff['initial_child_start_ticks']:
            raise ValueError('Initial PID was reused')
        elapsed = (datetime.now(timezone.utc)-began).total_seconds()
        if stat['state'] == 'Z':
            raw = stat['raw_wait_status']
            code = os.waitstatus_to_exitcode(raw)
            record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                raw_wait_status=raw, exit_observer='Linux unreaped /proc child wait status',
                supervisor_pid=os.getpid(), previous_supervisor_pid=old,
                elapsed_seconds=elapsed, finished_at_utc=now())
            write_json(output/'initial_process.json', record)
            # The original launcher is stopped; its child has finished. Retire
            # that launcher before it can schedule the old three-GPU continuation.
            identity(old, str(ROOT/'tools/supervise_course_progressive_rgb.py'),
                     handoff['original_supervisor_start_ticks'])
            os.kill(old, signal.SIGTERM); os.kill(old, signal.SIGCONT)
            handoff.update(status='initial_finished', initial_exit_code=code,
                           raw_wait_status=raw, initial_finished_at_utc=now())
            write_json(output/'six_gpu_handoff.json', handoff)
            if code: raise RuntimeError(f'Initial trial failed with exit {code}; no automatic retry')
            run = read(output/'initial/run.json')
            if (run['status'] != 'completed' or run['optimizer_steps'] != 2190
                    or run['training_image_exposures'] != 14432*5):
                raise ValueError('Initial training has incomplete coverage')
            state['completed_phases'] = ['initial']
            write_json(state_path, state)
            return
        record.update(last_process_check_at_utc=now(), elapsed_seconds=elapsed,
            supervisor_pid=os.getpid(), previous_supervisor_pid=old,
            exit_observer='Linux unreaped /proc child wait status')
        write_json(output/'initial_process.json', record)
        state.update(last_process_check_at_utc=now(), initial_child_pid=child,
                     initial_child_state=stat['state'])
        write_json(state_path, state)
        if elapsed > protocol['limits']['external_trial_seconds']:
            record['hard_timeout'] = True; write_json(output/'initial_process.json', record)
            os.killpg(child, signal.SIGTERM)
            raise TimeoutError('Original initial training reached its declared hard timeout')
        time.sleep(10)


def continuation_protocol(output):
    p = old_protocol('continuation', output, [2, 3, 4, 5, 6, 7])
    p.update(schema='course_progressive_rgb_six_gpu_v1', source_sha256=sources(),
             hardware_change_requested_by_user=True)
    p['training'].update(per_rank_batch_sizes=[2, 2, 2, 2, 2, 1], accumulation=3,
                         effective_batch_size=33)
    p['hardware'].update(reason='User requested six free GPUs after completed initial5; preserve global batch33',
                         initial_physical_gpus=['2', '3', '7'])
    p['limitations'].append('RGB initial5 uses three GPUs; continuation10 uses six GPUs with reseeded rank RNGs.')
    p['user_request'] = '再用多几张卡来训练吧'
    return p


def execute(p, path, folder, smoke, record_path):
    from tools.launch_course_longer_training import check_devices
    check_devices(p['hardware']['physical_gpus'])
    folder.mkdir(exist_ok=False)
    command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
        '--nproc_per_node=6', '--max_restarts=0', str(ROOT/'tools/run_course_progressive_rgb_multigpu.py'),
        '--protocol', str(path), '--output', str(folder), '--resolution', '1024']
    if smoke: command += ['--smoke-steps', '2', '--smoke-tail-images', '11', '--smoke-dense']
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(p['hardware']['physical_gpus']),
        OMP_NUM_THREADS='4', MKL_NUM_THREADS='4', PYTHONUNBUFFERED='1', TOKENIZERS_PARALLELISM='false')
    log = folder.parent/(folder.name+'.log'); started = time.monotonic()
    with log.open('x') as stream:
        proc = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
            stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        record = dict(status='running', phase='continuation', smoke=smoke, pid=proc.pid,
            supervisor_pid=os.getpid(), command=command, log=str(log), started_at_utc=now(),
            protocol=str(path), protocol_sha256=file_sha(path))
        while proc.poll() is None:
            record.update(last_process_check_at_utc=now(), elapsed_seconds=time.monotonic()-started)
            write_json(record_path, record)
            if time.monotonic()-started > (3600 if smoke else p['limits']['external_trial_seconds']):
                record['hard_timeout'] = True; write_json(record_path, record)
                os.killpg(proc.pid, signal.SIGTERM)
                try: proc.wait(timeout=30)
                except subprocess.TimeoutExpired: os.killpg(proc.pid, signal.SIGKILL)
                break
            time.sleep(10)
        code = proc.wait()
    record.update(status='completed' if code == 0 else 'failed', exit_code=code,
        elapsed_seconds=time.monotonic()-started, finished_at_utc=now())
    write_json(record_path, record)
    if code: raise RuntimeError(f'Six-GPU child failed with exit {code}; no automatic retry')
    if read(folder/'run.json')['status'] != 'completed': raise ValueError('Missing completed run receipt')
    return record


def audit_smoke(p, folder, record):
    run = read(folder/'run.json')
    checkpoint = torch.load(folder/'last.pt', map_location='cpu', weights_only=False)
    ev = read(next(folder.glob('val_epoch_*/evaluation_run.json')))
    if (record['exit_code'] != 0 or run['world_size'] != 6 or run['optimizer_steps'] != 2
            or run['training_image_exposures'] != 44 or len(checkpoint['rank_rng']) != 6
            or ev['images'] != 12 or ev['depth_used']
            or checkpoint['source_sha256'] != p['source_sha256']):
        raise ValueError('Six-GPU engineering coverage/provenance differs')
    for state in checkpoint['optimizer']['state'].values():
        if any(not torch.isfinite(state[key]).all() for key in ('exp_avg', 'exp_avg_sq')):
            raise ValueError('Nonfinite AdamW state')
    return dict(status='passed', at_utc=now(), source_sha256=p['source_sha256'],
        training_images=44, updates=2, validation_images=12, rng_states=6, actual_exit_code=0,
        optimizer_finite=True, depth_used=False, checkpoint_sha256=file_sha(folder/'last.pt'),
        world_size=6, rank_peak_allocated_bytes=run['rank_peak_allocated_bytes'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initial-supervisor', type=int, required=True)
    parser.add_argument('--initial-child', type=int, required=True)
    args = parser.parse_args()
    output = args.output.resolve(); state_path = output/'supervisor.json'
    if (output/'six_gpu_handoff.json').exists(): raise FileExistsError('Six-GPU handoff already exists')
    state, record, handoff = take_over(output, args.initial_supervisor, args.initial_child, state_path)
    try:
        finish_initial(output, state, record, handoff, state_path)
        p = continuation_protocol(output)
        path = output/'continuation_smoke_protocol.json'; write_json(path, p)
        state.update(phase='continuation_engineering', active_gpus=[2, 3, 4, 5, 6, 7]); write_json(state_path, state)
        smoke = output/'smoke_continuation'
        rec = execute(p, path, smoke, True, output/'continuation_engineering_process.json')
        audit = audit_smoke(p, smoke, rec)
        audit_path = output/'continuation_engineering_audit.json'; write_json(audit_path, audit)
        p['engineering_validation'] = dict(path=str(audit_path), sha256=file_sha(audit_path))
        path = output/'continuation_protocol.json'; write_json(path, p)
        state.update(phase='continuation_formal'); write_json(state_path, state)
        execute(p, path, output/'continuation', False, output/'continuation_process.json')
        state.update(status='completed', phase='rgb15_complete', scope='rgb_15_epochs',
                     completed_phases=['initial', 'continuation'], finished_at_utc=now())
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), finished_at_utc=now())
        write_json(state_path, state)
        raise
    write_json(state_path, state)


if __name__ == '__main__':
    main()
