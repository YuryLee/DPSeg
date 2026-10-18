#!/usr/bin/env python3
"""Run the frozen continuation arms sequentially, with process and timeout records."""
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n')
    temporary.replace(path)


def check_devices(gpus):
    text = subprocess.check_output(['nvidia-smi', '--query-gpu=index,memory.used',
                                   '--format=csv,noheader,nounits'], text=True)
    used = {row.split(',')[0].strip(): int(row.split(',')[1]) for row in text.splitlines()}
    busy = {gpu: used.get(gpu) for gpu in gpus if gpu not in used or used[gpu] > 512}
    if busy:
        raise RuntimeError(f'Requested GPUs are already in use (MiB): {busy}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocols', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--supervise', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    protocols = [json.loads(p.read_text()) for p in args.protocols]
    if any(p['status'] != 'frozen' for p in protocols):
        raise ValueError('Use frozen protocols')
    if not args.supervise:
        for protocol in protocols:
            check_devices(protocol['hardware']['physical_gpus'])
        args.control.mkdir(parents=True, exist_ok=False)
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(args.output)
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
            '--protocols', *[str(p.resolve()) for p in args.protocols],
            '--output', str(args.output.resolve()), '--control', str(args.control.resolve()), '--supervise']
        if args.smoke:
            command.append('--smoke')
        with (args.control/'supervisor.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'supervisor_pid': process.pid, 'command': command, 'started_at_utc': now()}
        write_json(args.control/'supervisor_start.json', receipt)
        print(json.dumps(receipt), flush=True)
        return 0
    lease_path = ROOT/'reports/course_longer_training/device_lease.lock'
    lease_path.parent.mkdir(exist_ok=True, parents=True)
    lease = lease_path.open('a')
    fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
    record = {'status': 'starting', 'supervisor_pid': os.getpid(), 'started_at_utc': now(),
              'scope': 'engineering_smoke' if args.smoke else 'full_train_val', 'trials': []}
    code = 1
    try:
        for path, protocol in zip(args.protocols, protocols):
            arm = protocol['arm']
            check_devices(protocol['hardware']['physical_gpus'])
            command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                '--nnodes=1', f'--nproc-per-node={len(protocol["hardware"]["physical_gpus"])}',
                '--max-restarts=0', str(ROOT/'tools/run_course_longer_training.py'),
                '--protocol', str(path.resolve()), '--output', str(args.output.resolve()/arm),
                '--resolution', '1024']
            if args.smoke:
                command += ['--smoke-steps', '2', '--smoke-tail-images', '11', '--smoke-dense']
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(protocol['hardware']['physical_gpus']),
                OMP_NUM_THREADS=str(protocol['training']['threads']), PYTHONUNBUFFERED='1')
            with (args.control/f'console_{arm}.log').open('x') as stream:
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                trial = {'arm': arm, 'pid': process.pid, 'command': command,
                    'protocol_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'started_at_utc': now(), 'status': 'running'}
                record['trials'].append(trial)
                record.update(status='running', active_arm=arm)
                began = time.monotonic()
                timeout = 1800 if args.smoke else protocol['limits']['external_trial_seconds']
                while True:
                    record.update(last_process_check_utc=now(), active_process_alive=process.poll() is None)
                    write_json(args.control/'process.json', record)
                    try:
                        code = process.wait(timeout=min(30, max(.1, timeout-(time.monotonic()-began))))
                        break
                    except subprocess.TimeoutExpired:
                        if time.monotonic()-began < timeout:
                            continue
                        trial['hard_timeout'] = True
                        write_json(args.control/'process.json', record)
                        os.killpg(process.pid, signal.SIGTERM)
                        try:
                            code = process.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGKILL)
                            code = process.wait()
                        break
                trial.update(status='completed' if code == 0 else 'failed', exit_code=code,
                             finished_at_utc=now())
                write_json(args.control/'process.json', record)
                if code:
                    break
        if code == 0:
            candidates = []
            for path, protocol in zip(args.protocols, protocols):
                run_path = args.output/protocol['arm']/'run.json'
                run = json.loads(run_path.read_text())
                if run['status'] != 'completed':
                    raise ValueError('A child exited without a completed run record')
                candidates.append({'arm': protocol['arm'], 'protocol': str(path.resolve()),
                    'checkpoint': str((run_path.parent/'best.pt').resolve()),
                    'epoch': run['best_epoch'], 'validation_APs': run['best_validation_key'][0],
                    'validation_AP': run['best_validation_key'][1],
                    'checkpoint_sha256': run['best_checkpoint_sha256']})
            candidates.sort(key=lambda v: (-v['validation_APs'], -v['validation_AP'], v['epoch'], v['arm']))
            write_json(args.control/'base_comparison.json', {
                'status': 'base_training_completed', 'scope': record['scope'],
                'selection_split': 'val', 'test_used': False, 'candidates': candidates,
                'selected': candidates[0],
                'remaining': ['Adapt and train point head for the selected backbone',
                              'Full validation comparison with tiled probability-depth inference',
                              'Locked local test evaluation and validated submission CSV']})
    except BaseException as error:
        code = 1
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                      active_process_alive=False, finished_at_utc=now())
        write_json(args.control/'exit.json', record)
        write_json(args.control/'process.json', record)
        lease.close()
    return code


if __name__ == '__main__':
    raise SystemExit(main())
