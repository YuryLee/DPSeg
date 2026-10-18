#!/usr/bin/env python3
"""Detached, bounded resolution pair supervisor; stop visibly on any failure."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n')
    temporary.replace(path)


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--supervise', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    if protocol['status'] != 'frozen':
        raise ValueError('Formal supervisor requires frozen protocol')
    if not args.supervise:
        args.control.mkdir(parents=True, exist_ok=False)
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(args.output)
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
            '--protocol', str(args.protocol.resolve()), '--output', str(args.output.resolve()),
            '--control', str(args.control.resolve()), '--supervise']
        with (args.control/'supervisor.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=root, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        receipt = {'supervisor_pid': process.pid, 'command': command, 'started_at_utc': now()}
        write_json(args.control/'supervisor_start.json', receipt)
        print(json.dumps(receipt), flush=True)
        return 0
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=','.join(protocol['hardware']['physical_gpus']),
        OMP_NUM_THREADS=str(protocol['training']['threads']), PYTHONUNBUFFERED='1')
    record = {'status': 'starting', 'supervisor_pid': os.getpid(), 'started_at_utc': now(),
        'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(), 'trials': []}
    code = 1
    try:
        for resolution in protocol['execution_order']:
            command = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone',
                '--nnodes=1', f'--nproc-per-node={len(protocol["hardware"]["physical_gpus"])}',
                '--max-restarts=0', str(root/'tools/run_course_resolution_ddp.py'),
                '--protocol', str(args.protocol.resolve()), '--output', str(args.output.resolve()/str(resolution)),
                '--resolution', str(resolution)]
            with (args.control/f'console_{resolution}.log').open('x') as stream:
                process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                trial = {'resolution': resolution, 'pid': process.pid, 'command': command,
                    'started_at_utc': now(), 'status': 'running'}
                record['trials'].append(trial)
                record.update(status='running', active_resolution=resolution)
                write_json(args.control/'process.json', record)
                try:
                    code = process.wait(timeout=protocol['limits']['external_trial_seconds'])
                except subprocess.TimeoutExpired:
                    trial['hard_timeout'] = True
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        code = process.wait()
                trial.update(status='completed' if code == 0 else 'failed', exit_code=code,
                    finished_at_utc=now())
                write_json(args.control/'process.json', record)
                if code:
                    break
    finally:
        record.update(status='completed' if code == 0 else 'failed', exit_code=code,
            finished_at_utc=now())
        write_json(args.control/'exit.json', record)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
