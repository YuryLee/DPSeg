#!/usr/bin/env python3
"""Persist one authorized experiment's command, output and bounded exit status."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('status') != 'frozen':
        raise ValueError('Only a frozen protocol can be supervised')
    args.control.mkdir(parents=True, exist_ok=False)
    train = protocol['training']
    command = [sys.executable, '-u', str(root/'tools/run_course_dafusion.py'),
               '--protocol', str(args.protocol.resolve()), '--output', str(args.output.resolve()),
               '--checkpoint', protocol['models']['dafusion']['pretrained_checkpoint'],
               '--archive', protocol['archive'], '--device', 'cuda:0']
    for key in ('epochs', 'batch_size', 'accumulation', 'seed', 'lr', 'precision', 'workers', 'threads'):
        command.extend(['--'+key.replace('_', '-'), str(train[key])])
    command.extend(['--save-every', str(protocol['checkpoint_selection']['resume_checkpoint_every_updates']),
                    '--timeout-seconds', str(protocol['limits']['per_process_timeout_seconds'])])
    if args.resume:
        command.append('--resume')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=protocol['models']['dafusion']['gpu'],
               OMP_NUM_THREADS=str(train['threads']), PYTHONUNBUFFERED='1')
    record = {'command': command, 'cwd': str(root), 'started_at_utc': datetime.now(timezone.utc).isoformat(),
              'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
              'supervisor_pid': os.getpid(), 'physical_gpu': env['CUDA_VISIBLE_DEVICES']}
    with (args.control/'console.log').open('x') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        record.update(pid=process.pid, status='running')
        (args.control/'process.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
        try:
            code = process.wait(timeout=protocol['limits']['external_timeout_seconds'])
        except subprocess.TimeoutExpired:
            record['external_timeout'] = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait()
        record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        (args.control/'exit.json').write_text(json.dumps(record, indent=2)+'\n')
        print(json.dumps(record), flush=True)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
