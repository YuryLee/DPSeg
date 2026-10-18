#!/usr/bin/env python3
"""Detach a bounded supervisor so training and its exit receipt survive the UI."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    parser.add_argument('--supervise', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    protocol = json.loads(args.protocol.read_text())
    if protocol.get('status') != 'frozen':
        raise ValueError('Only frozen protocols may launch')
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
        receipt = {'supervisor_pid': process.pid, 'command': command,
            'detached': True, 'started_at_utc': datetime.now(timezone.utc).isoformat()}
        write_json(args.control/'supervisor_start.json', receipt)
        print(json.dumps(receipt), flush=True)
        return 0
    train = protocol['training']
    command = [sys.executable, '-u', str(root/'tools/run_course_small_target.py'),
        '--protocol', str(args.protocol.resolve()), '--output', str(args.output.resolve()),
        '--checkpoint', protocol['models']['dafusion']['pretrained_checkpoint'],
        '--archive', protocol['archive'], '--device', 'cuda:0']
    for key in ('epochs','batch_size','accumulation','seed','lr','precision','workers','threads'):
        command.extend(['--'+key.replace('_','-'), str(train[key])])
    command.extend(['--save-every', str(protocol['checkpoint_selection']['resume_checkpoint_every_updates']),
                    '--timeout-seconds', str(protocol['limits']['per_process_timeout_seconds'])])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=protocol['models']['dafusion']['gpu'],
               OMP_NUM_THREADS=str(train['threads']), PYTHONUNBUFFERED='1')
    if env['CUDA_VISIBLE_DEVICES'] != '5':
        raise ValueError('This frozen launch uses physical GPU 5')
    record = {'command': command, 'cwd': str(root), 'supervisor_pid': os.getpid(),
        'protocol_sha256': hashlib.sha256(args.protocol.read_bytes()).hexdigest(),
        'started_at_utc': datetime.now(timezone.utc).isoformat(), 'physical_gpu': env['CUDA_VISIBLE_DEVICES']}
    code = 1
    try:
        with (args.control/'console.log').open('x') as stream:
            process = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            record.update(pid=process.pid, status='running')
            write_json(args.control/'process.json', record)
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
    except BaseException as error:
        record.update(error_type=type(error).__name__, error=str(error))
        raise
    finally:
        record.update(status='completed' if code == 0 else 'failed', exit_code=code,
                      finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(args.control/'exit.json', record)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
