#!/usr/bin/env python3
"""Execute the predeclared v4 matrix sequentially, preserving every attempt."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/v4/experiment'


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = ROOT / 'docs/paper_trial_protocol.md'
    entries = []
    for split in ('val', 'test'):
        for seed, mode, folder in [(17, 'E2', 'graspnet_v2'), (23, 'E2', 'graspnet_seed23'),
                                   (41, 'E2', 'graspnet_seed41'), (17, 'E0', 'graspnet_v2')]:
            entries.append(('graspnet', split, seed, mode, folder))
    for mode in ('E0', 'E2'):
        entries.append(('tless', 'test', 17, mode, 'graspnet_v2'))
    ledger = {'created_at_utc': now(), 'status': 'running',
              'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
              'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
              'per_process_timeout_seconds': 1200, 'entries': []}
    if ledger['cuda_visible_devices'] != '4':
        raise ValueError('The frozen v4 run uses only physical GPU 4')

    def save():
        tmp = OUT / 'matrix.tmp.json'
        tmp.write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + '\n')
        tmp.replace(OUT / 'matrix.json')

    for dataset, split, seed, mode, folder in entries:
        output = OUT / dataset / f'seed{seed}_{mode}' / split
        data = 'data/graspnet_expanded_v2' if dataset == 'graspnet' else 'data/tless_confirmation'
        command = [sys.executable, '-u', 'tools/evaluate_multiscale.py', '--data', data,
                   '--split', split, '--checkpoint', f'outputs/{folder}/{mode}/model.pt',
                   '--output', str(output), '--device', 'cuda:0', '--threads', '4']
        if dataset == 'graspnet':
            command += ['--extra']
        ledger['entries'].append({'dataset': dataset, 'split': split, 'seed': seed, 'mode': mode,
                                  'command': command, 'output': str(output), 'status': 'pending'})
    save()
    for index, entry in enumerate(ledger['entries']):
        if hashlib.sha256(protocol.read_bytes()).hexdigest() != ledger['protocol_sha256']:
            raise RuntimeError('Frozen protocol changed during experiment')
        entry.update(status='running', started_at_utc=now())
        save()
        print(f"START {index + 1}/10 {entry['dataset']} {entry['split']} seed{entry['seed']} {entry['mode']}", flush=True)
        start = time.monotonic()
        log = OUT / f"{index + 1:02d}_{entry['dataset']}_{entry['split']}_{entry['seed']}_{entry['mode']}.log"
        with log.open('x') as stream:
            process = subprocess.Popen(['timeout', '--signal=TERM', '--kill-after=30s', '1200s', *entry['command']],
                                       cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
            entry['pid'] = process.pid
            save()
            code = process.wait()
        entry.update(status='completed' if code == 0 else 'failed', exit_code=code,
                     finished_at_utc=now(), wall_seconds=time.monotonic() - start, log=str(log))
        save()
        print(f"END {index + 1}/10 {entry['status']} {entry['wall_seconds']:.1f}s", flush=True)
        if code:
            ledger.update(status='failed', finished_at_utc=now())
            save()
            raise SystemExit(code)
    ledger.update(status='completed', finished_at_utc=now())
    save()


if __name__ == '__main__':
    main()
