#!/usr/bin/env python3
"""Run the fixed v5 matrix sequentially, preserving every process receipt."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_v5 import source_snapshot, write


def main():
    os.chdir(ROOT)
    ledger_path = ROOT / 'reports/v5/experiment_ledger.json'
    if ledger_path.exists():
        raise FileExistsError('Existing matrix receipt: never silently resume or overwrite')
    env = dict(os.environ, CUDA_VISIBLE_DEVICES='5', OMP_NUM_THREADS='4')
    python = str(ROOT / '.venv/bin/python')
    variants = ['baseline', 'bce2', 'bace', 'freqfusion']
    common = ['--device', 'cuda:0', '--threads', '4', '--cuda-memory-fraction', '.55',
              '--protocol', 'docs/training_methods_protocol.md']
    jobs = []
    for seed in (17, 23, 41):
        config = 'configs/GraspNetV2_E2.json' if seed == 17 else f'configs/seed_replication/seed{seed}_E2.json'
        for variant in variants:
            output = f'outputs/paper_v5/seed{seed}/{variant}'
            jobs.append({'id': f'train_seed{seed}_{variant}', 'kind': 'train', 'timeout_s': 1800,
                         'output': output, 'command': [python, 'tools/run_v5.py', 'train',
                         '--config', config, '--variant', variant, '--init-rgb',
                         'pretrained/maskrcnn_resnet50_fpn_v2_coco_mirror_state_dict.pt',
                         '--steps', '800', '--data', 'data/graspnet_expanded_v2', '--output', output, *common]})
    for dataset, seeds, splits in [('graspnet', (17, 23, 41), ('val', 'test')),
                                    ('tless', (17,), ('test',))]:
        data = 'data/graspnet_expanded_v2' if dataset == 'graspnet' else 'data/tless_confirmation'
        for split in splits:
            for seed in seeds:
                for variant in variants:
                    output = f'reports/v5/evaluation/{dataset}/seed{seed}/{variant}/{split}'
                    jobs.append({'id': f'eval_{dataset}_seed{seed}_{variant}_{split}',
                                 'kind': 'eval', 'timeout_s': 900, 'output': output,
                                 'command': [python, 'tools/run_v5.py', 'eval', '--split', split,
                                 '--checkpoint', f'outputs/paper_v5/seed{seed}/{variant}/model.pt',
                                 '--data', data, '--output', output, *common]})
    protocol = ROOT / 'docs/training_methods_protocol.md'
    ledger = {'created_at_utc': datetime.now(timezone.utc).isoformat(), 'status': 'running',
              'protocol_sha256': hashlib.sha256(protocol.read_bytes()).hexdigest(),
              'source_sha256': source_snapshot(), 'physical_gpu': 5,
              'expected_trainings': 12, 'expected_evaluations': 28, 'jobs': jobs}
    log_root = ROOT / 'reports/v5/logs'
    log_root.mkdir(exist_ok=True)
    write(ledger_path, ledger)
    for index, job in enumerate(jobs):
        log_path = log_root / (job['id'] + '.log')
        job.update(status='running', started_at_utc=datetime.now(timezone.utc).isoformat(), log=str(log_path))
        print(f"START {index + 1}/{len(jobs)} {job['id']}", flush=True)
        start = time.monotonic()
        with log_path.open('x') as log:
            process = subprocess.Popen(job['command'], env=env, stdout=log, stderr=subprocess.STDOUT)
            job['pid'] = process.pid
            write(ledger_path, ledger)
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - start
                    if elapsed > job['timeout_s']:
                        print(f"HARD TIMEOUT: stopping {job['id']} after {elapsed:.1f}s", flush=True)
                        process.kill()
                        process.wait()
                        job['status'] = 'timeout'
                        break
                    job['last_alive_at_utc'] = datetime.now(timezone.utc).isoformat()
                    job['elapsed_s'] = elapsed
                    job['log_bytes'] = log_path.stat().st_size
                    write(ledger_path, ledger)
                    time.sleep(5)
            except BaseException:
                # Do not silently orphan an experiment when the monitor is interrupted.
                job['status'] = 'monitor_interrupted'
                ledger['status'] = 'monitor_interrupted'
                write(ledger_path, ledger)
                raise
        job.update(returncode=process.returncode, duration_s=time.monotonic() - start,
                   ended_at_utc=datetime.now(timezone.utc).isoformat())
        if job.get('status') != 'timeout':
            job['status'] = 'completed' if process.returncode == 0 else 'failed'
        print(f"END {job['id']} {job['status']} {job['duration_s']:.1f}s", flush=True)
        if job['status'] != 'completed':
            ledger['status'] = 'failed'
            write(ledger_path, ledger)
            raise SystemExit(1)
        write(ledger_path, ledger)
    ledger.update(status='completed', finished_at_utc=datetime.now(timezone.utc).isoformat())
    write(ledger_path, ledger)


if __name__ == '__main__':
    main()
