#!/usr/bin/env python3
"""Portable entry point for the unchanged paired point-head training loop.

Run an engineering check first, then pass its supervisor receipt to a full run.
The only runtime adapter supplies explicit paths to the audited base loader.
"""
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
from rgbd_seg.course_portable import DEFAULT_MANIFEST, check_file, load_base, read_json, source_path


def now():
    return datetime.now(timezone.utc).isoformat()


def build_protocol(args):
    manifest_path = args.manifest.resolve()
    bundle = read_json(manifest_path)
    base = bundle['base']
    base_protocol = check_file(manifest_path.parent / base['protocol'], base['protocol_sha256'], 'Base protocol')
    weights = check_file(args.base_weights, base['sha256'], 'Selected base weights')
    template = check_file(manifest_path.parent / bundle['head']['protocol'],
                          bundle['head']['protocol_sha256'], 'Point protocol template')
    p = copy.deepcopy(read_json(template))
    gpus = [int(value) for value in args.gpus.split(',')]
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0 or args.batch_size < 1:
        raise ValueError('Provide distinct GPU indices and a positive batch size')
    if args.epochs < 1 or not math.isfinite(args.timeout_hours) or args.timeout_hours <= 0:
        raise ValueError('Epochs and timeout must be positive')
    images = 16 if args.engineering else 14432
    if images % (len(gpus) * args.batch_size):
        raise ValueError('The sample count must be divisible by the global batch size')
    if not args.archive.is_file() or not args.coco_checkpoint.is_dir():
        raise ValueError('Provide the course ZIP and public COCO checkpoint directory')
    sources = {}
    for name, expected in p['source_sha256'].items():
        path = check_file(source_path(name), expected, 'Historical model source')
        sources[str(path) if Path(name).is_absolute() else name] = expected
    for name in ('rgbd_seg/course_portable.py', 'tools/course_train_points.py',
                 'tools/audit_course_longer_point.py', 'tools/launch_course_longer_training.py'):
        sources[name] = file_sha(ROOT / name)
    stat = args.archive.stat()
    p.update(
        schema='course_portable_point_training_v1', status='frozen', created_at_utc=now(),
        scope='engineering_smoke' if args.engineering else 'full_experiment',
        archive=str(args.archive.resolve()), archive_stat={'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns},
        weights={'path': str(weights), 'sha256': base['sha256']},
        training_protocol={'path': str(base_protocol), 'sha256': base['protocol_sha256']},
        model_manifest={'path': str(manifest_path), 'sha256': file_sha(manifest_path)},
        public_initialization=str(args.coco_checkpoint.resolve()),
        base_selection={'path': str(manifest_path), 'sha256': file_sha(manifest_path),
                        'kind': 'published_audited_base'},
        source_sha256=sources, gpus=gpus, batch_size=args.batch_size,
        global_batch_size=len(gpus) * args.batch_size,
        epochs=1 if args.engineering else args.epochs, training_images=images,
        evaluation_policy='validation_only',
        material_passport={'origin_skill': 'academic-research-suite/experiment-agent', 'mode': 'run',
                           'verification_status': 'ENGINEERING_PENDING' if args.engineering else 'MATCHING_ENGINEERING_PASSED'},
    )
    p.pop('engineering', None)
    p.pop('training_indices', None)
    if args.engineering:
        p['training_indices'] = list(range(16))
    else:
        if args.engineering_receipt is None:
            raise ValueError('A full run requires --engineering-receipt')
        receipt = read_json(args.engineering_receipt)
        engineer = read_json(receipt['protocol'])
        audit_binding = receipt.get('audit')
        if (receipt['status'] != 'completed' or receipt['scope'] != 'engineering_smoke'
                or not audit_binding
                or file_sha(audit_binding['path']) != audit_binding['sha256']
                or read_json(audit_binding['path'])['status'] != 'passed'
                or engineer['source_sha256'] != sources or engineer['weights'] != p['weights']
                or engineer['gpus'] != gpus or engineer['batch_size'] != args.batch_size):
            raise ValueError('Engineering must pass with the same source, base and GPU/batch layout')
        p['engineering'] = {'path': str(args.engineering_receipt.resolve()),
                            'sha256': file_sha(args.engineering_receipt)}
    return p


def worker(args):
    from tools import run_course_longer_point as runner
    from tools.run_course_point_refinement import checked
    p = checked(args.protocol)
    manifest = check_file(p['model_manifest']['path'], p['model_manifest']['sha256'], 'Model manifest')
    base_spec = read_json(manifest)['base']

    def relocated_base(weights_path, protocol_path, *, device='cpu'):
        return load_base(weights_path, protocol_path, coco_checkpoint=p['public_initialization'],
                         device=device, expected=base_spec)

    # Keep the original sampler, losses, optimizer, DDP and validation intact.
    runner.load_for_inference = relocated_base
    runner.train(p, args.output.resolve())


def terminate_child(process):
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def export_selected_manifest(output, protocol, final):
    bundle = copy.deepcopy(read_json(protocol['model_manifest']['path']))
    chosen = final['selection']['selected']
    bundle['name'] = f"course-base-e{bundle['base']['epoch']}-head-e{chosen['epoch']}"
    bundle['base']['protocol'] = Path(os.path.relpath(protocol['training_protocol']['path'], output)).as_posix()
    bundle['head'].update(protocol='protocol.json', protocol_sha256=file_sha(output / 'protocol.json'),
                          epoch=chosen['epoch'], sha256=chosen['checkpoint']['sha256'],
                          head_tensor_sha256=chosen['checkpoint']['head_tensor_sha256'])
    entry = next(row for row in final['validation_history'] if row['epoch'] == chosen['epoch'])
    bundle['validation'].update(selected_variant=chosen['variant'], mask_ap=chosen['mask_ap'],
                                mask_ap_small=chosen['mask_ap_small'], metrics=entry['metrics'],
                                completion_audit_sha256=file_sha(output / 'audit.json'))
    write_json(output / 'model_manifest.json', bundle)


def launch(args):
    from tools.launch_course_longer_training import check_devices
    from tools.audit_course_longer_point import audit
    from tools.run_course_point_refinement import checked
    protocol = build_protocol(args)
    check_devices([str(value) for value in protocol['gpus']])
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', protocol)
    command = [sys.executable, '-m', 'torch.distributed.run', '--standalone',
               f"--nproc_per_node={len(protocol['gpus'])}", str(Path(__file__).resolve()), 'worker',
               '--protocol', str(output / 'protocol.json'), '--output', str(output / 'experiment')]
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': ','.join(map(str, protocol['gpus'])),
           'OMP_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'PYTHONUNBUFFERED': '1',
           'TOKENIZERS_PARALLELISM': 'false'}
    state = {'status': 'starting', 'supervisor_pid': os.getpid(), 'created_at_utc': now(),
             'protocol': str(output / 'protocol.json'), 'protocol_sha256': file_sha(output / 'protocol.json'),
             'scope': protocol['scope'], 'command': command, 'gpus': protocol['gpus'],
             'timeout_hours': args.timeout_hours, 'log': str(output / 'training.log')}
    start = time.monotonic()
    interrupted = False
    with (output / 'training.log').open('x') as log:
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        state.update(status='running', pid=process.pid)
        write_json(output / 'supervisor.json', state)
        print(json.dumps({'status': 'running', 'pid': process.pid, 'output': str(output)}), flush=True)
        try:
            while process.poll() is None:
                state.update(elapsed_seconds=time.monotonic() - start, last_process_alive_at_utc=now())
                write_json(output / 'supervisor.json', state)
                if state['elapsed_seconds'] > args.timeout_hours * 3600:
                    state['status'] = 'timed_out'
                    terminate_child(process)
                    break
                time.sleep(10)
        except BaseException as exc:
            interrupted = isinstance(exc, KeyboardInterrupt)
            state['supervisor_error'] = f'{type(exc).__name__}: {exc}'
            terminate_child(process)
        code = process.wait()
    state.update(exit_code=code, elapsed_seconds=time.monotonic() - start, finished_at_utc=now())
    if code == 0 and not interrupted and 'supervisor_error' not in state and state['status'] != 'timed_out':
        final = read_json(output / 'experiment/comparison.json')
        if final['status'] != 'completed' or final['test_used']:
            raise RuntimeError('Child did not produce a completed validation-only report')
        files = [output / 'experiment/comparison.json', output / 'experiment/comparison.md',
                 *[Path(row['checkpoint']['path']) for row in final['validation_history']]]
        state.update(status='completed', artifact_sha256={str(path): file_sha(path) for path in files})
    elif state['status'] != 'timed_out':
        state['status'] = 'interrupted' if interrupted else 'failed'
    write_json(output / 'supervisor.json', state)
    write_json(output / 'exit.json', {'status': state['status'], 'exit_code': code})
    if state['status'] != 'completed':
        raise SystemExit(code if code > 0 else 1)
    try:
        result = audit(checked(output / 'protocol.json'), output / 'supervisor.json')
        write_json(output / 'audit.json', result)
        state['audit'] = {'path': str(output / 'audit.json'), 'sha256': file_sha(output / 'audit.json')}
        write_json(output / 'supervisor.json', state)
        if protocol['scope'] == 'full_experiment':
            export_selected_manifest(output, protocol, final)
    except Exception as exc:
        state.update(status='audit_failed', audit_error=str(exc))
        write_json(output / 'supervisor.json', state)
        write_json(output / 'exit.json', {'status': state['status'], 'exit_code': code})
        raise
    print(json.dumps({'status': 'completed_audited', 'scope': protocol['scope'],
                      'audit': state['audit'], 'output': str(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    train = sub.add_parser('launch', help='Build a frozen protocol and supervise training')
    train.add_argument('--archive', type=Path, required=True)
    train.add_argument('--base-weights', type=Path, required=True)
    train.add_argument('--coco-checkpoint', type=Path, required=True)
    train.add_argument('--manifest', type=Path, default=DEFAULT_MANIFEST)
    train.add_argument('--output', type=Path, required=True)
    train.add_argument('--gpus', default='0,1,2,3')
    train.add_argument('--batch-size', type=int, default=2)
    train.add_argument('--epochs', type=int, default=6)
    train.add_argument('--engineering', action='store_true')
    train.add_argument('--engineering-receipt', type=Path)
    train.add_argument('--timeout-hours', type=float, default=24)
    child = sub.add_parser('worker', help=argparse.SUPPRESS)
    child.add_argument('--protocol', type=Path, required=True)
    child.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'worker':
        worker(args)
    else:
        launch(args)


if __name__ == '__main__':
    main()
