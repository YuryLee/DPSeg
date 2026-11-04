#!/usr/bin/env python3
"""Complete the Large arm on two free A800s after a recorded zero-update OOM.

The original attempt and the completed no-intrinsics arm remain immutable.
This controller makes no changes to model or training implementations.
"""
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
from tools.queue_course_depth_backbone import (
    check_results, command, completed_base, read, sha, system_protocol, write,
)

OUT = ROOT/'reports/course_depth_backbone_v1_gpu02'
ORIGINAL = ROOT/'reports/course_depth_backbone_v1'


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    if (OUT/'queue.json').exists():
        raise FileExistsError('Existing recovery queue requires explicit review')
    old = read(ORIGINAL/'queue.json')
    failure = read(ORIGINAL/'swinl_xyz/base15/failure_rank1.json')
    if old['status'] != 'failed' or failure['error_type'] != 'OutOfMemoryError':
        raise ValueError('Original attempt is not a recorded terminal OOM')
    if any((ORIGINAL/f'swinl_xyz/base15/train_rank{rank}.jsonl').stat().st_size for rank in range(2)):
        raise ValueError('Original attempt completed updates; require explicit checkpoint migration')
    gate = read(OUT/'swinl_xyz/engineering_gate.json')
    if gate['status'] != 'passed':
        raise ValueError('New physical-device engineering gate is incomplete')
    for path, digest in gate['evidence_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Engineering evidence changed: '+path)
    protocol = read(OUT/'swinl_xyz/protocol.json')
    from tools.run_course_depth_backbone import sources
    from tools.run_course_depth_backbone_system import sources as system_sources
    if (protocol['status'] != 'frozen' or protocol['hardware']['physical_gpus'] != ['0', '2']
            or protocol['source_sha256'] != sources()
            or protocol['frozen_system_source_sha256'] != system_sources()
            or protocol['engineering_validation']['sha256'] != sha(OUT/'swinl_xyz/engineering_gate.json')):
        raise ValueError('Recovery protocol, hardware, gate or frozen sources differ')
    first_audit = read(OUT/'swint_depth_no_k/first_arm_completion_audit.json')
    if first_audit['status'] != 'passed':
        raise ValueError('First arm is incomplete')
    for path, digest in first_audit['evidence_sha256'].items():
        if sha(Path(path)) != digest:
            raise ValueError('Completed first-arm evidence changed: '+path)
    results = {'swint_depth_no_k': check_results(OUT/'swint_depth_no_k/final_evaluation')}
    state = {'status': 'running', 'pid': os.getpid(), 'started_at_utc': now(),
             'objective': read(OUT/'plan.json')['user_objective'],
             'queue_script_sha256': sha(Path(__file__)), 'stages': [],
             'original_failed_queue_sha256': sha(ORIGINAL/'queue.json'),
             'reused_completed_first_arm_audit_sha256': sha(OUT/'swint_depth_no_k/first_arm_completion_audit.json'),
             'hardware_only_change': True, 'physical_gpus': [0, 2]}
    write(OUT/'queue.json', state)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES='0,2', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

    def execute(name, argv, output):
        stage = {'name': name, 'status': 'starting', 'command': argv,
                 'output': str(output), 'started_at_utc': now(), 'hard_timeout_seconds': 605000}
        state['stages'].append(stage)
        state.update(current_stage=name, status='running')
        write(OUT/'queue.json', state)
        log_path = OUT/(name+'.log')
        with log_path.open('x') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            began = time.monotonic()
            stage.update(pid=child.pid, status='running', log=str(log_path))
            write(OUT/'queue.json', state)
            while child.poll() is None:
                elapsed = time.monotonic()-began
                if elapsed > stage['hard_timeout_seconds']:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError('Declared stage timeout: '+name)
                stage.update(last_observed_alive_at_utc=now(), elapsed_seconds=elapsed,
                             log_bytes=log_path.stat().st_size)
                write(OUT/'queue.json', state)
                time.sleep(30)
            stage.update(exit_code=child.returncode, finished_at_utc=now(),
                         elapsed_seconds=time.monotonic()-began,
                         status='completed' if child.returncode == 0 else 'failed')
            write(OUT/'queue.json', state)
            if child.returncode:
                raise RuntimeError(f'{name} exited {child.returncode}; no automatic retry')
        receipt = read(output/'run.json')
        if receipt['status'] != 'completed':
            raise ValueError('Missing completed stage receipt: '+name)
        stage['run_receipt_sha256'] = sha(output/'run.json')
        write(OUT/'queue.json', state)

    try:
        folder = OUT/'swinl_xyz'
        base = folder/'base15'
        execute('swinl_xyz_base15', command('run_course_depth_backbone.py', '--protocol',
                folder/'protocol.json', '--output', base, '--resolution', 1024), base)
        completed_base(base, protocol)
        system = system_protocol(protocol, base)
        if system['system_source_sha256'] != protocol['frozen_system_source_sha256']:
            raise ValueError('System sources changed during base training')
        system.update(official_test_metadata=str(OUT/'test_images.json'),
                      head_selection=str(folder/'points6/selection.json'))
        write(folder/'protocol_system.json', system)
        points = folder/'points6'
        execute('swinl_xyz_points6', command('run_course_depth_backbone_system.py', '--protocol',
                folder/'protocol_system.json', '--base', base/'final.pt', '--output', points,
                '--mode', 'points'), points)
        selected = read(points/'selection.json')
        if (selected['status'] != 'locked' or selected['epochs'] != 6
                or selected['training_images'] != 14432 or selected['global_steps'] != 10824
                or selected['test_used'] or not selected['base_unchanged']
                or not selected['base_gradients_absent']):
            raise ValueError('Point training or validation-only selection is incomplete')
        evaluation = folder/'final_evaluation'
        execute('swinl_xyz_final_evaluation', command('run_course_depth_backbone_system.py',
                '--protocol', folder/'protocol_system.json', '--base', base/'final.pt',
                '--heads', selected['selected']['checkpoint'], '--output', evaluation,
                '--mode', 'evaluate'), evaluation)
        results['swinl_xyz'] = check_results(evaluation)
        write(OUT/'results_partial.json', results)
        submissions = OUT/'submissions'
        submissions.mkdir(exist_ok=False)
        rows = []
        for arm, splits in results.items():
            for variant in ('native', 'full'):
                source = OUT/arm/'final_evaluation'/f'test_{variant}'/'submission.csv'
                target = submissions/f'dpseg_{arm}_base15_{variant}.csv'
                os.link(source, target)
                metrics = splits['test'][variant]['metrics']
                rows.append({'arm': arm, 'variant': variant,
                    'test_ap': metrics['mask_ap'], 'test_aps': metrics['mask_ap_small'],
                    'validation_metrics': splits['val'][variant]['metrics'],
                    'submission': str(target), 'submission_sha256': sha(target)})
        report = {'status': 'completed', 'scope': 'two_full_course_training_and_test_experiments',
            'rows': rows, 'test_used_for_selection': False, 'finished_at_utc': now(),
            'plan_sha256': sha(OUT/'plan.json'), 'limitations': read(OUT/'plan.json')['limitations'],
            'recovery': 'Large hardware-only change from GPU0,1 to GPU0,2 after zero-update OOM'}
        write(OUT/'comparison.json', report)
        state.update(status='completed', finished_at_utc=now(),
                     comparison_sha256=sha(OUT/'comparison.json'))
        write(OUT/'queue.json', state)
    except BaseException as error:
        state.update(status='failed', failed_at_utc=now(), error_type=type(error).__name__, error=str(error))
        write(OUT/'queue.json', state)
        raise


if __name__ == '__main__':
    main()
