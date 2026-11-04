#!/usr/bin/env python3
"""Finish engineering checks, freeze both arms, then start the formal queue."""
from __future__ import annotations
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'reports/course_depth_backbone_v1'


def read(path):
    return json.loads(Path(path).read_text())


def write(path, data):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    from tools.run_course_depth_backbone import sources
    from tools.run_course_depth_backbone_system import sources as system_sources
    from tools.queue_course_depth_backbone import command, system_protocol
    state = {'status': 'engineering', 'pid': os.getpid(), 'started_at_utc': now(), 'steps': [],
             'script_sha256': sha(__file__)}
    if (OUT/'preparation.json').exists():
        raise FileExistsError('Existing preparation requires explicit review before another attempt')
    write(OUT/'preparation.json', state)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')

    def execute(name, argv, folder):
        record = {'name': name, 'command': argv, 'status': 'starting', 'started_at_utc': now()}
        state['steps'].append(record)
        write(OUT/'preparation.json', state)
        with (OUT/(name+'.log')).open('x') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, start_new_session=True)
            began = time.monotonic()
            record.update(status='running', pid=child.pid, hard_timeout_seconds=1800)
            write(OUT/'preparation.json', state)
            while child.poll() is None:
                if time.monotonic()-began > 1800:
                    record['timeout_at_utc'] = now()
                    write(OUT/'preparation.json', state)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError('Engineering stage exceeded30min: '+name)
                record['last_observed_alive_at_utc'] = now()
                write(OUT/'preparation.json', state)
                time.sleep(15)
            record.update(status='completed' if child.returncode == 0 else 'failed',
                          exit_code=child.returncode, finished_at_utc=now())
            write(OUT/'preparation.json', state)
            if child.returncode != 0:
                raise RuntimeError(name+' failed; no automatic retry')
        if read(folder/'run.json')['status'] != 'completed':
            raise ValueError('Missing completed engineering run: '+name)

    try:
        formal = {}
        for arm, base_name, protocol_name in (
            ('swint_depth_no_k', 'engineering_dense', 'protocol_engineering.json'),
            ('swinl_xyz', 'engineering_batch11', 'protocol_engineering_batch11.json')):
            folder = OUT/arm
            base = folder/base_name
            run = read(base/'run.json')
            if run['status'] != 'completed' or run['source_sha256'] != sources():
                raise ValueError('Base engineering is not complete or source changed: '+arm)
            protocol = read(folder/protocol_name)
            if arm == 'swint_depth_no_k':
                tail = folder/'engineering_tail'
                execute(arm+'_engineering_tail', command('run_course_depth_backbone.py',
                    '--protocol', folder/protocol_name, '--output', tail, '--resolution', 1024,
                    '--smoke-steps', 2, '--smoke-tail-images', 2), tail)
            system = system_protocol(protocol, base, scope='engineering_smoke')
            system['limits'] = {**system['limits'], 'per_trial_seconds': 1800}
            write(folder/'protocol_system_engineering.json', system)
            points = folder/'engineering_points'
            execute(arm+'_engineering_points', command('run_course_depth_backbone_system.py',
                '--protocol', folder/'protocol_system_engineering.json', '--base', base/'final.pt',
                '--output', points, '--mode', 'points', '--smoke'), points)
            selection = read(points/'selection.json')
            if not selection['base_unchanged'] or not selection['base_gradients_absent']:
                raise ValueError('Point training changed its frozen base')
            evaluation = folder/'engineering_system'
            execute(arm+'_engineering_system', command('run_course_depth_backbone_system.py',
                '--protocol', folder/'protocol_system_engineering.json', '--base', base/'final.pt',
                '--heads', selection['selected']['checkpoint'], '--output', evaluation,
                '--mode', 'evaluate', '--smoke'), evaluation)
            check = read(evaluation/'comparison.json')
            if set(check['val']) != {'native', 'full'}:
                raise ValueError('Full system variants were not evaluated')
            evidence = [base/'run.json', base/'model_sync_epoch_01.json', points/'run.json',
                points/'selection.json', evaluation/'run.json', evaluation/'comparison.json']
            if arm == 'swint_depth_no_k':
                evidence += [folder/'engineering_tail/run.json']
            gate = {'status': 'passed', 'scope': 'engineering_only', 'source_sha256': sources(),
                'system_source_sha256': system_sources(), 'base_probe': str(base),
                'base_peak_allocated_bytes': run['rank_peak_allocated_bytes'],
                'base_steps': run['optimizer_steps'], 'native_and_full_system_real_gpu_smoke': True,
                'point_training_frozen_base_and_rank_sync': True,
                'evidence_sha256': {str(path): sha(path) for path in evidence}, 'verified_at_utc': now()}
            write(folder/'engineering_gate.json', gate)
            protocol.update(status='frozen', frozen_system_source_sha256=system_sources(),
                engineering_validation={'path': str(folder/'engineering_gate.json'),
                                        'sha256': sha(folder/'engineering_gate.json')})
            write(folder/'protocol.json', protocol)
            formal[arm] = {'protocol': str(folder/'protocol.json'), 'sha256': sha(folder/'protocol.json')}
        attached = Path('/data/yinli/.codex/attachments/4bbf8349-fa71-4ff8-badc-39a5e011bc54/test_images.json')
        (OUT/'test_images.json').write_bytes(attached.read_bytes())
        write(OUT/'frozen_protocols.json', formal)
        with (OUT/'queue.log').open('x') as log:
            child = subprocess.Popen([sys.executable, str(ROOT/'tools/queue_course_depth_backbone.py'),
                                      '--root', str(OUT)], cwd=ROOT, env=env,
                stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = {'pid': child.pid, 'start_ticks': Path(f'/proc/{child.pid}/stat').read_text().split(') ')[1].split()[19],
                    'started_at_utc': now(), 'log': str(OUT/'queue.log')}
        write(OUT/'queue_process.json', identity)
        plan = read(OUT/'plan.json')
        plan.update(status='running', material_passport={**plan['material_passport'],
            'verification_status': 'ENGINEERING_PASSED_FORMAL_RUNNING'},
            formal_protocols=formal, test_metadata_sha256=sha(OUT/'test_images.json'))
        write(OUT/'plan.json', plan)
        state.update(status='completed', formal_queue=identity, finished_at_utc=now())
        write(OUT/'preparation.json', state)
        print(json.dumps({'status': 'formal_queue_started', **identity}), flush=True)
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), failed_at_utc=now())
        write(OUT/'preparation.json', state)
        raise


if __name__ == '__main__':
    main()
