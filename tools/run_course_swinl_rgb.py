#!/usr/bin/env python3
"""Supervise the RGB Swin-L smoke check, fresh training, and fixed-epoch test."""
import copy
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.run_course_dafusion import now, sha256, write_json
from tools.train_course_swinl_rgb import sources

OUT = ROOT / 'reports/course_swinl_rgb_v1'
REFERENCE = ROOT / 'reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/protocol.json'
PHYSICAL_GPUS = ('5', '6', '7')
PER_RANK_BATCHES = (4, 4, 3)
SMOKE_TAIL_IMAGES = len(PHYSICAL_GPUS)


def read(path):
    return json.loads(Path(path).read_text())


def command(script, *args):
    return [str(ROOT / '.venv/bin/python'), '-m', 'torch.distributed.run',
            '--standalone', f'--nproc_per_node={len(PHYSICAL_GPUS)}', str(ROOT / 'tools' / script), *map(str, args)]


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    lock = (OUT / 'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'queue.json').exists():
        raise FileExistsError('Existing run requires review; no implicit restart')
    if shutil.disk_usage(ROOT).free < 30 * 2**30:
        raise OSError('Need 30 GiB free for checkpoints and predictions')
    protocol = copy.deepcopy(read(REFERENCE))
    for key in ('engineering_validation', 'frozen_system_source_sha256', 'hardware_amendment'):
        protocol.pop(key, None)
    protocol.update(schema='course_swinl_rgb_training_v1', status='engineering',
        source_sha256=sources(), model={'backbone': 'large', 'input_modality': 'RGB',
        'backbone_checkpointing': True}, evaluation={'base_selection': 'fixed_epoch15',
        'per_epoch_full_validation': True, 'test_used_for_selection': False},
        reference_protocol={'path': str(REFERENCE), 'sha256': sha256(REFERENCE)})
    protocol['hardware'].update(physical_gpus=list(PHYSICAL_GPUS))
    protocol['training']['per_rank_batch_sizes'] = list(PER_RANK_BATCHES)
    protocol['hardware_amendment'] = {
        'reason': 'Reference GPUs0/2 became occupied before launch; use free A100 GPUs5/6/7',
        'effective_batch_size_preserved': 33,
        'difference': 'Local microbatch partition and per-rank RNG differ from the RGB-D run'}
    plan = {'schema': 'course_swinl_rgb_v1',
        'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent',
            'mode': 'run', 'verification_status': 'ENGINEERING_PENDING'},
        'user_objective': 'Mask2Former 换成 Swin-Large，重新在课程训练集微调',
        'created_at_utc': now(), 'model': protocol['model'],
        'training': protocol['training'], 'hardware': protocol['hardware'],
        'data': {'train': 14432, 'val': 2584, 'test': 2112, 'resolution': 1024},
        'main_table_checkpoint': 'Fresh COCO initialization -> fixed epoch15',
        'reference_protocol': protocol['reference_protocol'],
        'limitations': ['Single seed17', 'Local annotated test differs from Kaggle hidden evaluation',
            'RGB uses three A100 GPUs with batches4/4/3; RGB-D used two A800 GPUs with batches6/5; both accumulate3 for batch33'],
        'outputs': ['base15/final.pt', 'base15/selection.json', 'final_evaluation/test/evaluation.json',
                    'submissions/mask2former_swinl_rgb_1024_e15.csv']}
    write_json(OUT / 'plan.json', plan)
    write_json(OUT / 'protocol_engineering.json', protocol)
    state = {'status': 'running', 'pid': os.getpid(), 'started_at_utc': now(), 'stages': [],
             'controller_sha256': sha256(__file__), 'physical_gpus': list(map(int, PHYSICAL_GPUS))}
    write_json(OUT / 'queue.json', state)
    env = dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES=','.join(PHYSICAL_GPUS), OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1',
        TRANSFORMERS_OFFLINE='1')

    def execute(name, argv, timeout):
        log_path = OUT / (name + '.log')
        record = {'name': name, 'command': argv, 'status': 'starting', 'started_at_utc': now(),
                  'hard_timeout_seconds': timeout, 'log': str(log_path)}
        state['stages'].append(record)
        state['current_stage'] = name
        write_json(OUT / 'queue.json', state)
        with log_path.open('x') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=env, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            start = time.monotonic()
            record.update(status='running', pid=child.pid)
            while child.poll() is None:
                elapsed = time.monotonic() - start
                record.update(last_observed_alive_at_utc=now(), elapsed_seconds=elapsed,
                              log_bytes=log_path.stat().st_size)
                write_json(OUT / 'queue.json', state)
                if elapsed > timeout:
                    record.update(status='timeout', timeout_at_utc=now())
                    write_json(OUT / 'queue.json', state)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError('Declared hard timeout: ' + name)
                time.sleep(15)
            record.update(status='completed' if child.returncode == 0 else 'failed',
                exit_code=child.returncode, finished_at_utc=now(), elapsed_seconds=time.monotonic()-start)
            write_json(OUT / 'queue.json', state)
            if child.returncode:
                raise RuntimeError(name + ' failed; queue stopped without automatic retry')

    try:
        execute('engineering_dense_tail', command('train_course_swinl_rgb.py',
            '--protocol', OUT / 'protocol_engineering.json', '--output', OUT / 'engineering_dense_tail',
            '--resolution', 1024, '--smoke-steps', 2,
            '--smoke-tail-images', SMOKE_TAIL_IMAGES, '--smoke-dense'), 1800)
        smoke = read(OUT / 'engineering_dense_tail/run.json')
        metadata = smoke['model_metadata']
        if (smoke['status'] != 'completed' or smoke['optimizer_steps'] != 2
                or smoke['training_image_exposures'] != 33 + SMOKE_TAIL_IMAGES or metadata['depth_used']
                or metadata['backbone'] != 'large' or smoke['source_sha256'] != sources()):
            raise ValueError('Dense/tail training and evaluation smoke check failed')
        gate = {'status': 'passed', 'created_at_utc': now(), 'source_sha256': sources(),
            'smoke_run': str(OUT / 'engineering_dense_tail/run.json'),
            'smoke_run_sha256': sha256(OUT / 'engineering_dense_tail/run.json'),
            'checks': ['Dense real-image batch and partial final batch',
                       'Three GPUs remain synchronized after both optimizer steps',
                       'Validation inference, merged masks and COCO scoring'],
            'rank_peak_allocated_bytes': smoke['rank_peak_allocated_bytes']}
        write_json(OUT / 'engineering_gate.json', gate)
        protocol.update(status='frozen', engineering_validation={
            'path': str(OUT / 'engineering_gate.json'), 'sha256': sha256(OUT / 'engineering_gate.json')})
        write_json(OUT / 'protocol.json', protocol)
        plan['material_passport']['verification_status'] = 'FORMAL_RUNNING'
        write_json(OUT / 'plan.json', plan)
        execute('base15', command('train_course_swinl_rgb.py', '--protocol', OUT / 'protocol.json',
            '--output', OUT / 'base15', '--resolution', 1024), protocol['limits']['external_trial_seconds'])
        trained = read(OUT / 'base15/run.json')
        if (trained['status'] != 'completed' or trained['optimizer_steps'] != math.ceil(14432/33)*15
                or trained['training_image_exposures'] != 14432*15):
            raise ValueError('Incomplete formal training budget')
        execute('final_evaluation', command('evaluate_course_swinl_rgb.py',
            '--protocol', OUT / 'protocol.json', '--weights', OUT / 'base15/final.pt',
            '--output', OUT / 'final_evaluation'), 86400)
        evaluation = read(OUT / 'final_evaluation/test/evaluation.json')
        receipt = read(OUT / 'final_evaluation/test/submission_validation.json')
        source_csv = OUT / 'final_evaluation/test/submission.csv'
        if receipt['sha256'] != sha256(source_csv):
            raise ValueError('CSV validation receipt differs')
        (OUT / 'submissions').mkdir()
        csv = OUT / 'submissions/mask2former_swinl_rgb_1024_e15.csv'
        os.link(source_csv, csv)
        val = read(OUT / 'base15/selection.json')['epochs'][-1]
        result = {'status': 'completed', 'method': 'RGB Mask2Former Swin-Large', 'epoch': 15,
            'parameters': trained['model_metadata']['parameters'],
            'val': val['metrics'], 'test': evaluation['metrics'], 'csv': str(csv),
            'csv_sha256': receipt['sha256'], 'checkpoint': str(OUT / 'base15/final.pt'),
            'checkpoint_sha256': trained['final_checkpoint_sha256'],
            'test_used_for_selection': False, 'finished_at_utc': now()}
        write_json(OUT / 'results.json', result)
        write_json(ROOT / 'docs/results/mask2former_swinl_rgb.json', result)
        state.update(status='completed', finished_at_utc=now(), results=str(OUT / 'results.json'))
        plan['material_passport']['verification_status'] = 'COMPLETED'
        write_json(OUT / 'plan.json', plan)
        write_json(OUT / 'queue.json', state)
    except BaseException as error:
        state.update(status='failed', error=str(error), stopped_at_utc=now())
        write_json(OUT / 'queue.json', state)
        raise


if __name__ == '__main__':
    main()
