#!/usr/bin/env python3
"""Sequential, monitored course experiments. A failed stage stops the queue."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT/'reports/course_depth_backbone_v1'
ARMS = ('swint_depth_no_k', 'swinl_xyz')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def command(script, *arguments):
    return [sys.executable, '-m', 'torch.distributed.run', '--standalone', '--nproc_per_node=2',
            str(ROOT/'tools'/script), *map(str, arguments)]


def completed_base(folder, protocol):
    run = read(folder/'run.json')
    if (run['status'] != 'completed' or run['scope'] != 'full_train_val'
            or run['optimizer_steps'] != 438*15 or run['training_image_exposures'] != 14432*15
            or run['source_sha256'] != protocol['source_sha256']
            or run['final_checkpoint_sha256'] != sha(folder/'final.pt')):
        raise ValueError('Formal base training is incomplete or its identity differs')
    selection = read(folder/'selection.json')
    if [e['epoch'] for e in selection['epochs']] != list(range(1, 16)):
        raise ValueError('Base full-validation epoch coverage differs')
    for epoch in selection['epochs']:
        evaluation = read(Path(epoch['output'])/'evaluation.json')
        if evaluation['scope'] != 'full_split' or evaluation['full_split_images'] != 2584:
            raise ValueError('Base validation used an incomplete split')
    return run


def system_protocol(base_protocol, folder, *, scope='full_experiment'):
    from tools.run_course_depth_backbone_system import sources
    value = dict(base_protocol)
    value.update(scope=scope, base_checkpoint_sha256=sha(folder/'final.pt'),
                 system_source_sha256=sources())
    return value


def verify_engineering(folder, expected_source):
    gate = read(folder/'engineering_gate.json')
    if (gate['status'] != 'passed' or gate['source_sha256'] != expected_source
            or gate['system_source_sha256'] != read(folder/'protocol_system_engineering.json')['system_source_sha256']):
        raise ValueError('Engineering gate is missing, failed or stale')
    for path, digest in gate['evidence_sha256'].items():
        if sha(path) != digest:
            raise ValueError('Engineering evidence changed: '+path)
    return gate


def check_results(folder):
    result = read(folder/'comparison.json')
    for split, count in (('val', 2584), ('test', 2112)):
        for variant in ('native', 'full'):
            evaluation = result[split][variant]
            if evaluation['scope'] != 'full_split' or evaluation['full_split_images'] != count:
                raise ValueError('Incomplete final split evaluation')
            for key in ('mask_ap', 'mask_ap_small'):
                if (not isinstance(evaluation['metrics'][key], (int, float))
                        or not math.isfinite(evaluation['metrics'][key]) or not 0 <= evaluation['metrics'][key] <= 1):
                    raise ValueError('Missing/nonfinite formal metric')
            receipt = read(folder/f'{split}_{variant}'/'evaluation_run.json')
            if receipt['status'] != 'completed' or receipt['images'] != count or receipt['scope'] != 'full_split':
                raise ValueError('Incomplete final prediction receipt')
            if split == 'test':
                validation = read(folder/f'{split}_{variant}'/'submission_validation.json')
                if (validation['rows'] != count or not validation['id_order_verified']
                        or not validation['rle_native_decode_verified']):
                    raise ValueError('Final CSV is not fully verified')
    for split in ('val', 'test'):
        for variant in ('native', 'full'):
            path = folder/f'{split}_{variant}'
            receipt = read(path/'evaluation_run.json')
            if (receipt['predictions_sha256'] != sha(path/'predictions.jsonl')
                    or receipt['evaluation_sha256'] != sha(path/'evaluation.json')
                    or read(path/'evaluation.json') != result[split][variant]):
                raise ValueError('Final prediction/evaluation evidence changed')
            if split == 'test' and read(path/'submission_validation.json')['sha256'] != sha(path/'submission.csv'):
                raise ValueError('Final CSV differs from its decode receipt')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=DEFAULT)
    args = parser.parse_args()
    root = args.root.resolve()
    lock = (root/'queue.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root/'queue.json').exists():
        raise FileExistsError('An existing queue requires explicit review before resuming; no automatic retries')
    state = {'status': 'preparing', 'pid': os.getpid(), 'started_at_utc': now(),
             'objective': read(root/'plan.json')['user_objective'], 'stages': [],
             'queue_script_sha256': sha(__file__)}
    write(root/'queue.json', state)
    environment = dict(os.environ)
    environment.update(CUDA_VISIBLE_DEVICES='0,1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4',
        OPENBLAS_NUM_THREADS='4', TOKENIZERS_PARALLELISM='false', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    results = {}

    def execute(name, argv, output):
        record = {'name': name, 'command': argv, 'status': 'starting', 'started_at_utc': now(),
                  'output': str(output), 'hard_timeout_seconds': 605000}
        state['stages'].append(record)
        state.update(status='running', current_stage=name)
        write(root/'queue.json', state)
        log_path = root/(name+'.log')
        with log_path.open('x') as log:
            child = subprocess.Popen(argv, cwd=ROOT, env=environment, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
            began = time.monotonic()
            record.update(pid=child.pid, status='running', log=str(log_path))
            write(root/'queue.json', state)
            while child.poll() is None:
                elapsed = time.monotonic()-began
                if elapsed > record['hard_timeout_seconds']:
                    record['timeout_at_utc'] = now()
                    write(root/'queue.json', state)
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    raise TimeoutError('Declared stage hard timeout: '+name)
                progress = sorted(output.glob('progress_rank*.json')) if output.exists() else []
                record.update(last_observed_alive_at_utc=now(), elapsed_seconds=elapsed,
                    output_files={str(p): {'size': p.stat().st_size, 'mtime_ns': p.stat().st_mtime_ns} for p in progress},
                    log_bytes=log_path.stat().st_size)
                write(root/'queue.json', state)
                time.sleep(30)
            record.update(exit_code=child.returncode, finished_at_utc=now(),
                          elapsed_seconds=time.monotonic()-began,
                          status='completed' if child.returncode == 0 else 'failed')
            write(root/'queue.json', state)
            if child.returncode != 0:
                raise RuntimeError(f'{name} exited with {child.returncode}; queue stopped without retry')
        receipt = read(output/'run.json')
        if receipt['status'] != 'completed':
            raise ValueError('Process exit0 without completed run receipt: '+name)
        record.update(run_receipt_sha256=sha(output/'run.json'))
        write(root/'queue.json', state)

    try:
        sys.path.insert(0, str(ROOT))
        from tools.run_course_depth_backbone import sources as base_sources
        from tools.run_course_depth_backbone_system import sources as system_sources
        frozen_base_sources = base_sources()
        frozen_system_sources = system_sources()
        # Both engineering gates must exist before spending either formal budget.
        protocols = {}
        for arm in ARMS:
            folder = root/arm
            protocol = read(folder/'protocol.json')
            if (protocol['status'] != 'frozen' or protocol['source_sha256'] != frozen_base_sources
                    or protocol['frozen_system_source_sha256'] != frozen_system_sources):
                raise ValueError('Formal protocol source binding differs')
            gate = verify_engineering(folder, frozen_base_sources)
            if (protocol['engineering_validation']['sha256'] != sha(folder/'engineering_gate.json')
                    or gate['system_source_sha256'] != frozen_system_sources):
                raise ValueError('Formal engineering identity differs')
            protocols[arm] = protocol
        for arm in ARMS:
            folder, protocol = root/arm, protocols[arm]
            formal = folder/'base15'
            execute(arm+'_base15', command('run_course_depth_backbone.py',
                '--protocol', folder/'protocol.json', '--output', formal, '--resolution', 1024), formal)
            completed_base(formal, protocol)
            system = system_protocol(protocol, formal)
            if system['system_source_sha256'] != frozen_system_sources:
                raise ValueError('System code changed after protocol freeze')
            system['official_test_metadata'] = str(root/'test_images.json')
            system['head_selection'] = str(folder/'points6'/'selection.json')
            write(folder/'protocol_system.json', system)
            points = folder/'points6'
            execute(arm+'_points6', command('run_course_depth_backbone_system.py',
                '--protocol', folder/'protocol_system.json', '--base', formal/'final.pt',
                '--output', points, '--mode', 'points'), points)
            selected = read(points/'selection.json')
            if (selected['status'] != 'locked' or selected['epochs'] != 6
                    or selected['training_images'] != 14432 or selected['global_steps'] != 1804*6
                    or selected['test_used'] or not selected['base_unchanged'] or not selected['base_gradients_absent']):
                raise ValueError('Point training/selection is incomplete')
            evaluation = folder/'final_evaluation'
            execute(arm+'_final_evaluation', command('run_course_depth_backbone_system.py',
                '--protocol', folder/'protocol_system.json', '--base', formal/'final.pt',
                '--heads', selected['selected']['checkpoint'], '--output', evaluation,
                '--mode', 'evaluate'), evaluation)
            results[arm] = check_results(evaluation)
            write(root/'results_partial.json', results)
        submissions = root/'submissions'
        submissions.mkdir(exist_ok=False)
        table = []
        for arm, splits in results.items():
            for variant in ('native', 'full'):
                source = root/arm/'final_evaluation'/f'test_{variant}'/'submission.csv'
                target = submissions/f'dpseg_{arm}_base15_{variant}.csv'
                os.link(source, target)
                metrics = splits['test'][variant]['metrics']
                table.append({'arm': arm, 'variant': variant, 'test_ap': metrics['mask_ap'],
                    'test_aps': metrics['mask_ap_small'], 'validation_metrics': splits['val'][variant]['metrics'],
                    'submission': str(target), 'submission_sha256': sha(target)})
        report = {'status': 'completed', 'scope': 'two_full_course_training_and_test_experiments',
            'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent',
                'mode': 'run', 'verification_status': 'FORMAL_COMPLETE'},
            'rows': table, 'test_used_for_selection': False, 'finished_at_utc': now(),
            'plan_sha256': sha(root/'plan.json'), 'limitations': read(root/'plan.json')['limitations']}
        write(root/'comparison.json', report)
        lines = ['# 深度表示与Swin-Large对比结果', '', '## Material Passport', '',
                 '- Origin Skill: academic-research-suite / experiment-agent', '- Origin Mode: run',
                 '- Verification Status: FORMAL_COMPLETE', '',
                 '| 配置 | 推理流程 | 本地测试 AP | 本地测试 APₛ |', '|---|---|---:|---:|']
        for row in table:
            lines.append(f"| {row['arm']} | {row['variant']} | {row['test_ap']:.6f} | {row['test_aps']:.6f} |")
        lines += ['', '主对比固定15轮基座；配套点头训练6轮后只按完整验证集选择。所有测试CSV完整解码校验，未上传Kaggle。', '',
                  '单种子实验；Swin-L与Swin-T的公开预训练和窗口尺寸不同。无内参组同时改变深度表示，不能把差异单独归因于删除一个参数。']
        (root/'comparison.md').write_text('\n'.join(lines)+'\n')
        state.update(status='completed', finished_at_utc=now(), comparison_sha256=sha(root/'comparison.json'))
        write(root/'queue.json', state)
    except BaseException as error:
        state.update(status='failed', failed_at_utc=now(), error_type=type(error).__name__, error=str(error))
        write(root/'queue.json', state)
        raise


if __name__ == '__main__':
    main()
