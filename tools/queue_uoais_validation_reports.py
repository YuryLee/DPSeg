#!/usr/bin/env python3
"""Publish audited base-model validation results while the original runs continue.

This CPU process reads completed validation artifacts. It never loads a model,
changes a training run, selects a final test method, or declares the goal done.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

from audit_uoais_validation import METRICS
from plot_uoais_validation_history import MODELS, load_verified_history

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPORT = ROOT / 'reports/dafusion_benchmark_comparison'
SCOPE = 'full_validation_history_not_final_real_test'
LABELS = {'dafusion': 'DA-Fusion public reimplementation', 'dpseg': 'DPSeg RGB-D base'}
INPUTS = {'dafusion': [480, 640], 'dpseg': [1024, 1024]}


def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def read(path):
    return json.loads(path.read_text())


def json_text(value):
    return json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n'


def publish(path, content):
    if path.exists() and path.read_text() == content:
        return False
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(content)
    temporary.replace(path)
    return True


def csv_text(rows, fields):
    buffer = io.StringIO(newline='')
    writer = csv.DictWriter(buffer, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def summary_signature(report):
    return [(family, epoch, digest(path))
            for family, (directory, *_rest) in MODELS.items()
            for epoch in range(1, 21)
            for path in [report / directory / f'validation_epoch{epoch:02d}.json']
            if path.exists()]


def verify_protocol(report):
    path = report / 'uoais_training_protocol_frozen.json'
    protocol, checksum = read(path), digest(path)
    if protocol['status'] != 'frozen' or protocol['training']['epochs'] != 20:
        raise ValueError('Expected the frozen 20-epoch base protocol')
    for name, expected in protocol['source_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError(f'Frozen training source changed: {name}')
    for family, (directory, *_rest) in MODELS.items():
        base = report / directory
        run = read(base / 'run.json')
        if (run['family'] != family or run['scope'] != 'full_train_val'
                or digest(base / 'protocol.json') != checksum
                or run['protocol_sha256'] != checksum
                or run['source_sha256'] != protocol['source_sha256']):
            raise ValueError(f'Unidentified base run: {family}')
        if list(base.glob('failure_rank*.json')):
            raise ValueError(f'Training failure recorded for {family}; review the original run')
    return protocol, checksum


def verified_row(report, family, epoch, protocol, checksum, audit_new):
    directory, backbone, *_rest = MODELS[family]
    base = report / directory
    summary_path = base / f'validation_epoch{epoch:02d}.json'
    receipt_path = report / f'uoais_{family}_epoch{epoch:02d}_validation_audit.json'
    summary = read(summary_path)
    created = not receipt_path.exists()
    if created:
        subprocess.run([sys.executable, str(ROOT / 'tools/audit_uoais_validation.py'),
                        '--run', str(base), '--epoch', str(epoch),
                        '--output', str(receipt_path)], check=True)
    audit = read(receipt_path)
    if (summary['epoch'] != epoch or summary['scope'] != 'full_validation'
            or summary['images'] != 5000 or summary['test_used'] is not False
            or summary['score_threshold'] != .5
            or summary['overlap_rule'] != 'ascending confidence; later overwrites'
            or set(summary['metrics']) != set(METRICS)
            or audit['status'] != 'passed' or audit['family'] != family
            or audit['images'] != 5000 or audit['expected_unique_images'] != 5000
            or audit['test_used'] is not False
            or any(audit[key] != 0 for key in ['duplicates', 'missing', 'extra'])
            or audit['summary_sha256'] != digest(summary_path)
            or audit['annotation_sha256'] != protocol['annotations']['val']['sha256']
            or not 0 <= audit['maximum_summary_difference'] < 2e-12):
        raise ValueError(f'Invalid validation receipt: {family}, epoch {epoch}')
    # The original epoch-1 receipts predate the epoch/protocol fields.
    if epoch > 1 and (audit['epoch'] != epoch or audit['protocol_sha256'] != checksum):
        raise ValueError('Receipt epoch or protocol differs')
    if audit_new and not created:
        # Recover safely if a prior publication stopped after writing its audit.
        if audit.get('audit_source_sha256') != digest(ROOT / 'tools/audit_uoais_validation.py'):
            raise ValueError('Unpublished receipt uses an unidentified audit implementation')
        for shard in audit['files']:
            if digest(Path(shard['path'])) != shard['sha256']:
                raise ValueError('Unpublished audited predictions changed')
    metrics = audit['recomputed_metrics']
    for key in METRICS:
        value = metrics[key]
        if (not math.isfinite(value) or not 0 <= value <= 1
                or abs(value - summary['metrics'][key]) >= 2e-12):
            raise ValueError(f'Invalid metric {key}')
    return {'family': family, 'backbone': backbone, 'epoch': epoch,
            'train_image_presentations': epoch * 45000, 'validation_images': 5000,
            **metrics, 'validation_seconds': summary['elapsed_seconds'],
            'independently_audited': True, 'best_so_far_by_overlap_f': False,
            'summary_sha256': audit['summary_sha256'], 'summary_path': str(summary_path)}


def comparison(report, rows, epoch, checksum):
    models = {}
    for family in MODELS:
        row = next(v for v in rows if v['family'] == family and v['epoch'] == epoch)
        models[family] = {'label': LABELS[family], 'backbone': row['backbone'],
                          'input_size': INPUTS[family],
                          'metrics': {key: row[key] for key in METRICS},
                          'integrity_receipt': f'uoais_{family}_epoch{epoch:02d}_validation_audit.json',
                          'summary_path': row['summary_path'], 'summary_sha256': row['summary_sha256']}
    path = report / f'uoais_epoch{epoch:02d}_validation_comparison.json'
    if path.exists():
        existing = read(path)
        if (existing['epoch'] != epoch or existing['images_per_model'] != 5000
                or existing['protocol_sha256'] != checksum
                or existing['test_used'] is not False or existing['protocol_identical'] is not True):
            raise ValueError('Existing paired comparison identifies a different experiment')
        for family in models:
            for key in METRICS:
                if abs(existing['models'][family]['metrics'][key] - models[family]['metrics'][key]) >= 2e-12:
                    raise ValueError('Existing paired comparison disagrees with its receipts')
    else:
        value = {'scope': f'UOAIS-Sim epoch{epoch:02d} full validation; final real-test comparison remains pending',
                 'created_at': now(), 'dataset': 'UOAIS-Sim val', 'epoch': epoch,
                 'images_per_model': 5000, 'train_image_presentations_per_model': epoch * 45000,
                 'protocol_sha256': checksum, 'protocol_identical': True,
                 'units': 'fractions; percentage-point deltas multiply differences by 100',
                 'test_used': False, 'models': models,
                 'dpseg_minus_dafusion_percentage_points': {
                     key: 100 * (models['dpseg']['metrics'][key] - models['dafusion']['metrics'][key])
                     for key in METRICS},
                 'interpretation': 'Base system comparison: backbone and input size differ. '
                     'Both models have completed the same number of passes through 45,000 images '
                     'and validation on the same 5,000 images. Point training, system validation '
                     'selection and final OSD/OCID/OCBD comparison remain separate requirements.'}
        publish(path, json_text(value))
    output_rows = [{'family': family, 'backbone': m['backbone'], 'epoch': epoch,
                    'train_image_presentations': epoch * 45000, 'validation_images': 5000,
                    **m['metrics']} for family, m in models.items()]
    csv_path = path.with_suffix('.csv')
    if csv_path.exists():
        saved = list(csv.DictReader(csv_path.open()))
        if len(saved) != len(output_rows):
            raise ValueError('Existing comparison CSV has missing or extra rows')
        for row, expected in zip(saved, output_rows):
            if (row['backbone'] != expected['backbone'] or int(row['epoch']) != epoch
                    or int(row.get('validation_images', row.get('images', '0'))) != 5000
                    or any(abs(float(row[key]) - expected[key]) >= 2e-12 for key in METRICS)):
                raise ValueError('Existing comparison CSV disagrees with its receipts')
    else:
        publish(csv_path, csv_text(output_rows, list(output_rows[0])))
    return models


def update_reports(report):
    protocol, checksum = verify_protocol(report)
    # Validate existing records before deriving or extending any published file.
    verified, _sources, existing_checksum = load_verified_history(report)
    history = read(report / 'uoais_validation_history.json')
    if existing_checksum != checksum or history['scope'] != SCOPE:
        raise ValueError('History identifies a different protocol')
    old_pairs = {(row['family'], row['epoch']) for group in verified.values() for row in group}
    rows, counts = [], {}
    for family, (directory, *_rest) in MODELS.items():
        epochs = [epoch for epoch in range(1, 21)
                  if (report / directory / f'validation_epoch{epoch:02d}.json').exists()]
        if epochs != list(range(1, len(epochs) + 1)):
            raise ValueError('Completed validation summaries have an epoch gap')
        if len(epochs) < len(verified[family]):
            raise ValueError('A previously published validation summary disappeared')
        counts[family] = len(epochs)
        group = [verified_row(report, family, epoch, protocol, checksum,
                              (family, epoch) not in old_pairs) for epoch in epochs]
        best = max(group, key=lambda row: (row['Objects F-measure'], -row['epoch']))
        for row in group:
            row['best_so_far_by_overlap_f'] = row is best
        rows.extend(group)
    changed = rows != history['rows']
    if changed:
        history['rows'] = rows
        history['updated_utc'] = now()
        publish(report / 'uoais_validation_history.json', json_text(history))
    publish(report / 'uoais_validation_history.csv', csv_text(rows, list(rows[0])))
    # Re-read the exact files that downstream readers will use.
    groups, _sources, _checksum = load_verified_history(report)
    saved = list(csv.DictReader((report / 'uoais_validation_history.csv').open()))
    if len(saved) != len(rows) or any(float(a[key]) != b[key]
            for a, b in zip(saved, rows) for key in METRICS):
        raise ValueError('Published history CSV and JSON disagree')
    shared = min(counts.values())
    for epoch in range(1, shared + 1):
        models = comparison(report, rows, epoch, checksum)
    figure_path = report / 'figures/uoais_validation_history_data.json'
    figure_valid = False
    if figure_path.exists():
        figure = read(figure_path)
        figure_valid = (figure['rows'] == groups and figure['protocol_sha256'] == checksum
                        and figure['latest_shared_completed_epoch'] == shared
                        and all(digest(report / 'figures' / name) == value['sha256']
                                for name, value in figure['artifacts'].items()))
    if not figure_valid:
        subprocess.run([sys.executable, str(ROOT / 'tools/plot_uoais_validation_history.py'),
                        '--reports', str(report)], check=True)
    text = ['# UOAIS-Sim 基础模型验证对比', '',
            '范围：每轮完整 5,000 张合成验证图，七项指标经逐图核对；数值单位为 %。', '',
            f'最新共同完成第 **{shared} 轮**，两组各累计 {shared * 45000:,} 次训练图片呈现。', '',
            '| 模型 | Overlap P | Overlap R | Overlap F | Boundary P | Boundary R | Boundary F | %75 |',
            '|---|---:|---:|---:|---:|---:|---:|---:|']
    for family, model in models.items():
        text.append('| ' + LABELS[family] + ' (' + model['backbone'] + ') | '
                    + ' | '.join(f'{model["metrics"][key] * 100:.2f}' for key in METRICS) + ' |')
    text.extend(['', '两组骨干与输入尺寸不同，属于系统对比。%75 使用匹配 F > 0.75，区别于 COCO AP75。', '',
                 f'完整验证轮次：DA-Fusion {counts["dafusion"]}/20，DPSeg {counts["dpseg"]}/20。', '',
                 '这些验证结果不构成 OSD、OCID、OCBD 的最终测试结论。基础训练完成状态以原运行记录为准，'
                 '点头训练、四级方法验证选择及真实测试集最终比较仍需分别完成。', '',
                 f'[本轮 CSV](uoais_epoch{shared:02d}_validation_comparison.csv) · '
                 '[逐轮 CSV](uoais_validation_history.csv) · [验证曲线](figures/uoais_validation_history.png)', ''])
    publish(report / 'LATEST_VALIDATION.md', '\n'.join(text))
    return {'counts': counts, 'audited_rows': len(rows), 'latest_shared_epoch': shared,
            'best_epochs': {family: next(row['epoch'] for row in group if row['best_so_far_by_overlap_f'])
                            for family, group in groups.items()}, 'history_changed': changed}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=DEFAULT_REPORT)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=60)
    args = parser.parse_args()
    if args.poll_seconds < 10:
        parser.error('Poll no faster than every ten seconds')
    report = args.reports.resolve()
    state_dir = (args.state_dir or report / 'validation_reporting_v1').resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    with (report / '.validation_report_publisher.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit('A validation publisher already owns this state directory')
        sources = {str(path.relative_to(ROOT)): digest(path) for path in [Path(__file__).resolve(),
                   ROOT / 'tools/audit_uoais_validation.py', ROOT / 'tools/plot_uoais_validation_history.py']}
        plan = {'scope': 'validation_reporting_only_not_final_test', 'reports': str(report),
                'source_sha256': sources, 'training_sources_modified': False}
        plan_path = state_dir / 'plan.json'
        if plan_path.exists() and read(plan_path) != plan:
            raise ValueError('Reporting plan changed; use a reviewed new state directory')
        publish(plan_path, json_text(plan))
        state = {'pid': os.getpid(), 'started_utc': now(), 'scope': plan['scope']}
        previous, result = None, None
        try:
            while True:
                if any(digest(ROOT / name) != expected for name, expected in sources.items()):
                    raise ValueError('Reporting implementation changed while running')
                for directory, *_rest in MODELS.values():
                    if list((report / directory).glob('failure_rank*.json')):
                        raise ValueError(f'Training failure recorded in {directory}')
                signature = summary_signature(report)
                if signature != previous:
                    result = update_reports(report)
                    previous = signature
                    state['last_publication_utc'] = now()
                    print(json.dumps({'event': 'validation_reports_verified', **result,
                                      'updated_utc': state['last_publication_utc']}), flush=True)
                complete = all(value == 20 for value in result['counts'].values())
                state.update(status='validation_reporting_completed' if complete else
                             ('waiting_for_validation' if args.watch else 'verified_once'),
                             updated_utc=now(), result=result)
                publish(state_dir / 'state.json', json_text(state))
                if complete or not args.watch:
                    return
                time.sleep(args.poll_seconds)
        except BaseException:
            state.update(status='failed', updated_utc=now(), traceback=traceback.format_exc())
            publish(state_dir / 'state.json', json_text(state))
            raise


if __name__ == '__main__':
    main()
