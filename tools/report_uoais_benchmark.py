#!/usr/bin/env python3
"""Verify and report the full DA-Fusion/DPSeg original-benchmark comparison.

--check is read-only and uses the standard library. Report generation requires
both 20-epoch bases, five point epochs, full system validation and all three
original test sets. It recomputes test scores from saved masks and original GT.
This consumer never changes training, inference, selection or existing reports.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = ('Objects Precision', 'Objects Recall', 'Objects F-measure',
           'Boundary Precision', 'Boundary Recall', 'Boundary F-measure',
           'obj_detected_075_percentage')
STAGES = ('rgbd_fusion', 'depth_point', 'depth_point_tiled', 'depth_point_tiled_probability')
COUNTS = {'osd': 111, 'ocid': 2390, 'ocbd': 1000}
BASES = {'dafusion': 'uoais_dafusion_swinl_20ep', 'dpseg': 'uoais_dpseg_swint_20ep'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def completed_budget(run, epochs, step):
    """A file or a completed label alone is insufficient evidence."""
    state = run.get('state', {})
    history = state.get('validation', [])
    return (run.get('status') == 'training_completed' and run.get('scope') == 'full_train_val'
            and state.get('epoch') == epochs
            and state.get('global_step', state.get('step')) == step
            and state.get('batch_in_epoch') == 0 and len(history) == epochs
            and all(v.get('epoch') == i and v.get('scope') == 'full_validation'
                    and v.get('images') == 5000 and v.get('test_used') is False
                    for i, v in enumerate(history, 1)))


class Evidence:
    def __init__(self):
        self.files = {}

    def sha(self, path):
        path = Path(path).resolve()
        if str(path) not in self.files:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    digest.update(block)
            self.files[str(path)] = digest.hexdigest()
        return self.files[str(path)]

    def read(self, path):
        self.sha(path)
        return json.loads(Path(path).read_text())

    def sources(self, hashes):
        require(bool(hashes), 'Missing source hashes')
        for name, expected in hashes.items():
            require(self.sha(ROOT / name) == expected, f'Changed source: {name}')


def required_paths(report, manifests):
    paths = [report / 'uoais_validation_history.json', report / 'paper_protocol.json']
    for directory in BASES.values():
        paths += [report / directory / name for name in ('run.json', 'protocol.json', 'best.pt')]
    paths += [report / 'uoais_dpseg_point_5ep' / name
              for name in ('run.json', 'protocol.json', 'best_point.pt', 'best_point_depth.pt')]
    paths += [report / 'uoais_dpseg_system_val_full_v1' / name
              for name in ('protocol.json', 'summary.json', 'selection.json',
                           'predictions_rank0.jsonl', 'predictions_rank1.jsonl')]
    for split in COUNTS:
        paths.append(manifests / f'{split}_manifest.json')
        paths += [report / f'uoais_dafusion_{split}_final_v1' / name for name in
                  ('protocol.json', 'completion.json', 'predictions.jsonl',
                   'scores/protocol.json', 'scores/evaluation.json', 'scores/per_image.jsonl')]
        paths += [report / f'uoais_dpseg_system_{split}_final_v1' / name for name in
                  ('protocol.json', 'summary.json', 'predictions_rank0.jsonl', 'predictions_rank1.jsonl')]
    return paths


def readiness(report, manifests):
    missing = [str(path) for path in required_paths(report, manifests) if not path.is_file()]
    incomplete = []
    for name, directory, epochs, steps in (
            ('dafusion', BASES['dafusion'], 20, 112500),
            ('dpseg', BASES['dpseg'], 20, 112500),
            ('point', 'uoais_dpseg_point_5ep', 5, 56250)):
        path = report / directory / 'run.json'
        if path.is_file() and not completed_budget(json.loads(path.read_text()), epochs, steps):
            incomplete.append(name)
    return {'scope': 'readiness_only_not_accuracy_or_goal_completion',
            'required_benchmarks': COUNTS, 'missing_files': missing,
            'incomplete_training': incomplete, 'ready_for_verification': not missing and not incomplete}


def records(paths, expected_ids, evidence):
    seen = set()
    for path in paths:
        evidence.sha(path)
        with Path(path).open() as stream:
            for line in stream:
                row = json.loads(line)
                key = row['image_id']
                require(key in expected_ids and key not in seen, f'Extra or duplicate image: {key}')
                seen.add(key)
                yield row
    require(seen == expected_ids, 'Missing images in saved predictions or scores')


def same_number(actual, expected):
    if actual is None or expected is None:
        return actual is expected
    return (math.isfinite(actual) and math.isfinite(expected)
            and math.isclose(actual, expected, abs_tol=2e-12, rel_tol=0))


def mean_metrics(rows):
    require(bool(rows), 'Cannot report an empty benchmark')
    require(all(row[key] is None or (isinstance(row[key], (int, float))
                and not isinstance(row[key], bool) and math.isfinite(row[key])
                and 0 <= row[key] <= 1) for row in rows for key in METRICS),
            'Invalid per-image metric')
    return {key: (None if any(row[key] is None for row in rows)
                  else math.fsum(row[key] for row in rows) / len(rows)) for key in METRICS}


def compare_metrics(actual, expected):
    for key in METRICS:
        require(same_number(actual[key], expected[key]), f'Metric mismatch: {key}')


def count_empty(counts, metrics):
    empty_gt, empty_prediction = metrics['obj_gt'] == 0, metrics['obj_detected'] == 0
    for key, value in [('empty_gt_images', empty_gt), ('empty_prediction_images', empty_prediction),
                       ('both_empty_images', empty_gt and empty_prediction)]:
        counts[key] = counts.get(key, 0) + int(value)


def checkpoint_metadata(path):
    """Read local checkpoint metadata on CPU without materializing its storages."""
    import torch
    saved = torch.load(path, map_location='cpu', mmap=True, weights_only=False)
    fields = ('scope', 'family', 'selection', 'protocol_sha256', 'source_sha256',
              'base_checkpoint_sha256', 'selected_arm')
    return {key: saved.get(key) for key in fields}


def verify_checkpoint_selection(saved, history, protocol_hash, sources, *, family=None, arm=None):
    require(saved['scope'] == 'full_train_val' and saved['protocol_sha256'] == protocol_hash
            and saved['source_sha256'] == sources, 'Checkpoint training provenance differs')
    require(saved.get('family') == family if family else saved.get('selected_arm') == arm,
            'Wrong checkpoint family or point arm')
    def values(row):
        metrics = row.get('metrics', row)
        return metrics[arm] if arm else metrics
    def key(row):
        value = values(row)['Objects F-measure']
        require(value is not None and math.isfinite(value), 'Undefined checkpoint selection metric')
        return value, -row['epoch']
    require(bool(history), 'Missing checkpoint validation history')
    winner = max(history, key=key)
    selected = saved['selection']
    require(selected['scope'] == 'full_validation' and selected['images'] == 5000
            and selected['test_used'] is False and selected['epoch'] == winner['epoch'],
            'Checkpoint is not the validation winner with earliest-epoch tie breaking')
    compare_metrics(values(selected), values(winner))
    return winner['epoch']


def selected_stage(summary):
    require(summary['scope'] == 'full_validation' and summary['images'] == 5000,
            'Method selection requires all 5000 validation images')
    def key(stage):
        overlap = summary['metrics'][stage]['Objects F-measure']
        boundary = summary['metrics'][stage]['Boundary F-measure']
        require(overlap is not None and math.isfinite(overlap), 'Undefined validation Overlap F')
        return overlap, (-1 if boundary is None else boundary), -STAGES.index(stage)
    return max(STAGES, key=key)


def verify_point_checkpoints(point, run, base_checkpoint_hash, evidence):
    """Both paired controls must belong to this base and win their own history."""
    results = {}
    for arm in ('point', 'point_depth'):
        path = point / f'best_{arm}.pt'
        saved = checkpoint_metadata(path)
        require(saved['base_checkpoint_sha256'] == base_checkpoint_hash,
                f'{arm} weights belong to another base checkpoint')
        epoch = verify_checkpoint_selection(saved, run['state']['validation'], run['protocol_sha256'],
                                            run['source_sha256'], arm=arm)
        results[arm] = {'selected_epoch': epoch, 'checkpoint_sha256': evidence.sha(path)}
    return results


def audit_system(folder, ids, evidence, dataset=None):
    """Validate coverage/means; real test masks are rescored on original GT."""
    from rgbd_seg.uois_benchmark_metrics import evaluate_label_maps, predictions_to_label_map
    from tools.score_uois_benchmark import json_safe
    summary = evidence.read(folder / 'summary.json')
    require(summary['status'] == 'completed' and summary['images'] == len(ids), 'Incomplete system run')
    values = {stage: [] for stage in STAGES}
    index = {record['id']: i for i, record in enumerate(dataset.records)} if dataset else {}
    diagnostics = {stage: [] for stage in STAGES}
    empty_cases = {stage: {} for stage in STAGES}
    for row in records([folder / f'predictions_rank{i}.jsonl' for i in (0, 1)], ids, evidence):
        require(set(row['stages']) == set(STAGES), 'Missing system stage')
        truth = dataset.ground_truth(index[row['image_id']]) if dataset else None
        for stage in STAGES:
            saved = row['stages'][stage]
            if truth is not None:
                labels = predictions_to_label_map(saved['predictions'], truth.shape, score_threshold=.5)
                result = json_safe(evaluate_label_maps(labels, truth))
                compare_metrics(saved['metrics'], result['metrics'])
                require(saved['diagnostics'] == result['diagnostics'], 'Changed endpoint diagnostics')
            else:
                result = saved
            values[stage].append({key: result['metrics'][key] for key in METRICS})
            diagnostics[stage].append(result['diagnostics']['obj_detected_ge075_percentage'])
            count_empty(empty_cases[stage], result['metrics'])
    means = {stage: mean_metrics(value) for stage, value in values.items()}
    for stage in STAGES:
        compare_metrics(means[stage], summary['metrics'][stage])
    return means, {stage: {'mean_inclusive_f_ge_075': math.fsum(value) / len(value),
                           **empty_cases[stage]} for stage, value in diagnostics.items()}


def verify_training(report, evidence):
    from tools.plot_uoais_validation_history import load_verified_history
    from rgbd_seg.uoais_point_validation_audit import audit_point_epoch, load_point_validation_cohort
    history, history_sources, protocol_hash = load_verified_history(report)
    require(all(len(value) == 20 for value in history.values()), 'Forty audited base validations required')
    for source in history_sources:
        require(evidence.sha(source['path']) == source['sha256'], 'Validation evidence changed')
    protocol, bases = None, {}
    for family, directory in BASES.items():
        folder = report / directory
        run, current = evidence.read(folder / 'run.json'), evidence.read(folder / 'protocol.json')
        require(completed_budget(run, 20, 112500) and run['family'] == family, 'Incomplete/wrong base')
        require(evidence.sha(folder / 'protocol.json') == run['protocol_sha256'] == protocol_hash,
                'Base protocols differ')
        require(current['status'] == 'frozen' and current['training']['epochs'] == 20,
                'Expected frozen twenty-epoch protocol')
        evidence.sources(run['source_sha256'])
        best_epoch = verify_checkpoint_selection(checkpoint_metadata(folder / 'best.pt'), history[family],
                                                protocol_hash, run['source_sha256'], family=family)
        bases[family] = {'run_sha256': evidence.sha(folder / 'run.json'),
                         'checkpoint_sha256': evidence.sha(folder / 'best.pt'), 'selected_epoch': best_epoch}
        protocol = current
    point = report / 'uoais_dpseg_point_5ep'
    run, head = evidence.read(point / 'run.json'), evidence.read(point / 'protocol.json')
    require(completed_budget(run, 5, 56250), 'Incomplete five-epoch point training')
    require(run['base_unchanged'] and run['base_gradients_absent'], 'Point training changed the base')
    require(hashlib.sha256(json.dumps(head, sort_keys=True).encode()).hexdigest() == run['protocol_sha256'],
            'Point protocol changed')
    require(head['scope'] == 'full_train_val' and head['base']['scope'] == 'full_train_val'
            and head['base']['checkpoint_sha256'] == bases['dpseg']['checkpoint_sha256']
            and head['base']['protocol_sha256'] == protocol_hash
            and head['source_sha256'] == run['source_sha256']
            and head['test_used'] is False and head['training']['epochs'] == 5
            and head['training']['global_batch'] == 4 and head['training']['world_size'] == 2,
            'Wrong point base/budget')
    evidence.sources(run['source_sha256'])
    point_ids, point_cohort = load_point_validation_cohort(report)
    for path, expected in point_cohort['sources_sha256'].items():
        require(evidence.sha(path) == expected, 'Point cohort provenance changed')
    paired_audits = []
    for epoch in range(1, 6):
        folder = point / f'val_epoch{epoch:02d}'
        audit = audit_point_epoch(folder, point_ids, epoch)
        require(evidence.read(folder / 'summary.json') == run['state']['validation'][epoch - 1],
                'Point run history differs from its actual epoch summary')
        require(evidence.sha(folder / 'summary.json') == audit['summary_sha256'], 'Point summary changed')
        for item in audit['files']:
            require(evidence.sha(item['path']) == item['sha256'], 'Point validation predictions changed')
        paired_audits.append(audit)
    checkpoints = verify_point_checkpoints(point, run, bases['dpseg']['checkpoint_sha256'], evidence)
    point_hash = checkpoints['point_depth']['checkpoint_sha256']
    bases['dpseg']['point_selected_epoch'] = checkpoints['point_depth']['selected_epoch']
    bases['dpseg']['point_plain_selected_epoch'] = checkpoints['point']['selected_epoch']
    bases['dpseg']['point_plain_checkpoint_sha256'] = checkpoints['point']['checkpoint_sha256']
    bases['dpseg']['paired_point_validation'] = {
        'scope': 'all_five_paired_validation_epochs_saved_metric_integrity_not_RLE_GT_rescoring',
        'cohort': point_cohort, 'epochs': paired_audits,
        'verifier_sources': {str(path): evidence.sha(path) for path in
                            [ROOT / 'rgbd_seg/uoais_point_validation_audit.py', Path(__file__).resolve()]},
    }
    validation = report / 'uoais_dpseg_system_val_full_v1'
    selection, val_protocol = evidence.read(validation / 'selection.json'), evidence.read(validation / 'protocol.json')
    for value in (selection, val_protocol):
        require(value['scope'] == 'full_validation' and value['images'] == 5000
                and value['base_checkpoint_sha256'] == bases['dpseg']['checkpoint_sha256']
                and value['point_checkpoint_sha256'] == point_hash, 'Wrong system validation provenance')
    require(selection['source_sha256'] == val_protocol['source_sha256'], 'Selection sources changed')
    evidence.sources(selection['source_sha256'])
    val_summary = evidence.read(validation / 'summary.json')
    require(all(val_summary[key] == value for key, value in val_protocol.items()),
            'Validation summary protocol differs')
    ids = set()
    for rank in (0, 1):
        path = report / BASES['dpseg'] / f'val_epoch20_rank{rank}.jsonl'
        evidence.sha(path)
        with path.open() as stream:
            ids.update(json.loads(line)['sample_id'] for line in stream)
    require(len(ids) == 5000, 'Incomplete base validation cohort')
    audit_system(validation, ids, evidence)
    require(selection['stage'] == selected_stage(evidence.read(validation / 'summary.json')),
            'Selected method differs from frozen validation rule')
    return protocol, protocol_hash, bases, point_hash, selection, val_protocol


def assemble(report, manifests):
    from rgbd_seg.uois_benchmark_data import BenchmarkDataset
    from rgbd_seg.uois_benchmark_metrics import evaluate_label_maps, predictions_to_label_map, reference_provenance
    from tools.score_uois_benchmark import json_safe
    require(readiness(report, manifests)['ready_for_verification'], 'Required training/tests/data are incomplete')
    evidence = Evidence()
    protocol, protocol_hash, bases, point_hash, selection, validation = verify_training(report, evidence)
    reference = reference_provenance()
    results = {}
    for split, count in COUNTS.items():
        manifest = manifests / f'{split}_manifest.json'
        data = BenchmarkDataset(manifest)
        require(data.manifest['dataset'] == split and len(data) == count, 'Wrong/incomplete original manifest')
        require(data.manifest['annotation_policy'] == 'original annotation files; documented background only; no object removal, correction or resize',
                'Unexpected annotation policy')
        manifest_hash = evidence.sha(manifest)
        ids = {record['id'] for record in data.records}
        native = report / f'uoais_dafusion_{split}_final_v1'
        pred = evidence.read(native / 'protocol.json')
        complete = evidence.read(native / 'completion.json')
        scoring = evidence.read(native / 'scores/protocol.json')
        score = evidence.read(native / 'scores/evaluation.json')
        require(pred['family'] == 'dafusion' and pred['dataset'] == split and pred['images'] == count
                and pred['training_protocol_sha256'] == protocol_hash
                and pred['training_run_sha256'] == bases['dafusion']['run_sha256']
                and pred['weights_sha256'] == bases['dafusion']['checkpoint_sha256']
                and pred['test_used_for_selection'] is False and pred['fixed_evaluation_threshold'] == .5
                and pred['precision'].startswith('float32'), 'Wrong DA-Fusion inference provenance')
        evidence.sources(pred['source_sha256'])
        require(complete['status'] == score['status'] == 'completed' and complete['images'] == score['images'] == count,
                'Incomplete DA-Fusion test')
        require(pred['manifest_sha256'] == scoring['manifest_sha256'] == manifest_hash
                and complete['predictions_sha256'] == scoring['predictions_sha256'] == evidence.sha(native / 'predictions.jsonl')
                and score['protocol_sha256'] == evidence.sha(native / 'scores/protocol.json')
                and score['per_image_sha256'] == evidence.sha(native / 'scores/per_image.jsonl'), 'Changed DA-Fusion test artifacts')
        require(scoring['reference'] == reference and scoring['score_threshold'] == .5
                and scoring['score_comparison'] == 'strictly greater than', 'Different evaluator or threshold')
        evidence.sources(scoring['sources_sha256'])
        saved = {row['image_id']: row for row in records([native / 'scores/per_image.jsonl'], ids, evidence)}
        index = {record['id']: i for i, record in enumerate(data.records)}
        means, inclusive, empty_cases = [], [], {}
        for row in records([native / 'predictions.jsonl'], ids, evidence):
            truth = data.ground_truth(index[row['image_id']])
            labels = predictions_to_label_map(row['predictions'], truth.shape, score_threshold=.5)
            actual = json_safe(evaluate_label_maps(labels, truth))
            compare_metrics(actual['metrics'], saved[row['image_id']]['metrics'])
            means.append(actual['metrics'])
            inclusive.append(actual['diagnostics']['obj_detected_ge075_percentage'])
            count_empty(empty_cases, actual['metrics'])
        native_metrics = mean_metrics(means)
        compare_metrics(native_metrics, score['summary']['metrics'])
        system = report / f'uoais_dpseg_system_{split}_final_v1'
        test = evidence.read(system / 'protocol.json')
        require(test['scope'] == 'final_test' and test['split'] == split and test['images'] == count
                and test['score_threshold'] == .5 and test['precision'] == 'float32'
                and test['base_checkpoint_sha256'] == bases['dpseg']['checkpoint_sha256']
                and test['point_checkpoint_sha256'] == point_hash
                and test['source_sha256'] == selection['source_sha256']
                and test['fixed_final_stage_from_validation'] == selection['stage'], 'Wrong DPSeg final-test provenance')
        for key in ('tile_policy', 'probability_policy', 'stages', 'overlap_rule'):
            require(test[key] == validation[key], f'Test policy differs from validation: {key}')
        system_summary = evidence.read(system / 'summary.json')
        require(all(system_summary[key] == value for key, value in test.items()), 'System summary protocol differs')
        metrics, diagnostics = audit_system(system, ids, evidence, data)
        results[split] = {'images': count, 'manifest_sha256': manifest_hash,
                          'dafusion': native_metrics, 'dpseg': metrics,
                          'endpoint_and_empty_case_diagnostics': {
                              'dafusion': {'mean_inclusive_f_ge_075': math.fsum(inclusive) / count, **empty_cases},
                              'dpseg': diagnostics},
                          'recorded_run_times': {
                              'dafusion': {'seconds': complete['elapsed_seconds'],
                                  'scope': 'prediction loop including data, restoration and RLE writing; scoring separate'},
                              'dpseg': {'seconds': system_summary['seconds'],
                                  'scope': 'joint four-stage system evaluation, including scoring and RLE writing'}},
                          'selected_dpseg_stage': selection['stage'],
                          'selected_dpseg_minus_dafusion_pp': {key: None if metrics[selection['stage']][key] is None or native_metrics[key] is None
                              else (metrics[selection['stage']][key] - native_metrics[key]) * 100 for key in METRICS}}
    paper = evidence.read(report / 'paper_protocol.json')
    literature = {}
    for split in COUNTS:
        table = paper['published_results']['table_II' if split == 'ocbd' else 'table_I']
        literature[split] = next(row[split.upper()] for row in table
                                if row['method'] == 'DA-Fusion DS+DC' and row['input'] == 'RGB-D')
    return {'status': 'verified_all_three_original_benchmarks', 'created_utc': datetime.now(timezone.utc).isoformat(),
            'scope': 'UOAIS-Sim-trained local system comparison; public DA-Fusion reimplementation',
            'required_benchmarks': COUNTS, 'units': 'fractions; deltas are percentage points',
            'training_protocol_sha256': protocol_hash, 'training_models': protocol['models'],
            'selected_base_checkpoints': bases,
            'selected_stage_from_full_validation': selection, 'results': results,
            'paper_reported_percent_separate_from_local_measurements': literature,
            'paper_source': paper['paper']['source'], 'evaluator': reference,
            'source_sha256': evidence.sha(Path(__file__)), 'evidence_sha256': evidence.files}


def write_report(output, result):
    require(set(result['results']) == set(COUNTS), 'All three original benchmarks are required')
    require(not output.exists(), 'Use a new report directory')
    output.mkdir(parents=True)
    (output / 'comparison.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    lines = ['# DA-Fusion 与 DPSeg：原始基准对照', '',
             '两组均在 UOAIS-Sim 45,000 张训练图上训练 20 轮；模型与方法只使用 5,000 张验证图选择。',
             'DPSeg 点头训练 5 轮。下表使用相同原始测试图和 GT，逐图重算掩码指标并取算术平均，单位为 %。',
             '标星的 DPSeg 阶段由验证集选择，全部测试集使用同一阶段。%75 使用严格 F > 0.75。', '',
             'DA-Fusion 为公开重新实现，Swin-L 输入 480×640；DPSeg 使用 Swin-T 和 1024×1024 输入，属于系统比较。',
             '本地全局 batch 为 8，论文为 2；本地公开实现使用 XYZ 与有效性，论文使用 HHA。UOAIS 相机内参采用已披露的名义假设。', '']
    with (output / 'comparison.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['dataset', 'model', 'stage', 'selected', 'images', *METRICS])
        writer.writeheader()
        for split, values in result['results'].items():
            lines += [f'## {split.upper()}（{values["images"]} 张）', '',
                      '| 模型／阶段 | Overlap P | Overlap R | Overlap F | Boundary P | Boundary R | Boundary F | %75 |',
                      '|---|---:|---:|---:|---:|---:|---:|---:|']
            rows = [('DA-Fusion public reimplementation', 'base', True, values['dafusion'])]
            rows += [('DPSeg', stage, stage == values['selected_dpseg_stage'], values['dpseg'][stage]) for stage in STAGES]
            for model, stage, chosen, metrics in rows:
                writer.writerow({'dataset': split, 'model': model, 'stage': stage, 'selected': chosen,
                                 'images': values['images'], **metrics})
                label = model + (' / ' + stage if model == 'DPSeg' else '') + (' *' if model == 'DPSeg' and chosen else '')
                display = ['未定义' if metrics[key] is None else f'{metrics[key] * 100:.2f}' for key in METRICS]
                lines.append('| ' + ' | '.join([label, *display]) + ' |')
            lines.append('')
    lines += ['## 论文报告值', '', '下表为文献转录，与上面的本地实测分别保存。', '',
              '| 数据集 | Overlap P | Overlap R | Overlap F | Boundary P | Boundary R | Boundary F | %75 |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for split, values in result['paper_reported_percent_separate_from_local_measurements'].items():
        lines.append('| ' + ' | '.join([split.upper(), *(f'{v:.1f}' for v in values)]) + ' |')
    lines += ['', f'[论文来源]({result["paper_source"]})。本地 JSON 保存全部输入哈希、模型选择与严格／含端点 %75 诊断。', '']
    (output / 'comparison.md').write_text('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=ROOT / 'reports/dafusion_benchmark_comparison')
    parser.add_argument('--manifests', type=Path, default=ROOT / 'data/uois')
    parser.add_argument('--check', action='store_true', help='Print readiness only; do not create a report')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.check:
        print(json.dumps(readiness(args.reports, args.manifests), ensure_ascii=False, indent=2))
        return
    if args.output is None:
        parser.error('--output is required for report generation')
    write_report(args.output, assemble(args.reports, args.manifests))


if __name__ == '__main__':
    main()
