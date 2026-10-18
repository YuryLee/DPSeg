#!/usr/bin/env python3
"""Report both completed model families using the already locked validation choice."""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.lock_course_longer_system import bind, read, require, sha, verify_lock


def comparison_rows(decision, reports):
    """Keep validation ranking even when another method has a higher test score."""
    rows = []
    expected = {}
    for candidate in decision['candidates']:
        family, variant = candidate['family'], candidate['variant']
        expected.setdefault(family, set()).add(variant)
        report = reports[family]
        require(variant in report['val'] and variant in report['test'] and variant in report['csvs'],
                'Candidate metrics or CSV missing')
        val, test = report['val'][variant], report['test'][variant]
        require(all(candidate[k] == val[k] for k in ('mask_ap', 'mask_ap_small')),
                'Validation results differ from locked selection')
        require(all(type(test[k]) in (int, float) and math.isfinite(test[k]) and 0 <= test[k] <= 1
                    for k in ('mask_ap', 'mask_ap_small')), 'Invalid full-test AP')
        rows.append({'family': family, 'variant': variant, 'validation_AP': val['mask_ap'],
                     'validation_APs': val['mask_ap_small'], 'test_AP': test['mask_ap'],
                     'test_APs': test['mask_ap_small'], 'csv': report['csvs'][variant]})
    require(set(reports) == set(expected), 'Comparison families differ from locked decision')
    for family, variants in expected.items():
        require(set(reports[family]['val']) == set(reports[family]['test']) == set(reports[family]['csvs']) == variants,
                'Comparison variant coverage differs')
    selected = decision['selected']
    chosen = [r for r in rows if r['family'] == selected['family'] and r['variant'] == selected['variant']]
    require(len(chosen) == 1, 'Locked winner is missing or duplicated')
    return rows, chosen[0]


def inspect_test(folder, ids, metrics, csv, artifacts):
    from tools.audit_course_point_refinement import compare_csv_predictions
    evaluation = read(folder / 'evaluation.json')
    require(evaluation['status'] == 'completed' and evaluation['split'] == 'test'
            and evaluation['images'] == len(ids) == 2112 and evaluation['metrics'] == metrics,
            'Full-test evaluation is incomplete or inconsistent')
    require(evaluation['iou_type'] == 'segm' and evaluation['max_detections'] == [1, 10, 100]
            and evaluation['area_ranges'][1] == [0, 1024]
            and evaluation['ground_truth_area'] == 'original annotation area preserved'
            and len(evaluation['iou_thresholds']) == 10
            and all(math.isclose(v, .5 + .05 * i) for i, v in enumerate(evaluation['iou_thresholds'])),
            'Test metric semantics differ')
    predictions = folder / 'predictions.jsonl'
    require(bind(predictions, artifacts) == evaluation['predictions_sha256'], 'Scored test predictions changed')
    require(bind(csv['path'], artifacts) == csv['sha256'], 'CSV bytes changed')
    receipt = csv['validation']
    require(receipt['sha256'] == csv['sha256'] and Path(receipt['path']).resolve() == Path(csv['path']).resolve()
            and receipt['rows'] == len(ids) and receipt['image_size'] == [1024, 1024]
            and receipt['category_ids'] == [1] and receipt['max_detections'] == 100
            and receipt['id_order_verified'] and receipt['rle_runs_verified'] and receipt['rle_native_decode_verified']
            and receipt['predictions'] == evaluation['predicted_instances'], 'Native CSV receipt is incomplete')
    compare_csv_predictions(csv['path'], predictions, ids)
    bind(folder / 'evaluation.json', artifacts)


def assemble(selection_path):
    from rgbd_seg.course_archive import CourseArchive
    decision, artifacts = verify_lock(selection_path), {}
    decision_sha = bind(selection_path, artifacts)
    primary = next(v for v in decision['families'] if v['name'] == 'selected_base')
    protocol = read(primary['protocol'])
    historical_path = Path(protocol['historical_comparison']['path'])
    historical = copy.deepcopy(read(historical_path))
    # The historical tiled comparison reused its baseline CSV from the earlier experiment.
    baseline_receipt_path = ROOT / 'reports/course_all_test_comparison/scores/tiled_probability_depth/submission_validation.json'
    baseline = read(baseline_receipt_path)
    historical['csvs']['tiled_probability_depth'] = {
        'path': baseline['path'], 'sha256': baseline['sha256'], 'validation': baseline}
    bind(baseline_receipt_path, artifacts)
    with CourseArchive(protocol['archive'], 'test') as data:
        ids = data.image_ids
    reports = {'historical': historical}
    for variant, metrics in historical['test'].items():
        inspect_test(historical_path.parent / 'test' / variant, ids, metrics, historical['csvs'][variant], artifacts)
    bind(historical_path, artifacts)
    for item in decision['families']:
        if item.get('mode') == 'retained_historical_parent':
            reports[item['name']] = historical
            continue
        output = Path(item['output'])
        result, run = read(output / 'comparison.json'), read(output / 'test_run.json')
        digest = sha(item['protocol'])
        require(result['status'] == 'completed' and result['scope'] == 'full_val_then_locked_full_test'
                and result['protocol_sha256'] == digest and not result['test_labels_used_for_selection'],
                'Family comparison is incomplete or changed selection using test labels')
        require(run['status'] == 'completed' and run['phase'] == 'test' and run['protocol_sha256'] == digest
                and datetime.fromisoformat(run['started_at_utc']) >= datetime.fromisoformat(decision['created_at_utc']),
                'Family test must complete after the global validation decision')
        require(result['selection_sha256'] == sha(output / 'selection.json')
                and result['selection'] == read(output / 'selection.json'), 'Family validation selection changed')
        require(read(output / 'test/image_ids.json') == ids, 'Test did not cover the official split')
        require(result['test'] == read(output / 'test/metrics.json'), 'Test summary differs from scorer outputs')
        for variant, metrics in result['test'].items():
            inspect_test(output / 'test' / variant, ids, metrics, result['csvs'][variant], artifacts)
        for name in ('comparison.json', 'test_run.json', 'test/image_ids.json', 'test/metrics.json'):
            bind(output / name, artifacts)
        reports[item['name']] = result
    rows, chosen = comparison_rows(decision, reports)
    require(sha(selection_path) == decision_sha, 'Global selection changed during reporting')
    verify_lock(selection_path)
    bind(Path(__file__), artifacts)
    return {'status': 'completed', 'scope': 'locked_full_system_comparison',
            'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'selection': str(selection_path.resolve()), 'selection_sha256': decision_sha,
            'selected': chosen, 'rows': rows, 'test_images': len(ids), 'validation_images': 2584,
            'test_used_for_selection': False, 'kaggle_submitted_by_this_tool': False,
            'artifact_sha256': artifacts,
            'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent',
                                 'mode': 'run', 'verification_status': 'SCORES_AND_SUBMISSIONS_BOUND'},
            'limitations': ['Local full-test scores are different from Kaggle public-subset scores',
                           'Single seed; best means best among evaluated candidates',
                           'This system audit does not replace the base-training convergence review']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'Output directory already exists')
    report = assemble(args.selection)
    args.output.mkdir(parents=True)
    (args.output / 'comparison.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    selected = report['selected']
    lines = ['# 延长训练后的完整系统比较', '',
             '两条训练路线和历史方案使用完整验证集统一选优，随后评测完整本地测试集。分数范围为 0–1。', '',
             f'验证集选定方案：**{selected["family"]} / {selected["variant"]}**。',
             f'验证 APₛ={selected["validation_APs"]:.6f}；本地测试 APₛ={selected["test_APs"]:.6f}。',
             f'对应提交文件：[CSV]({selected["csv"]["path"]})。', '',
             '| 路线 | 配置 | 验证 AP | 验证 APₛ | 本地测试 AP | 本地测试 APₛ | CSV |',
             '|---|---|---:|---:|---:|---:|---|']
    for row in report['rows']:
        lines.append(f'| {row["family"]} | {row["variant"]} | {row["validation_AP"]:.6f} | '
                     f'{row["validation_APs"]:.6f} | {row["test_AP"]:.6f} | {row["test_APs"]:.6f} | '
                     f'[下载]({row["csv"]["path"]}) |')
    lines += ['', '表格沿用验证集排序。测试成绩未用于更换所选方案；本地测试成绩与 Kaggle 公榜分数范围不同。',
              '已核对测试图像覆盖、指标定义、预测文件摘要及 CSV 与评分预测逐行一致性。',
              '原生 RLE 校验沿用已通过且绑定相同 CSV 字节的校验记录。', '']
    (args.output / 'report.md').write_text('\n'.join(lines))
    print(json.dumps({'status': report['status'], 'family': selected['family'], 'variant': selected['variant'],
                     'validation_APs': selected['validation_APs'], 'test_APs': selected['test_APs'],
                     'csv': selected['csv']['path'], 'report': str((args.output / 'report.md').resolve())}, ensure_ascii=False))


if __name__ == '__main__':
    main()
