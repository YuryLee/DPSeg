#!/usr/bin/env python3
"""Audit completed official-SAHI comparisons before publishing measured results."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_course_swinl_ablation import read, sha, require, METRICS
from tools.summarize_course_swinl_removal import check_sources, record_digest

VARIANTS = ('sahi_nms_iou_early', 'mask_nms_late', 'sahi_polygon_mask_nms_late',
            'sahi_nms_ios_late', 'sahi_greedynmm_ios_late')
LABELS = {
    'legacy_full': '原完整 DPSeg（当前默认）',
    'sahi_nms_iou_early': '提前 SAHI NMS（框 IoU）',
    'mask_nms_late': '延后掩码 NMS（顺序对照）',
    'sahi_polygon_mask_nms_late': '延后掩码 NMS＋多边形转换（表示对照）',
    'sahi_nms_ios_late': '延后 SAHI NMS（框 IoS）',
    'sahi_greedynmm_ios_late': '延后 SAHI GreedyNMM（框 IoS）',
}
SEMANTICS = ('iou_type', 'max_detections', 'iou_thresholds', 'area_ranges',
             'ground_truth_area', 'prediction_area', 'gt_instances', 'gt_small_including_crowd')
CONVERSION_KEYS = ('input_instances', 'changed_instances', 'dropped_instances',
                   'added_pixels', 'removed_pixels')


def integral(value):
    return type(value) is int and value >= 0


def check_mirror_install():
    """Bind an explicit direct mirror repeat to the untouched installed payload."""
    folder = ROOT/'reports/course_sahi_merge_deps'
    path = folder/'mirror_direct/verification.json'
    receipt = read(path)
    require(receipt['status'] == 'passed' and receipt['mirror'] == 'https://pypi.tuna.tsinghua.edu.cn/simple'
            and set(receipt['proxy_env_explicitly_removed']) ==
            {'http_proxy', 'https_proxy', 'all_proxy', 'HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY'},
            'Mirror direct-download verification incomplete')
    require(len(receipt['wheels']) == 2, 'Expected both pinned dependency wheels')
    packages = set()
    for item in receipt['wheels']:
        wheel = Path(item['path'])
        require(sha(wheel) == item['sha256'], 'Verified mirror wheel changed')
        package = wheel.name.split('-')[0]
        require(package in {'sahi', 'shapely'} and package not in packages, 'Unexpected dependency wheel')
        packages.add(package)
        count = 0
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if name.endswith('/') or '.dist-info/' in name:
                    continue
                target = (folder/'site'/name).resolve()
                require(target.is_relative_to((folder/'site').resolve()), 'Wheel payload path escapes dependency directory')
                require(target.read_bytes() == archive.read(name), 'Mirror wheel payload differs from frozen site')
                count += 1
        require(count == item['installed_payload_files_verified'], 'Wheel payload coverage differs')
    return dict(path=str(path), sha256=sha(path), receipt=receipt)


def check_result(row, ids, expected_semantics=None):
    """Bind evaluation, decoded CSV receipt and every actual CSV prediction."""
    from tools.audit_course_point_refinement import compare_csv_predictions
    require(row['status'] == 'completed' and row['images'] == len(ids), 'Incomplete result')
    for kind in ('evaluation', 'predictions', 'csv'):
        require(sha(row[kind]) == row[f'{kind}_sha256'], f'{kind} artifact changed')
    evaluated = read(row['evaluation'])
    require(evaluated['status'] == 'completed' and evaluated['split'] == 'test'
            and evaluated['images'] == len(ids) and evaluated['depth_used']
            and evaluated['metrics'] == row['metrics']
            and evaluated['predictions_sha256'] == row['predictions_sha256']
            and evaluated['predicted_instances'] == row['predicted_instances'], 'Evaluation/result mismatch')
    check_sources(evaluated['source_sha256'])
    semantics = {key: evaluated[key] for key in SEMANTICS}
    require(semantics['iou_type'] == 'segm' and semantics['max_detections'] == [1, 10, 100]
            and semantics['area_ranges'][1] == [0, 1024]
            and semantics['ground_truth_area'] == 'original annotation area preserved'
            and semantics['prediction_area'] == 'independent COCO RLE; no bbox passed to loadRes',
            'Wrong COCO evaluation settings')
    require(len(semantics['iou_thresholds']) == 10 and all(
        abs(value-(.5+.05*i)) < 1e-12 for i, value in enumerate(semantics['iou_thresholds'])),
        'Wrong IoU thresholds')
    if expected_semantics is not None:
        require(semantics == expected_semantics, 'Evaluation settings or GT statistics differ')
    require(all(isinstance(row['metrics'][metric], (int, float))
                and 0 <= row['metrics'][metric] <= 1 for metric in METRICS), 'Invalid report metric')
    validation_path = Path(row['evaluation']).parent/'csv_validation.json'
    validation = read(validation_path)
    require(Path(validation['path']).resolve() == Path(row['csv']).resolve()
            and validation['sha256'] == row['csv_sha256'] and validation['rows'] == len(ids)
            and validation['predictions'] == row['predicted_instances']
            and validation['image_size'] == [1024, 1024] and validation['category_ids'] == [1]
            and validation['max_detections'] == 100 and validation['id_order_verified']
            and validation['rle_runs_verified'] and validation['rle_native_decode_verified'],
            'CSV RLE/ID validation is incomplete or belongs to another artifact')
    check_sources(validation['source_sha256'])
    binding = compare_csv_predictions(row['csv'], row['predictions'], ids)
    require(binding == row['csv_rows_identical_to_scored_predictions'] == len(ids), 'CSV binding incomplete')
    return semantics, dict(path=str(validation_path), sha256=sha(validation_path),
                           rows=binding, rle_native_decode_verified=True)


def audit(run, semantics_audit):
    state = read(run/'status.json')
    require(state['status'] == 'completed' and state['exit_code'] == 0, 'Experiment has not completed')
    plan, result = read(run/'protocol.json'), read(run/'results.json')
    protocol_sha = sha(run/'protocol.json')
    require(result['status'] == 'completed' and result['split'] == 'test' and result['images'] == 2112
            and result['protocol_sha256'] == protocol_sha, 'Incomplete result or changed protocol')
    require(not result['training'] and not result['test_used_for_selection']
            and not plan['training'] and not plan['test_used_for_selection']
            and not plan['ground_truth_in_prediction'] and not plan['speed_benchmark'],
            'Unexpected training, selection, annotation input or speed scope')
    require(plan['variants'] == list(VARIANTS) and set(result['variants']) == set(VARIANTS),
            'Missing or changed variants')
    require(plan['input_keys'] == ['image', 'depth_m', 'depth_valid']
            and not plan['camera_intrinsics_used'] and plan['tile_size_px'] == 640
            and plan['base_epoch'] == 15 and plan['head_epoch'] == 6
            and plan['precision'] == 'float32' and plan['tf32'] is False
            and plan['score_threshold'] == .01 and plan['max_detections'] == 100
            and plan['official_match_threshold'] == .5, 'Fixed inference settings changed')
    check_sources(plan['new_source_sha256'])
    for item in [*plan['weights'].values(), plan['system_protocol']]:
        require(sha(item['path']) == item['sha256'], 'Fixed weight or system protocol changed')
    system = read(plan['system_protocol']['path'])
    check_sources(system['system_source_sha256'])
    require(system['model']['backbone'] == 'large' and system['model']['depth_encoding'] == 'depth'
            and system['base_checkpoint_sha256'] == plan['weights']['base']['sha256'], 'Base identity differs')
    selection = read(Path(plan['system_protocol']['path']).parent/'points6/selection.json')
    require(selection['status'] == 'locked' and not selection['test_used']
            and selection['selected']['epoch'] == 6
            and selection['selected_head_sha256'] == plan['weights']['heads']['sha256'], 'Head selection differs')
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_sahi_merge import dependency_receipt, VARIANTS as implemented
    from rgbd_seg.course_tiled_probability import TILE_POLICY, PROBABILITY_POLICY, view_bounds
    require(tuple(implemented) == VARIANTS, 'Runtime variants differ')
    dependencies = dependency_receipt()
    require(dependencies == plan['dependency_receipt'] == result['dependency_receipt'], 'Dependency receipt changed')
    require(dependencies['sahi'] == '0.14.0' and dependencies['shapely'] == '2.1.2'
            and dependencies['backend'] == 'numpy', 'Unexpected official SAHI runtime')
    mirror = check_mirror_install()
    sahi_wheel = next(item for item in mirror['receipt']['wheels'] if Path(item['path']).name.startswith('sahi-'))
    require(sahi_wheel['sha256'] == dependencies['official_wheel_sha256'], 'Mirror and official SAHI wheel differ')
    require(plan['view_bounds'] == json.loads(json.dumps(view_bounds(1024, 1024))), 'Window geometry changed')
    expected_policy = {**TILE_POLICY, 'geometry': 'aligned RGB/depth/valid crop; no camera K or intrinsics used'}
    require(plan['tile_policy'] == expected_policy and plan['probability_policy'] == PROBABILITY_POLICY,
            'Fixed candidate/refinement policies differ')
    with CourseArchive(plan['archive'], split='test') as data:
        ids = data.image_ids
    require(ids == plan['image_ids'] and len(ids) == len(set(ids)) == 2112, 'Test coverage/order differs')
    require(result['full_reference'] == plan['full_reference'], 'Final full-model reference changed')
    full = plan['full_reference']
    reference_root = Path(full['evaluation']).parent.parent
    reference_plan = read(reference_root/'protocol.json')
    reference_result = read(reference_root/'results.json')
    require(sha(reference_root/'protocol.json') == plan['reference_run_protocol_sha256']
            == reference_result['protocol_sha256'] and reference_result['status'] == 'completed'
            and reference_result['variants']['full'] == full
            and reference_plan['weights'] == plan['weights'], 'Reference run identity differs')
    semantics, full_validation = check_result(full, ids)
    rescore_path = run/'reference_rescore/verification.json'
    rescore = read(rescore_path)
    require(rescore['status'] == 'passed' and rescore['all_metrics_exact']
            and rescore['images'] == 2112 and rescore['metrics'] == full['metrics']
            and rescore['reference_predictions_sha256'] == full['predictions_sha256']
            and rescore['checker_sha256'] == sha(ROOT/'tools/check_course_sahi_reference.py')
            and sha(rescore['evaluation']) == rescore['evaluation_sha256'],
            'Reference was not independently rescored with current annotations')
    rescored = read(rescore['evaluation'])
    require(rescored['status'] == 'completed' and rescored['split'] == 'test'
            and rescored['images'] == 2112 and rescored['metrics'] == full['metrics']
            and rescored['predictions_sha256'] == full['predictions_sha256']
            and {key: rescored[key] for key in SEMANTICS} == semantics,
            'Fresh reference scoring differs')
    check_sources(rescored['source_sha256'])
    native = plan['native_reference']
    require(sha(native['predictions']) == native['predictions_sha256']
            and sha(native['evaluation']) == native['evaluation_sha256'], 'Native reference changed')
    native_eval = read(native['evaluation'])
    require(native_eval['status'] == 'completed' and native_eval['images'] == 2112
            and native_eval['split'] == 'test' and native_eval['predictions_sha256'] == native['predictions_sha256']
            and {key: native_eval[key] for key in SEMANTICS} == semantics, 'Native reference scoring differs')
    check_sources(native_eval['source_sha256'])
    reference_counts = {}
    with Path(full['predictions']).open() as stream:
        for line in stream:
            row = json.loads(line)
            require(row['image_id'] not in reference_counts, 'Duplicate full-reference image')
            reference_counts[row['image_id']] = len(row['predictions'])
    require(list(reference_counts) == ids, 'Full reference IDs/order differ')

    expected_exits = {'engineering': {'engineering': 0},
        'inferring': {f'worker{i}': 0 for i in range(len(plan['gpus']))},
        'scoring': {f'score_{variant}': 0 for variant in VARIANTS}}
    for phase, values in expected_exits.items():
        require(read(run/f'{phase}_exits.json') == values, f'{phase} did not finish successfully')
    gate = read(run/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['split'] == 'val' and not gate['ground_truth_used']
            and not gate['model_or_threshold_changes'] and gate['protocol_sha256'] == protocol_sha
            and gate['dependency_receipt'] == dependencies, 'Invalid engineering gate')
    require([row['image_id'] for row in gate['checks']] == plan['engineering_val_ids']
            and len(gate['checks']) == 3 and all(row['native_and_legacy_exact']
                and row['score_errors'] == {'legacy_full': 0., 'native': 0.} for row in gate['checks']),
            'Engineering legacy/native comparisons incomplete')
    synthetic = gate['synthetic']
    require(synthetic['status'] == 'passed' and all(synthetic[key] for key in
        ('bbox_iou_vs_ios', 'same_view_suppression_difference', 'disjoint_masks_overlapping_boxes',
         'union_mask_and_max_score', 'source_identity_preserved', 'input_not_mutated')),
        'Official postprocessor synthetic checks incomplete')
    require(synthetic['single_pixel_conversion']['dropped_instances'] == 1,
            'Single-pixel conversion check differs')
    supplement = read(semantics_audit)
    require(supplement['status'] == 'passed', 'Supplementary geometry audit incomplete')
    check_sources(supplement['source_sha256'])
    require(supplement['source_sha256']['rgbd_seg/course_sahi_merge.py']
            == plan['new_source_sha256']['rgbd_seg/course_sahi_merge.py'],
            'Supplementary audit used a different integration module')
    cases = {case['case']: case for case in supplement['cases']}
    require(set(cases) == {'overlapping_rings', 'bounding_box_coordinate_convention'},
            'Supplementary geometry cases missing')
    union, boxes = cases['overlapping_rings'], cases['bounding_box_coordinate_convention']
    require(union['input_unchanged'] and union['input_conversion']['changed_instances'] == 0
            and union['binary_union_area'] == 480 and union['official_polygon_union_area'] == 580
            and union['added_vs_binary_union'] == 100 and union['removed_vs_binary_union'] == 0,
            'Unexpected polygon-union audit outcome')
    require(boxes['input_conversion']['changed_instances'] == 0
            and boxes['rle_nms_count'] == 1 and boxes['polygon_nms_count'] == 2
            and boxes['rle_pixel_edge_iou'] == .5
            and abs(boxes['polygon_contour_iou']-13/27) < 1e-12,
            'Unexpected coordinate-convention audit outcome')

    digests = {variant: {} for variant in VARIANTS}
    counts, conversion, candidates, rank_receipts, coverage = Counter(), Counter(), 0, [], []
    for rank in range(len(plan['gpus'])):
        rank_ids = ids[rank::len(plan['gpus'])]
        receipt = read(run/f'rank{rank}.json')
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids
                and receipt['images'] == len(rank_ids) and receipt['native_exact'] and receipt['legacy_full_exact']
                and not receipt['ground_truth_in_prediction'] and receipt['dependency_receipt'] == dependencies,
                'Invalid rank receipt')
        coverage.extend(receipt['image_ids'])
        checks_path = run/f'checks_rank{rank}.jsonl'
        require(sha(checks_path) == receipt['checks_sha256'], 'Per-image check file changed')
        checks = [json.loads(line) for line in checks_path.read_text().splitlines()]
        require([row['image_id'] for row in checks] == rank_ids, 'Per-image check coverage differs')
        for row in checks:
            require(row['native_exact'] and row['legacy_full_exact'], 'Native/full reproduction check failed')
            stats = row['stats']
            require(stats['model_views'] == 5 and integral(stats['candidates']) and stats['candidates'] <= 500,
                    'Wrong view/candidate counts')
            require(stats['original_merging']['candidates'] == stats['late_merging']['candidates']
                    == stats['candidates'] and len(stats['original_merging']['selected_views'])
                    == reference_counts[row['image_id']], 'Legacy/order-control candidate count differs')
            cvt = stats['polygon_conversion']
            require(set(cvt) == set(CONVERSION_KEYS) and all(integral(cvt[key]) for key in CONVERSION_KEYS)
                    and cvt['input_instances'] == stats['candidates']
                    and cvt['changed_instances']+cvt['dropped_instances'] <= cvt['input_instances']
                    and stats['polygon_merging']['candidates'] == cvt['input_instances']-cvt['dropped_instances'],
                    'Polygon conversion accounting inconsistent')
            require(stats['output_counts'] == row['instances'] and set(row['instances']) == set(VARIANTS)
                    and all(integral(count) and count <= 100 for count in row['instances'].values()),
                    'Output count audit differs')
            conversion.update(cvt)
            candidates += stats['candidates']
        by_id = {row['image_id']: row for row in checks}
        for variant in VARIANTS:
            shard = run/f'{variant}_rank{rank}.jsonl'
            require(sha(shard) == receipt['artifacts'][variant], 'Prediction shard changed')
            observed, total = [], 0
            with shard.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    image_id = row['image_id']
                    require(image_id not in digests[variant], 'Duplicate shard image')
                    observed.append(image_id)
                    require(len(row['predictions']) == by_id[image_id]['instances'][variant],
                            'Prediction/check instance count differs')
                    digests[variant][image_id] = record_digest(row)
                    total += len(row['predictions'])
            require(observed == rank_ids and total == receipt['instances'][variant], 'Shard coverage/count differs')
            counts[variant] += total
        rank_receipts.append(dict(rank=rank, images=len(rank_ids), gpu=receipt['gpu'],
                                  receipt_sha256=sha(run/f'rank{rank}.json'), checks_sha256=receipt['checks_sha256']))
    require(Counter(coverage) == Counter(ids), 'Ranks do not cover each test image exactly once')
    require(candidates == conversion['input_instances'], 'Aggregate conversion coverage differs')
    rows = [dict(full, variant='legacy_full', method=LABELS['legacy_full'], validation=full_validation)]
    for variant in VARIANTS:
        row = result['variants'][variant]
        folder = run/variant
        require(row == read(folder/'result.json') and row['variant'] == variant
                and Path(row['evaluation']).resolve() == folder/'evaluation.json'
                and Path(row['predictions']).resolve() == folder/'predictions.jsonl'
                and Path(row['csv']).resolve() == run/'submissions'/f'dpseg_swinl_depth_no_k_{variant}.csv',
                'Result belongs to the wrong variant/path')
        _, validation = check_result(row, ids, semantics)
        require(row['predicted_instances'] == counts[variant], 'Merged instance count differs')
        observed = []
        with Path(row['predictions']).open() as stream:
            for line in stream:
                item = json.loads(line)
                observed.append(item['image_id'])
                require(record_digest(item) == digests[variant].get(item['image_id']),
                        'Scored prediction differs from inference shard')
        require(observed == ids, 'Merged IDs/order differ')
        rows.append(dict(row, method=LABELS[variant], validation=validation))
        print(f'Verified {variant}: 2112 CSV rows, {counts[variant]} instances', flush=True)
    return dict(plan=plan, result=result, protocol_sha256=protocol_sha, rows=rows, semantics=semantics,
                conversion=dict(conversion), candidates=candidates, rank_receipts=rank_receipts,
                phase_exits=expected_exits, gate=gate, dependencies=dependencies, mirror=mirror,
                supplementary_semantics=dict(path=str(semantics_audit), sha256=sha(semantics_audit), receipt=supplement),
                baseline_rescore=dict(path=str(rescore_path), sha256=sha(rescore_path), receipt=rescore))


def publish(run, verified):
    plan, rows = verified['plan'], verified['rows']
    baseline = rows[0]['metrics']
    for row in rows:
        row['delta_vs_legacy'] = {metric: row['metrics'][metric]-baseline[metric] for metric in METRICS}
    by_variant = {row['variant']: row for row in rows}
    controlled_pairs = (
        ('提前官方 NMS', 'sahi_nms_iou_early', 'legacy_full'),
        ('合并顺序', 'mask_nms_late', 'legacy_full'),
        ('多边形表示', 'sahi_polygon_mask_nms_late', 'mask_nms_late'),
        ('官方框 IoS NMS 规则', 'sahi_nms_ios_late', 'sahi_polygon_mask_nms_late'),
        ('GreedyNMM 相对 NMS', 'sahi_greedynmm_ios_late', 'sahi_nms_ios_late'))
    contrasts = {label: dict(current=current, reference=reference,
        delta={metric: by_variant[current]['metrics'][metric]-by_variant[reference]['metrics'][metric]
               for metric in METRICS}) for label, current, reference in controlled_pairs}
    table = ['| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |', '|:---:|:---:|:---:|:---:|']
    for row in rows:
        cells = [row['method'], *(f'{row["metrics"][metric]:.6f}' for metric in METRICS)]
        if row['variant'] == 'legacy_full':
            cells = [f'**{cell}**' for cell in cells]
        table.append('| '+' | '.join(cells)+' |')
    table = '\n'.join(table)
    comparisons = '\n'.join(f'- **{label}：** {LABELS[item["current"]]} 相对 {LABELS[item["reference"]]}，'
        f'APₛ {item["delta"]["mask_ap_small"]:+.6f}、ARₛ {item["delta"]["mask_ar_small"]:+.6f}、'
        f'AP {item["delta"]["mask_ap"]:+.6f}。' for label, item in contrasts.items())
    best = {metric: [row['method'] for row in rows
                    if row['metrics'][metric] == max(item['metrics'][metric] for item in rows)] for metric in METRICS}
    conclusion = (f'本表六组中，APₛ 最高为 **{"、".join(best["mask_ap_small"])}**，'
                  f'ARₛ 最高为 **{"、".join(best["mask_ar_small"])}**，'
                  f'整体 AP 最高为 **{"、".join(best["mask_ap"])}**。')
    common = ('固定 Swin-Large＋无内参深度、第 15 轮基座与第 6 轮点头，在完整本地测试集 **2,112 张图**上'
              '比较官方 SAHI 后处理与对照配置。各组均保留整图和四个 640×640 角点切片、'
              '来源视图点细化及深度引导概率图细化。分数范围 0–1，加粗标记原完整配置。')
    scope = ('本轮调用 **SAHI 0.14.0 的官方 NMSPostprocess、GreedyNMMPostprocess 类**，'
             'RGB-D 输入和切片前向仍由 DPSeg 负责，未调用完整 get_sliced_prediction 接口。'
             '这是在已查看过结果的本地测试集上追加的固定权重比较；没有根据本表调阈值、选择检查点或替换默认配置。'
             '单种子结果不代表统计显著性，Kaggle 成绩另行报告。共享特征的实验运行时间不用于速度比较。')
    cvt = verified['conversion']
    conversion_text = (f'五视图合计产生 **{verified["candidates"]:,}** 个候选记录；同一物体在不同视图中可能重复出现。'
        f'对全部细化候选进行官方多边形转换后，**{cvt["dropped_instances"]:,}** 个候选因无法形成多边形而丢弃，'
        f'**{cvt["changed_instances"]:,}** 个仍保留候选的掩码发生变化。转换累计增加 '
        f'**{cvt["added_pixels"]:,}** 个掩码像素、移除 **{cvt["removed_pixels"]:,}** 个掩码像素；'
        '这些是合并前的表示转换统计，不包含 GreedyNMM 多边形并集后的拓扑或栅格化变化。'
        '移除量包含丢弃候选的像素，各候选分别计数，不是唯一物体数或图像覆盖面积。')
    paths = '\n'.join(f'- **{row["method"]}：** `{row["csv"]}`' for row in rows)
    deps = verified['dependencies']
    run_label = str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run)
    ring = verified['gate']['synthetic']['ring_conversion']
    ring_text = ('合成环形掩码在该版本中的像素往返没有变化。' if ring['changed_instances'] == 0
                 and ring['dropped_instances'] == 0 else '合成环形掩码的转换变化详见 JSON 中的合成检查记录。')
    document = f'''# DPSeg 与官方 SAHI 后处理对比

{common}

{table}

{conclusion}

## 对照设计

| 配置 | 操作顺序与对照目的 |
|---|---|
| 原完整 DPSeg | 粗候选跨视图掩码 IoU NMS → 仅对保留候选做点细化和概率图细化 |
| 提前 SAHI NMS（框 IoU） | 只把提前选择改为官方框 IoU NMS；保留原 RLE 和来源特征，不做多边形转换 |
| 延后掩码 NMS | 先细化全部候选，再用原跨视图掩码 IoU NMS；检验顺序影响 |
| 延后掩码 NMS＋多边形转换 | 相同延后流程，加入官方 bool-mask→COCO polygon→mask 转换后再做原掩码 NMS；检验表示变化 |
| 延后 SAHI NMS（框 IoS） | 对同一批转换后的 ObjectPrediction 执行官方框 IoS NMS |
| 延后 SAHI GreedyNMM（框 IoS） | 对相同 ObjectPrediction 执行官方框 IoS GreedyNMM；使用官方多边形并集并保留组内最高分 |

{comparisons}

提前 NMS 是候选选择的直接替换。延后组在各候选来源视图中完成点细化，再用原图深度做概率图细化，最后才合并。掩码 NMS 只抑制跨视图重复；官方后处理同时允许同类别、同视图候选互相抑制。框 IoS 为交集面积除以较小框面积，与框 IoU、掩码 IoU 均不同。因此“官方框 IoS NMS 规则”这一对比同时改变重叠定义、同视图抑制行为和框的几何约定，不能解释为单独 IoS 的贡献。

提前官方 NMS 的框由 COCO RLE 像素边界得到，延后官方组的框由 SAHI 多边形轮廓得到；两种坐标约定在重叠阈值附近可能产生不同选择。补充合成检查中，转换前后像素掩码均不变，但像素边界框 IoU 为 0.5，轮廓框 IoU 为 0.481481；同一官方 NMS、阈值 0.5 分别保留 1、2 个候选。因而提前/延后官方组不能仅归因于细化顺序或 IoU/IoS。两个延后官方组共享相同框和预测对象，可在这一共同表示下比较。

两个延后官方方法使用同一批转换后的预测对象，每次调用使用新的列表容器。合成检查已验证 GreedyNMM 不修改该批输入对象，避免影响 NMS 对照。合成例子也检验了框重叠而掩码不相交、同视图重复、IoU 与 IoS 的区别，以及矩形候选并集和最高分规则。

GreedyNMM 使用官方多边形并集，不能一般性等同于二值掩码逐像素 OR：并集及再次栅格化可能填充孔洞等改变拓扑。补充环形例子中，合并前表示往返均无变化，但官方并集面积为 580 像素，逐像素 OR 为 480 像素，多出 100 像素。因此 NMS 与 GreedyNMM 的差异包含“选择或合并”以及并集后的表示变化，不把它解释为纯分组算法的贡献。该例子用于解释几何行为，不代表测试集的总体新增像素量。

## 固定参数与范围

所有组置信度阈值为 0.01、合并匹配阈值为 0.5、每图最多 100 个实例。模型输入统一为 1024×1024，四个 640×640 切片相邻重叠 256 像素，内部裁剪边缘过滤为 2 像素。RGB、米制深度和有效性同步裁剪，无相机内参；FP32 推理，关闭 TF32，seed=17。

所有合并方法接收的候选都经过 DPSeg 的切片边缘过滤。对照结论限定在这批候选和固定阈值下；完整 SAHI 流程、关闭边缘过滤或另行验证选参后的效果需单独评估。

SAHI 0.14.0 的 predict.py 在模型置信度阈值低于 0.1 时会自动使用 NMS/IOU，除非 force_postprocess_type=True。本轮阈值为 0.01，因此 IOS 与 GreedyNMM 组是显式调用所选官方类的对照，不能称为该阈值下的自动默认行为。

{scope}

## 多边形转换影响

{conversion_text}

合成单像素掩码无法形成官方多边形，已明确丢弃并记录，没有用边界框代替其掩码。{ring_text}不据此概括所有带孔或细小连通域都能无损转换。三组包含转换的延后方法共享这批转换结果；表示对照用于区分转换损失和后处理规则的影响。

## 官方实现依据

版本与发布文件见 [SAHI 0.14.0](https://pypi.org/project/sahi/0.14.0/)。后处理类见[官方 combine.py](https://github.com/obss/sahi/blob/main/sahi/postprocess/combine.py)，掩码与分数的合并见[官方 utils.py](https://github.com/obss/sahi/blob/main/sahi/postprocess/utils.py)。GitHub 主分支可能继续更新；本次实验实际运行上述固定 wheel，具体源码哈希已保存在配套 JSON 中。

## 核验

- 冻结实验协议、旧模型源码、新集成源码、基座/点头文件，以及 SAHI、Shapely、OpenCV 文件哈希和版本。
- 3 张指定验证图上，独立原实现与共享特征入口的 legacy full/native 输出逐掩码、逐分数一致；合成行为检查全部通过。
- 独立 CPU 几何审计另记录了带孔多边形并集与框坐标约定的差异，源码哈希及完整 receipt 均纳入配套 JSON；没有更改已冻结推理。
- 全部 2,112 张测试图在运行时逐图严格核对原 S1 整图预测和原 S4 完整预测。发布程序检查对应日志的覆盖、哈希和成功标记。
- 五组分片实际 ID、顺序、实例数及哈希均与记录一致；每条合并评分预测与来源分片再次逐图绑定。
- 六份结果（原完整配置加五组新结果）的预测 CSV 均通过全量原生 RLE 解码。发布时核对原解码记录与文件哈希，并重新逐行核对 CSV 的 ID、类别、置信度和掩码与评分输入一致。
- 六组均使用相同 COCO 掩码设置和 GT 统计。APₛ、ARₛ 对原始标注面积 [0, 32²] 像素计算，AP 对所有尺寸计算；IoU=0.50:0.05:0.95，每图最多 100 个实例。
- 原完整配置另外使用当前课程归档重新评分，全部 12 项 COCO 指标与原记录完全一致，核验记录纳入配套 JSON。

## 依赖与复现

SAHI 固定为 **{deps['sahi']}**，Shapely 固定为 **{deps['shapely']}**；本次运行 OpenCV 为 {deps['opencv']}、NumPy 为 {deps['numpy']}，后处理使用 NumPy backend。SAHI wheel SHA-256 为 `{deps['official_wheel_sha256']}`，与官方发布文件哈希一致；全部运行文件的哈希见配套 JSON。

依赖来自清华 PyPI 镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`；另以显式清除代理变量的独立下载核对 SAHI 和 Shapely wheel，其全部 payload 与冻结安装目录逐文件一致，receipt 见配套 JSON。依赖隔离存放在 `reports/course_sahi_merge_deps/site`。下列复现命令用于新建环境，不覆盖运行中实验的冻结 site；`--no-deps` 避免更换现有 NumPy、PyTorch 等依赖，其余包需与项目环境预先兼容。

```bash
mkdir -p reports/course_sahi_merge_deps/wheels
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \\
  .venv/bin/python -m pip download --no-deps \\
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \\
  --dest reports/course_sahi_merge_deps/wheels sahi==0.14.0 shapely==2.1.2
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \\
  .venv/bin/python -m pip install --no-deps \\
  --target reports/course_sahi_merge_deps/site \\
  reports/course_sahi_merge_deps/wheels/sahi-0.14.0-py3-none-any.whl \\
  reports/course_sahi_merge_deps/wheels/shapely-2.1.2-*.whl
.venv/bin/python -u tools/evaluate_course_sahi_merge.py \\
  --output reports/course_sahi_merge_new --gpus 0,2,3,4,5,6,7 \\
  --reference-run reports/course_swinl_removal_v1
.venv/bin/python tools/check_course_sahi_reference.py --run reports/course_sahi_merge_new
.venv/bin/python tools/summarize_course_sahi_merge.py \\
  --run reports/course_sahi_merge_new \\
  --semantics-audit reports/course_sahi_merge_v1/supplementary_semantics.json
```

按实际空闲显卡调整 --gpus，输出实验目录须尚不存在。参考实验目录需包含已完成的完整预测、数据与权重路径；入口先冻结依赖 receipt，再通过工程检查后执行全量推理和评分。--semantics-audit 指向本轮独立 CPU 几何审计，发布时要求其源码与当前依赖一致；可运行 tools/check_course_sahi_semantics.py 重现该合成检查。五组共享视图特征只用于节省实验时间，不据总运行时间计算各方法 FPS。

## CSV 本机路径

{paths}

原始预测 CSV、权重和运行日志保留在本机；可分发的指标与核验索引见 [CSV](course_sahi_merge.csv) 和 [JSON](course_sahi_merge.json)。

## Material Passport

- 来源：ARS experiment-agent run；`{run_label}` 的真实推理和 COCO 评分产物。
- 核验状态：VERIFIED；协议 SHA-256：`{verified['protocol_sha256']}`。
- 范围：固定权重的官方 SAHI 后处理、顺序与表示对照；完整本地测试集，不作为验证选参或 Kaggle 分数。
- 限制：单次训练；各路径的合并顺序、表示及规则按表分别变化；无统计显著性或速度结论。RGB-D 融合去除实验仍待后续训练。
'''
    artifact = dict(schema='dpseg-official-sahi-merge-v1', status='verified',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), split='local_test', images=2112,
        backbone='Swin-Large', camera_intrinsics_used=False, training=False, test_used_for_selection=False,
        default_changed=False, speed_benchmark=False, interpretation_scope=plan['test_analysis'],
        semantics=verified['semantics'], results=rows, controlled_contrasts=contrasts,
        candidate_occurrences=verified['candidates'], polygon_conversion=verified['conversion'],
        synthetic_checks=verified['gate']['synthetic'], supplementary_semantics=verified['supplementary_semantics'],
        baseline_rescore=verified['baseline_rescore'],
        checkpoints=plan['weights'],
        protocol=dict(path=str(run/'protocol.json'), sha256=verified['protocol_sha256']),
        source_sha256=plan['new_source_sha256'], system_protocol=plan['system_protocol'],
        dependency_receipt=verified['dependencies'], mirror_verification=verified['mirror'],
        phase_exits=verified['phase_exits'],
        rank_receipts=verified['rank_receipts'],
        publisher_source_sha256={name: sha(ROOT/name) for name in
            ('tools/summarize_course_sahi_merge.py', 'tools/summarize_course_swinl_removal.py',
             'tools/summarize_course_swinl_ablation.py')},
        material_passport=dict(source=f'academic-research-suite / {run_label}', verification_status='VERIFIED',
            scope='fixed-weight official SAHI integration comparison', limitations=scope))
    docs = ROOT/'docs/results'
    (docs/'course_sahi_merge.md').write_text(document)
    (docs/'course_sahi_merge.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (docs/'course_sahi_merge.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['方法', 'AP_s', 'AR_s', 'AP'])
        for row in rows:
            writer.writerow([row['method'], *(row['metrics'][metric] for metric in METRICS)])
    start, end = '<!-- sahi-merge:start -->', '<!-- sahi-merge:end -->'
    block = f'''{start}
### 官方 SAHI 后处理对比

{common}

{table}

{conclusion}

{scope}

{conversion_text}

[对照顺序、核验、依赖及 CSV 路径](docs/results/course_sahi_merge.md) · [指标 CSV](docs/results/course_sahi_merge.csv) · [核验 JSON](docs/results/course_sahi_merge.json)。
{end}'''
    readme = ROOT/'README.md'
    original = readme.read_text()
    if start in original or end in original:
        require(original.count(start) == original.count(end) == 1, 'Ambiguous SAHI section markers')
        updated = original[:original.index(start)]+block+original[original.index(end)+len(end):]
    else:
        anchor = '<!-- tile-size:end -->'
        require(original.count(anchor) == 1, 'Missing README tile-size anchor')
        index = original.index(anchor)+len(anchor)
        updated = original[:index]+'\n\n'+block+original[index:]
    readme.write_text(updated)
    print(table)
    print(conclusion)
    print('Verified all six results and published the SAHI section; default configuration unchanged.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'reports/course_sahi_merge_v1')
    parser.add_argument('--semantics-audit', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    semantics_audit = args.semantics_audit.resolve() if args.semantics_audit else run/'supplementary_semantics.json'
    publish(run, audit(run, semantics_audit))


if __name__ == '__main__':
    main()
