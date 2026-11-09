#!/usr/bin/env python3
"""Audit fixed-rule 640-pixel guarded merging and write measured comparisons."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
import hashlib
from zipfile import ZipFile
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_course_swinl_ablation import read, sha, require
from tools.summarize_course_swinl_removal import check_sources, record_digest
from tools.evaluate_course_guarded_merge import SEMANTICS, check_dependencies, baseline_rescore_receipt
METRICS = ('mask_ap_small', 'mask_ar_small', 'mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ap_medium', 'mask_ap_large')
HEADERS = ('APₛ', 'ARₛ', 'AP', 'AP50', 'AP75', 'APₘ', 'APₗ')
LABELS = {
 'legacy_full': '原完整 DPSeg（640，当前默认）', 'without_tiles': '无切片（参考）',
 'global_guard': '可靠整图候选优先', 'global_guard_edge': '整图优先＋截断处理',
 'global_guard_consensus': '整图优先＋跨视图一致性',
 'global_guard_edge_consensus': '整图优先＋截断处理＋一致性',
 'small_tiles_only': '整图优先＋仅补充小目标切片候选',
}

def integral(value):
    return type(value) is int and value >= 0

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



def check_capacity(run, plan, ids, reference_counts):
    """Recompute count-only diagnostics and bind their annotation shard bytes."""
    from rgbd_seg.course_archive import CourseArchive
    counts = dict.fromkeys(ids, 0)
    with CourseArchive(plan['archive'], split='test') as data:
        require(data.image_ids == ids, 'Capacity diagnostic image coverage differs')
        with ZipFile(data.zip_path) as archive:
            shards = {member: hashlib.sha256(archive.read(member)).hexdigest()
                      for member in data.annotation_members}
        truth = data.coco_ground_truth()
        for item in truth['annotations']:
            if not item.get('iscrowd', 0):
                counts[item['image_id']] += 1
    observed = dict(images=len(ids), gt_non_crowd=sum(counts.values()),
        gt_per_image_max=max(counts.values()), gt_per_image_mean=sum(counts.values())/len(ids),
        images_with_gt_over100=sum(count > 100 for count in counts.values()),
        instances_above_image_cap=sum(max(0, count-100) for count in counts.values()),
        original640_images_at100=sum(count == 100 for count in reference_counts.values()),
        original640_predictions_mean=sum(reference_counts.values())/len(ids))
    path = run/'instance_capacity_audit.json'
    if path.exists():
        recorded = read(path)
        require({key: recorded[key] for key in observed} == observed, 'Recorded instance capacity differs')
        record = dict(path=str(path), sha256=sha(path))
    else:
        record = None
    return dict(**observed, scope='Post-hoc count-only diagnostic; no inference rule changes',
        annotation_shard_sha256=shards,
        per_image_non_crowd_counts_sha256=hashlib.sha256(json.dumps(counts, sort_keys=True).encode()).hexdigest(),
        reference_predictions_sha256=plan['full_reference']['predictions_sha256'], original_record=record)

def audit(run):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_guarded_merge import VARIANTS
    from rgbd_seg.course_tiled_probability import TILE_POLICY, PROBABILITY_POLICY, view_bounds

    state, plan, result = read(run/'status.json'), read(run/'protocol.json'), read(run/'results.json')
    require(state['status'] == 'completed' and state['exit_code'] == 0, 'Experiment has not completed')
    protocol_sha = sha(run/'protocol.json')
    require(result['status'] == 'completed' and result['split'] == 'test' and result['images'] == 2112
            and result['protocol_sha256'] == protocol_sha, 'Incomplete result or changed protocol')
    require(not result['training'] and not result['test_used_for_selection'] and not result['default_changed']
            and not plan['training'] and not plan['test_used_for_selection'] and not plan['default_changed']
            and not plan['ground_truth_in_prediction'] and not plan['speed_benchmark'],
            'Unexpected training, test selection, GT input, default or speed scope')
    require(plan['variants'] == list(VARIANTS) and set(result['variants']) == set(VARIANTS), 'Missing variants')
    require(plan['input_keys'] == ['image', 'depth_m', 'depth_valid']
            and not plan['camera_intrinsics_used'] and plan['tile_size_px'] == 640
            and plan['base_epoch'] == 15 and plan['head_epoch'] == 6
            and plan['precision'] == 'float32' and plan['tf32'] is False
            and plan['score_threshold'] == .01 and plan['max_detections'] == 100, 'Fixed settings changed')
    check_dependencies(plan)
    for item in [*plan['weights'].values(), plan['system_protocol']]:
        require(sha(item['path']) == item['sha256'], 'Fixed weights or model protocol changed')
    system = read(plan['system_protocol']['path'])
    check_sources(system['system_source_sha256'])
    require(system['model']['backbone'] == 'large' and system['model']['depth_encoding'] == 'depth'
            and system['base_checkpoint_sha256'] == plan['weights']['base']['sha256'], 'Base identity differs')
    require(plan['view_bounds'] == json.loads(json.dumps(view_bounds(1024, 1024))), 'Window geometry changed')
    expected_policy = {**TILE_POLICY, 'geometry': 'aligned RGB/depth/valid crop; no camera K or intrinsics used'}
    require(plan['tile_policy'] == expected_policy and plan['probability_policy'] == PROBABILITY_POLICY,
            'Fixed candidate/refinement policy changed')
    with CourseArchive(plan['archive'], split='test') as data:
        ids = data.image_ids
    require(ids == plan['image_ids'] and len(ids) == len(set(ids)) == 2112, 'Test coverage/order differs')
    full, no_tiles = plan['full_reference'], plan['without_tiles_reference']
    require(result['full_reference'] == full and result['without_tiles_reference'] == no_tiles,
            'Final reference identities differ')
    reference_root = Path(full['evaluation']).parent.parent
    reference_plan, reference_result = read(reference_root/'protocol.json'), read(reference_root/'results.json')
    require(sha(reference_root/'protocol.json') == plan['reference_run_protocol_sha256']
            == reference_result['protocol_sha256'] and reference_result['status'] == 'completed'
            and reference_result['variants']['full'] == full
            and reference_result['variants']['without_tiles'] == no_tiles
            and reference_plan['weights'] == plan['weights'], 'Reference run identity differs')
    semantics, full_validation = check_result(full, ids)
    _, no_tiles_validation = check_result(no_tiles, ids, semantics)
    rescore = baseline_rescore_receipt(Path(plan['baseline_rescore']['path']), full)
    require(rescore == plan['baseline_rescore'] == result['baseline_rescore'], 'Baseline rescore differs')
    require({key: read(rescore['receipt']['evaluation'])[key] for key in SEMANTICS} == semantics,
            'Rescored baseline settings/GT differ')
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
    capacity = check_capacity(run, plan, ids, reference_counts)
    expected_exits = {'engineering': {'engineering': 0},
        'inferring': {f'worker{i}': 0 for i in range(len(plan['gpus']))},
        'scoring': {f'score_{variant}': 0 for variant in VARIANTS}}
    for phase, values in expected_exits.items():
        require(read(run/f'{phase}_exits.json') == values, f'{phase} did not finish successfully')
    gate = read(run/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['split'] == 'val' and not gate['ground_truth_used']
            and not gate['model_or_threshold_changes'] and gate['protocol_sha256'] == protocol_sha,
            'Invalid engineering gate')
    require([row['image_id'] for row in gate['checks']] == plan['engineering_val_ids']
            and len(gate['checks']) == 3 and all(row['native_and_legacy_exact']
                and row['score_errors'] == {'legacy_full': 0., 'native': 0.}
                and row['stats']['independent_refinement_verified'] for row in gate['checks']),
            'Engineering reference/independent-refinement comparisons incomplete')
    require(gate['synthetic']['status'] == 'passed', 'Synthetic checks incomplete')
    digests = {variant: {} for variant in VARIANTS}
    counts, candidates, union_refined, rank_receipts, coverage = Counter(), 0, 0, [], []
    merging = {variant: Counter() for variant in VARIANTS}
    for rank in range(len(plan['gpus'])):
        rank_ids = ids[rank::len(plan['gpus'])]
        receipt = read(run/f'rank{rank}.json')
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids
                and receipt['images'] == len(rank_ids) and receipt['native_exact'] and receipt['legacy_full_exact']
                and not receipt['ground_truth_in_prediction'], 'Invalid rank receipt')
        coverage.extend(receipt['image_ids'])
        checks_path = run/f'checks_rank{rank}.jsonl'
        require(sha(checks_path) == receipt['checks_sha256'], 'Per-image check file changed')
        checks = [json.loads(line) for line in checks_path.read_text().splitlines()]
        require([row['image_id'] for row in checks] == rank_ids, 'Per-image check coverage differs')
        for row in checks:
            require(row['native_exact'] and row['legacy_full_exact'], 'Native/full reproduction failed')
            stats = row['stats']
            require(stats['model_views'] == 5 and integral(stats['candidates']) and stats['candidates'] <= 500
                    and integral(stats['union_refined']) and stats['union_refined'] <= stats['candidates'],
                    'Wrong view/candidate counts')
            require(stats['original_merging']['candidates'] == stats['candidates']
                    and len(stats['original_merging']['selected_views']) == reference_counts[row['image_id']],
                    'Legacy candidate accounting differs')
            require(stats['output_counts'] == row['instances'] and set(row['instances']) == set(VARIANTS)
                    and all(integral(count) and count <= 100 for count in row['instances'].values()),
                    'Output count audit differs')
            require(set(stats['merging']) == set(VARIANTS), 'Missing merging statistics')
            for variant, item in stats['merging'].items():
                require(all(integral(value) for value in item.values()), 'Invalid candidate statistic')
                require(item['selected'] == row['instances'][variant]
                        == item['selected_whole']+item['selected_tiles']
                        == item['added_vs_legacy']+item['shared_vs_legacy']
                        and item['removed_vs_legacy']+item['shared_vs_legacy'] == reference_counts[row['image_id']]
                        and item['selected_tile_small'] <= item['selected_tiles']
                        and item['protected_whole_selected'] <= item['selected_whole']
                        and item['area_filtered']+item['partial_filtered']+item['duplicates_removed']
                            +item['top100_removed']+item['selected'] == item['candidates'],
                        'Candidate identity/source count mismatch')
                merging[variant].update(item)
            candidates += stats['candidates']
            union_refined += stats['union_refined']
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
    require(Counter(coverage) == Counter(ids), 'Ranks do not cover every test image exactly once')
    rows = [dict(full, variant='legacy_full', method=LABELS['legacy_full'], validation=full_validation)]
    for variant in VARIANTS:
        row = result['variants'][variant]
        folder = run/variant
        require(row == read(folder/'result.json') and row['variant'] == variant
                and Path(row['evaluation']).resolve() == folder/'evaluation.json'
                and Path(row['predictions']).resolve() == folder/'predictions.jsonl'
                and Path(row['csv']).resolve() == run/'submissions'/f'dpseg_swinl_depth_no_k_guarded640_{variant}.csv',
                'Result belongs to the wrong variant/path')
        _, validation = check_result(row, ids, semantics)
        require(row['predicted_instances'] == counts[variant] == merging[variant]['selected'],
                'Merged instance count differs')
        observed = []
        with Path(row['predictions']).open() as stream:
            for line in stream:
                item = json.loads(line)
                observed.append(item['image_id'])
                require(record_digest(item) == digests[variant].get(item['image_id']),
                        'Scored prediction differs from inference shard')
        require(observed == ids, 'Merged IDs/order differ')
        rows.append(dict(row, method=LABELS[variant], validation=validation,
                         candidate_statistics=dict(merging[variant])))
        print(f'Verified {variant}: 2112 CSV rows, {counts[variant]} instances', flush=True)
    reference = dict(no_tiles, method=LABELS['without_tiles'], validation=no_tiles_validation)
    return dict(plan=plan, result=result, protocol_sha256=protocol_sha, rows=rows, reference_without_tiles=reference,
        semantics=semantics, candidates=candidates, union_refined=union_refined, rank_receipts=rank_receipts,
        phase_exits=expected_exits, gate=gate, baseline_rescore=rescore, instance_capacity=capacity)

def save_summary(run, verified, docs=None):
    plan, rows, no_tiles = verified['plan'], verified['rows'], verified['reference_without_tiles']
    baseline = rows[0]['metrics']
    capacity = verified['instance_capacity']
    for row in rows:
        row['delta_vs_legacy'] = {metric: row['metrics'][metric]-baseline[metric] for metric in METRICS}
        row['delta_vs_without_tiles'] = {metric: row['metrics'][metric]-no_tiles['metrics'][metric]
                                         for metric in METRICS}
    table_lines = ['| 方法 | '+' | '.join(name+' ↑' for name in HEADERS)+' |',
                   '|---|'+'---:|'*len(METRICS)]
    for row in [*rows, no_tiles]:
        table_lines.append('| '+row['method']+' | '+' | '.join(f'{row["metrics"][key]:.6f}' for key in METRICS)+' |')
    table = '\n'.join(table_lines)
    count_lines = ['| 配置 | 输出实例数 | 整图来源 | 切片来源 | 新增候选 | 移除候选 | 共用候选 | 保护候选落出top100 |',
                   '|---|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows[1:]:
        item = row['candidate_statistics']
        count_lines.append('| '+row['method']+' | '+' | '.join(str(item[key]) for key in
            ('selected','selected_whole','selected_tiles','added_vs_legacy','removed_vs_legacy','shared_vs_legacy',
             'protected_whole_lost_to_top100'))+' |')
    count_table = '\n'.join(count_lines)
    rules = '\n'.join(f'- **{LABELS[key]}**：{value}' for key,value in plan['variant_descriptions'].items())
    csv_paths = '\n'.join(f'- **{row["method"]}**：`{row["csv"]}`' for row in [*rows, no_tiles])
    both = [row for row in rows[1:] if row['metrics']['mask_ap'] > baseline['mask_ap']
            and row['metrics']['mask_ap_small'] > baseline['mask_ap_small']]
    observed = ('相对原640切片，本轮同时提高 AP 与 APₛ 的固定配置为：'+ '、'.join(row['method'] for row in both)+'。'
                if both else '本轮五种固定规则中，没有配置同时超过原640切片的 AP 与 APₛ。')
    best_ap = max(rows[1:], key=lambda row: row['metrics']['mask_ap'])
    best_small = max(rows[1:], key=lambda row: row['metrics']['mask_ap_small'])
    comparison = (f"本轮新规则中，整体AP最高为 **{best_ap['method']}**：AP={best_ap['metrics']['mask_ap']:.6f}，"
                  f"相对原640的 APₛ 变化为 {best_ap['delta_vs_legacy']['mask_ap_small']:+.6f}。"
                  f"APₛ最高为 **{best_small['method']}**：APₛ={best_small['metrics']['mask_ap_small']:.6f}，"
                  f"整体AP={best_small['metrics']['mask_ap']:.6f}。两个目标应分别比较，不据单一指标称为统一最优。")
    above_no_tiles = [row['method'] for row in rows[1:] if row['delta_vs_without_tiles']['mask_ap'] > 0
                      and row['delta_vs_without_tiles']['mask_ap_small'] > 0]
    comparison += ('相对无切片参考，同时提高AP和APₛ的配置为：'+'、'.join(above_no_tiles)+'。'
                   if above_no_tiles else '本轮没有新规则同时超过无切片参考的AP和APₛ。')
    document = f'''# 640×640 切片候选合并对照

## Material Passport

- 来源：ARS experiment-agent；`{run}` 的真实推理、逐图核验与 COCO 评测。
- 核验状态：VERIFIED；协议 SHA-256：`{verified['protocol_sha256']}`。
- 范围：固定权重、固定规则的探索性测试集对比；不作为验证集选出的最优配置，不替换默认。

## 指标

固定 Swin-Large＋无内参深度，第15轮基座与第6轮点头，完整本地测试集2,112张。保留整图＋四个640×640切片、来源视图点细化及深度引导概率图细化。各组预测分数不重新计算，最终仍按原分数排序，每图最多100个实例，阈值0.01。

{table}

{observed}

{comparison}

无切片行复用已完成的同权重结果，作为整体AP与小目标指标的参考；该行不属于本轮新合并规则。分数范围0–1，整体AP并非各尺寸AP的简单平均。

## 规则

{rules}

实际规则及全部常量以配套JSON中的 `guarded_merge_policy` 为准。“可靠”仅指预测分数≥0.5且预测面积>1024的启发式条件，不保证掩码正确；跨视图一致性也不等价于GT准确率。整图保护指冲突选择时优先保留可靠中大目标；最终原分数排序与100实例上限可能使已选保护候选落出输出，相关次数单独记录。边缘与一致性系数只影响候选选择优先级，不修改交付CSV中的置信度。细化与掩码表示沿用原实现，无多边形转换、掩码并集或新增训练。

## 候选变化

{count_table}

新增、移除与共用按同一图像的来源视图/查询编号计算，相对于原640切片候选集合；不是正确检出数、误检数或唯一物体数。相同物体可能在多个视图中产生候选。全部五视图共生成 {verified['candidates']:,} 条候选，跨变体共享细化的候选并集共 {verified['union_refined']:,} 条。各组边缘过滤、面积过滤、去重及top100截断统计详见JSON。

## 100实例上限诊断

本地测试集每图非crowd GT最多 {capacity['gt_per_image_max']} 个，平均 {capacity['gt_per_image_mean']:.4f} 个；超过100个GT的图像为 {capacity['images_with_gt_over100']} 张。原640切片有 {capacity['original640_images_at100']} 张图输出达到100个实例，平均输出 {capacity['original640_predictions_mean']:.4f} 个。因此，在本地测试集上，100上限没有因为真实实例数超过限制而造成必然漏检，但候选排序和重复预测仍可能影响哪些实例进入最终输出。这些计数不能证明多出的候选都是误检。GT仅用于推理后的数量诊断，未用于候选选择；标注分片与参考预测哈希见JSON。

## 核验与范围

- 固定模型与实验源码、权重和协议哈希；RGB、深度、有效性是唯一推理输入，无GT输入。
- 三张验证图仅用于工程门禁；原native、原完整输出以及各变体独立细化与共享缓存细化逐掩码、逐分数完全一致。
- 全部2,112张测试图逐图复现原native及原完整640结果，评分前检查所有分片ID、顺序、实例数及哈希。
- 五份新CSV以及两份参考结果均检查全量RLE解码记录，并重新逐行绑定CSV与实际评分输入。
- 所有组使用相同COCO掩码评测：IoU=0.50:0.05:0.95，maxDets=[1,10,100]，小目标范围[0,1024]，保留原GT面积；预测面积来自独立RLE，不传预测bbox。
- 原640基线复用已独立重评分的核验：全部12项COCO指标完全一致，预测、评分源码与核验文件哈希再次检查。
- 规则在本轮推理前冻结；这是在已查看过的本地测试集上进行的探索性对照，没有根据本轮测试结果继续调阈值或替换默认。没有多种子、显著性或Kaggle分数结论。
- 共享视图特征和候选细化仅用于节省实验时间，本轮耗时不能作为各变体独立推理速度。

## 复现

```bash
.venv/bin/python -u tools/evaluate_course_guarded_merge.py --output reports/course_swinl_guarded_merge640_new --gpus 1,3,4,5,6,7
.venv/bin/python tools/summarize_course_guarded_merge.py --run reports/course_swinl_guarded_merge640_new --docs docs/results
```

输出目录需尚不存在，按实际空闲GPU调整设备列表。入口先检查每张卡至少24GiB空闲显存，并执行工程门禁；不终止其他进程。重评分参考文件默认复用本机 `reports/course_sahi_merge_v1/reference_rescore/verification.json`，它只是一份已有指标核验文件，本轮无SAHI运行依赖。

## CSV 路径

{csv_paths}
'''
    artifact = dict(schema='dpseg-guarded-merge640-v1', status='verified',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), split='local_test', images=2112,
        backbone='Swin-Large', camera_intrinsics_used=False, training=False, test_used_for_selection=False,
        default_changed=False, speed_benchmark=False, interpretation_scope=plan['test_analysis'],
        semantics=verified['semantics'], results=rows, reference_without_tiles=no_tiles,
        guarded_merge_policy=plan['guarded_merge_policy'], variant_descriptions=plan['variant_descriptions'],
        candidate_occurrences=verified['candidates'], union_refined_occurrences=verified['union_refined'],
        instance_capacity=capacity,
        synthetic_checks=verified['gate']['synthetic'], engineering_gate=verified['gate'],
        baseline_rescore=verified['baseline_rescore'], checkpoints=plan['weights'],
        protocol=dict(path=str(run/'protocol.json'), sha256=verified['protocol_sha256']),
        source_sha256=plan['new_source_sha256'], system_protocol=plan['system_protocol'],
        phase_exits=verified['phase_exits'], rank_receipts=verified['rank_receipts'],
        summary_source_sha256=sha(Path(__file__)),
        material_passport=dict(source=f'academic-research-suite / {run}', verification_status='VERIFIED',
            scope='fixed-weight 640-pixel candidate-selection comparison',
            limitations='Exploratory examined-test comparison; no statistical significance, speed or validation-optimal claim'))
    targets = [(run/'summary.md',run/'summary.json',run/'summary.csv')]
    if docs is not None:
        docs.mkdir(parents=True, exist_ok=True)
        targets.append((docs/'course_guarded_merge640.md',docs/'course_guarded_merge640.json',docs/'course_guarded_merge640.csv'))
    for markdown_path, json_path, csv_path in targets:
        markdown_path.write_text(document)
        json_path.write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
        with csv_path.open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['variant','method',*METRICS,'predicted_instances'])
            for row in [*rows,no_tiles]:
                writer.writerow([row['variant'],row['method'],*(row['metrics'][key] for key in METRICS),row['predicted_instances']])
    print(table)
    print(observed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'reports/course_swinl_guarded_merge640_v1')
    parser.add_argument('--docs', type=Path, help='Optional docs/results destination; no README or git changes')
    args = parser.parse_args()
    run = args.run.resolve()
    save_summary(run, audit(run), args.docs.resolve() if args.docs else None)


if __name__ == '__main__':
    main()
