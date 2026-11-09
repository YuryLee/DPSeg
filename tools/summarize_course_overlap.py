#!/usr/bin/env python3
"""Audit fixed four-tile overlap/size experiments without changing the default."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_course_swinl_ablation import read, sha, require
from tools.summarize_course_swinl_removal import check_sources, record_digest
from tools.evaluate_course_overlap import SEMANTICS, check_dependencies, check_window_configs, check_runtime_stats
METRICS = ('mask_ap_small', 'mask_ar_small', 'mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ap_medium', 'mask_ap_large')
HEADERS = ('APₛ', 'ARₛ', 'AP', 'AP50', 'AP75', 'APₘ', 'APₗ')

def integral(value):
    return type(value) is int and value >= 0


def audit_recovery(run, plan):
    """Verify reused prefixes against the preserved interrupted run, if any."""
    recovery = plan.get('recovery')
    if recovery is None:
        return None
    original = Path(recovery['original_run']).resolve()
    require(original != run.resolve(), 'Recovery must preserve the original run directory')
    protocol = recovery['original_protocol']
    require(Path(protocol['path']).resolve() == original/'protocol.json'
            and sha(protocol['path']) == protocol['sha256'], 'Interrupted protocol changed')
    old = read(protocol['path'])
    fixed = ('archive', 'image_ids', 'weights', 'system_protocol', 'engineering_val_ids',
             'input_keys', 'base_epoch', 'head_epoch', 'precision', 'tf32', 'score_threshold',
             'max_detections', 'camera_intrinsics_used', 'new_variants', 'all_variants',
             'gpus', 'full_reference', 'native_reference', 'current_default_manifest',
             'overlap_configs', 'tile_policy', 'probability_policy', 'guarded_merge_policy',
             'merge_variant', 'views_per_configuration', 'default_at_start', 'default_after_run')
    require(all(plan[key] == old[key] for key in fixed), 'Recovery changed experimental settings')
    require(all(plan['new_source_sha256'].get(name) == checksum
                for name, checksum in old['new_source_sha256'].items()),
            'Recovery changed a frozen inference/evaluation source')
    require('tools/resume_course_overlap.py' in plan['new_source_sha256'],
            'Recovery implementation was not frozen')
    interruption = recovery['interruption']
    require(sha(interruption['path']) == interruption['sha256'], 'Interruption record changed')
    event = read(interruption['path'])
    require(event['exit_code'] == 143, 'Original external termination must remain disclosed')
    evidence = recovery['original_evidence_manifest']
    require(sha(evidence['path']) == evidence['sha256'], 'Original evidence manifest changed')
    files = read(evidence['path'])['files']
    preserved = set()
    for item in files:
        original_path, copied_path = Path(item['original_path']).resolve(), Path(item['copy_path']).resolve()
        require(original_path not in preserved and original_path.is_relative_to(original)
                and copied_path.is_relative_to(run.resolve())
                and sha(original_path) == sha(copied_path) == item['sha256'],
                'Original interruption evidence was altered or not copied exactly')
        preserved.add(original_path)
    prefixes = recovery['prefixes']
    require([row['rank'] for row in prefixes] == list(range(len(plan['gpus']))),
            'Recovery prefixes do not cover every rank exactly once')
    reused = 0
    identity_checks = []
    for prefix in prefixes:
        rank, count = prefix['rank'], prefix['images']
        rank_ids = plan['image_ids'][rank::len(plan['gpus'])]
        require(type(count) is int and 0 < count < len(rank_ids)
                and prefix['image_ids'] == rank_ids[:count], 'Invalid recovery prefix IDs')
        required = [*plan['new_variants'], 'checks']
        require(set(prefix['artifacts']) == set(required), 'Incomplete prefix evidence')
        for name in required:
            source = prefix['artifacts'][name]
            old_path = original/f'{name}_rank{rank}.jsonl'
            prefix_path = Path(source['path']).resolve()
            require(old_path in preserved and prefix_path.is_relative_to(run.resolve())
                    and sha(prefix_path) == source['sha256'], 'Prefix evidence changed')
            with old_path.open() as before, prefix_path.open() as prefix_stream, \
                    (run/f'{name}_rank{rank}.jsonl').open() as after:
                for expected_id in rank_ids[:count]:
                    original_line, prefix_line, recovered_line = before.readline(), prefix_stream.readline(), after.readline()
                    require(original_line.endswith('\n') and prefix_line.endswith('\n') and recovered_line.endswith('\n'),
                            'Recovery contains an incomplete prefix line')
                    original_row, recovered_row = json.loads(original_line), json.loads(recovered_line)
                    require(original_row == json.loads(prefix_line) == recovered_row
                            and original_row['image_id'] == expected_id,
                            'Recovery altered or reordered a completed prefix row')
                require(prefix_stream.read() == '', 'Prefix evidence includes unverified trailing rows')
        receipt = read(run/f'rank{rank}.json')['recovery']
        require(receipt['prefix_images'] == count and receipt['new_images'] == len(rank_ids)-count,
                'Recovery reused/new image count differs')
        checks = receipt['prefix_identity_checks']
        expected_ids = list(dict.fromkeys((rank_ids[0], rank_ids[count-1])))
        require([row['image_id'] for row in checks] == expected_ids
                and all(row['predictions_exact'] and row['native_exact'] and row['reference40_exact']
                        for row in checks), 'Prefix endpoint predictions were not exactly reproduced')
        identity_checks.extend(checks)
        reused += count
    return dict(status='verified', original_run=str(original), original_protocol=protocol,
                interruption=interruption, original_evidence_manifest=evidence,
                reused_images_per_configuration=reused,
                newly_completed_images_per_configuration=len(plan['image_ids'])-reused,
                prefix_identity_checks=identity_checks, original_shards_preserved=True,
                frozen_inference_unchanged=True)

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



def audit(run):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_dpseg import NEW_OVERLAP_VARIANTS, OVERLAP_VARIANTS
    state, plan, result = read(run/'status.json'), read(run/'protocol.json'), read(run/'results.json')
    protocol_sha = sha(run/'protocol.json')
    require(state['status'] == 'completed' and state['exit_code'] == 0
            and result['status'] == 'completed' and result['split'] == 'test' and result['images'] == 2112
            and result['protocol_sha256'] == protocol_sha, 'Experiment incomplete or protocol changed')
    require(not plan['training'] and not result['training'] and not plan['ground_truth_in_prediction']
            and not plan['test_used_for_selection'] and not result['test_used_for_selection']
            and not plan['default_changed_by_experiment'] and not result['default_changed_by_experiment']
            and plan['default_at_start'] == plan['default_after_run'] == result['default_after_run'] == 'overlap40'
            and not plan['speed_benchmark'], 'Training/default/selection/speed scope changed')
    require(plan['new_variants'] == list(NEW_OVERLAP_VARIANTS) and plan['all_variants'] == list(OVERLAP_VARIANTS)
            and set(result['variants']) == set(NEW_OVERLAP_VARIANTS), 'Missing or changed arms')
    require(plan['input_keys'] == ['image', 'depth_m', 'depth_valid'] and not plan['camera_intrinsics_used']
            and plan['base_epoch'] == 15 and plan['head_epoch'] == 6
            and plan['precision'] == 'float32' and plan['tf32'] is False
            and plan['score_threshold'] == .01 and plan['max_detections'] == 100
            and plan['merge_variant'] == 'global_guard_edge_consensus' and plan['views_per_configuration'] == 5,
            'Fixed inference settings changed')
    check_dependencies(plan)
    geometry_check = check_window_configs(plan['overlap_configs'])
    for item in [*plan['weights'].values(), plan['system_protocol']]:
        require(sha(item['path']) == item['sha256'], 'Weights/model protocol changed')
    system = read(plan['system_protocol']['path'])
    check_sources(system['system_source_sha256'])
    require(system['model']['backbone'] == 'large' and system['model']['depth_encoding'] == 'depth'
            and system['base_checkpoint_sha256'] == plan['weights']['base']['sha256'], 'Base identity differs')
    with CourseArchive(plan['archive'], split='test') as data:
        ids = data.image_ids
    require(ids == plan['image_ids'] and len(ids) == len(set(ids)) == 2112, 'Test coverage/order differs')
    full = plan['full_reference']
    require(result['full_reference'] == full and result['reference_variant'] == plan['reference_variant'] == 'overlap40',
            'Reused reference differs')
    reference_root = Path(full['evaluation']).parent.parent
    ref_plan, ref_result = read(reference_root/'protocol.json'), read(reference_root/'results.json')
    require(sha(reference_root/'protocol.json') == plan['reference_run_protocol_sha256'] == ref_result['protocol_sha256']
            and ref_result['status'] == 'completed'
            and ref_result['variants']['global_guard_edge_consensus'] == full
            and ref_plan['weights'] == plan['weights'], 'Reference identity differs')
    ref_summary = plan['reference_run_summary']
    require(sha(ref_summary['path']) == ref_summary['sha256'], 'Reference audit artifact changed')
    prior = read(ref_summary['path'])
    require(prior['status'] == 'verified' and prior['protocol']['sha256'] == ref_result['protocol_sha256'],
            'Reference audit protocol differs')
    prior_full = next(row for row in prior['results'] if row['variant'] == 'global_guard_edge_consensus')
    require(prior_full['metrics'] == full['metrics'] and prior_full['predictions_sha256'] == full['predictions_sha256'],
            'Reference audit prediction differs')
    semantics, full_validation = check_result(full, ids)
    native = plan['native_reference']
    require(sha(native['predictions']) == native['predictions_sha256']
            and sha(native['evaluation']) == native['evaluation_sha256'], 'Native reference changed')
    native_eval = read(native['evaluation'])
    require(native_eval['status'] == 'completed' and native_eval['split'] == 'test' and native_eval['images'] == 2112
            and native_eval['predictions_sha256'] == native['predictions_sha256']
            and {key: native_eval[key] for key in SEMANTICS} == semantics, 'Native semantics differ')
    check_sources(native_eval['source_sha256'])
    expected_exits = {'engineering': {'engineering': 0},
        'inferring': {f'worker{i}': 0 for i in range(len(plan['gpus']))},
        'scoring': {f'score_{variant}': 0 for variant in NEW_OVERLAP_VARIANTS}}
    for phase, expected in expected_exits.items():
        require(read(run/f'{phase}_exits.json') == expected, f'{phase} did not finish successfully')
    gate = read(run/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['split'] == 'val' and not gate['ground_truth_used']
            and not gate['model_or_threshold_changes'] and gate['protocol_sha256'] == protocol_sha
            and gate['geometry_check'] == geometry_check and gate['synthetic']['status'] == 'passed', 'Gate invalid')
    require([row['image_id'] for row in gate['checks']] == plan['engineering_val_ids'] and len(gate['checks']) == 3
            and all(row['native_and_reference40_exact'] and row['score_errors'] == {'reference40': 0., 'native': 0.}
                    for row in gate['checks']), 'Engineering exact comparisons incomplete')
    for row in gate['checks']:
        dummy = {name: [None]*row['instances'][name] for name in NEW_OVERLAP_VARIANTS}
        check_runtime_stats(row['stats'], dummy, plan, True)
    counts, merging = Counter(), {variant: Counter() for variant in NEW_OVERLAP_VARIANTS}
    digests = {variant: {} for variant in NEW_OVERLAP_VARIANTS}
    coverage, reference_checks, rank_receipts = [], [], []
    for rank in range(len(plan['gpus'])):
        rank_ids = ids[rank::len(plan['gpus'])]
        reference_ids = list(dict.fromkeys((rank_ids[0], rank_ids[-1])))
        receipt = read(run/f'rank{rank}.json')
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids
                and receipt['images'] == len(rank_ids) and receipt['native_exact'] and receipt['reference40_exact']
                and receipt['reference40_image_ids'] == reference_ids and not receipt['ground_truth_in_prediction'],
                'Invalid inference rank receipt')
        coverage.extend(rank_ids)
        checks_path = run/f'checks_rank{rank}.jsonl'
        require(sha(checks_path) == receipt['checks_sha256'], 'Per-image checks changed')
        checks = [json.loads(line) for line in checks_path.read_text().splitlines()]
        require([row['image_id'] for row in checks] == rank_ids, 'Per-image check coverage differs')
        observed_reference = []
        for row in checks:
            require(row['native_exact'] and set(row['instances']) == set(NEW_OVERLAP_VARIANTS), 'Native/count audit failed')
            is_reference = row['image_id'] in reference_ids
            require(row['reference40_checked'] is is_reference
                    and (row['reference40_exact'] is True if is_reference else row['reference40_exact'] is None),
                    'Reference checkpoint verification mismatch')
            if is_reference:
                observed_reference.append(row['image_id'])
            dummy = {name: [None]*row['instances'][name] for name in NEW_OVERLAP_VARIANTS}
            check_runtime_stats(row['stats'], dummy, plan, is_reference)
            for name in NEW_OVERLAP_VARIANTS:
                item = row['stats'][name]['merging']
                require(all(integral(value) for value in item.values())
                        and item['selected'] == row['instances'][name] <= 100
                        and item['selected_whole']+item['selected_tiles'] == item['selected']
                        and item['area_filtered']+item['partial_filtered']+item['duplicates_removed']
                           +item['top100_removed']+item['selected'] == item['candidates'], 'Candidate accounting differs')
                merging[name].update(item)
        require(observed_reference == reference_ids, 'Reference endpoint coverage differs')
        reference_checks.extend(reference_ids)
        by_id = {row['image_id']: row for row in checks}
        for variant in NEW_OVERLAP_VARIANTS:
            shard = run/f'{variant}_rank{rank}.jsonl'
            require(sha(shard) == receipt['artifacts'][variant], 'Prediction shard changed')
            observed, total = [], 0
            with shard.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    image_id = row['image_id']
                    require(image_id not in digests[variant], 'Duplicate shard image')
                    observed.append(image_id)
                    require(len(row['predictions']) == by_id[image_id]['instances'][variant], 'Count audit differs')
                    digests[variant][image_id] = record_digest(row)
                    total += len(row['predictions'])
            require(observed == rank_ids and total == receipt['instances'][variant], 'Shard coverage/count differs')
            counts[variant] += total
        rank_receipts.append(dict(rank=rank, gpu=receipt['gpu'], images=len(rank_ids),
            receipt_sha256=sha(run/f'rank{rank}.json'), checks_sha256=receipt['checks_sha256'], reference40_image_ids=reference_ids))
    require(Counter(coverage) == Counter(ids), 'Ranks do not cover every test image exactly once')
    rows = {}
    for variant in NEW_OVERLAP_VARIANTS:
        row, folder = result['variants'][variant], run/variant
        require(row == read(folder/'result.json') and row['variant'] == variant
                and Path(row['evaluation']).resolve() == folder/'evaluation.json'
                and Path(row['predictions']).resolve() == folder/'predictions.jsonl'
                and Path(row['csv']).resolve() == run/'submissions'/f'dpseg_swinl_depth_no_k_{variant}.csv',
                'Result belongs to wrong variant/path')
        _, validation = check_result(row, ids, semantics)
        require(row['predicted_instances'] == counts[variant] == merging[variant]['selected'], 'Total instances differ')
        observed = []
        with Path(row['predictions']).open() as stream:
            for line in stream:
                item = json.loads(line)
                observed.append(item['image_id'])
                require(record_digest(item) == digests[variant].get(item['image_id']), 'Merged row differs from shard')
        require(observed == ids, 'Merged image order differs')
        rows[variant] = dict(row, validation=validation, reused=False, candidate_statistics=dict(merging[variant]))
        print(f'Verified {variant}: 2112 CSV rows, {counts[variant]} instances', flush=True)
    rows['overlap40'] = dict(full, variant='overlap40', validation=full_validation, reused=True,
                            candidate_statistics=prior_full['candidate_statistics'])
    return dict(plan=plan, result=result, protocol_sha256=protocol_sha, rows=[rows[name] for name in OVERLAP_VARIANTS],
        semantics=semantics, gate=gate, geometry_check=geometry_check, rank_receipts=rank_receipts,
        phase_exits=expected_exits, reference40_checked_test_ids=reference_checks,
        recovery=audit_recovery(run, plan))

def publish(run, verified, docs, update_readme):
    plan, rows = verified['plan'], verified['rows']
    baseline = next(row for row in rows if row['variant'] == 'overlap40')
    for row in rows:
        row['geometry'] = plan['overlap_configs'][row['variant']]
        row['current_default'] = row['variant'] == 'overlap40'
        row['delta_vs_current_default'] = {key: row['metrics'][key]-baseline['metrics'][key] for key in METRICS}
    lines = ['| 目标重叠率 | 实际重叠率 | 切片尺寸 | 视图数 | APₛ ↑ | ARₛ ↑ | AP ↑ |',
             '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        g, m = row['geometry'], row['metrics']
        cells = [f'{100*g["target_overlap_ratio"]:.0f}%'+('（当前默认）' if row['current_default'] else ''),
                 f'{100*g["actual_overlap_ratio"]:.4f}%', f'{g["tile_size_px"]}×{g["tile_size_px"]}', '5',
                 *(f'{m[key]:.6f}' for key in METRICS[:3])]
        if row['current_default']:
            cells = [f'**{cell}**' for cell in cells]
        lines.append('| '+' | '.join(cells)+' |')
    table = '\n'.join(lines)
    extra = ['| 目标重叠率 | AP50 ↑ | AP75 ↑ | APₘ ↑ | APₗ ↑ |', '|---:|---:|---:|---:|---:|']
    for row in rows:
        extra.append('| '+f'{100*row["geometry"]["target_overlap_ratio"]:.0f}%'+' | '+
                     ' | '.join(f'{row["metrics"][key]:.6f}' for key in METRICS[3:])+' |')
    extra_table = '\n'.join(extra)
    counters = ['| 目标重叠率 | 候选总数 | 最终实例数 | 整图来源 | 切片来源 | 截断候选过滤 | 保护候选落出top100 |',
                '|---:|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        values = row['candidate_statistics']
        counters.append('| '+f'{100*row["geometry"]["target_overlap_ratio"]:.0f}%'+' | '+
            ' | '.join(str(values[key]) for key in ('candidates','selected','selected_whole','selected_tiles',
                                                    'partial_filtered','protected_whole_lost_to_top100'))+' |')
    count_table = '\n'.join(counters)
    best_ap = max(rows, key=lambda row: row['metrics']['mask_ap'])
    best_small = max(rows, key=lambda row: row['metrics']['mask_ap_small'])
    description = (f'本轮五档中，整体AP最高出现在目标重叠率 {100*best_ap["geometry"]["target_overlap_ratio"]:.0f}%'
                   f'（AP={best_ap["metrics"]["mask_ap"]:.6f}）；APₛ最高出现在 '
                   f'{100*best_small["geometry"]["target_overlap_ratio"]:.0f}%'
                   f'（APₛ={best_small["metrics"]["mask_ap_small"]:.6f}）。'
                   '这是本轮已观察测试集的描述性比较，当前默认保持40%。')
    common = ('固定当前DPSeg完整版：Swin-Large＋无内参深度、整图保护＋截断处理＋跨视图一致性合并、'
              '来源视图点细化及深度引导概率图细化。每档均为整图＋四个四角切片，共5视图，'
              '切片完整覆盖1024×1024原图；各视图统一缩放到1024×1024输入。基座第15轮、点头第6轮，'
              '在完整本地测试集2,112张上比较。40%档复用此前已核验的全部预测。')
    scope = ('固定四片和完整覆盖时，切片尺寸、实际重叠率与目标在输入中的相对尺度一起改变，'
             '因此该实验是不同切片几何设置的条件比较，不能解释为重叠率的独立因果作用。'
             '规则在推理前冻结，没有根据本轮测试结果调参或自动更换默认，不作为验证选出的最优配置。')
    recovery_note = ''
    if verified['recovery']:
        recovery = verified['recovery']
        recovery_note = ('原运行收到外部 SIGTERM（退出码 143），具体来源未确定。'
            f'原始预测与检查分片完整保留并复制留档；四个新配置各复用 '
            f'{recovery["reused_images_per_configuration"]} 张共同完整前缀预测，'
            f'在新目录补齐剩余 {recovery["newly_completed_images_per_configuration"]} 张。'
            '每个分片的前缀首尾图重新推理，四组预测、整图输出和40%参考均精确一致。'
            '恢复使用相同权重、窗口与合并规则，最终按完整2,112张重新评分；'
            '本报告不将原中断记为正常退出。')
    paths = '\n'.join(f'- **目标{100*row["geometry"]["target_overlap_ratio"]:.0f}%**'
                      +('（复用40%已完成结果）' if row['reused'] else '')+f'：`{row["csv"]}`' for row in rows)
    document = f'''# 固定四切片的重叠率与尺寸对照

## Material Passport

- 来源：ARS experiment-agent；`{run}` 的真实推理与COCO评分。
- 核验状态：VERIFIED；协议SHA-256：`{verified['protocol_sha256']}`。
- 范围：固定四切片、完整覆盖的探索性测试集比较；当前默认40%保持不变。

## 主要结果

{common}

{table}

{description}

{scope}

## 切片构造

设原图边长L=1024，目标相邻重叠比例为r，切片边长取 S=round(L/(2-r))。横纵起点均为[0,L-S]，形成固定2×2四角网格。相邻重叠宽度为2S-L，实际重叠比例为(2S-L)/S；整数取整使部分实际比例与目标比例略有差异。每片缩放到1024输入后，相对放大倍数为1024/S。

0%档为四个互不重叠的512×512象限，仍保留整图预测。它与旧切片尺寸实验中“512＋20%名义重叠、多于四片”的设置不同。全部五档都保留整图预测并完整覆盖原图，没有通过省略区域改变评测范围。

## 其他尺寸与IoU指标

{extra_table}

## 候选统计

{count_table}

候选数统计包含跨视图重复，同一物体可能产生多个候选；不能据此把额外候选都判断为误检。整图保护仅按预测分数≥0.5、预测面积>1024确定优先级，不保证掩码正确。最终仍按原始分数排序、每图最多100个实例，故优先选择的保护候选仍可能落出top100。

跨尺寸视图的查询编号不代表相同物体，不能据查询编号将不同重叠配置的新增/移除候选直接一一对应。JSON保留每档原始合并统计，其中added_vs_legacy等字段是该档相同候选集合相对原始掩码NMS的选择差异，不是相对40%配置的物体新增量。

## 核验

{recovery_note}

- 冻结基座、点头、原模型协议、新实验源码及当前默认manifest。manifest绑定的所有源码与提升来源协议均核验哈希。
- CPU独立检查五档四角窗口、相邻重叠、放大倍数、四片完整覆盖和40%窗口与既有640实现一致。
- 三张验证图工程门禁：新入口的40%预测与旧guarded实现逐掩码、逐分数完全一致；native与独立整图入口完全一致。验证图仅用于工程检查，没有用其GT选参数。
- 全部测试图严格核对native结果；每个推理分片首尾重算40%并与此前2112张预测中的相应行精确比较，共{len(verified['reference40_checked_test_ids'])}张。40%整组结果复用已有完整预测，不声称本轮重新推理了全部40%图像。
- 每个新配置逐图验证5视图、窗口几何与实际1024×1024模型输入。RGB、深度、有效性同步裁剪，GT不进入推理。
- 四份新CSV与一份40%参考均绑定实际评分预测；检查全量原生RLE解码receipt、图像顺序、实例数、分片哈希和每行掩码/置信度。
- 各组使用相同COCO掩码指标，IoU=0.50:0.05:0.95，maxDets=[1,10,100]，小目标范围[0,1024]，原GT面积保持不变；预测面积来自独立RLE。
- 多配置仅共享整图特征以减少实验成本，本轮总耗时不用于各配置独立FPS对比。没有多种子、统计显著性或Kaggle成绩结论。

## 复现

```bash
.venv/bin/python -u tools/evaluate_course_overlap.py --output reports/course_swinl_overlap_new --reference-run reports/course_swinl_guarded_merge640_v1 --current-default-manifest configs/dpseg_current.json --gpus 1,3,4,5,6,7
.venv/bin/python tools/summarize_course_overlap.py --run reports/course_swinl_overlap_new --docs docs/results --update-readme
```

按实际空闲GPU调整设备列表，输出目录需尚不存在。入口先执行工程门禁，再进行四档全量推理和统一评分。历史640普通合并、早期切片尺寸实验与本轮当前默认的合并策略不同，不能把结果混为同一组重叠率实验。

## CSV路径

{paths}
'''
    artifact = dict(schema='dpseg-four-tile-overlap-v1', status='verified',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), split='local_test', images=2112,
        backbone='Swin-Large', camera_intrinsics_used=False, training=False, test_used_for_selection=False,
        default_changed_by_experiment=False, current_default='overlap40', speed_benchmark=False,
        interpretation_scope=scope, results=rows, semantics=verified['semantics'], recovery=verified['recovery'],
        protocol=dict(path=str(run/'protocol.json'),sha256=verified['protocol_sha256']),
        current_default_manifest=plan['current_default_manifest'], checkpoints=plan['weights'],
        system_protocol=plan['system_protocol'], source_sha256=plan['new_source_sha256'],
        geometry_check=verified['geometry_check'], engineering_gate=verified['gate'],
        reference_run_summary=plan['reference_run_summary'], reference40_checked_test_ids=verified['reference40_checked_test_ids'],
        phase_exits=verified['phase_exits'], rank_receipts=verified['rank_receipts'],
        publisher_sha256=sha(Path(__file__)),
        material_passport=dict(source=f'academic-research-suite / {run}', verification_status='VERIFIED',
            scope='fixed four-tile coverage comparison with coupled overlap and crop size', limitations=scope))
    targets = [(run/'summary.md',run/'summary.json',run/'summary.csv')]
    if docs is not None:
        docs.mkdir(parents=True,exist_ok=True)
        targets.append((docs/'course_overlap_swinl.md',docs/'course_overlap_swinl.json',docs/'course_overlap_swinl.csv'))
    for md,js,cs in targets:
        md.write_text(document)
        js.write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        with cs.open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['variant','target_overlap_ratio','actual_overlap_ratio','tile_size_px','views',
                             *METRICS,'predicted_instances','reused','current_default'])
            for row in rows:
                g=row['geometry']
                writer.writerow([row['variant'],g['target_overlap_ratio'],g['actual_overlap_ratio'],g['tile_size_px'],5,
                                 *(row['metrics'][key] for key in METRICS),row['predicted_instances'],row['reused'],row['current_default']])
    if update_readme:
        path=ROOT/'README.md'
        original=path.read_text()
        start,end='<!-- tile-overlap:start -->','<!-- tile-overlap:end -->'
        block=f'''{start}
### 固定四切片的重叠率与尺寸对照

{common}

{table}

{description}

{scope}

[完整指标、窗口设置、核验及CSV路径](docs/results/course_overlap_swinl.md) · [指标CSV](docs/results/course_overlap_swinl.csv) · [核验JSON](docs/results/course_overlap_swinl.json)。
{end}'''
        if start in original or end in original:
            require(original.count(start)==original.count(end)==1,'Ambiguous overlap README markers')
            updated=original[:original.index(start)]+block+original[original.index(end)+len(end):]
        else:
            anchor='<!-- tile-size:start -->'
            require(original.count(anchor)==1,'Expected unique historical tile-size section')
            index=original.index(anchor)
            updated=original[:index]+block+'\n\n'+original[index:]
        path.write_text(updated)
    print(table)
    print(description)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'reports/course_swinl_overlap_v1')
    parser.add_argument('--docs',type=Path)
    parser.add_argument('--update-readme',action='store_true')
    args=parser.parse_args()
    run=args.run.resolve()
    publish(run,audit(run),args.docs.resolve() if args.docs else None,args.update_readme)


if __name__=='__main__':
    main()
