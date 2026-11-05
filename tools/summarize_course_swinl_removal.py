#!/usr/bin/env python3
"""Audit fixed-weight module removals and publish a separate README table."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
VARIANTS = ('full', 'without_point', 'without_probability', 'without_tiles')
LABELS = {'full': '完整 DPSeg', 'without_point': '去除深度感知点细化',
          'without_probability': '去除深度引导概率图细化', 'without_tiles': '去除切片增强推理'}
METRICS = ('mask_ap_small', 'mask_ar_small', 'mask_ap')
SEMANTICS = ('iou_type', 'max_detections', 'iou_thresholds', 'area_ranges',
             'ground_truth_area', 'prediction_area', 'gt_instances', 'gt_small_including_crowd')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def record_digest(row):
    payload = {'image_id': row['image_id'], 'predictions': row['predictions']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, allow_nan=False).encode()).hexdigest()


def source_path(name):
    path = Path(name)
    if path.is_absolute():
        return path
    return ROOT/path if (ROOT/path).is_file() else ROOT/'rgbd_seg'/path


def check_sources(values):
    for name, checksum in values.items():
        require(sha(source_path(name)) == checksum, f'Source changed: {name}')


def check_prediction_reproduction(actual, reference, ids):
    with Path(actual).open() as first, Path(reference).open() as second:
        count = 0
        for a, b in zip_longest(first, second):
            require(a is not None and b is not None, 'Reference row count differs')
            left, right = json.loads(a), json.loads(b)
            require(count < len(ids) and left['image_id'] == right['image_id'] == ids[count],
                    'Reference image ID/order differs')
            require(left['predictions'] == right['predictions'], 'Reference masks/scores/order differ')
            count += 1
    require(count == len(ids), 'Reference comparison is incomplete')
    return count


def audit(run):
    from rgbd_seg.course_archive import CourseArchive
    from tools.audit_course_point_refinement import compare_csv_predictions

    plan, results, status = (read(run/name) for name in ('protocol.json', 'results.json', 'status.json'))
    protocol_sha = sha(run/'protocol.json')
    require(status['status'] == 'completed' and status['exit_code'] == 0, 'Controller not complete')
    require(results['status'] == 'completed' and results['split'] == 'test' and results['images'] == 2112,
            'Incomplete full-test results')
    require(results['protocol_sha256'] == protocol_sha, 'Protocol differs from completed experiment')
    require(not results['training'] and not results['test_used_for_selection']
            and not plan['training'] and not plan['test_used_for_selection']
            and not plan['ground_truth_in_prediction'] and not plan['camera_intrinsics_used'],
            'Unexpected training, test selection or inference input policy')
    require(plan['input_keys'] == ['image', 'depth_m', 'depth_valid']
            and plan['base_epoch'] == 15 and plan['head_epoch'] == 6
            and plan['precision'] == 'float32' and plan['tf32'] is False
            and plan['score_threshold'] == .01 and plan['max_detections'] == 100,
            'Fixed inference settings differ')
    expected = {name: dict(rgbd=True, point=name != 'without_point',
                          probability=name != 'without_probability', tiles=name != 'without_tiles')
                for name in VARIANTS}
    require(plan['variants'] == expected and set(results['variants']) == set(VARIANTS),
            'Expected independent removal of exactly one module')
    require(plan['deferred'] == results['deferred'] == ['without_rgbd_fusion'], 'Deferred scope differs')
    check_sources(plan['new_source_sha256'])
    system_item = plan['system_protocol']
    require(sha(system_item['path']) == system_item['sha256'], 'Original system protocol changed')
    system = read(system_item['path'])
    check_sources(system['system_source_sha256'])
    require(system['model']['backbone'] == 'large' and system['model']['depth_encoding'] == 'depth',
            'Expected Swin-Large and no-intrinsics depth')
    for name, item in plan['weights'].items():
        require(sha(item['path']) == item['sha256'], f'Fixed {name} checkpoint changed')
    require(system['base_checkpoint_sha256'] == plan['weights']['base']['sha256'], 'Base identity differs')
    selection = read(Path(system_item['path']).parent/'points6/selection.json')
    require(selection['status'] == 'locked' and not selection['test_used']
            and selection['selected']['epoch'] == 6
            and selection['selected_head_sha256'] == plan['weights']['heads']['sha256'],
            'Point head is not the validation-selected checkpoint')
    with CourseArchive(plan['archive'], split='test') as data:
        ids = data.image_ids
    require(len(ids) == len(set(ids)) == 2112 and ids == plan['image_ids'], 'Test coverage/order differs')
    gate = read(run/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['split'] == 'val'
            and gate['protocol_sha256'] == protocol_sha and not gate['ground_truth_used']
            and not gate['parameter_changes'], 'Engineering comparison invalid')
    require([row['image_id'] for row in gate['checks']] == plan['engineering_val_ids']
            and len(gate['checks']) == 3, 'Engineering validation IDs differ')
    for row in gate['checks']:
        require(row['exact_masks_and_scores'] and row['score_errors'] == dict.fromkeys(VARIANTS, 0.),
                'Engineering prediction comparison failed')
    phase_exits = {'engineering': {'engineering': 0},
                   'inferring': {f'worker{i}': 0 for i in range(len(plan['gpus']))},
                   'scoring': {f'score_{variant}': 0 for variant in VARIANTS}}
    for phase, expected_exits in phase_exits.items():
        require(read(run/f'{phase}_exits.json') == expected_exits, f'{phase} did not finish successfully')

    digests = {variant: {} for variant in VARIANTS}
    counts = Counter()
    checks, receipts, coverage = {}, [], []
    for rank in range(len(plan['gpus'])):
        receipt = read(run/f'rank{rank}.json')
        rank_ids = ids[rank::len(plan['gpus'])]
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids
                and receipt['images'] == len(rank_ids) and receipt['full_and_without_tiles_exact']
                and not receipt['ground_truth_in_prediction'], f'Rank {rank} coverage/status invalid')
        coverage.extend(receipt['image_ids'])
        check_path = run/f'checks_rank{rank}.jsonl'
        require(sha(check_path) == receipt['checks_sha256'], f'Rank {rank} checks changed')
        with check_path.open() as stream:
            records = [json.loads(line) for line in stream]
        require([row['image_id'] for row in records] == rank_ids, 'Per-image check coverage differs')
        for row in records:
            require(row['full_and_without_tiles_exact'] and row['unchanged_tiled_candidate_scores'],
                    'Per-image fixed-candidate check failed')
            checks[row['image_id']] = row
        for variant in VARIANTS:
            path = run/f'{variant}_rank{rank}.jsonl'
            require(sha(path) == receipt['artifacts'][variant], f'{variant} rank {rank} shard changed')
            observed, total = [], 0
            with path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    image_id = row['image_id']
                    require(image_id not in digests[variant], 'Duplicate shard image')
                    observed.append(image_id)
                    values = row['predictions']
                    require(len(values) <= 100, 'Too many instances')
                    if variant == 'without_tiles':
                        require(len(values) == checks[image_id]['whole_count'], 'Whole candidate count changed')
                    else:
                        expected_scores = [(1, item[0]) for item in checks[image_id]['coarse_selected']]
                        require([(p['category_id'], p['score']) for p in values] == expected_scores,
                                'Tiled candidate categories/scores/order changed')
                    digests[variant][image_id] = record_digest(row)
                    total += len(values)
            require(observed == rank_ids and total == receipt['instances'][variant], 'Shard coverage/count differs')
            counts[variant] += total
        receipts.append(dict(rank=rank, images=len(rank_ids), gpu=receipt['gpu'],
                             receipt_sha256=sha(run/f'rank{rank}.json'), checks_sha256=receipt['checks_sha256']))
    require(Counter(coverage) == Counter(ids), 'Ranks do not cover test set exactly once')

    rows, reference_semantics = [], None
    for variant in VARIANTS:
        folder = run/variant
        result, evaluation, validation = (read(folder/name) for name in
                                         ('result.json', 'evaluation.json', 'csv_validation.json'))
        require(result == results['variants'][variant] and result['status'] == 'completed'
                and result['variant'] == variant and result['images'] == 2112, 'Result summary differs')
        prediction_path, csv_path = Path(result['predictions']), Path(result['csv'])
        require(prediction_path.resolve() == (folder/'predictions.jsonl').resolve()
                and Path(result['evaluation']).resolve() == (folder/'evaluation.json').resolve(),
                'Result points outside its expected prediction/evaluation folder')
        require(sha(folder/'evaluation.json') == result['evaluation_sha256'], 'Evaluation changed')
        require(sha(prediction_path) == result['predictions_sha256'] == evaluation['predictions_sha256'],
                'Scored prediction hash differs')
        require(evaluation['status'] == 'completed' and evaluation['split'] == 'test'
                and evaluation['images'] == 2112 and evaluation['depth_used']
                and evaluation['predicted_instances'] == result['predicted_instances'] == counts[variant]
                and evaluation['metrics'] == result['metrics'], 'Evaluation coverage/metrics differ')
        check_sources(evaluation['source_sha256'])
        semantics = {key: evaluation[key] for key in SEMANTICS}
        require(semantics['iou_type'] == 'segm' and semantics['max_detections'] == [1, 10, 100]
                and semantics['area_ranges'][1] == [0, 1024]
                and semantics['ground_truth_area'] == 'original annotation area preserved'
                and semantics['prediction_area'] == 'independent COCO RLE; no bbox passed to loadRes',
                'Incorrect COCO evaluation settings')
        require(len(semantics['iou_thresholds']) == 10 and all(
            abs(value-(.5+.05*i)) < 1e-12 for i, value in enumerate(semantics['iou_thresholds'])),
            'Incorrect IoU thresholds')
        reference_semantics = semantics if reference_semantics is None else reference_semantics
        require(semantics == reference_semantics, 'COCO semantics or GT statistics differ')
        require(all(isinstance(result['metrics'][metric], (int, float))
                    and 0 <= result['metrics'][metric] <= 1 for metric in METRICS), 'Invalid report metric')
        with prediction_path.open() as stream:
            observed = []
            for line in stream:
                row = json.loads(line)
                observed.append(row['image_id'])
                require(record_digest(row) == digests[variant][row['image_id']], 'Merged prediction differs from shard')
        require(observed == ids, 'Merged prediction coverage/order differs')
        require(sha(csv_path) == result['csv_sha256'] == validation['sha256']
                and Path(validation['path']).resolve() == csv_path.resolve(), 'CSV validation hash/path differs')
        require(validation['rows'] == 2112 and validation['predictions'] == counts[variant]
                and validation['image_size'] == [1024, 1024] and validation['category_ids'] == [1]
                and validation['max_detections'] == 100 and validation['id_order_verified']
                and validation['rle_runs_verified'] and validation['rle_native_decode_verified'],
                'CSV validation is incomplete')
        check_sources(validation['source_sha256'])
        binding = compare_csv_predictions(csv_path, prediction_path, ids)
        require(binding == result['csv_rows_identical_to_scored_predictions'] == 2112, 'CSV binding incomplete')
        reproduced = None
        if variant in plan['references']:
            reference = plan['references'][variant]
            require(sha(reference['predictions']) == reference['predictions_sha256']
                    and sha(reference['evaluation']) == reference['evaluation_sha256'], 'Original reference changed')
            original = read(reference['evaluation'])
            require(original['predictions_sha256'] == reference['predictions_sha256']
                    and {key: original[key] for key in SEMANTICS} == semantics
                    and original['metrics'] == result['metrics'], 'Reproduced COCO results differ')
            reproduced = check_prediction_reproduction(prediction_path, reference['predictions'], ids)
        rows.append(dict(result, method=LABELS[variant],
                         csv_validation_sha256=sha(folder/'csv_validation.json'),
                         exact_reference_rows=reproduced))
        print(f'Verified {variant}: {binding} CSV rows, {counts[variant]} instances', flush=True)
    cumulative_path = ROOT/'docs/results/course_ablation_swinl.json'
    cumulative = read(cumulative_path)
    stage_rows = {row['stage']: row for row in cumulative['results']}
    for stage in ('S1', 'S2', 'S3', 'S4'):
        row = stage_rows[stage]
        require(sha(row['evaluation']) == row['evaluation_sha256']
                and read(row['evaluation'])['metrics'] == row['metrics'], 'Cumulative comparison changed')
    require(stage_rows['S4']['metrics'] == rows[0]['metrics']
            and stage_rows['S3']['metrics'] == rows[3]['metrics'], 'Cumulative/removal result identities differ')
    return dict(plan=plan, results=results, rows=rows, semantics=reference_semantics,
                rank_receipts=receipts, phase_exits=phase_exits, protocol_sha256=protocol_sha,
                cumulative_sha256=sha(cumulative_path), cumulative_rows=stage_rows)


def publish(run, verified):
    rows, plan = verified['rows'], verified['plan']
    full = rows[0]['metrics']
    deltas = {row['variant']: {key: full[key]-row['metrics'][key] for key in METRICS} for row in rows[1:]}
    table = ['| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |', '|:---:|:---:|:---:|:---:|']
    for row in rows:
        cells = [row['method'], *(f'{row["metrics"][key]:.6f}' for key in METRICS)]
        if row['variant'] == 'full':
            cells = [f'**{cell}**' for cell in cells]
        table.append('| '+' | '.join(cells)+' |')
    table = '\n'.join(table)
    setup = ('本表从完整 DPSeg 出发，每次独立去除一项模块，其余组件与权重保持不变。'
             '统一使用 **Swin-Large＋无内参深度、完整本地测试集 2,112 张图**，数值范围 0–1。'
             '这是固定权重的推理消融，各去除配置独立比较；RGB-D 融合的去除实验暂缓。加粗仅标记完整配置。')
    point, probability, tiles = (deltas[key] for key in VARIANTS[1:])
    cumulative = verified['cumulative_rows']
    cumulative_point = cumulative['S2']['metrics']['mask_ap_small']-cumulative['S1']['metrics']['mask_ap_small']
    interpretation = f'''- **点细化：** 完整配置相对去除点细化的 APₛ、ARₛ、AP 分别增加 {point['mask_ap_small']:.6f}、{point['mask_ar_small']:.6f}、{point['mask_ap']:.6f}。这是点头在包含切片与概率图细化的组合中的贡献；累计表 S1→S2 的 APₛ 仍下降 {abs(cumulative_point):.6f}，两项观察对应不同上下文，不能互相替代。
- **概率图细化：** 完整配置相对去除此模块的 APₛ、ARₛ、AP 分别增加 {probability['mask_ap_small']:.6f}、{probability['mask_ar_small']:.6f}、{probability['mask_ap']:.6f}。
- **切片：** 完整配置相对无切片的 APₛ、ARₛ 分别增加 {tiles['mask_ap_small']:.6f}、{tiles['mask_ar_small']:.6f}，整体 AP 则下降 {abs(tiles['mask_ap']):.6f}。切片改善本任务的小目标指标，但本次结果不支持其同时改善整体 AP。'''
    limits = ('所有结果来自同一次 seed=17 训练：共用第 15 轮 RGB-D 基座，保留点细化时使用同一验证集选取的第 6 轮点头。'
              '没有针对各去除配置重新训练或调参，测试标注只用于评分，结果不表示重新训练后的架构优劣，也不构成统计显著性结论。')
    run_label = str(run.relative_to(ROOT)) if run.is_relative_to(ROOT) else str(run)
    passport = dict(source=f'本地冻结实验 {run_label}；academic-research-suite 实验流程',
                    scope='固定权重、逐项去除点细化/概率图细化/切片，完整本地测试集',
                    verification_status='VERIFIED', origin='实际模型推理及 COCO 掩码评分，非估计或合成数值',
                    authority='用户授权的三项模块去除实验；without RGB-D 延后',
                    completed_at_utc=verified['results']['finished_at_utc'],
                    protocol_sha256=verified['protocol_sha256'], limitations=limits,
                    dataset_archive_sha256_recorded_in_training_protocol=read(plan['system_protocol']['path'])['input_sha256']['archive'],
                    archive_rehashed_by_this_publisher=False)
    paths = '\n'.join(f'- **{row["method"]}：** `{row["csv"]}`' for row in rows)
    document = f'''# Swin-Large 逐项去除消融

{setup}

{table}

APₛ、ARₛ、AP 均采用 COCO 掩码评测，在 IoU=0.50:0.05:0.95 上平均，每图最多 100 个实例；小目标沿用原始标注面积 [0, 32²] 像素。完整本地测试与 Kaggle 公榜分开报告。

## 结果解释

{interpretation}

{limits}

## 固定配置与实现

基座第 15 轮、点头第 6 轮，推理为 FP32，关闭 TF32；置信度阈值 0.01、掩码跨视图 NMS 阈值 0.5、每图最多 100 个实例。切片配置为整图加四个 640×640 角点切片，相邻切片重叠 256 像素，切片输入放大到 1024×1024。无切片配置只使用整图候选；去除点细化时仍对原粗概率图做既有深度引导概率细化；去除概率图细化时直接输出点细化掩码。阈值、深度表示、边缘过滤、去重规则和其余权重均沿用冻结完整系统。

四个输出共享各视图的基座前向结果以节省评测计算；无切片输出只取整图候选和整图特征。共享计算的运行耗时不用于逐变体速度对比。

## 核验记录

- 工程检查在 3 张预先指定的验证图上，将四条共享特征路径分别与独立原实现比较，掩码和置信度完全一致。
- 工程检查、{len(plan['gpus'])} 个推理进程、4 个评分进程全部正常退出。每个变体的 {len(plan['gpus'])} 个分片合计覆盖全部 2,112 张图，各图恰好出现一次。
- 完整配置逐图精确复现已有累计 S4；无切片配置逐图精确复现上一轮累计 S3，所有 COCO 指标也完全相同。
- 三个含切片配置的候选类别、数量、分数和顺序完全相同。四份 CSV 均已有原生 RLE 解码校验，文件哈希与校验记录一致；发布前再次核对全部 CSV 行与评分预测、分片与合并预测的对应关系。
- 已核对冻结协议、模型源码、权重、进程记录、结果文件、评分设置及 CSV 哈希。数据归档的 SHA-256 沿用训练协议记录，本发布程序核对测试图像 ID 和评分 GT 统计，没有重新计算整个数据压缩包的哈希。

## 提交文件本机路径

{paths}

预测 CSV、权重和完整实验记录保留在本机实验目录；本页配套 [指标 CSV](course_removal_swinl.csv) 和 [核验 JSON](course_removal_swinl.json) 可随代码分发。

## 复现命令

在仓库根目录运行，选择实际空闲显卡，输出目录必须尚不存在：

```bash
.venv/bin/python -u tools/evaluate_course_swinl_removal.py \\
  --output reports/course_swinl_removal_new --gpus 3,4,5,6,7 \\
  --run reports/course_swinl_depth_no_k_v1/swinl_depth_no_k \\
  --archive /data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip \\
  --without-tiles-reference reports/course_swinl_report_ablation_v1/S3
.venv/bin/python tools/summarize_course_swinl_removal.py \\
  --run reports/course_swinl_removal_new
```

程序需要已存在的冻结基座/点头、原完整 S4 预测、上一轮 S3 预测及累计结果记录。按实际位置调整路径；工程复现检查通过后才开始全量推理，完成后统一重新评分。上述发布命令只更新本页及 README 的逐项去除段落。

## Material Passport

| 字段 | 内容 |
|---|---|
| 来源 | 本地冻结实验 `{run_label}`；academic-research-suite 实验流程 |
| 材料类型 | 实际模型推理、完整测试 COCO 评分与已验证预测 CSV |
| 范围 | 固定权重独立去除三项模块；RGB-D 去除延后 |
| 核验状态 | VERIFIED；四组完整测试、分片与文件哈希、全部 CSV 绑定、两组历史预测精确复现 |
| 协议 SHA-256 | `{verified['protocol_sha256']}` |
| 完成时间（UTC） | {verified['results']['finished_at_utc']} |
| 使用边界 | 单次训练、无重训练、无统计显著性检验；本地结果与 Kaggle 分开报告 |
'''
    artifact = dict(schema='dpseg-swinl-single-module-removal-v1', status='verified',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), material_passport=passport,
        split='local_test', images=2112, backbone='Swin-Large', camera_intrinsics_used=False,
        training=False, test_used_for_selection=False, variants=plan['variants'],
        deferred=plan['deferred'], results=rows, full_minus_removal=deltas,
        cumulative_s1_to_s2_ap_small_change=cumulative_point,
        semantics=verified['semantics'], weights=plan['weights'],
        protocol=dict(path=str(run/'protocol.json'), sha256=verified['protocol_sha256']),
        system_protocol=plan['system_protocol'], source_sha256=plan['new_source_sha256'],
        publisher_sha256=sha(__file__), phase_exits=verified['phase_exits'],
        rank_receipts=verified['rank_receipts'], references=plan['references'],
        cumulative_summary_sha256=verified['cumulative_sha256'])
    docs = ROOT/'docs/results'
    (docs/'course_removal_swinl.md').write_text(document)
    (docs/'course_removal_swinl.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    with (docs/'course_removal_swinl.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['方法', 'AP_s', 'AR_s', 'AP'])
        for row in rows:
            writer.writerow([row['method'], *(row['metrics'][key] for key in METRICS)])
    start, end = '<!-- module-removal:start -->', '<!-- module-removal:end -->'
    block = f'''{start}
### 逐项去除消融

{setup}

{table}

{interpretation}

{limits}完整配置与无切片配置均逐图精确复现已有 S4、S3，四组 CSV 与评分预测完全一致。

[完整报告、提交文件路径与复现命令](docs/results/course_removal_swinl.md) · [指标 CSV](docs/results/course_removal_swinl.csv) · [核验 JSON](docs/results/course_removal_swinl.json)。
{end}'''
    readme = ROOT/'README.md'
    original = readme.read_text()
    if start in original or end in original:
        require(original.count(start) == original.count(end) == 1, 'Ambiguous removal markers')
        updated = original[:original.index(start)] + block + original[original.index(end)+len(end):]
    else:
        anchor = '<!-- progressive-ablation:end -->'
        require(original.count(anchor) == 1, 'Missing cumulative table anchor')
        index = original.index(anchor)+len(anchor)
        updated = original[:index]+'\n\n'+block+original[index:]
    readme.write_text(updated)
    print(table)
    print('Published verified removal results; other README sections preserved.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'reports/course_swinl_removal_v1')
    args = parser.parse_args()
    run = args.run.resolve()
    publish(run, audit(run))


if __name__ == '__main__':
    main()
