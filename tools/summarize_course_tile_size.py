#!/usr/bin/env python3
"""Verify completed tile-size comparisons and publish their measured results."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.summarize_course_swinl_ablation import read, sha, require, METRICS
from tools.summarize_course_swinl_removal import record_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=ROOT/'reports/course_swinl_tile_size_v1')
    args = parser.parse_args()
    run = args.run.resolve()
    plan, result = read(run/'protocol.json'), read(run/'results.json')
    state = read(run/'status.json')
    require(state['status'] == result['status'] == 'completed' and state['exit_code'] == 0,
            'Experiment has not completed')
    require(result['protocol_sha256'] == sha(run/'protocol.json'), 'Protocol changed')
    require(not result['training'] and not result['test_used_for_selection'], 'Unexpected training/selection')
    require(plan['sizes'] == [512, 640, 768] and plan['recomputed_sizes'] == [512, 768], 'Different comparison')
    require(plan['nominal_overlap_ratio'] == .2 and not plan['camera_intrinsics_used'], 'Different input/geometry')
    for name, checksum in plan['new_source_sha256'].items():
        require(sha(ROOT/name) == checksum, f'Frozen source changed: {name}')
    for item in [*plan['weights'].values(), plan['system_protocol']]:
        require(sha(item['path']) == item['sha256'], 'Frozen model/protocol changed')
    system = read(plan['system_protocol']['path'])
    for name, checksum in system['system_source_sha256'].items():
        require(sha(ROOT/name) == checksum, f'Original system source changed: {name}')
    expected_exits = dict(engineering=['engineering'],
        inferring=[f'worker{i}' for i in range(len(plan['gpus']))], scoring=['score512', 'score768'])
    for phase, names in expected_exits.items():
        require(read(run/f'{phase}_exits.json') == dict.fromkeys(names, 0), f'{phase} did not finish cleanly')
    gate = read(run/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['protocol_sha256'] == sha(run/'protocol.json'),
            'Invalid engineering gate')
    require([c['image_id'] for c in gate['checks']] == plan['engineering_val_ids']
            and all(c['exact_masks_and_scores'] and c['score_errors'] == {'512': 0., '640': 0., '768': 0.}
                    for c in gate['checks']), 'Incomplete engineering comparisons')
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_tile_size import geometry
    from tools.audit_course_point_refinement import compare_csv_predictions
    with CourseArchive(plan['archive'], split='test') as data:
        ids = data.image_ids
    require(ids == plan['image_ids'] and len(ids) == len(set(ids)) == 2112, 'Test coverage differs')
    for size in plan['sizes']:
        # JSON stores tuples as lists.
        expected = json.loads(json.dumps(geometry(size)))
        require(plan['geometry'][str(size)] == gate['geometry'][str(size)] == expected, 'Geometry changed')
    checks_640, rank_receipts = [], []
    counts = {'512': 0, '768': 0}
    prediction_digests = {'512': {}, '768': {}}
    for rank in range(len(plan['gpus'])):
        receipt = read(run/f'rank{rank}.json')
        rank_ids = ids[rank::len(plan['gpus'])]
        require(receipt['status'] == 'completed' and receipt['image_ids'] == rank_ids
                and receipt['images'] == len(rank_ids) and receipt['native_exact']
                and not receipt['ground_truth_in_prediction'], 'Invalid inference shard receipt')
        require(receipt['reference_640_exact_images'] == [rank_ids[0], rank_ids[-1]], 'Missing 640 sentinels')
        checks_640.extend(receipt['reference_640_exact_images'])
        require(sha(run/f'checks_rank{rank}.jsonl') == receipt['checks_sha256'], 'Check records changed')
        checks = [json.loads(line) for line in (run/f'checks_rank{rank}.jsonl').read_text().splitlines()]
        require([c['image_id'] for c in checks] == rank_ids and all(c['native_exact'] for c in checks),
                'Native prediction checks incomplete')
        require([c['image_id'] for c in checks if c['reference_640_checked']]
                == receipt['reference_640_exact_images'], 'Sentinel checks incomplete')
        for size in ('512', '768'):
            shard = run/f'tile{size}_rank{rank}.jsonl'
            require(sha(shard) == receipt['artifacts'][size], 'Shard changed')
            observed, instances = [], 0
            with shard.open() as stream:
                for line in stream:
                    item = json.loads(line)
                    image_id = item['image_id']
                    require(image_id not in prediction_digests[size], 'Duplicate shard image')
                    observed.append(image_id)
                    prediction_digests[size][image_id] = record_digest(item)
                    instances += len(item['predictions'])
            require(observed == rank_ids and instances == receipt['instances'][size],
                    'Actual shard coverage/order/count differs from receipt')
            counts[size] += receipt['instances'][size]
        rank_receipts.append(dict(rank=rank, images=len(rank_ids), gpu=receipt['gpu'],
                                  receipt_sha256=sha(run/f'rank{rank}.json')))
    rows, semantics = [], None
    for size in ('512', '640', '768'):
        row = result['sizes'][size]
        require(row['status'] == 'completed' and row['images'] == 2112
                and row['tile_size_px'] == int(size), 'Incomplete or mislabeled result')
        if size in counts:
            folder = run/f'tile{size}'
            require(row == read(folder/'result.json'), 'Final summary differs from per-size result')
            require(Path(row['evaluation']).resolve() == folder/'evaluation.json'
                    and Path(row['predictions']).resolve() == folder/'predictions.jsonl'
                    and Path(row['csv']).resolve() == run/'submissions'/f'dpseg_swinl_depth_no_k_tile{size}_overlap20.csv',
                    'Result paths do not belong to their declared size')
        for kind in ('evaluation', 'predictions', 'csv'):
            require(sha(row[kind]) == row[f'{kind}_sha256'], f'{size} {kind} changed')
        evaluated = read(row['evaluation'])
        require(evaluated['status'] == 'completed' and evaluated['split'] == 'test'
                and evaluated['images'] == 2112 and evaluated['metrics'] == row['metrics']
                and evaluated['predictions_sha256'] == row['predictions_sha256'], 'Evaluation/result mismatch')
        current = {key: evaluated[key] for key in ('iou_type', 'max_detections', 'iou_thresholds',
            'area_ranges', 'ground_truth_area', 'prediction_area', 'gt_instances', 'gt_small_including_crowd')}
        require(current['iou_type'] == 'segm' and current['max_detections'] == [1, 10, 100]
                and current['area_ranges'][1] == [0, 1024]
                and current['ground_truth_area'] == 'original annotation area preserved', 'Wrong COCO settings')
        require(len(current['iou_thresholds']) == 10 and abs(current['iou_thresholds'][0]-.5) < 1e-12
                and abs(current['iou_thresholds'][-1]-.95) < 1e-12, 'Wrong IoU thresholds')
        if semantics is None:
            semantics = current
        require(current == semantics, 'GT or evaluation settings differ between arms')
        validation = read(Path(row['evaluation']).parent/'csv_validation.json')
        require(validation['sha256'] == row['csv_sha256'] and validation['rows'] == 2112
                and validation['rle_native_decode_verified'] and validation['rle_runs_verified']
                and validation['id_order_verified'], 'Missing CSV decoding/ID validation')
        require(compare_csv_predictions(row['csv'], row['predictions'], ids) == 2112
                == row['csv_rows_identical_to_scored_predictions'], 'CSV binding incomplete')
        require(validation['predictions'] == evaluated['predicted_instances'] == row['predicted_instances'],
                'Instance counts differ')
        if size in counts:
            require(counts[size] == row['predicted_instances'] and row['recomputed'], 'Shard totals differ')
            observed = []
            with Path(row['predictions']).open() as stream:
                for line in stream:
                    item = json.loads(line)
                    observed.append(item['image_id'])
                    require(record_digest(item) == prediction_digests[size].get(item['image_id']),
                            'Scored merged prediction differs from its inference shard')
            require(observed == ids, 'Merged prediction IDs/order differ')
        else:
            reference = plan['reference_640']
            require(not row['recomputed'] and all(row[k] == reference[k]
                for k in ('metrics', 'predictions_sha256', 'csv_sha256', 'evaluation_sha256')),
                '640 reference differs')
        require(all(0 <= row['metrics'][m] <= 1 for m in METRICS), 'Invalid metric')
        rows.append(dict(row, geometry=plan['geometry'][size]))
    baseline = result['sizes']['640']['metrics']
    for row in rows:
        row['delta_vs_640'] = {m: row['metrics'][m]-baseline[m] for m in METRICS}
    table = ['| 切片尺寸 | 每图视图数（含整图） | APₛ ↑ | ARₛ ↑ | AP ↑ |',
             '|:---:|:---:|:---:|:---:|:---:|']
    for row in rows:
        label = f"{row['tile_size_px']}×{row['tile_size_px']}" + ('（当前默认）' if row['tile_size_px'] == 640 else '')
        table.append('| ' + ' | '.join([label, str(row['geometry']['views_including_whole']),
                      *(f'{row["metrics"][m]:.6f}' for m in METRICS)]) + ' |')
    table = '\n'.join(table)
    common = ('固定完整 DPSeg 的 Swin-Large＋无内参深度、第 15 轮基座与第 6 轮点头，保留两种掩码细化。'
        '在完整本地测试集 **2,112 张图**上比较切片尺寸，数值范围 0–1。512、768 重新推理；'
        '640 的窗口和完整流程与已完成的默认配置完全一致，复用其全量结果并通过样例复现核验。')
    interpretation = '\n'.join(f'- {row["tile_size_px"]} 相对 640：APₛ **{row["delta_vs_640"]["mask_ap_small"]:+.6f}**，'
        f'ARₛ **{row["delta_vs_640"]["mask_ar_small"]:+.6f}**，AP **{row["delta_vs_640"]["mask_ap"]:+.6f}**。'
        for row in rows if row['tile_size_px'] != 640)
    best = {metric: max(rows, key=lambda row: row['metrics'][metric])['tile_size_px'] for metric in METRICS}
    conclusion = (f'三组中 APₛ 最高的是 **{best["mask_ap_small"]}**，ARₛ 最高的是 **{best["mask_ar_small"]}**，'
                  f'整体 AP 最高的是 **{best["mask_ap"]}**。不同尺寸的优势取决于评价指标。')
    if result['sizes']['512']['metrics']['mask_ap_small'] < baseline['mask_ap_small']:
        conclusion += '512 虽然有更多切片并将目标放得更大，APₛ 仍低于 640；切片更小不保证本任务的分割精度更高。'
    caveat = ('本轮固定名义重叠率 20%，采用滑窗步长并将末端窗口移至图像边界。'
        '**实际重叠和视图数量会随尺寸变化，结果同时反映尺度、覆盖和候选数量的影响，不能当作同计算量的纯尺度比较。**'
        '这是在已查看过结果的本地测试集上追加的敏感性分析，没有依据这张表更换默认配置；'
        '若要选择新的切片参数，应另在验证集确认。分数与 Kaggle 公榜分开报告。')
    geometry_table = ['| 尺寸 | 单轴起点（像素） | 相邻窗口实际重叠（像素） | 实际重叠比例 | 放大倍率 |',
                      '|:---:|:---:|:---:|:---:|:---:|']
    for row in rows:
        g = row['geometry']
        geometry_table.append('| ' + ' | '.join([str(row['tile_size_px']),
            '、'.join(map(str, g['axis_starts_px'])), '、'.join(map(str, g['actual_adjacent_overlap_px'])),
            '、'.join(f'{x:.2%}' for x in g['actual_adjacent_overlap_ratios']), f'{g["magnification"]:.3f}×']) + ' |')
    paths = '\n'.join(f'- **{row["tile_size_px"]}：** `{row["csv"]}`' for row in rows)
    document = f'''# DPSeg 切片尺寸敏感性分析

{common}

{table}

{conclusion}

{interpretation}

## 比较口径

{caveat}

APₛ、ARₛ 对原始标注面积 [0, 32²] 像素的小实例计算，AP 包含所有尺寸，均采用 COCO 掩码评测与 IoU=0.50:0.05:0.95；每图最多 100 个实例。模型输入统一为 1024×1024，分数阈值 0.01，跨视图掩码 IoU 去重阈值 0.5，切片内部边缘过滤为 2 像素。RGB、深度和有效性图同步裁剪；点细化使用来源视图特征，概率图细化使用原图深度。各组均为 FP32、seed=17 固定权重推理，不重新训练。

## 窗口与计算量

{chr(10).join(geometry_table)}

单轴起点在横、纵方向组合成网格，所有组额外保留整图预测。512 的 9 个切片加整图需要 10 次模型前向；640、768 都是 4 个切片加整图，需要 5 次。这是前向次数，不等同于实测耗时比。本轮不同组共享整图特征以节省实验时间，且在混合显卡上并行执行，因此不将运行时间用作模型速度对比。

窗口生成参考 [SAHI 官方切片实现](https://github.com/obss/sahi/blob/main/sahi/slicing.py)的步长和末端对齐规则。推理适配器、RGB-D 输入、掩码去重和两种细化仍使用 DPSeg 实现；本轮未调用 SAHI 的检测器或结果合并接口。[原论文](https://arxiv.org/pdf/2202.06934)在 VisDrone 和 xView 分别使用 640、400 的切片及 25% 重叠实验；[当前官方指南](https://obss.github.io/sahi/guides/sliced-inference/)建议从 20% 重叠开始。这些提供实验起点，不能直接确定本数据集的最佳参数。

## 核验

- 三张固定验证图上，新入口在 512、640、768 三种窗口下与原推理实现逐掩码、逐分数完全一致；验证只比较实现，不使用 GT 调参。
- 全部 2,112 张测试图的整图基座预测与原 S1 精确一致；640 另在每个分片首尾测试图上与原完整结果精确一致，共 {len(checks_640)} 张。
- 两组新结果均完成整套 COCO 评分、全量 CSV RLE 解码及与评分输入的逐行核对；三组使用一致的评价设置与 GT 统计。
- 单种子、固定权重的条件比较，不据此声称统计显著或无偏的新方法泛化提升。RGB-D 融合去除实验仍待后续训练。

## CSV 本机路径

{paths}

预测 CSV 和权重保留在本机；公开数值与证据索引见 [CSV](course_tile_size_swinl.csv) 和 [JSON](course_tile_size_swinl.json)。

## 复现

```bash
.venv/bin/python -u tools/evaluate_course_tile_size.py \\
  --output reports/course_swinl_tile_size_new --gpus 0,2,3,4,5,6,7
.venv/bin/python tools/summarize_course_tile_size.py \\
  --run reports/course_swinl_tile_size_new
```

按实际空闲显卡修改 `--gpus`。入口的 `--reference-run` 指向已完成的逐项去除实验，读取其数据、权重及 640 对照路径；这些本机文件需先准备好。程序在推理前冻结协议，先做实现核验，再推理与评分。

## Material Passport

- 来源：ARS experiment-agent run；实测数据来自本仓库冻结的实验协议和评测产物。
- 核验状态：VERIFIED；协议 SHA-256：`{sha(run/'protocol.json')}`。
- 范围：完整本地测试集的切片尺寸敏感性分析；不作为验证集选参或 Kaggle 分数。
'''
    artifact = dict(schema='dpseg-tile-size-sensitivity-v1', status='completed',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), split='local_test', images=2112,
        backbone='Swin-Large', camera_intrinsics_used=False, training=False, test_used_for_selection=False,
        interpretation_scope=plan['test_analysis'], semantics=semantics, results=rows,
        checkpoints=plan['weights'], protocol_sha256=sha(run/'protocol.json'),
        source_sha256=plan['new_source_sha256'], rank_receipts=rank_receipts,
        reference_640_sentinel_ids=checks_640,
        material_passport=dict(source='academic-research-suite / experiment-agent run', verification_status='VERIFIED'))
    docs = ROOT/'docs/results'
    (docs/'course_tile_size_swinl.md').write_text(document)
    (docs/'course_tile_size_swinl.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n')
    with (docs/'course_tile_size_swinl.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['tile_size_px', 'views_including_whole', 'AP_s', 'AR_s', 'AP'])
        for row in rows:
            writer.writerow([row['tile_size_px'], row['geometry']['views_including_whole'],
                             *(row['metrics'][m] for m in METRICS)])
    start, end = '<!-- tile-size:start -->', '<!-- tile-size:end -->'
    block = f'''{start}
### 切片尺寸敏感性分析

{common}

{table}

{conclusion}

{interpretation}

{caveat}

[窗口几何、核验和 CSV 路径](docs/results/course_tile_size_swinl.md) · [结果 CSV](docs/results/course_tile_size_swinl.csv) · [结果 JSON](docs/results/course_tile_size_swinl.json)。
{end}'''
    readme = ROOT/'README.md'
    text = readme.read_text()
    if start in text or end in text:
        require(text.count(start) == text.count(end) == 1, 'Ambiguous tile-size markers')
        text = text[:text.index(start)] + block + text[text.index(end)+len(end):]
    else:
        anchor = '### 去内参与 Swin-Large 补充对比'
        require(text.count(anchor) == 1, 'Missing README insertion point')
        text = text.replace(anchor, block+'\n\n'+anchor, 1)
    if any(row['metrics']['mask_ap_small'] > baseline['mask_ap_small'] for row in rows):
        text = text.replace('当前最佳', '当前默认')
        text = text.replace('## 当前默认配置（按 Mask APₛ）', '## 当前默认配置（640×640 切片）')
    readme.write_text(text)
    print(table)
    print(interpretation)
    print('Verified all complete test results; published README and result documents.')


if __name__ == '__main__':
    main()
