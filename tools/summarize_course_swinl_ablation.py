#!/usr/bin/env python3
"""Verify five complete test predictions and publish the report's S0–S4 table."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = ['S0：Mask2Former', 'S1：+ RGB-D 融合（DA-Fusion）',
          'S2：+ 深度感知点细化', 'S3：+ 深度引导概率图细化', 'S4：+ 切片增强推理（DPSeg）']
METRICS = ('mask_ap_small', 'mask_ar_small', 'mask_ap')


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rgb', type=Path, default=ROOT/'reports/course_swinl_rgb_v1')
    parser.add_argument('--rgbd', type=Path, default=ROOT/'reports/course_swinl_depth_no_k_v1/swinl_depth_no_k')
    parser.add_argument('--supplement', type=Path, default=ROOT/'reports/course_swinl_report_ablation_v1')
    args = parser.parse_args()
    from rgbd_seg.course_archive import CourseArchive
    from tools.audit_course_point_refinement import compare_csv_predictions

    rgb, rgbd, supplement = args.rgb.resolve(), args.rgbd.resolve(), args.supplement.resolve()
    baseline = read(rgb/'results.json')
    completion = read(supplement/'results.json')
    plan = read(supplement/'protocol.json')
    audit = read(rgbd/'independent_audit.json')
    require(read(rgb/'queue.json')['status'] == 'completed', 'RGB controller is not complete')
    require(baseline['status'] == 'completed' and baseline['epoch'] == 15, 'RGB must use final epoch15')
    require(read(supplement/'status.json')['status'] == 'completed', 'Supplement controller is not complete')
    require(completion['status'] == 'completed' and not completion['test_used_for_selection'], 'Supplement incomplete')
    require(completion['protocol_sha256'] == sha(supplement/'protocol.json'), 'Supplement protocol changed')
    require(audit['status'] == 'passed' and not audit['test_used_for_selection'], 'RGB-D audit invalid')
    require(not baseline['test_used_for_selection'], 'RGB test selection is not allowed')
    for name in ('base', 'heads'):
        require(sha(plan[name]['path']) == plan[name]['sha256'], f'{name} weights changed')
    require(sha(baseline['checkpoint']) == baseline['checkpoint_sha256'], 'RGB weights changed')
    with CourseArchive(plan['archive'], split='test') as data:
        ids = data.image_ids
    require(len(ids) == 2112 and ids == plan['image_ids'], 'Incomplete or changed test set')
    references = {row['variant']: row for row in audit['rows']}
    configs = [
        (rgb/'final_evaluation/test', Path(baseline['csv']), baseline['csv_sha256']),
        (rgbd/'final_evaluation/test_native', Path(references['native']['csv']), references['native']['sha256']),
        (supplement/'S2', Path(completion['stages']['S2']['csv']), completion['stages']['S2']['csv_sha256']),
        (supplement/'S3', Path(completion['stages']['S3']['csv']), completion['stages']['S3']['csv_sha256']),
        (rgbd/'final_evaluation/test_full', Path(references['full']['csv']), references['full']['sha256']),
    ]
    rows, reference_semantics = [], None
    for i, (folder, csv_path, csv_hash) in enumerate(configs):
        evaluation = read(folder/'evaluation.json')
        predictions = folder/'predictions.jsonl'
        require(evaluation['status'] == 'completed' and evaluation['split'] == 'test'
                and evaluation['images'] == 2112, f'S{i} is not a full test evaluation')
        require(evaluation['predictions_sha256'] == sha(predictions), f'S{i} scored predictions changed')
        require(sha(csv_path) == csv_hash, f'S{i} CSV changed')
        semantics = {key: evaluation[key] for key in ('iou_type', 'max_detections', 'iou_thresholds',
            'area_ranges', 'ground_truth_area', 'prediction_area', 'gt_instances', 'gt_small_including_crowd')}
        require(semantics['iou_type'] == 'segm' and semantics['max_detections'] == [1, 10, 100]
                and semantics['area_ranges'][1] == [0, 1024], 'Incorrect COCO metric settings')
        require(len(semantics['iou_thresholds']) == 10
                and abs(semantics['iou_thresholds'][0] - .5) < 1e-12
                and abs(semantics['iou_thresholds'][-1] - .95) < 1e-12, 'Incorrect IoU thresholds')
        if reference_semantics is None:
            reference_semantics = semantics
        require(semantics == reference_semantics, f'S{i} evaluation semantics/GT differ')
        binding = compare_csv_predictions(csv_path, predictions, ids)
        metrics = evaluation['metrics']
        require(all(isinstance(metrics[m], (int, float)) and 0 <= metrics[m] <= 1 for m in METRICS),
                f'S{i} missing metrics')
        if i == 0:
            require(metrics == baseline['test'], 'RGB summary differs from evaluation')
        if i in (2, 3):
            require(metrics == completion['stages'][f'S{i}']['metrics'], 'Supplement summary differs')
        rows.append(dict(stage=f'S{i}', method=LABELS[i], metrics=metrics, images=2112,
            evaluation=str(folder/'evaluation.json'), evaluation_sha256=sha(folder/'evaluation.json'),
            predictions=str(predictions), predictions_sha256=evaluation['predictions_sha256'],
            csv=str(csv_path), csv_sha256=csv_hash, csv_rows_identical_to_scored_predictions=binding,
            predicted_instances=evaluation['predicted_instances']))
    table = ['| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |', '|:---:|:---:|:---:|:---:|']
    for i, row in enumerate(rows):
        cells = [row['method'], *(f'{row["metrics"][m]:.6f}' for m in METRICS)]
        if i == 4:
            cells = [f'**{cell}**' for cell in cells]
        table.append('| ' + ' | '.join(cells) + ' |')
    table = '\n'.join(table)
    delta = {m: rows[4]['metrics'][m]-rows[1]['metrics'][m] for m in METRICS}
    decreases = []
    for previous, current in zip(rows, rows[1:]):
        change_small = current['metrics']['mask_ap_small']-previous['metrics']['mask_ap_small']
        change_ap = current['metrics']['mask_ap']-previous['metrics']['mask_ap']
        if change_small < 0:
            decreases.append(f'{current["stage"]} 相对 {previous["stage"]} 的 APₛ 变化为 '
                f'**{change_small:+.6f}**，整体 AP 变化为 **{change_ap:+.6f}**。')
    qualification = (''.join(decreases) + '因此不能将本表概括为“每个模块均提升小目标 AP”；'
                     '各行反映既定组合下的增量效果。') if decreases else ''
    best_ap = max(rows, key=lambda row: row['metrics']['mask_ap'])
    if best_ap['stage'] != 'S4':
        qualification += (f'整体 AP 最高的是 {best_ap["stage"]}（**{best_ap["metrics"]["mask_ap"]:.6f}**）。')
        if all(rows[4]['metrics'][key] == max(row['metrics'][key] for row in rows)
               for key in ('mask_ap_small', 'mask_ar_small')):
            qualification += '完整 S4 的 APₛ 和 ARₛ 最高。'
    common = ('所有配置均采用 **Swin-Large、1024×1024 输入**，RGB-D 分支使用无需相机内参的深度表示。'
              '以下为**完整本地测试集 2,112 张图**的 COCO 掩码评测，数值范围 0–1；各行按表中顺序累计添加模块。'
              '加粗标记完整 DPSeg 配置，任务主指标为 APₛ。')
    notes = '''APₛ、ARₛ 分别表示小目标掩码平均精度和平均召回率，均在 IoU=0.50:0.05:0.95 上平均，最多保留 100 个实例；AP 为所有尺寸的掩码平均精度。小目标使用原始标注面积 [0, 32²] 像素。以上分数与 Kaggle 公榜分开报告。

- **固定权重：** S0 为 RGB Mask2Former 第 15 轮；S1–S4 共用 RGB-D 第 15 轮基座，S2–S4 共用验证集选取的第 6 轮深度点头。RGB 与 RGB-D 均从相同公开 COCO Swin-Large 权重开始训练，使用完整训练集、15 轮和全局 batch 33。
- **累计配置：** S2 为整图点细化；S3 在 S2 后增加概率图细化，不使用切片；S4 保留两种细化并增加整图＋重叠切片候选。表中顺序表示组件加入顺序；完整系统仍先合并候选，再做来源视图点细化和原图概率图细化。
- **比较范围：** RGB-D 增加独立深度编码器和融合模块，点头增加 6 轮训练，切片增加推理计算。结果来自单次 seed=17 训练，显卡与每卡批次分配存在差异。测试标注仅用于评分，不用于训练、选择检查点或本轮调整推理参数。'''
    local_paths = '\n'.join(f'- **S{i}：** `{r["csv"]}`' for i, r in enumerate(rows))
    document = f'''# Swin-Large 基线对比与累计模块消融

{common}

{table}

{notes}

{qualification}

完整 DPSeg 相对 DA-Fusion 适配基座 S1 的 APₛ、ARₛ、AP 绝对增量分别为 **{delta['mask_ap_small']:+.6f}、{delta['mask_ar_small']:+.6f}、{delta['mask_ap']:+.6f}**。增量描述本次固定配置的结果，不代表多次训练的统计显著性。

## 结果核验

S0 使用本轮刚完成的 15 轮 RGB 训练及自动测试结果。S1 和 S4 复用已完成并独立从 CSV 复评的 Large＋无内参结果。S2、S3 补跑了全部 2,112 张图；每张图的基座预测均与已有 S1 结果逐掩码核对，概率图细化保持候选数量、类别、分数和顺序。五份 CSV 的全部图像 ID、掩码、类别和分数均与对应评分输入一致，五组 COCO 评测设置及 GT 统计一致。

## 提交文件本机路径

{local_paths}

预测 CSV 和模型权重保留在实验目录，不随本页上传到 GitHub。可分发的数值及校验信息见 [JSON](course_ablation_swinl.json) 和 [CSV](course_ablation_swinl.csv)。

## 补跑复现

```bash
.venv/bin/python -u tools/evaluate_course_swinl_ablation.py \\
  --output reports/course_swinl_report_ablation_new --gpus 3,4
```

选择空闲显卡，并按实际位置传入 `--run`、`--archive`。该程序仅补齐 S2/S3，固定原第 15 轮基座、第 6 轮点头和推理参数；不训练、不根据测试成绩选择参数。RGB 基线入口为 [run_course_swinl_rgb.py](../../tools/run_course_swinl_rgb.py)，其最终评测由控制程序自动完成。

## 与历史表的关系

本表替换 README 主表中的旧 Tiny＋XYZ 结果。旧表的 S3 为“点细化＋切片”，本表的 S3 为“点细化＋概率图细化”，二者不应只改标签后混用。[历史累计消融](progressive_ablation.md)保留原数字和顺序。[效率表](course_efficiency_swinl.md)使用相同五种结构；其中 RGB 耗时仍来自此前第 12 轮检查点，精度表使用最终第 15 轮。
'''
    docs = ROOT/'docs/results'
    (docs/'course_ablation_swinl.md').write_text(document)
    artifact = dict(schema='dpseg-swinl-cumulative-ablation-v1', status='completed',
        generated_at_utc=datetime.now(timezone.utc).isoformat(), split='local_test', images=2112,
        backbone='Swin-Large', camera_intrinsics_used=False, test_used_for_selection=False,
        cumulative_order=['RGB', 'RGB-D fusion', 'depth point', 'depth probability', 'tiles'],
        semantics=reference_semantics, results=rows, full_minus_rgbd=delta,
        checkpoints={'rgb': {'path': baseline['checkpoint'], 'sha256': baseline['checkpoint_sha256']},
                     'rgbd': plan['base'], 'points': plan['heads']},
        supplement_protocol_sha256=sha(supplement/'protocol.json'))
    (docs/'course_ablation_swinl.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2)+'\n')
    with (docs/'course_ablation_swinl.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(['方法', 'AP_s', 'AR_s', 'AP'])
        for row in rows:
            writer.writerow([row['method'], *(row['metrics'][m] for m in METRICS)])
    readme = ROOT/'README.md'
    original = readme.read_text()
    original = original.replace('## 当前最佳配置\n', '## 当前最佳配置（按 Mask APₛ）\n', 1)
    start, end = '<!-- progressive-ablation:start -->', '<!-- progressive-ablation:end -->'
    block = f'''{start}
### 基线对比与累计模块消融结果

{common}

{table}

{notes}

{qualification}

完整 DPSeg 相对 S1 的 APₛ、ARₛ 和 AP 分别提高 **{delta['mask_ap_small']:+.6f}、{delta['mask_ar_small']:+.6f}、{delta['mask_ap']:+.6f}**。S2、S3 已补齐完整测试推理；五组 CSV 与各自的评分预测逐行一致。

[完整报告与提交文件路径](docs/results/course_ablation_swinl.md) · [结果 CSV](docs/results/course_ablation_swinl.csv) · [结果 JSON](docs/results/course_ablation_swinl.json)。[历史 Tiny＋XYZ 累计消融](docs/results/progressive_ablation.md)保留旧数值及原模块顺序；本表 S3 为新增的“点细化＋概率图细化”结果。
{end}'''
    require(original.count(start) == original.count(end) == 1, 'Ambiguous README markers')
    readme.write_text(original[:original.index(start)] + block + original[original.index(end)+len(end):])
    print(table)
    print('Verified all five complete test results and CSV bindings; wrote README, Markdown, CSV, JSON.')


if __name__ == '__main__':
    main()
