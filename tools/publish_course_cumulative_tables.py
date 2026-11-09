#!/usr/bin/env python3
"""Verify current cumulative accuracy/timing artifacts and publish both tables."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import statistics
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['S0：Mask2Former', 'S1：+ RGB-D 融合（DA-Fusion）', 'S2：+ 深度感知点细化',
          'S3：+ 深度引导概率图细化', 'S4：+ 切片增强推理（DPSeg）']


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def replace_block(text, start, end, body):
    assert text.count(start) == text.count(end) == 1, (start, end)
    return text[:text.index(start)] + start + '\n' + body.strip() + '\n' + end + text[text.index(end) + len(end):]


def validate(run):
    summary = read(run / 'summary.json')
    protocol = read(run / 'protocol.json')
    assert summary['status'] == read(run / 'status.json')['status'] == 'completed'
    assert summary['protocol'] == protocol
    assert protocol['schema'] == 'dpseg-current-cumulative-efficiency-v1'
    assert (protocol['rgb_epoch'], protocol['base_epoch'], protocol['head_epoch']) == (15, 15, 6)
    assert (protocol['tile_size'], protocol['tiles'], protocol['views_including_whole']) == (640, 4, 5)
    assert protocol['merge_strategy'] == 'global_guard_edge_consensus'
    assert not protocol['peak_gpu_memory_measured'] and protocol['precision'] == 'FP32'
    assert len(protocol['image_ids']) == len(set(protocol['image_ids'])) == 32
    assert protocol['repeats'] == 3 and protocol['warmup_images'] == 5
    assert all(x['exit_code'] == 0 for x in summary['exits']) and len(summary['exits']) == 5
    for name, digest in protocol['sources'].items():
        assert sha(ROOT / name) == digest, name
    for entry in protocol['weights'].values():
        assert Path(entry['path']).stat().st_size == entry['bytes'] and sha(entry['path']) == entry['sha256']
    main = read(protocol['main_table']['path'])
    assert sha(protocol['main_table']['path']) == protocol['main_table']['sha256']
    assert main['images'] == 2112 and len(main['results']) == 5
    for row in main['results']:
        evaluation = read(row['evaluation'])
        assert sha(row['evaluation']) == row['evaluation_sha256']
        assert evaluation['images'] == row['images'] == 2112
        assert evaluation['metrics'] == row['metrics']
        assert sha(row['csv']) == row['csv_sha256']
    rows = summary['results']
    assert [r['stage'] for r in rows] == list(range(5))
    for row in rows:
        stage = row['stage']
        assert read(run / f'M{stage}/result.json') == row
        measurements = [json.loads(line) for line in (run / f'M{stage}/measurements.jsonl').read_text().splitlines()]
        assert sha(run / f'M{stage}/measurements.jsonl') == row['measurements_sha256']
        assert row['protocol_sha256'] == sha(run / 'protocol.json')
        assert len(measurements) == row['measurements'] == 96
        assert abs(statistics.mean(x['seconds'] for x in measurements) * 1000 - row['mean_ms']) < 1e-9
        assert abs(1000 / row['mean_ms'] - row['fps']) < 1e-12
        assert row['images'] == 32 and row['repeats'] == 3
        assert row['repeat_masks_exact'] and row['detected_gpu_activity_conflicts'] == 0
        assert row['test_gate_images'] == 3 and row['rgb_epoch'] == 15
        assert all(c['exact_masks'] and c['counts_equal'] and c['score_error'] <= 1e-6 for c in row['reference_checks'])
        expected_checks = 35 if stage in (0, 1, 2) else 3
        assert len(row['reference_checks']) == expected_checks
        for repeat in range(3):
            current = [x for x in measurements if x['repeat'] == repeat]
            assert len(current) == 32 and {x['image_id'] for x in current} == set(protocol['image_ids'])
        for image_id in protocol['image_ids']:
            assert len({x['mask_sha256'] for x in measurements if x['image_id'] == image_id}) == 1
        assert row['merge_strategy'] == ('global_guard_edge_consensus' if stage == 4 else None)
    assert len({row['hardware'] for row in rows}) == 1
    assert 'A100' in rows[0]['hardware']
    assert rows[2]['depth_head_parameters'] == 249858
    assert rows[1]['parameters'] + 249858 == rows[2]['parameters']
    assert len({r['parameters'] for r in rows[2:]}) == len({r['checkpoint_bytes'] for r in rows[2:]}) == 1
    assert rows[2]['mean_instances'] == rows[3]['mean_instances']
    return main, protocol, rows


def accuracy_table(main):
    lines = ['| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |', '|---|---:|---:|---:|']
    for stage, (label, row) in enumerate(zip(LABELS, main['results'])):
        if stage == 4:
            label = f'**{label}**'
        m = row['metrics']
        values = [f'{m[k]:.6f}' for k in ('mask_ap_small', 'mask_ar_small', 'mask_ap')]
        if stage == 4:
            values = [f'**{value}**' for value in values]
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    return '\n'.join(lines)


def efficiency_table(rows):
    lines = ['| 配置 | 参数量（M） | 推理权重（MiB） | 平均耗时（ms/图）↓ | FPS ↑ |',
             '|---|---:|---:|---:|---:|---:|']
    for stage, (label, row) in enumerate(zip(LABELS, rows)):
        values = [f'{row["parameters"]/1e6:.3f}', f'{row["checkpoint_bytes"]/2**20:.2f}',
                  f'{row["mean_ms"]:.1f}', f'{row["fps"]:.2f}']
        if stage == 4:
            label = f'**{label}**'
        lines.append('| ' + ' | '.join([label, *values]) + ' |')
    return '\n'.join(lines)


def status_block(text, sentence):
    return replace_block(text, '<!-- current-comparison-status:start -->',
                         '<!-- current-comparison-status:end -->', sentence)


def publish(run, main, protocol, rows):
    docs = ROOT / 'docs/results'
    table, timing = accuracy_table(main), efficiency_table(rows)
    date = datetime.now(ZoneInfo('Asia/Shanghai')).date().isoformat()
    condition = (f'统一使用 **{rows[0]["hardware"]} 单卡、FP32、batch=1、1024×1024模型输入**。'
                 '从验证集固定选取32张等间隔图像，预热5张后重复3轮，每组共96次计时。'
                 '五组按相同图片与顺序独立执行，完整版使用整图＋4个640×640切片、40%重叠，'
                 '采用整图优先＋截断处理＋跨视图一致性合并。')
    scope = ('参数量包含冻结骨干；推理权重统计基座文件及实际使用的深度点头独立文件，'
             '不计优化器和未使用的普通点头。点头增加249,858个参数，概率图细化与切片不增加参数。'
             '计时包含预处理、GPU前向、适用的细化/切片/合并、结果传回CPU和COCO RLE编码；'
             '排除图片读取与解码、模型加载、文件写入、一致性核对及AP评分。不统计GPU峰值显存。'
             'FPS按每张完整原图计算，为1000除以平均毫秒数。')
    weight_note = 'RGB和RGB-D基座均使用最终第15轮权重；S2–S4共用验证集选定的第6轮深度点头。'
    comparison = f'''### 基线对比与累计模块消融结果

统一采用 **Swin-Large、无需相机内参的深度输入、完整本地测试集2,112张**。每行在上一行基础上增加模块；S4使用当前640×640切片及整图优先＋截断处理＋跨视图一致性合并。

{table}

S1→S2的小目标APₛ下降0.004966，整体AP提高0.017296；本表反映各模块在既定组合下的累计变化。完整版相对DA-Fusion的APₛ、ARₛ、AP分别提高0.119913、0.193310、0.041826。

{weight_note} 表中添加顺序用于实验对比；完整推理仍先合并候选，再做来源视图点细化及原图概率图细化。当前合并由用户根据已查看的本地测试对比指定，Kaggle与验证集成绩另行列报。

[累计结果与五份CSV](docs/results/course_ablation_swinl_current.md) · [指标CSV](docs/results/course_ablation_swinl_current.csv) · [来源JSON](docs/results/course_ablation_swinl_current.json)。已完成的[七组独立去除模块实验](docs/results/course_baselines_removal_current.md)作为补充结果保留。'''
    efficiency = f'''### 模型规模与推理开销对比

{condition}

{timing}

{weight_note}

{scope}

本轮五组统一实测，旧合并的4794.0 ms/图保留在历史报告中。测速样本与重复用于衡量运行时间，不构成新增完整验证集精度结果。

[当前效率报告](docs/results/course_efficiency_swinl_current.md) · [CSV](docs/results/course_efficiency_swinl_current.csv) · [JSON](docs/results/course_efficiency_swinl_current.json) · [历史效率记录](docs/results/course_efficiency_swinl.md)。'''
    backups = run / 'document_before'
    backups.mkdir(exist_ok=True)
    paths = ['README.md', 'docs/README.md', 'docs/methods.md', 'docs/course_metric.md',
             'docs/historical_results.md', 'docs/results/course_ablation_swinl_current.md',
             'docs/results/course_baselines_removal_current.md', 'docs/results/course_efficiency_swinl.md']
    for name in paths:
        target = backups / name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((ROOT / name).read_text())
    readme = ROOT / 'README.md'
    text = readme.read_text()
    if '<!-- baseline-removal:start -->' in text:
        text = text.replace('<!-- baseline-removal:start -->', '<!-- cumulative-main:start -->').replace(
            '<!-- baseline-removal:end -->', '<!-- cumulative-main:end -->')
    text = replace_block(text, '<!-- cumulative-main:start -->', '<!-- cumulative-main:end -->', comparison)
    text = replace_block(text, '<!-- inference-efficiency:start -->', '<!-- inference-efficiency:end -->', efficiency)
    text = text.replace('当前主表采用基线对比与独立去除模块的形式。基线与独立消融总表已核验 7/7 行；七行完整测试结果均已核验。新合并策略尚未完成完整验证及独立速度复测。',
                        '当前主表采用S0–S4累计式消融，五组完整本地测试与统一推理测速均已完成；七组独立去除实验保留作补充。当前合并尚无新的完整验证集精度结果。')
    text = text.replace('下方效率表保留 Swin-Large＋无内参的既有测量，并区分新旧合并。',
                        '下方效率表与当前累计主表的五组配置及最终权重一致。')
    readme.write_text(text)
    sentence = ('当前主表采用S0–S4累计式消融；5组完整本地测试及统一推理测速均已完成。'
                '7组独立去除模块实验作为补充保留。')
    for name in ('methods', 'course_metric', 'historical_results'):
        path = ROOT / f'docs/{name}.md'
        text = status_block(path.read_text(), sentence)
        text = text.replace('[当前总表及CSV路径](results/course_baselines_removal_current.md)',
                            '[当前累计主表及CSV路径](results/course_ablation_swinl_current.md)')
        text = text.replace('当前合并尚无完整验证、Kaggle 或独立速度测量；独立消融的最新完成范围见本页状态及主表。',
                            '当前合并尚无新的完整验证集精度与Kaggle结果；五组统一推理测速已完成，见[效率报告](results/course_efficiency_swinl_current.md)。')
        text = text.replace('[当前基线与独立消融结果、来源](results/course_baselines_removal_current.md)',
                            '[独立去除模块补充结果](results/course_baselines_removal_current.md)')
        text = text.replace('[当前完整结果](results/course_baselines_removal_current.md)',
                            '[当前累计主表](results/course_ablation_swinl_current.md)')
        path.write_text(text)
    index = ROOT / 'docs/README.md'
    text = index.read_text().replace('- [当前完整版：基线对比与独立模块消融、CSV（已核验7/7行）](results/course_baselines_removal_current.md)',
          '- [当前主表：基线对比与累计模块消融、五份CSV](results/course_ablation_swinl_current.md)\n'
          '- [当前模型规模与推理开销](results/course_efficiency_swinl_current.md)\n'
          '- [补充：七组独立去除模块实验](results/course_baselines_removal_current.md)')
    index.write_text(text)
    accuracy_doc = docs / 'course_ablation_swinl_current.md'
    text = accuracy_doc.read_text().replace('# 原累计模块消融展示快照（640 切片＋当前合并）', '# 当前主表：基线对比与累计模块消融')
    text = text.replace('本文保留原 S0–S4 累计添加模块的结果。当前论文主表已改为[基线对比与独立去除模块](course_baselines_removal_current.md)，待测项在新表中单独标明。',
                        '当前论文主表采用S0–S4累计添加模块的结果。已完成的[七组独立去除模块实验](course_baselines_removal_current.md)作为补充结果保留。配套[模型规模与推理开销](course_efficiency_swinl_current.md)使用相同五组配置和最终权重。')
    text = text.replace('当前合并策略的完整验证集、Kaggle、逐项去除消融和独立推理耗时尚未补测，旧配置的这些数值不移作当前结果。',
                        '当前合并的七组独立去除实验和五组统一推理测速均已完成；当前合并尚无新的完整验证集精度或Kaggle成绩。')
    text = text.replace('当前机器可读摘要记录各行全部 12 项 COCO 指标、原始评分/预测/提交文件路径及 SHA-256、所用权重、策略常量、选择依据和未完成测量：',
                        '冻结的指标来源摘要记录各行全部12项COCO指标、原始评分/预测/提交路径及SHA-256、所用权重、策略和生成时的测量状态；最新测速完成状态见[当前效率报告](course_efficiency_swinl_current.md)：')
    accuracy_doc.write_text(text)
    independent_doc = docs / 'course_baselines_removal_current.md'
    text = independent_doc.read_text().replace('# DPSeg 基线对比与独立模块消融',
          '# 补充实验：基线对比与独立模块去除\n\n当前论文主表采用[五组累计式消融](course_ablation_swinl_current.md)，本页保留七组独立去除模块的补充结果。', 1)
    text = text.replace('尚无该完整配置的新验证集、Kaggle或独立速度结果。',
                        '尚无该完整配置的新验证集精度和Kaggle结果；五组累计配置的独立速度见[当前效率报告](course_efficiency_swinl_current.md)。')
    text = text.replace('第二条在核验通过后更新本表、README和关联文档。',
                        '第二条在核验通过后更新本补充表，保留当前累计主表。')
    independent_doc.write_text(text)
    legacy_efficiency = docs / 'course_efficiency_swinl.md'
    text = legacy_efficiency.read_text().replace(
        '当前整图优先＋截断处理＋一致性不增加模型参数或权重，但独立推理耗时尚未补测，不能直接沿用本页速度。[当前配置与结果](course_ablation_swinl_current.md)。',
        '当前整图优先＋截断处理＋一致性配置已使用最终权重独立复测，见[当前效率报告](course_efficiency_swinl_current.md)。本页保留原测速记录。')
    legacy_efficiency.write_text(text)
    distribution = ['| 配置 | 中位数（ms/图） | P90（ms/图） | 三轮平均耗时（ms/图） | 平均实例数 |',
                    '|---|---:|---:|---|---:|']
    for label, row in zip(LABELS, rows):
        rounds = ' / '.join(f'{value:.1f}' for value in row['repeat_mean_ms'])
        distribution.append(f'| {label} | {row["median_ms"]:.1f} | {row["p90_ms"]:.1f} | {rounds} | {row["mean_instances"]:.1f} |')
    doc = f'''# 当前累计配置：模型规模与推理开销

## Material Passport

- 来源：冻结的五组累计配置、最终权重与独立进程逐图计时。
- 测量日期：{date}（Asia/Shanghai）。状态：五组已完成并核验。
- 原始记录：`{run.relative_to(ROOT)}`；协议、逐图时间、GPU活动和退出码均保留。
- 本表测量运行时间，不新增完整验证集AP评分，不统计GPU峰值显存。

## 报告用表

{condition}

{timing}

{weight_note}

## 测量口径

{scope}

固定CPU逻辑核{','.join(map(str, protocol['cpu_affinity']))}；PyTorch/OMP/MKL/OpenBLAS各4线程、interop为1。未启用混合精度、TF32、编译或TensorRT。软件环境为PyTorch {rows[0]['torch']}、Transformers {rows[0]['transformers']}、CUDA {rows[0]['cuda']}。物理GPU为{protocol['gpu']}，UUID `{protocol['gpu_uuid']}`，开始测量时没有其他计算进程。

## 核验与适用范围

- 每组首先对固定的3张测试图片与此前已评分预测进行逐掩码一致性检查；类别、实例数及次序一致，置信度误差不超过10⁻⁶。此步骤不读取GT、不计算新评分，且不计入耗时。
- S0–S2的32张验证图还与已有对应权重的完整验证预测核对一致；S3/S4仅使用上述3张测试图检查加上32张验证图的三轮预测一致性，不将其称为完整验证集AP评测。
- 每组3轮、共96次计时完整；同图跨轮掩码完全一致，GPU活动检查没有发现其他进程。NVML约每秒采样，不能排除未被采样捕捉的极短活动；共享主机CPU也可能受到其他任务影响。
- 每种配置独立执行完整推理，无共享前向或其他配置的特征缓存。S4使用当前合并策略；旧4794.0 ms/图保留在[历史效率报告](course_efficiency_swinl.md)。
- 相同图片与权重的重复反映运行波动，不代表多次独立训练的统计方差。效率表与[累计精度主表](course_ablation_swinl_current.md)配置和权重一致，测速样本来自验证集。

## 耗时分布

{chr(10).join(distribution)}

## 复现

```bash
.venv/bin/python tools/benchmark_course_cumulative_current.py prepare --output reports/course_efficiency_swinl_current/new_run --gpu 7
.venv/bin/python tools/benchmark_course_cumulative_current.py launch --output reports/course_efficiency_swinl_current/new_run
.venv/bin/python tools/publish_course_cumulative_tables.py --run reports/course_efficiency_swinl_current/new_run --publish
```

选择空闲A100，等待五组完成后发布。完整原始数据见[JSON](course_efficiency_swinl_current.json)，表格数据见[CSV](course_efficiency_swinl_current.csv)。
'''
    (docs / 'course_efficiency_swinl_current.md').write_text(doc)
    artifact = dict(schema='dpseg-current-cumulative-efficiency-report-v1', status='verified',
                    protocol=protocol, results=rows, raw_run=str(run), summary_sha256=sha(run / 'summary.json'))
    (docs / 'course_efficiency_swinl_current.json').write_text(json.dumps(artifact, ensure_ascii=False, indent=2) + '\n')
    with (docs / 'course_efficiency_swinl_current.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['配置', '参数量_M', '推理权重_MiB', '平均耗时_ms', 'FPS'])
        for label, row in zip(LABELS, rows):
            writer.writerow([label, row['parameters']/1e6, row['checkpoint_bytes']/2**20, row['mean_ms'], row['fps']])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    run = args.run.resolve()
    accuracy, protocol, rows = validate(run)
    if args.publish:
        publish(run, accuracy, protocol, rows)
    print(efficiency_table(rows))
    print(json.dumps(dict(status='verified', rows=5, published=args.publish, run=str(run))))


if __name__ == '__main__':
    main()
