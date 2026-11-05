#!/usr/bin/env python3
"""Publish the completed Swin-L timing run in report/README tables."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Mask2Former', '+ RGB-D 融合（DA-Fusion）',
          '+ 深度感知点细化', '+ 深度引导概率图细化', '+ 切片增强推理（DPSeg）']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    summary = json.loads((run/'summary.json').read_text())
    assert summary['status'] == 'completed'
    p, rows = summary['protocol'], summary['results']
    assert [r['stage'] for r in rows] == list(range(5))
    assert all(e['exit_code'] == 0 for e in summary['exits'])
    for row in rows:
        path = run/f'M{row["stage"]}'/'measurements.jsonl'
        measurements = [json.loads(line) for line in path.read_text().splitlines()]
        assert sha(path) == row['measurements_sha256']
        assert len(measurements) == len(p['image_ids'])*p['repeats']
        assert abs(statistics.mean(x['seconds'] for x in measurements)*1000-row['mean_ms']) < 1e-9
        assert abs(1000/row['mean_ms']-row['fps']) < 1e-12
        assert all({x['image_id'] for x in measurements if x['repeat']==i} == set(p['image_ids'])
                   for i in range(p['repeats']))
        assert row['repeat_masks_exact'] and row['detected_gpu_activity_conflicts'] == 0
        assert all(c['exact_masks'] and c['counts_equal'] and c['score_error']<=1e-6
                   for c in row['reference_checks'])
    assert rows[1]['parameters']+rows[2]['depth_head_parameters'] == rows[2]['parameters']
    assert len({r['parameters'] for r in rows[2:]}) == 1
    assert len({r['checkpoint_bytes'] for r in rows[2:]}) == 1
    assert rows[2]['mean_instances'] == rows[3]['mean_instances']
    assert len({r['hardware'] for r in rows}) == 1
    table = ['| 配置 | 参数量（M） | 推理权重（MiB） | 平均耗时（ms/图） | FPS |',
             '|:---:|:---:|:---:|:---:|:---:|']
    for row, label in zip(rows, LABELS):
        if row['stage'] == 4:
            label = f'**{label}**'
        table.append(f'| {label} | {row["parameters"]/1e6:.3f} | {row["checkpoint_bytes"]/2**20:.2f} | {row["mean_ms"]:.1f} | {row["fps"]:.2f} |')
    table = '\n'.join(table)
    note = (f'RGB 基线本次使用已保存的第 {p["rgb_epoch"]} 轮检查点；测速时 15 轮训练尚未完成，'
            '其最终权重的耗时仍需复核。RGB-D 各组共用第 15 轮基座，含点细化的各组共用第 6 轮深度点头。'
            if not p['rgb_final_training_complete'] else
            'RGB 与 RGB-D 基座均为第 15 轮，含点细化的各组共用第 6 轮深度点头。')
    condition = (f'统一在 **{rows[0]["hardware"]} 单卡、FP32、batch=1、1024×1024 输入** 下测量。'
                 f'从完整验证集等间隔选取 {len(p["image_ids"])} 张图，预热 5 张后重复 {p["repeats"]} 次，'
                 f'每组共 {len(p["image_ids"])*p["repeats"]} 次计时。各组使用相同图片与每轮顺序，按表中顺序累计添加模块。')
    distribution = ['| 配置 | 中位数（ms） | P90（ms） | 各轮平均耗时（ms） | 平均实例数 |',
                    '|---|---:|---:|---|---:|']
    for i, row in enumerate(rows):
        rounds = ' / '.join(f'{v:.1f}' for v in row['repeat_mean_ms'])
        distribution.append(f'| M{i} | {row["median_ms"]:.1f} | {row["p90_ms"]:.1f} | {rounds} | {row["mean_instances"]:.1f} |')
    document = f'''# Swin-Large 模型规模与推理开销

## Material Passport

- 来源：academic-research-suite / experiment-agent，当前五组配置的实际测量。
- 测量日期：2026-11-05（北京时间）。
- 状态：五组推理计时完成；RGB 权重阶段见下方说明。
- 范围：Swin-Large＋无内参深度；不统计 GPU 峰值显存。

## 报告用表

{condition}

{table}

**权重阶段：** {note}

## 统计口径

- M0–M4 按报告的新顺序累计添加组件，概率图细化排在切片增强之前；每组在独立进程中运行。
- 参数量来自实际加载模型的参数计数，包含冻结骨干。只有实际使用的深度点头被计入；其参数量为 {rows[2]['depth_head_parameters']:,}。
- 权重大小为实际推理文件大小：RGB 或 RGB-D 基座检查点，加上适用的深度点头独立文件；包含模型缓冲区和序列化元数据，不含优化器、梯度或未使用的普通点头。三组细化配置共用相同文件。
- M = 1,000,000 个参数；MiB = 2²⁰ 字节；FPS = 1000 / 平均毫秒数，按完整原图计，不按切片数计。
- 计时从解码后的 CPU RGB 图像和米制深度开始，包含张量转换、归一化、传入 GPU、网络前向、适用的点细化/概率细化/切片合并、传回 CPU 和 COCO 压缩 RLE 编码。
- 排除模型加载、ZIP/PNG 读取与解码、参考预测核对、文件写入和 AP 评测；计时前后同步 CUDA。未启用混合精度、TF32、编译或 TensorRT。
- 输入统一为 1024×1024，置信度阈值 0.01，每图最多 100 个实例。切片组包含整图和四张 640×640 切片，各视图缩放至 1024×1024。
- 固定 CPU 逻辑核 {', '.join(map(str,p['cpu_affinity']))}，PyTorch/OMP/MKL/OpenBLAS 各 4 线程，interop 线程数为 1。
- 软件环境：PyTorch {rows[0]['torch']}、Transformers {rows[0]['transformers']}、CUDA {rows[0]['cuda']}。

## 耗时分布

{chr(10).join(distribution)}

这些重复使用相同权重与图片，反映运行波动和图片差异，不代表多次独立训练的方差。

## 核验与运行限制

- 五个工作进程均正常结束，每组样本和重复数完整，三轮预测掩码完全一致。
- M0、M1、M2、M4 各 32 张图与已有完整验证预测的掩码逐一一致，置信度误差不超过 10⁻⁶。M3 使用与定性展示相同的整图点细化后概率细化路径，其候选数与 M2 一致；不把此检查称为完整验证集 AP 评测。
- GPU 上保留原有演示服务，未停止服务。通过 NVML 记录其他进程活动，计时区间及前后 1.5 秒内未检测到其他进程的 SM/显存控制器活动。活动采样约为 1 秒，不能保证排除更短的服务请求；共享主机 CPU 也可能受其他任务影响。
- 本轮与历史 Tiny 的 A800 测速硬件不同；本轮五组内部可按统一条件比较，不将两轮时间差单独归因于骨干变化。
- 原始协议、权重哈希、逐图计时、GPU 活动记录及退出码保存在 `{run.relative_to(ROOT)}`。

## 复现

```bash
.venv/bin/python -u tools/benchmark_course_swinl.py \\
  --output reports/course_efficiency_swinl/new_run \\
  --gpu 4 --images 32 --repeats 3 --cpu-affinity 32,33,34,35
```

选择可用显卡；若有常驻但闲置的演示服务，可用 `--resident-pids` 显式列出其进程。程序会检查其他进程的计算活动。新运行优先使用 RGB 的 `final.pt`，尚未完成时固定当时的 `best.pt`，并在协议中记录实际轮次。
'''
    docs = ROOT/'docs/results'
    (docs/'course_efficiency_swinl.md').write_text(document)
    artifact = dict(schema='dpseg-swinl-efficiency-report-v1', protocol=p,
                    results=rows, summary_sha256=sha(run/'summary.json'), raw_run=str(run))
    (docs/'course_efficiency_swinl.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2)+'\n')
    with (docs/'course_efficiency_swinl.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream, lineterminator='\n')
        writer.writerow(['配置','参数量_M','推理权重_MiB','平均耗时_ms','FPS','权重说明'])
        for row,label in zip(rows,LABELS):
            writer.writerow([label,row['parameters']/1e6,row['checkpoint_bytes']/2**20,
                             row['mean_ms'],row['fps'],f'RGB epoch {p["rgb_epoch"]}' if row['stage']==0 else
                             'RGB-D epoch 15'+('; depth head epoch 6' if row['stage']>=2 else '')])
    readme=ROOT/'README.md'
    old=readme.read_text()
    start,end='<!-- inference-efficiency:start -->','<!-- inference-efficiency:end -->'
    body=f'''{start}
### 模型规模与推理开销对比

{condition}

{table}

{note}

参数量包含冻结骨干；权重文件只计推理使用的基座和深度点头。概率图细化与切片不增加参数。计时包含预处理、模型、对应的细化/切片及 CPU COCO RLE 编码，不含磁盘读取、图像解码与模型加载。本次不统计 GPU 峰值显存。

测量口径、共享服务器限制及逐轮结果见[当前效率报告](docs/results/course_efficiency_swinl.md)，另提供[CSV](docs/results/course_efficiency_swinl.csv)和[JSON](docs/results/course_efficiency_swinl.json)。[历史 Tiny＋XYZ 效率报告](docs/results/course_efficiency.md)保留原始测量，硬件和模块顺序与本表不同。
{end}'''
    left=old.index(start)
    right=old.index(end)+len(end)
    new=old[:left]+body+old[right:]
    new=new.replace('下方效率表仍对应原 S0–S4。','下方效率表已更新为当前 Swin-Large＋无内参配置。')
    readme.write_text(new)
    print(table)
    print(note)
    print('Verified and wrote Markdown, CSV, JSON, and README.')


if __name__=='__main__':
    main()
