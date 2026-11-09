# 当前累计配置：模型规模与推理开销

## Material Passport

- 来源：冻结的五组累计配置、最终权重与独立进程逐图计时。
- 测量日期：2026-11-09（Asia/Shanghai）。状态：五组已完成并核验。
- 原始记录：`reports/course_efficiency_swinl_current/v1`；协议、逐图时间、GPU活动和退出码均保留。
- 本表测量运行时间，不新增完整验证集AP评分，不统计GPU峰值显存。

## 报告用表

统一使用 **NVIDIA A100-PCIE-40GB 单卡、FP32、batch=1、1024×1024模型输入**。从验证集固定选取32张等间隔图像，预热5张后重复3轮，每组共96次计时。五组按相同图片与顺序独立执行，完整版使用整图＋4个640×640切片、40%重叠，采用整图优先＋截断处理＋跨视图一致性合并。

| 配置 | 参数量（M） | 推理权重（MiB） | 平均耗时（ms/图）↓ | FPS ↑ |
|---|---:|---:|---:|---:|---:|
| S0：Mask2Former | 215.502 | 826.23 | 418.5 | 2.39 |
| S1：+ RGB-D 融合（DA-Fusion） | 460.955 | 1766.71 | 850.9 | 1.18 |
| S2：+ 深度感知点细化 | 461.205 | 1767.67 | 998.4 | 1.00 |
| S3：+ 深度引导概率图细化 | 461.205 | 1767.67 | 1185.7 | 0.84 |
| **S4：+ 切片增强推理（DPSeg）** | 461.205 | 1767.67 | 4691.1 | 0.21 |

RGB和RGB-D基座均使用最终第15轮权重；S2–S4共用验证集选定的第6轮深度点头。

## 测量口径

参数量包含冻结骨干；推理权重统计基座文件及实际使用的深度点头独立文件，不计优化器和未使用的普通点头。点头增加249,858个参数，概率图细化与切片不增加参数。计时包含预处理、GPU前向、适用的细化/切片/合并、结果传回CPU和COCO RLE编码；排除图片读取与解码、模型加载、文件写入、一致性核对及AP评分。不统计GPU峰值显存。FPS按每张完整原图计算，为1000除以平均毫秒数。

固定CPU逻辑核32,33,34,35；PyTorch/OMP/MKL/OpenBLAS各4线程、interop为1。未启用混合精度、TF32、编译或TensorRT。软件环境为PyTorch 2.7.1+cu126、Transformers 4.53.2、CUDA 12.6。物理GPU为7，UUID `GPU-69de9f7f-ba81-68af-6185-2883a3588963`，开始测量时没有其他计算进程。

## 核验与适用范围

- 每组首先对固定的3张测试图片与此前已评分预测进行逐掩码一致性检查；类别、实例数及次序一致，置信度误差不超过10⁻⁶。此步骤不读取GT、不计算新评分，且不计入耗时。
- S0–S2的32张验证图还与已有对应权重的完整验证预测核对一致；S3/S4仅使用上述3张测试图检查加上32张验证图的三轮预测一致性，不将其称为完整验证集AP评测。
- 每组3轮、共96次计时完整；同图跨轮掩码完全一致，GPU活动检查没有发现其他进程。NVML约每秒采样，不能排除未被采样捕捉的极短活动；共享主机CPU也可能受到其他任务影响。
- 每种配置独立执行完整推理，无共享前向或其他配置的特征缓存。S4使用当前合并策略；旧4794.0 ms/图保留在[历史效率报告](course_efficiency_swinl.md)。
- 相同图片与权重的重复反映运行波动，不代表多次独立训练的统计方差。效率表与[累计精度主表](course_ablation_swinl_current.md)配置和权重一致，测速样本来自验证集。

## 耗时分布

| 配置 | 中位数（ms/图） | P90（ms/图） | 三轮平均耗时（ms/图） | 平均实例数 |
|---|---:|---:|---|---:|
| S0：Mask2Former | 456.7 | 465.9 | 419.1 / 417.8 / 418.7 | 76.8 |
| S1：+ RGB-D 融合（DA-Fusion） | 845.0 | 947.6 | 849.6 / 851.9 / 851.2 | 51.6 |
| S2：+ 深度感知点细化 | 984.0 | 1207.9 | 998.5 / 998.8 / 998.0 | 51.6 |
| S3：+ 深度引导概率图细化 | 1170.4 | 1541.4 | 1176.1 / 1189.0 / 1192.1 | 51.6 |
| S4：+ 切片增强推理（DPSeg） | 4775.2 | 5297.9 | 4749.4 / 4720.3 / 4603.5 | 69.8 |

## 复现

```bash
.venv/bin/python tools/benchmark_course_cumulative_current.py prepare --output reports/course_efficiency_swinl_current/new_run --gpu 7
.venv/bin/python tools/benchmark_course_cumulative_current.py launch --output reports/course_efficiency_swinl_current/new_run
.venv/bin/python tools/publish_course_cumulative_tables.py --run reports/course_efficiency_swinl_current/new_run --publish
```

选择空闲A100，等待五组完成后发布。完整原始数据见[JSON](course_efficiency_swinl_current.json)，表格数据见[CSV](course_efficiency_swinl_current.csv)。
