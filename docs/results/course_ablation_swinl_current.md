# 当前主表：基线对比与累计模块消融

当前论文主表采用S0–S4累计添加模块的结果。已完成的[七组独立去除模块实验](course_baselines_removal_current.md)作为补充结果保留。配套[模型规模与推理开销](course_efficiency_swinl_current.md)使用相同五组配置和最终权重。

## Material Passport

- 来源：已完成的 Swin-Large 累计消融和 640 切片合并对照；本次仅更新当前配置的结果呈现。
- 选择日期：2026-11-06（Asia/Shanghai）；用户指定 `global_guard_edge_consensus` 为当前完整版。
- 核验：五行原始评分 JSON 与提交 CSV 的 SHA-256 均重新核对一致；全部指标从原结果直接读取，未重新训练或生成新指标。
- 范围：完整本地测试集 2,112 张；新合并选择发生在查看探索性测试结果后，无验证集选优或统计显著性结论。

## 当前对比表

所有配置均采用 **Swin-Large、1024×1024 输入**，RGB-D 分支使用无需相机内参的深度表示。以下为**完整本地测试集 2,112 张图**的 COCO 掩码评测，数值范围 0–1；各行按表中顺序累计添加模块。当前 S4 使用 **640×640 切片＋整图优先＋截断处理＋跨视图一致性**，加粗标记完整 DPSeg。

| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |
|:---:|:---:|:---:|:---:|
| S0：Mask2Former | 0.205838 | 0.293236 | 0.734611 |
| S1：+ RGB-D 融合（DA-Fusion） | 0.408797 | 0.481296 | 0.845626 |
| S2：+ 深度感知点细化 | 0.403831 | 0.481031 | 0.862922 |
| S3：+ 深度引导概率图细化 | 0.476384 | 0.569641 | 0.886068 |
| **S4：+ 切片增强推理（DPSeg）** | **0.528710** | **0.674606** | **0.887453** |

APₛ、ARₛ 分别表示小目标掩码平均精度和平均召回率，均在 IoU=0.50:0.05:0.95 上平均，最多保留 100 个实例；AP 为所有尺寸的掩码平均精度。小目标使用原始标注面积 [0, 32²] 像素。以上分数与 Kaggle 公榜分开报告。

- **固定权重：** S0 为 RGB Mask2Former 第 15 轮；S1–S4 共用 RGB-D 第 15 轮基座，S2–S4 共用验证集选取的第 6 轮深度点头。RGB 与 RGB-D 均从相同公开 COCO Swin-Large 权重开始训练，使用完整训练集、15 轮和全局 batch 33。
- **累计配置：** S2 为整图点细化；S3 在 S2 后增加概率图细化，不使用切片；S4 保留两种细化，增加整图＋四个重叠切片，并采用当前合并策略。因此 S3→S4 表示整套切片增强推理的增量，包含候选选择规则。完整系统先合并候选，再做来源视图点细化和原图概率图细化。
- **选择依据：** 第 6 轮点头由完整验证集选择；当前合并策略由用户在查看探索性本地测试对比后指定。合并推理不读取 GT，但不能将此次配置选择称为验证集选优，或将该测试集称为从未查看的独立最终测试集。
- **比较范围：** RGB-D 增加独立深度编码器和融合模块，点头增加 6 轮训练，切片增加推理计算。结果来自单次 seed=17 训练，显卡与每卡批次分配存在差异；不构成统计显著性结论。

S2 相对 S1 的 APₛ 变化为 **-0.004966**，整体 AP 变化为 **+0.017296**，不能概括为每个模块均提升小目标 AP。当前 S4 在这五行中的 APₛ、ARₛ 和 AP 均最高；相对 S1 分别提高 **+0.119913、+0.193310、+0.041826**，相对无切片 S3 分别提高 **+0.052326、+0.104965、+0.001385**。

旧 S4 的 APₛ/ARₛ/AP 为 0.525286/0.674087/0.880289。只替换候选合并策略后，三项分别提高 **+0.003424、+0.000519、+0.007164**；模型权重、四个 640 切片和两种细化均保持不变。当前合并的七组独立去除实验和五组统一推理测速均已完成；当前合并尚无新的完整验证集精度或Kaggle成绩。

## 数据来源与历史快照

S0–S3 直接保留[原 Swin-Large 累计消融](course_ablation_swinl.json)的逐行指标、预测路径和文件校验值；S4 采用[640 合并对照](course_guarded_merge640.json)中 `global_guard_edge_consensus` 的真实结果。原报告和原 JSON 保留当时的配置、数值及 `default_changed=false`，不回写历史实验状态。

冻结的指标来源摘要记录各行全部12项COCO指标、原始评分/预测/提交路径及SHA-256、所用权重、策略和生成时的测量状态；最新测速完成状态见[当前效率报告](course_efficiency_swinl_current.md)：[JSON](course_ablation_swinl_current.json) · [CSV](course_ablation_swinl_current.csv)。

## 提交文件本机路径

- **S0：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_rgb_v1/submissions/mask2former_swinl_rgb_1024_e15.csv`
- **S1：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_native.csv`
- **S2：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_report_ablation_v1/submissions/dpseg_swinl_depth_no_k_s2_depth_point.csv`
- **S3：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_report_ablation_v1/submissions/dpseg_swinl_depth_no_k_s3_depth_point_probability.csv`
- **S4：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard_edge_consensus.csv`

当前 S4 CSV 的 SHA-256 为 `6f75831ace88fa4145311b749311c70ff21ebeaf9640473f18cabc5238226024`。这些 CSV 与实际评分输入逐行绑定，完整 RLE 校验记录见原始实验报告。新合并 CSV 尚未提交 Kaggle；历史公榜 0.56785 属于原分数优先的跨视图掩码 IoU 合并。

## 推理与复现

当前推理入口为 [predict_dpseg.py](../../tools/predict_dpseg.py)，配置见 [dpseg_current.json](../../configs/dpseg_current.json)，运行时实现见 [course_dpseg.py](../../rgbd_seg/course_dpseg.py)。候选选择规则沿用已完成实验的 [course_guarded_merge.py](../../rgbd_seg/course_guarded_merge.py)。原始实验脚本与冻结协议保留用于追溯旧结果。

原始五组合并对照的完整复现和核验命令见[640 合并实验记录](course_guarded_merge640.md)。该实验的共享前向与细化耗时不能作为当前配置的独立速度测量。
