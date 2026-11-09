# 补充实验：基线对比与独立模块去除

当前论文主表采用[五组累计式消融](course_ablation_swinl_current.md)，本页保留七组独立去除模块的补充结果。

## Material Passport

- 来源：已有累计结果、当前640独立细化去除实验，以及RGB专属点头流程；原始实验与协议均保留。
- 状态：**COMPLETED**，完成并核验 7/7 行。
- 核验：每个完成行核对权重、评分/预测/CSV文件哈希、2112行覆盖、全量RLE校验收据，以及CSV与实际评分输入逐行一致。
- 范围：完整本地测试。RGB去除组在六轮训练、逐轮完整验证及锁定点头的完整测试全部通过后入表。

## 对比表

统一采用 **Swin-Large、1024×1024 模型输入、完整本地测试集 2,112 张图**，分数范围为 0–1。每个 `w/o` 配置从完整 DPSeg 独立去掉相应模块；完整版保持 **640×640、4个切片、40%重叠率**，保留整图并使用整图优先＋截断处理＋跨视图一致性合并。

| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |
|:---|---:|---:|---:|
| Mask2Former | 0.205838 | 0.293236 | 0.734611 |
| DA-Fusion | 0.408797 | 0.481296 | 0.845626 |
| w/o RGB-D 融合 | 0.286780 | 0.440489 | 0.785559 |
| w/o 深度感知点细化 | 0.519366 | 0.663216 | 0.873441 |
| w/o 深度引导概率图细化 | 0.462478 | 0.586535 | 0.865874 |
| w/o 切片增强推理 | 0.476384 | 0.569641 | 0.886068 |
| **DPSeg（完整模型）** | **0.528710** | **0.674606** | **0.887453** |

当前已核验 **7/7 行**完整测试结果。七行均通过完整测试及提交文件核验。

- **Mask2Former、DA-Fusion：** 分别为RGB基线与本项目适配的RGB-D融合基座，均使用既有第15轮权重，不带两种掩码细化或切片。
- **w/o RGB-D融合：** 移除双分支跨模态特征融合，复用已完成的RGB第15轮基座，冻结后训练专属448维特征点头；保留局部深度点线索、深度引导概率图细化和当前切片合并，仍使用深度。专属点头已完成6轮训练，完整验证集选取第 **6** 轮用于测试。 没有重新训练基座，也不直接沿用RGB-D点头。
- **w/o点细化、w/o概率图细化：** 两组均已按当前640合并完成完整测试，候选身份、类别、分数及次序与完整版一致，仅改变掩码细化路径。 保留点头时使用原第6轮RGB-D配套点头。
- **w/o切片增强推理：** 保留RGB-D融合与两种细化，只运行整图；不涉及跨视图合并，复用已核验的完整整图结果。

完整DPSeg相对DA-Fusion的APₛ、ARₛ、AP分别变化 **+0.119913、+0.193310、+0.041826**。独立去除的描述性差值如下，正号表示完整版数值更高：

- 完整模型相对 **w/o RGB-D 融合**：APₛ **+0.241930**、ARₛ **+0.234117**、AP **+0.101894**。
- 完整模型相对 **w/o 深度感知点细化**：APₛ **+0.009343**、ARₛ **+0.011390**、AP **+0.014012**。
- 完整模型相对 **w/o 深度引导概率图细化**：APₛ **+0.066231**、ARₛ **+0.088070**、AP **+0.021579**。
- 完整模型相对 **w/o 切片增强推理**：APₛ **+0.052326**、ARₛ **+0.104965**、AP **+0.001385**。

RGB专属点头和原RGB-D点头分别按各自完整验证集结果选轮；移除融合同时改变基座表征、参数量和点头训练，不能将其差值解释为固定权重下关闭单个算子的效果。其他去除组使用固定权重。实验仅一次训练种子，不作统计显著性结论，也不预设每个模块一定使所有指标提高。当前合并由用户根据已查看的本地测试对比指定，尚无该完整配置的新验证集精度和Kaggle结果；五组累计配置的独立速度见[当前效率报告](course_efficiency_swinl_current.md)。默认继续保持640切片。

[完整定义、核验与CSV路径](course_baselines_removal_current.md) · [指标CSV](course_baselines_removal_current.csv) · [来源JSON](course_baselines_removal_current.json)。原[累计表](course_ablation_swinl_current.md)与[旧合并去除表](course_removal_swinl.md)保留作历史记录。

## 各配置提交CSV

- **Mask2Former：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_rgb_v1/submissions/mask2former_swinl_rgb_1024_e15.csv`
- **DA-Fusion：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_native.csv`
- **w/o RGB-D 融合：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_without_rgbd_v1/submissions/dpseg_swinl_without_rgbd_fusion_guarded640.csv`
- **w/o 深度感知点细化：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_current_removal_v1/submissions/dpseg_swinl_depth_no_k_guarded640_without_point.csv`
- **w/o 深度引导概率图细化：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_current_removal_v1/submissions/dpseg_swinl_depth_no_k_guarded640_without_probability.csv`
- **w/o 切片增强推理：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_report_ablation_v1/submissions/dpseg_swinl_depth_no_k_s3_depth_point_probability.csv`
- **DPSeg（完整模型）：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard_edge_consensus.csv`

## 更新与复核

```bash
.venv/bin/python tools/summarize_course_baselines_removal.py
.venv/bin/python tools/summarize_course_baselines_removal.py --publish
```

第一条仅核验当前已完成结果并显示状态；第二条在核验通过后更新本补充表，保留当前累计主表。缺结果会保留进行中/待测，已有但校验失败的结果会报错，不能降级成已完成。原累计与旧合并实验快照不被覆盖；不训练、不推理、不改默认配置、不提交Kaggle。
