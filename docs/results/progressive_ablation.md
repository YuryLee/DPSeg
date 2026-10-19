# 五步累计消融

## Material Passport

- 来源：ARS experiment-agent，run模式。
- 状态：完整训练、验证、锁定后测试与五份CSV一致性检查完成。
- 范围：单种子课程实验；本地测试分数与Kaggle公榜分别报告。

### 五步累计消融

以下各行累计保留上一行的组件。主表统一使用完整本地测试集2,112张图，分数范围为0–1。

| 编号 | 配置（累计添加） | 本地测试 AP | 本地测试 APₛ | APₛ 相对上一组增量 |
|---|---|---:|---:|---:|
| S0 | Mask2Former fine-tune | 0.705677 | 0.185779 | — |
| S1 | + RGB-D 融合（DA-Fusion课程适配） | 0.817895 | 0.368086 | +0.182307 |
| S2 | + 深度点细化 | 0.839838 | 0.380896 | +0.012810 |
| S3 | + 切片推理 | 0.850802 | 0.437560 | +0.056664 |
| S4 | + 概率图深度细化 | 0.868243 | 0.488299 | +0.050739 |

完整系统S4相对RGB基线S0的APₛ绝对变化为 **+0.302520**。负增量同样保留，不能把累计添加组件描述为每一步均有提升。

- **统一基座预算：** RGB与RGB-D均从COCO公开权重开始，Swin-Tiny、1024输入、完整训练集、15轮、全局batch33和相同两阶段AdamW学习率计划；主表固定第15轮权重。
- **固定组件：** S1–S4共用第15轮RGB-D基座，S2–S4共用验证集选择的第6轮深度点头；S3关闭概率细化，S4仅增加既定概率图深度细化。
- **评价顺序：** 完成2,584张验证图的五组评价并锁定配置后，再评价完整本地测试集。置信度阈值0.01、每图最多100实例；测试标注不进入推理或参数选择。
- **比较范围：** RGB-D包含额外编码器及DS/DC融合；历史RGB-D与本次RGB的显卡及局部batch划分不同，不能视为逐步梯度完全一致的控制实验。点头另有6轮训练，切片增加推理计算量。

完整验证指标、五份CSV本机路径、权重与工程核验见[五步消融报告](progressive_ablation.md)，原始数值见[结构化结果](progressive_ablation.json)。本表为本地完整测试结果，Kaggle公榜成绩见[历史结果](../historical_results.md)。

## 完整验证结果

| 编号 | 配置（累计添加） | 验证AP | 验证APₛ |
|---|---|---:|---:|
| S0 | Mask2Former fine-tune | 0.668338 | 0.343147 |
| S1 | + RGB-D 融合（DA-Fusion课程适配） | 0.814871 | 0.503742 |
| S2 | + 深度点细化 | 0.846843 | 0.547258 |
| S3 | + 切片推理 | 0.848502 | 0.611528 |
| S4 | + 概率图深度细化 | 0.861748 | 0.646540 |

## 测试CSV本机路径

每份文件均包含2,112张图，与该行评分使用的逐图预测完全一致。掩码为1024×1024的COCO压缩RLE，每图最多100实例。

- **S0 Mask2Former fine-tune**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/mask2former_finetune/dpseg_s0_swin_t_1024_e15_mask2former_finetune.csv`
- **S1 + RGB-D 融合（DA-Fusion课程适配）**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/rgbd_fusion/dpseg_s1_swin_t_1024_e15_rgbd_fusion.csv`
- **S2 + 深度点细化**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/depth_point/dpseg_s2_swin_t_1024_e15_depth_point.csv`
- **S3 + 切片推理**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/depth_point_tiled/dpseg_s3_swin_t_1024_e15_depth_point_tiled.csv`
- **S4 + 概率图深度细化**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/depth_point_tiled_probability/dpseg_s4_swin_t_1024_e15_depth_point_tiled_probability.csv`

## 核验与复现

- RGB基线完成5+10轮完整训练和逐轮完整验证；总计6,570次优化器更新。使用固定第15轮，而非依据测试结果选择轮次。
- 新切片实现通过8张真实验证图的旧实现对照，检查五个视图的零残差点头、最终RLE与原实现一致。RGB几何输入不变性检查通过。
- 所有推理分片完整覆盖目标划分，合并后进行COCO评分；实际进程退出状态和权重摘要均保存。
- 概率细化两组共享候选、置信度与顺序。未调整既有阈值、点数、切片和深度细化参数。
- 运行目录：`reports/course_progressive_ablation/v1/`；评价入口：`tools/supervise_course_progressive_ablation.py`；RGB训练入口：`tools/supervise_course_progressive_rgb.py`。

结构化指标与权重摘要见[progressive_ablation.json](progressive_ablation.json)。数据、权重、大型预测和CSV由Git忽略，通过课程本机目录共享。

### 限制

- One seed, fixed epoch15 base weights and prior validation-selected epoch6 point head.
- RGB and RGB-D match 1024 input, 15 epochs, global batch33 and two-stage AdamW schedule; historical RGB-D GPU/local microbatch layout differs, so gradients are not bitwise matched.
- Adding RGB-D also adds a second encoder and DS/DC fusion parameters.
- Later stages reuse frozen RGB-D weights; only the point head has additional training.
- Local test scores use archive annotations and differ from Kaggle public subset.
