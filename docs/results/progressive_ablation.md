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

完整系统S4相对RGB基线S0的APₛ绝对提升为 **+0.302520**。本轮完整测试中，四个新增步骤的APₛ增量均为正，其中RGB-D融合的增量最大。

- **统一基座预算：** RGB与RGB-D均从COCO公开权重开始，Swin-Tiny、1024输入、完整训练集、15轮、全局batch33和相同两阶段AdamW学习率计划；主表固定第15轮权重。
- **固定组件：** S1–S4共用第15轮RGB-D基座，S2–S4共用验证集选择的第6轮深度点头；S3关闭概率细化，S4仅增加既定概率图深度细化。
- **评价顺序：** 完成2,584张验证图的五组评价并锁定配置后，再评价完整本地测试集。置信度阈值0.01、每图最多100实例；测试标注不进入推理或参数选择。
- **比较范围：** RGB-D包含额外编码器及DS/DC融合；历史RGB-D与本次RGB的显卡及局部batch划分不同，不能视为逐步梯度完全一致的控制实验。点头另有6轮训练，切片增加推理计算量。

完整验证指标、五份CSV本机路径和核验说明见下文，原始数值及权重摘要见[结构化结果](progressive_ablation.json)。本表为本地完整测试结果，Kaggle公榜成绩见[历史结果](../historical_results.md)。

## 完整验证结果

| 编号 | 配置（累计添加） | 验证AP | 验证APₛ |
|---|---|---:|---:|
| S0 | Mask2Former fine-tune | 0.668338 | 0.343147 |
| S1 | + RGB-D 融合（DA-Fusion课程适配） | 0.814871 | 0.503742 |
| S2 | + 深度点细化 | 0.846843 | 0.547258 |
| S3 | + 切片推理 | 0.848502 | 0.611528 |
| S4 | + 概率图深度细化 | 0.861748 | 0.646540 |

## 训练设置

RGB与RGB-D均从公开COCO权重初始化，使用Swin-Tiny、1024输入、完整14,432张训练图、seed17和全局batch33，固定第15轮权重。AdamW权重衰减0.05，骨干学习率倍率0.1，梯度裁剪1。前5轮峰值学习率0.0002，预热250步后余弦衰减；后10轮保留权重和优化器状态，从0.00002预热100步至0.00005后余弦衰减，学习率下限均为峰值的0.1。

RGB前5轮使用GPU 2、3、7，局部batch为4、4、3，累积3次；第6–15轮使用GPU 2–7共6张卡，局部batch为2、2、2、2、2、1，仍累积3次，全局batch保持33。续训保留AdamW状态并按新rank布局重设随机状态。初始训练和续训均正常退出，完成总计6,570次更新及15次完整验证。

RGB-D为此前完成的独立训练，显卡布局和局部batch划分与RGB不同。S1–S4共用RGB-D第15轮基座，S2–S4共用额外训练的第6轮深度点头；切片和概率细化均不增加训练参数。协议文件见[文件说明](../artifacts.md)。

## 测试CSV本机路径

每份文件均包含2,112张图，与该行评分使用的逐图预测完全一致。掩码为1024×1024的COCO压缩RLE，每图最多100实例。

- **S0 Mask2Former fine-tune**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/mask2former_finetune/dpseg_s0_swin_t_1024_e15_mask2former_finetune.csv`
- **S1 + RGB-D 融合（DA-Fusion课程适配）**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/rgbd_fusion/dpseg_s1_swin_t_1024_e15_rgbd_fusion.csv`
- **S2 + 深度点细化**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/depth_point/dpseg_s2_swin_t_1024_e15_depth_point.csv`
- **S3 + 切片推理**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/depth_point_tiled/dpseg_s3_swin_t_1024_e15_depth_point_tiled.csv`
- **S4 + 概率图深度细化**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_progressive_ablation/v1/evaluation/test/depth_point_tiled_probability/dpseg_s4_swin_t_1024_e15_depth_point_tiled_probability.csv`

## Kaggle提交

S4已于2026-10-19提交，编号57398816，状态COMPLETE，公榜APₛ为 **0.52886**，较此前最高0.47971增加0.04915；本次查询中为本队最高公榜分。其评分范围约为官方测试集的30%，与上方完整本地测试结果分别列报。

上传文件为上述S4 CSV，SHA256为`0ba8eaa224c171181984dfa3334f96ebe041370b34024d44f5d480a31b079cf1`，与本地评分产物一致。来源与核验时间见[提交记录](kaggle_s4_e15.json)。

## 核验与复现

- RGB基线完成5+10轮完整训练和逐轮完整验证；总计6,570次优化器更新。使用固定第15轮，而非依据测试结果选择轮次。
- 新切片实现通过8张真实验证图的旧实现对照，检查五个视图的零残差点头、最终RLE与原实现一致。RGB几何输入不变性检查通过。
- 所有推理分片完整覆盖目标划分，合并后进行COCO评分；实际进程退出状态和权重摘要均保存。
- 概率细化两组共享候选、置信度与顺序。未调整既有阈值、点数、切片和深度细化参数。
- 运行目录：`reports/course_progressive_ablation/v1/`；评价入口：`tools/supervise_course_progressive_ablation.py`；RGB前5轮入口：`tools/supervise_course_progressive_rgb.py`；后10轮六卡续训入口：`tools/supervise_course_progressive_rgb_multigpu.py`。

结构化指标与权重摘要见[progressive_ablation.json](progressive_ablation.json)。五组对应的三份权重已整理至[Hugging Face权重](../model_weights.md)，保留原始字节和SHA-256；数据、大型预测和CSV仍通过课程本机目录共享。

### 限制

- 只运行一个随机种子，基座固定第15轮，点头沿用此前按验证集选择的第6轮；未估计多次训练的波动。
- 两种基座匹配输入分辨率、训练轮数、全局batch与两阶段学习率计划；显卡、局部batch划分和随机轨迹不同，不能视为逐步梯度完全一致的实验。
- RGB-D同时增加第二路编码器和DS/DC融合参数，S0→S1衡量整个融合方案的差异。
- 后续各组共享冻结基座，点头额外训练6轮；切片增加推理计算量。
- 本地测试使用下载归档中的标注，评分范围与Kaggle公榜约30%的子集不同。本轮仅S4已提交Kaggle，其余四份CSV未提交。
