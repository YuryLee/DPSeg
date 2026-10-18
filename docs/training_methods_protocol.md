# v5：论文方法的训练对照协议

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Mode: run / validate
- Date: 2026-10-02
- Version: training_methods_v5
- Status: 固定于正式训练与新模型开发评分之前；先做不读取开发/测试成绩的工程冒烟。
- User authorization: “可以再尝试一下相关方法”。本轮继续实际实现、训练和比较；无需新增数据下载，不启用代理。

## 问题与四个训练组

固定 E2（RGB-D Mask R-CNN R50-FPN v2）、数据与训练预算，比较：

1. **baseline**：原模型、原损失，在本轮同卡重新训练。
2. **bce2**：原 mask BCE 额外加一次，作为掩码损失权重增加的对照；其他损失不变。
3. **bace**：原 mask BCE + BACE 辅助项，各权重1。按正ROI对应类别取28×28原始logits，沿用torchvision投影得到的软GT。scale=4的maxpool后nearest还原，辅助logit `q = block_max(target) + logit - block_max(logit)`，以q对原target计算BCEWithLogits。maxpool为非线性，不称线性投影；不先sigmoid。该项仅训练使用，推理仍输出原mask head。
4. **freqfusion**：在RGB FPN三个自顶向下连接加入完整FreqFusion核心（ALPF/AHPF/偏移采样），保留原ResNet和FPN侧向/输出卷积、E2深度路径以及ROI head。

BACE依据 [Pan/Pei 2025论文](https://arxiv.org/abs/2504.19643)及作者BARIS-ERA版本代码的默认算式；只迁移其损失，不同时加入decoder或环境adapter。FreqFusion依据 [Chen等 TPAMI2024](https://arxiv.org/abs/2408.12879)及作者固定commit `3fb0c70637a3c194fb74294d3ce4681958b26241` 的带Apache许可证子树，属于torchvision/RGB-D适配而非原论文完整复现。来源和版本差异分别记录在 `reports/v5/boundary_method_notes.md`、`reports/v5/freqfusion_implementation_audit.md`。

FreqFusion固定：每级256通道，压缩(256+256)/8=64；lowpass5/highpass3；resample=True/groups4；up_group1；semi_conv/comp_feat_upsample/hr_residual/Hamming/normalization启用。采用与零越界语义一致的纯Torch CARAFE，32通道分块并checkpoint重算控制显存。保留全部核心，不采用作者根备用实现的reflect边界。保留作者实际Xavier初始化，不假设它是恒等初始化。额外314070个参数。

## 同预算、随机流与原权重

- 四组均从同一份本地COCO预训练初始化重训，seeds **17/23/41**，共 **12次、每次800步**。不用旧800步checkpoint续训，不挑中途checkpoint。
- 数据 `data/graspnet_expanded_v2` 固定：64训练帧、4开发帧、24测试帧；object单类。训练/开发/测试small分别8/4/1。
- 每次batch2，AdamW，lr0.0002，weight_decay0.0001，clip norm5；原RGB参数lr×0.1，depth/heads及新FreqFusion参数使用全lr；warmup100、原cosine规则；数据增强、crop概率0.35、训练尺度480/640/720、评测720/1280均照旧。
- 先构造共享模型并载入shape-compatible预训练，不打开foreground-mean-init，再装新模块；原FPN参数对象/名称保留。扩展只用CPU默认生成器的独立seed与fork，不额外重置CUDA随机流。共享初值与参数组以测试/记录核对。
- 样本选择 `Random(seed)`，增强 `Random(seed+1000)`，不被新模块调用。每步保存样本索引与增强随机状态摘要，跨四组同seed核对。方法改变proposal分布后，内部ROI/RPN随机采样可能自然分化；grid_sample CUDA反向也不能宣称逐位确定性。
- 新模型、新权重和runner独立保存，不覆盖旧源码或已有Demo权重。全部运行来源、参数组、数据/协议/源码/权重SHA及退出状态留存；非空输出目录禁止覆盖。

## 硬件、工程检查与执行边界

仅用当前空闲的物理 **GPU5（A100 40GB）** 顺序执行正式训练/评测，CPU线程4。已有Demo在GPU4运行。训练的进程显存上限设总显存0.55，约22GiB，不代表已在某款24GB卡上验证性能。

正式训练之前，允许每组8步训练冒烟以及纯训练样本的最大尺寸前向/反向、存取、有限梯度、内存和速度检查；这些输出与正式目录分离，不读取val/test分数，不作为800步模型成绩。修正确定的实现错误须记录，不能用冒烟调开发精度参数。正式每训练进程上限1800秒，每评测进程上限900秒；超时/失败记录保留，不静默替换种子、删结果或减步数。持续监测进程、loss增长和完成状态。

## 固定评测矩阵

- 12个最终模型全部评 GraspNet val/test，共24组。
- 四种方法的seed17全部评既有T-LESS100帧压力子集，共4组；完整保留52small和重叠visible GT。
- 合计 **28组完整评测**，所有方法包括权重对照均报告。始终使用原全图推理，score0.01、mask0.5、max100；不叠加上一轮highres/crop_fuse。
- 原图COCO十二项，主要开发指标APsmall，护栏总体AP；保存原图预测RLE与有序ID，GT不进入推理。T-LESS raw depth×0.0001米，GraspNet×0.001米；不能让checkpoint训练深度系数覆盖跨集输入系数。
- 同卡batch1/float32，3次预热；计时覆盖内存输入、传输、模型、二值化/CPU回传，不含读图、RLE编码、COCO或网页。记录参数量、文件大小、均值/p50/p95、已分配峰值及保留缓存。所有组重新计时，不直接比较异卡历史耗时。

## 事先规定的判断与展示

三种改法均与同seed重训baseline作配对：开发APsmall平均增益至少2个百分点、至少2/3种子正增益、平均总体AP下降不超过1个百分点，才达到可选实验方案门槛。若多个通过，按开发APsmall均值降序，其次总体AP均值降序，仍同分则开发平均耗时升序。bce2同样可入选，但须如实称简单损失加权对照。

若BACE胜过baseline，但没有胜过bce2，不把收益特指为其边界算式有效；同时报告BACE−bce2配对差。FreqFusion的效果描述包含新模块和新参数学习率安排，不能把全部收益归于某个滤波子机制。两方法不组合，也不依据新test/T-LESS成绩调权重、scale、kernel或裁块参数。

在线默认基线保留。新增实验报告与可视对照；通过门槛也仅称当前小数据上的候选，不自动宣布一般化优势。所有评测集均已被反复查看，属于探索性复用；开发只有4small、GraspNet test只有1small，三种子不是三份独立数据集。报告逐种子、均值和样本SD；不作显著性声明、不把T-LESS成绩混入GraspNet均值。

## 工程修正记录：正式评分前

首次执行中，seed17 baseline/BCE2完成800步；BACE在第13步被本项目新增的严格目标范围检查中止。重复相同输入与学习率流程定位到ROIAlign单个float32目标为1.0000001192092896，恰好比1多一个机器epsilon。原BCE及作者辅助算式可接受该浮点误差；问题出在新增校验，不是标注超出0/1或训练损失发散。诊断与全部原日志/源码存于 `reports/v5/attempt1/`、`reports/v5/attempt1_source/` 和 `reports/v5/bace_failure_diagnostic_range.log`。

修正仅让范围校验容忍不超过8个目标dtype机器epsilon（float16/bfloat16按float32精度）的越界，目标值原样传入BCE与辅助算式，不夹断、不改loss权重、scale或ROI采样。新增越界精度单元测试，允许BACE做32步训练样本检查以覆盖原先8步未触发的问题。修正后再次冻结源码与本协议，全部12组从头统一重训；原两份完整训练权重和失败组归档在 `outputs/paper_v5_attempt1/`，不进入最终成绩。新模型尚未做任何开发/测试评分，此次没有根据模型精度改动方案或门槛。
