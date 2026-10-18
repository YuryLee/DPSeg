# 近期论文启发的推理改进：冻结实验协议

冻结时间：2026-10-02 11:17 UTC，执行模型评分之前。

Material Passport：Origin Skill `academic-research-suite/experiment-agent`；模式 `plan → run → validate`；版本 v4；当前状态 PLANNED。用户授权“调研近期论文……尝试一下”。不下载新数据、不启用代理。

## 研究问题与来源

当前 GraspNet 小目标多数已经分配到最细 P2，仍可能在缩放与特征提取时丢失细节。固定已经训练好的模型，增加推理时的有效像素，是否改善小实例 mask AP？提高整图分辨率与裁块合并的收益/成本分别怎样？

工程灵感来自 [Huang 等，ICCV 2025 Workshop R6D](https://openaccess.thecvf.com/content/ICCV2025W/R6D/papers/Huang_Lessons_and_Winning_Solutions_in_Industrial_Object_Detection_and_Pose_ICCVW_2025_paper.pdf) 的 Crop & Fuse。论文用 YOLOv12-X 检测、SAM2 分割，带类别/工作区先验；本文保留原 RGB-D Mask R-CNN。论文的原图工作区边界过滤、包含比例过滤与本项目规则不同。以下是项目改编，不是论文复现。其检测 AP 消融不当作本项目 mask APsmall 的预期收益。

## 预先固定的三个设置

| 名称 | 固定规则 |
|---|---|
| full | 原尺寸输入，模型短边 720，长边上限 1280，原有候选规则。 |
| highres | 原尺寸输入，临时将短边改为 1152、长边上限 2048，推理后恢复设置。 |
| crop_fuse | full 加四角重叠裁块，裁块宽高各为原图相应边长的 ceil(0.625×尺寸)，重复坐标去重。裁块仍用短边 720/上限 1280。 |

共同规则：RGB、米制深度与有效像素图同步裁切；mask 阈值 0.5，返回原图坐标二值掩码；移除空 mask。模型权重、类别表、评分阈值 0.01、候选数设定均不改变。不从 GT 选择裁块或候选，不更新权重。

crop_fuse 的额外固定规则：只保留裁块预测中原图坐标 mask 面积在 (0,96²] 的候选；若 bbox 距离裁块内部边界 ≤2 像素则舍弃，原图外边界不受此过滤。全图候选不受上述面积/内部边界规则限制。将所有候选映射回原图，按类别执行 bbox NMS（IoU=0.5），按分数保留最多 100 个。这里不是 mask NMS；重叠 bbox 的独立实例可能被抑制，跨缝目标可能丢失。全图分支提供部分回退。

highres 与 crop_fuse 不是等算力对照。裁块同时改变像素尺度、上下文、视图数与候选合并，结果不能全部归因于分辨率。

## 运行矩阵与结果选择

- GraspNet val/test：E2 seeds 17、23、41，每个 checkpoint 测三个设置。
- GraspNet val/test：E0 seed 17，三个设置，检查收益是否同样存在于 RGB。
- T-LESS test：E0/E2 seed 17，三个设置。仅压力测试，不参与策略选择。
- 共 10 个 checkpoint×split 组合、30 组评分。所有预定结果都报告；失败保留日志，不以替换样本或静默更改规则补救。

**开发集工程门槛**：相对 full，E2 三种子 val APsmall 平均增加至少 2 个百分点，至少 2/3 种子增益为正，平均总体 AP 下降不超过 1 个百分点。通过门槛的设置可以加入 Demo 的可选增强模式；默认仍保留 full。若两者都通过，以开发 APsmall 均值更高者优先，完全相同则以开发平均耗时更少者优先。未通过的设置也保留在实验报告中，不用 test/T-LESS 选择。

该门槛是事前工程判断，不是统计显著性阈值。所有 GraspNet val/test、T-LESS 都已在前轮查看；本轮是复用数据的探索性实验，没有新盲测集。禁止查看结果后调整这些参数，若另作后续实验须单独编号并说明。

## 测量与可复现性

- 选用当前空闲 GPU 4（NVIDIA A100 40GB），仅此一张；CPU 线程 4，batch=1，float32。
- 每个 checkpoint 每个设置预热 3 次。逐图计时从输入搬运前至预测回贴/二值化/合并并复制到 CPU 后，前后 CUDA 同步；不含文件读取、COCO评分、PNG渲染和HTTP通信。报告均值、中位数、p95、逐图值与峰值已分配/保留显存。对照在本卡重新计时，不与前轮不同GPU的速度直接比较。
- 精度使用官方 COCO 十二项 mask AP/AR（0–100），面积以原图 GT mask 为准，结果输出 RLE。GraspNet 可补充原有边界/重合指标；T-LESS 保留重叠 GT 独立mask，仅用支持该格式的COCO评价。
- 保留 checkpoint SHA256、训练 run、协议和实现 SHA256、评测原始数据指纹、库版本、GPU名称、命令、参数量及文件大小。full 与历史同 checkpoint 同 split 的十二项结果核对。
- E2 报告每个种子的配对差、均值与样本标准差；3个种子不是3份独立数据集。开发集仅4个小GT、GraspNet test仅1个，不做显著性检验或泛化胜出声明。T-LESS 52个小GT也有同物体多视角相关性。
- 权重不变意味着参数量/文件大小不变；吞吐、显存、误检和截断实例仍可能变化，须一并报告。

## 已知限制与后续方向

课程正式数据/类别仍未确认，本项目是单类 object 子集。该实验仅检验既定权重上的推理策略，不证明深度融合本身有效。FreqFusion（TPAMI2024）的可学习 FPN 融合、QuBER（ICRA2025）的边界错误监督、DGGM（LSP2025）的深度梯度引导作为另列候选；不同时加入本实验，以保持解释范围清楚。
