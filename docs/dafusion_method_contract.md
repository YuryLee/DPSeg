# DA-Fusion 课程适配：方法约定

## Material Passport

- origin：academic-research-suite / experiment-agent
- mode：方法忠实性审查；不执行训练
- date_utc：2026-10-14
- verification_status：ANALYZED
- 范围：对照原论文与作者公开代码确定适配边界；不代表独立复现，也不证明课程数据上的性能。

## 1. 命名与必须保留的结构

推荐名称为 **“DA-Fusion 公开 DS/DC 重实现的课程适配（双 Swin、DS+DC）”**。下列结构是本项目的实施约定：

| 部分 | 必须保留 | 实施验收 |
|---|---|---|
| 双模态编码器 | 独立 RGB 与 depth Swin 分支、四级空间尺度 | 两路独立参数；输入严格空间对齐；分别记录初始化来源 |
| DS | 各尺度、各模态的可变形自注意力；学习偏移、可微采样、注意力聚合 | 不得用普通卷积、固定采样或普通 gate 替代后仍称完整 DS |
| DC | DS 之后进行 RGB→depth 与 depth→RGB 双向注意力；融合两方向输出 | 检查两方向均参与输出且获得梯度；不得退化为单向加法 |
| 层次融合 | 四尺度分别融合，保持供 decoder 使用的尺度与通道契约 | 逐尺度检查形状与下游消费；不把“多尺度”实现成只在最终层融合 |
| 分割头 | Mask2Former 风格多尺度 pixel decoder、query transformer decoder、实例 mask 预测 | 保留集合匹配与分类/BCE/Dice 训练路径；输出课程可见实例 |

依据：[论文 §III](https://arxiv.org/html/2607.17754v1)、[作者 FusionStage](https://github.com/teamROBI/DA-Fusion/blob/main/dafusion/dafusion/modeling/fusion/fusion_stage.py)。单独给现有模型添加深度 gate 应称“受 DA-Fusion 启发”，不使用上述完整模型名称。

## 2. 论文与当前作者代码必须区分

1. **深度表达**：论文 §III-A / §V-A 使用 HHA；当前 stage1 使用先反投影再逐图标准化的 XYZ、有效性通道及 4→3 输入适配器，并启用深度噪声/空洞增强。标准化后网络输入已非米制 XYZ 数值。课程须记录内参、原深度尺度、无效值规则和几何增强顺序；未经校准不能把重复三通道深度称为 XYZ 或 HHA。[论文](https://arxiv.org/html/2607.17754v1)、[官方 stage1 配置](https://github.com/teamROBI/DA-Fusion/blob/main/dafusion/configs/dafusion_rgbd_uoais_stage1.yaml)
2. **DC 采样来源有实质差异**：论文 Eq.9–10 描述由查询模态预测另一模态的采样偏移；已读官方 `DeformableCrossAttention.forward` 则先让每个模态各自 `sample_kv`，再将 RGB query 与 depth K/V、depth query 与 RGB K/V 交叉注意。课程沿用官方实现时必须保留此披露，不宣称逐公式复现。[官方 DC 源码](https://github.com/teamROBI/DA-Fusion/blob/main/dafusion/dafusion/modeling/fusion/deformable_cross_attention.py)
3. **融合细节**：当前 DC 将两个方向拼接，经 1×1 投影、RGB residual 和 GroupNorm 输出。FusionStage 还提供可选 `GeometryPrior`；它不是本次默认允许悄悄加入的论文核心模块。正式配置应显式记录其开关，以及 heads/groups/stride/range_factor/position_bias。[官方 DC](https://github.com/teamROBI/DA-Fusion/blob/main/dafusion/dafusion/modeling/fusion/deformable_cross_attention.py)、[FusionStage](https://github.com/teamROBI/DA-Fusion/blob/main/dafusion/dafusion/modeling/fusion/fusion_stage.py)
4. **训练配方不同**：论文为 UOAIS-Sim、20 epochs、batch 2；当前 README 为双 Swin-L、两段 cosine、450k + 150k 样本。课程训练不能把任一配方的结果归作原论文复现。[论文 §V-A](https://arxiv.org/html/2607.17754v1)、[官方 README](https://github.com/teamROBI/DA-Fusion)
5. **公开实现的来源关系存在内部矛盾**：仓库根 README 展示论文成绩，但内部 `dafusion/README.md` 将本目录称为重实现、写明完整 20-epoch 训练尚未完成；它又称产生论文数字的 legacy pipeline 仅用晚期平均与灰度深度，而论文写 DS/DC 与 HHA。这些是作者仓库的不同陈述，不能自行判定哪一套实际产生论文数字。本项目仅验证固定源码的课程运行，不背书论文成绩已由公开实现复现。[内部说明](https://github.com/teamROBI/DA-Fusion/blob/c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570/dafusion/README.md)

## 3. 可记录的课程调整与公平比较

- Swin-L→Swin-T、batch/累积步数、AMP、checkpointing、训练轮数、分辨率、设备均可调整；须在模型名、配置和结果中明确。小 backbone 仍需保留双分支、全部四尺度 DS/DC 与 decoder，不能沿用大模型性能宣传。
- PyTorch/Transformers 适配替代 Detectron2 可行，但这属于移植；记录上游 commit、被复用文件 SHA、移植代码 SHA、权重转换清单，并测试形状、采样、梯度与输出一致性。
- 两路初始化、decoder 初始化及未加载参数要逐项记录。若新增分支或融合模块随机初始化，明确其训练成本，不能称“已有完整预训练 DA-Fusion”。
- 比较沿用课程固定划分、原图 mask 面积、同一 COCO maxDets=100 和阈值约定。模型选择只用开发集；先前已查看测试成绩的事实继续披露。不同初始化/预算的总效果不能单独归因于深度融合。

## 4. 指标与后处理边界

论文与仓库均主要报告 Overlap/Boundary P、R、F，以及 Overlap F≥0.75 的对象比例；**该 F@75 不是 COCO AP75，也不是 mask AP_s**。README 的 OCID F@75 写为 92.7，而论文 DS+DC 行为 92.9；本项目不把任一值写作课程预期结果。[论文表 I 与 §V-B](https://arxiv.org/html/2607.17754v1)、[README Results](https://github.com/teamROBI/DA-Fusion)

当前仓库评测还包含 soft-mask 后处理、depth foreground filter、OCID valid-region crop、OCBD depth alignment。它们属于额外评测条件；课程默认不得照搬数据集专用前景规则。使用课程统一 COCO 可见 mask AP/AP_s，与原论文 UOIS 指标分开命名；预测不能访问 GT。

## 5. 固定源码审计：实际行为

2026-10-14 已读取本地固定 commit `c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570`。依据 `third_party/DA-Fusion/source_receipt.json` 独立重算全部 **100 文件 Git blob SHA-1 与 SHA-256，0 不一致**。源码入口均在其 `dafusion/dafusion/` 下；以下以实际代码为准。

| 检查项 | 固定源码行为及课程要求 |
|---|---|
| 双 Swin forward | `modeling/backbone/dual_fusion_backbone.py` 先完整计算 RGB、depth 四尺度输出，再分别融合；**融合结果不回馈后续 Swin stage**。HF 包装应保持这一连接关系。 |
| offset / K,V | `fusion/deformable_self_attention.py` 与 `offset_network.py`：按 4 个 channel group 预测共享采样场；参考 key 网格按 `[8,4,2,1]` 降采样。每组共享的一组 K/V 供所有 query 访问，不是每 query 仅 K 个独立点。偏移为 `tanh(off) * 2/max(Hk,Wk)`；双线性采样后作 K/V 1×1 投影。 |
| 采样坐标偏差 | `sampling.py` 用像素中心公式 `2*(i+.5)/size-1` 建 grid，但 `grid_sample(...align_corners=True, padding_mode='border')` 取样。两者 convention 不一致；例如零偏移、同尺寸时并非精确恒等采样。它也未显式 clamp 偏移坐标，越界由 border 模式处理。保留源码需披露；若修正需另记算法变体并单独验证。 |
| RPB | `relative_position_bias.py` 使用 2→32→heads 的连续 MLP、**未偏移** reference query/key 坐标；并非逐组真实变形点或离散偏置表插值。`Nq*Nk > 4,000,000` 时返回 None。它是公开实现本身的近似。 |
| XYZ 训练顺序 | `data/dataset_mappers/dafusion_mapper.py`：原始深度可先加噪→按原内参反投影→与 RGB 同一几何变换作用于 XYZ；valid 先生成再 nearest 变换；水平翻转另将 X 取反→对变换后的 valid 点逐通道 median 居中，除以距中心半径的 p95+1e-6→无效点归零→附 validity。反投影已在原图完成，此路径不能再套一遍 resize 后内参。 |
| XYZ 推理顺序 | `engine/predictor.py` 先对原图 XYZ 标准化，之后才可 resize；与训练有次序差异。课程须选择并记录其顺序，避免把“先 resize raw-depth 再反投影/标准化”误称逐行复用官方 mapper。 |
| 默认开关 | `config.py` 中 `GEOMETRY_PRIOR=False`、`DEPTH_FROM_SCRATCH=False`、`FREEZE_RGB_BACKBONE=False`、`BOUNDARY_WEIGHT=0`、`AMODAL_WEIGHT=0`；stage1/2 未覆盖这些。不能因类注释写 XYZ from-scratch 就实际采用随机 depth 初始化。 |
| adapter / loss | stage1 启用 DepthAdapter，hidden 来自配置默认 128；末投影零初始化，skip 为前三个 XYZ 通道。stage1 的 no-object weight=0.05；base 为0.1。分类/BCE/Dice 权重分别2/5/5，100 queries。HF 移植如不同，应列入适配清单。 |

源码引用：[backbone](https://github.com/teamROBI/DA-Fusion/blob/c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570/dafusion/dafusion/modeling/backbone/dual_fusion_backbone.py)、[fusion 目录](https://github.com/teamROBI/DA-Fusion/tree/c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570/dafusion/dafusion/modeling/fusion)、[mapper](https://github.com/teamROBI/DA-Fusion/blob/c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570/dafusion/dafusion/data/dataset_mappers/dafusion_mapper.py)、[depth_xyz](https://github.com/teamROBI/DA-Fusion/blob/c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570/dafusion/dafusion/data/depth_xyz.py)、[config](https://github.com/teamROBI/DA-Fusion/blob/c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570/dafusion/dafusion/config.py)。

## 6. 800×800 输入的成本与等价优化

按默认采样 stride、8 heads、batch 1，空间尺寸与注意力矩阵规模如下。这里只计算一个注意力调用的一张 fp16 logits 矩阵，**不是实测峰值显存**；每个 FusionStage 有 RGB DS、depth DS、两个 DC 方向，共四次注意力，训练还需保存激活和梯度。

| feature 尺寸 | key stride / 数量 | query×key | RPB | 单张 fp16 logits |
|---|---|---:|---|---:|
| 200×200 | 8 / 625 | 25,000,000 | 自动跳过 | 381.47 MiB |
| 100×100 | 4 / 625 | 6,250,000 | 自动跳过 | 95.37 MiB |
| 50×50 | 2 / 625 | 1,562,500 | 开启 | 23.84 MiB |
| 25×25 | 1 / 625 | 390,625 | 开启 | 5.96 MiB |

Swin-T 减少 channel 与骨干成本，**不改变上述 attention pair 数**。最高层的公开代码采用无 bias 的 SDPA，但其 reshape/transpose 后 Q/K/V 末维不连续（本地 CPU 检查 Q stride 为 `(3840000,480000,1,40000)`）。这可能阻止高效 CUDA backend，不能仅凭函数名就声称低显存。

可优先做的数学等价优化：

1. Q/K/V 末维 contiguous，保持所有值、顺序与缩放不变；实测并记录实际 SDPA backend。禁止因内存压力静默关闭 RPB。
2. 对 query 维分块，**每块仍访问完整 K/V**；先全图预测偏移/采样 K/V，再分块注意力。带 RPB 时按块生成相同 MLP bias；阈值跳过判断仍用完整 `Nq*Nk`，不能因切块而意外启用本应跳过的 RPB。
3. 对 Swin 与 FusionStage 使用 activation checkpoint；保证 RNG 保持、参数共享与梯度不变。AMP/累积梯度可记录使用，但都需有限值检查。
4. 可将带 bias 的显式 matmul/softmax 改用支持相同 additive bias 的 SDPA；须做输出与梯度容差测试，且不能保证所有硬件都有高效后端。

扩大 key stride、缩小输入、删除最高分辨率融合、只留单向 DC、截断 query 的 key 集合均会改变算法或分辨率，不能列为纯实现优化。固定模型包装完成后仍需独立前向/反向与真实深度对齐测试；本次未进行 GPU 实验。

## 7. 课程包装的独立工程检查

2026-10-14：`tests/test_course_dafusion.py` 的 **7 项 CPU 测试通过**（4 threads，4.219 秒）。覆盖有/无 bias 的原函数与 contiguous attention 输出及输入梯度对齐、四条 DS/DC offset 梯度、RPB cutoff、私有 import patch 边界、真实 COCO 双骨干等值且独立存储、decoder 原值继承、64px 四尺度与深度敏感性、BF16 native loss 反向、XYZ helper/独立重叠 GT/预测接口集成。包装的 adapter hidden 已对齐官方 stage1 实际默认128。

本次模型源码 SHA-256：`18ff74462c8c31c9a88550a2dfded24e52e01b2fcf47c671ab0442ed1ca83cec`；depth helper SHA-256：`f6fcb8b9f9f0cd5d62cd5f0ff0a6d1a65d767fa9cd359c10a3eba5a00eaf7f96`。源码后续改变需重跑相应检查。这些工程测试不包含800px CUDA显存/吞吐实测，不证明课程 AP 或论文指标复现。
