# 深度表示与骨干对比

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: FORMAL_CSV_RESCORED
- 更新：2026-11-04（北京时间）

## 已核验的本地测试结果

| 骨干与深度表示 | 推理配置 | Mask AP | Mask APₛ |
|---|---|---:|---:|
| 原 Swin-Tiny，内参反投影 XYZ | 基础配置 | 0.817895 | 0.368086 |
| 新 Swin-Tiny，无内参深度 | 基础配置 | 0.820201 | 0.362218 |
| 新 Swin-Large，内参反投影 XYZ | 基础配置 | 0.835743 | 0.393056 |
| 新 Swin-Large，无内参深度 | 基础配置 | 0.845626 | 0.408797 |
| 原 Swin-Tiny，内参反投影 XYZ | 完整配置 | 0.868243 | 0.488299 |
| 新 Swin-Tiny，无内参深度 | 完整配置 | 0.872224 | 0.488260 |
| 新 Swin-Large，内参反投影 XYZ | 完整配置 | 0.863532 | 0.499089 |
| 新 Swin-Large，无内参深度 | 完整配置 | **0.880289** | **0.525286** |

三项补充实验均已完成：无内参的 RGB＋深度 Swin-Tiny、双路 Swin-Large＋XYZ，以及双路 Swin-Large＋无内参深度。

- **去内参：** 完整配置 APₛ 比原完整配置低 0.000040，AP 高 0.003981；基础配置 APₛ 低 0.005868。本次完整配置保留了相近的小目标分割精度，但单种子实验不能证明两种表示等效。
- **更换 Large，保留 XYZ：** 基础配置 APₛ 比原基础配置高 0.024970；完整配置 APₛ 高 0.010789（约 1.08 个百分点），整体 AP 低 0.004711。
- **Large＋无内参：** 完整配置 AP 为 0.880289、APₛ 为 0.525286，在上述四组完整配置中均最高。相对 Large＋XYZ，AP 增加 0.016757、APₛ 增加 0.026197（约 2.62 个百分点）；相对原 Tiny＋XYZ，分别增加 0.012046 和 0.036986。完整验证集 AP/APₛ 为 0.885717/0.702910。15 轮基座及 6 轮点头已完成，仅依据验证集锁定第 6 轮点头；两份测试 CSV 共 257,282 个掩码已解码检查，CSV 独立复评差异为 0。详见 [Large＋无内参完整记录](swinl_depth_no_k.md)。

这些分数使用原课程归档中的本地测试标注计算，全部来自同一批 2,112 张图、75,462 个 GT 实例。它们不是 Kaggle 分数；本地归档标注与全部隐藏评分标注是否一致尚未得到官方确认。官方提供的 12 张样例标注共 520 个掩码与本地一致，但这不能证明全部隐藏标注一致。Kaggle 提交成绩另行报告。

## CSV 与核验记录

前四份 CSV 位于本机 `reports/course_depth_backbone_v1_gpu02/submissions/`，新增的 Large＋无内参两份位于 `reports/course_swinl_depth_no_k_v1/submissions/`：

| 配置 | CSV 文件 | 字节数 | 预测实例数 |
|---|---|---:|---:|
| 无内参 Tiny 基础 | [dpseg_swint_depth_no_k_base15_native.csv](../../reports/course_depth_backbone_v1_gpu02/submissions/dpseg_swint_depth_no_k_base15_native.csv) | 44,306,235 | 121,775 |
| 无内参 Tiny 完整 | [dpseg_swint_depth_no_k_base15_full.csv](../../reports/course_depth_backbone_v1_gpu02/submissions/dpseg_swint_depth_no_k_base15_full.csv) | 52,304,059 | 152,038 |
| XYZ Large 基础 | [dpseg_swinl_xyz_base15_native.csv](../../reports/course_depth_backbone_v1_gpu02/submissions/dpseg_swinl_xyz_base15_native.csv) | 42,323,850 | 121,719 |
| XYZ Large 完整 | [dpseg_swinl_xyz_base15_full.csv](../../reports/course_depth_backbone_v1_gpu02/submissions/dpseg_swinl_xyz_base15_full.csv) | 51,920,742 | 148,162 |
| 无内参 Large 基础 | [dpseg_swinl_depth_no_k_base15_native.csv](../../reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_native.csv) | 40,311,368 | 116,324 |
| 无内参 Large 完整 | [dpseg_swinl_depth_no_k_base15_full.csv](../../reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_full.csv) | 49,151,193 | 140,958 |

每份均覆盖官方全部 2,112 个 image_id，顺序一致；只有 `image_id` 和 `PredictionString` 两列，单类 component（category_id=1），掩码为 1024×1024 COCO 压缩 RLE，每图至多 100 个实例，无预测时为 `[]`。六份 CSV 共 800,976 个掩码已在导出流程中逐个解码；独立复评程序从 CSV 读取并核对评分 JSONL，重新运行 COCO 评测。六组 AP、APₛ 与原评分的差异均为 0。

- [四组独立复评及原方案对比](../../reports/course_depth_backbone_v1_gpu02/verified_comparison.json)
- [独立复评汇总](../../reports/course_depth_backbone_v1_gpu02/verified_comparison.md)
- [无内参 Tiny 两份 CSV 独立复评](../../reports/course_depth_backbone_v1_gpu02/swint_depth_no_k/independent_audit.json)
- [XYZ Large 两份 CSV 独立复评](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/independent_audit.json)
- [无内参 Large 两份 CSV 独立复评](../../reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/independent_audit.json)
- [第一组完整训练与测试核验](../../reports/course_depth_backbone_v1/swint_depth_no_k/first_arm_completion_audit.json)
- [数据归档与选定权重一致性核验](../../reports/course_depth_backbone_v1_gpu02/dataset_checkpoint_integrity_audit.json)
- [两项任务最终完成核验](../../reports/course_depth_backbone_v1_gpu02/final_completion_audit.json)

Tiny 无内参方案的原始权重、预测和核验记录位于本机 `reports/course_depth_backbone_v1/`；Large＋XYZ 与前四份统一命名的提交文件位于 `reports/course_depth_backbone_v1_gpu02/`；Large＋无内参的训练、最终评测及两份提交文件位于 `reports/course_swinl_depth_no_k_v1/`。这些大型本地文件不随 Git 仓库分发。历史阶段核验文件中的“测试未完成”等字段记录的是当时的核验范围，最终结果以本节六份 CSV 的独立复评为准。

## 实验设置与完成核验

1. 去掉相机内参，只用 RGB、米制深度及深度有效性训练和测试：已完成。
2. 在原 XYZ 方案上将两路骨干换为 Swin-Large，重新训练和测试：全部 15 轮基座训练、6 轮点细化训练、各轮完整验证及基础和完整配置的最终测试均已完成。仅依据验证集锁定第 6 轮点头，两份测试 CSV 已独立复评通过。
3. 使用双路 Swin-Large 和无内参深度表示：15 轮基座、6 轮点细化及两种配置的完整验证与测试已完成，两份 CSV 独立复评通过。与 Large＋XYZ 使用相同预训练、硬件和预算；仅改变深度表示，详见 [Large＋无内参记录](swinl_depth_no_k.md)。

Large＋XYZ 首次运行在 GPU 0、1；GPU 1 上另一个服务占用约 41 GB，导致首批显存不足，未完成任何优化器更新。保留失败记录后，将 Large 改到同型号的 GPU 0、2。已用同样的 1024 输入、每卡 batch 6/5、累积 3 次，完成密集目标真实训练检查，峰值显存约 54.65 / 46.59 GiB。新运行保持模型、预训练、训练数据、种子、优化器、batch、轮数和评测代码不变，从公开初始化重新训练；结果目录为 `reports/course_depth_backbone_v1_gpu02/`。原运行记录位于 `reports/course_depth_backbone_v1/`。

每组基座训练 15 轮，固定第 15 轮用于比较；随后冻结基座训练点头 6 轮，只依据完整验证集选择点头。完整配置使用深度点头、整图与四个 640 切片，以及概率图深度细化。输入为 1024×1024，训练集 14,432 张、验证集 2,584 张、测试集 2,112 张。

已重新校验 20,554,250,971 字节的课程归档，以及两组固定第 15 轮基座和选定第 6 轮点头共 4 份权重文件：全部校验值与锁定记录一致，核验时文件未发生变化，训练记录中的冻结源码校验值也已复查。详见[数据归档与最终权重文件一致性核验](../../reports/course_depth_backbone_v1_gpu02/dataset_checkpoint_integrity_audit.json)。这项核验覆盖文件身份；最终测试及四份 CSV 的独立复评也已完成，证据见下文。

比较限制：本次仅用种子 17。Large 与 Tiny 的公开预训练和窗口尺寸（12 与 7）也不同，因此不是只改变容量的严格消融。新实验的有效 batch 均为 33，但微批布局与历史实验不同，原生局部掩码损失归一化使梯度不保证完全等价。去内参同时改变了深度表示，具体见下一节。

## 无内参方案实现

深度先按各图 `depth_scale_m` 转为米，以有效像素的中位数居中，再以绝对偏差的第 95 百分位归一化。将归一化的 Z 重复为三通道，并附有效性通道。输入读取、训练、整图推理和切片推理均不读取或传入 `camera_K`。

这同时改变了 XYZ 表示和内参依赖，不能将差异单独归因于删除一个参数。新实验采用两卡、有效 batch 33；历史实验的微批布局不同，原生局部掩码损失归一化使二者不保证梯度完全等价。

## Swin-Large 最终验证与测试核验

基础配置 `native` 使用固定第 15 轮基座；完整配置 `full` 使用选定的第 6 轮深度点头、整图与四个切片及概率图深度细化。

| 已完成配置 | 数据划分 | 图片数 | Mask AP | Mask APₛ |
|---|---|---:|---:|---:|
| 基础配置 `native` | 完整验证集 | 2,584 | 0.842144 | 0.553877 |
| 完整配置 `full` | 完整验证集 | 2,584 | 0.877152 | 0.680892 |

已逐图核对最终基础配置验证与第 15 轮基座验证：全部 2,584 个图片 ID、文件名、129,808 个预测的类别、置信度和压缩 RLE 掩码完全一致，全部主评测指标和定义也一致。完整配置的 176,268 个预测也已核对全部图片覆盖、压缩 RLE 格式、每图最多 100 个实例及两卡合并结果一致性。

完整配置的验证 AP 和 APₛ 比基础配置分别增加 0.035009 和 0.127015。这些均为完整验证集结果。

基础配置已完成全部 2,112 张本地测试图的主评测：**AP 0.835743、APₛ 0.393056**。已导出 42,323,850 字节的 CSV，包含 121,719 个预测；全部图片顺序、CSV 与评分 JSONL 的逐图一致性、两卡预测合并、压缩 RLE 格式及绑定于同一 CSV 的原始尺寸掩码解码记录均已核对。测试指标定义和 GT 面积统计与已完成的无内参实验一致。

完整配置也已完成全部 2,112 张测试图：AP 0.863532、APₛ 0.499089；CSV 为 51,920,742 字节、148,162 个预测。四份新 CSV 均已从 CSV 重新构建检测结果并独立运行 COCO 评测，AP 和 APₛ 与原评分结果的差异均为 0。

- [最终基础配置完整验证](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/val_native/evaluation.json)
- [逐图预测与第 15 轮基座一致性核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/val_native/base15_parity_audit.json)
- [最终完整配置完整验证](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/val_full/evaluation.json)
- [最终完整配置验证覆盖与合并结果核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/val_full/completion_audit.json)
- [基础配置测试主评测](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/test_native/evaluation.json)
- [基础配置测试 CSV](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/test_native/submission.csv)
- [基础配置 CSV 导出与预测一致性核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/test_native/export_audit.json)

- [完整配置测试主评测](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/test_full/evaluation.json)
- [完整配置 CSV 导出与预测一致性核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/final_evaluation/test_full/export_audit.json)
- [Large 两份 CSV 独立 COCO 复评](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/independent_audit.json)

## Swin-Large 基座完整验证记录

以下均为基座模型在完整验证集 2,584 张图上的结果。此时尚未训练点头，也未启用切片和概率图深度细化。

| 已完成轮数 | 验证 Mask AP | 验证 Mask APₛ |
|---|---:|---:|
| 1 | 0.746352 | 0.383819 |
| 2 | 0.784603 | 0.449959 |
| 3 | 0.800311 | 0.486409 |
| 4 | 0.819278 | 0.508244 |
| 5 | 0.826512 | 0.523857 |
| 6 | 0.823410 | 0.517800 |
| 7 | 0.827570 | 0.529369 |
| 8 | 0.832616 | 0.530279 |
| 9 | 0.833700 | 0.539351 |
| 10 | 0.838245 | 0.547672 |
| 11 | 0.837554 | 0.543685 |
| 12 | 0.840264 | 0.555247 |
| 13 | 0.840540 | 0.549715 |
| 14 | 0.840965 | 0.556150 |
| 15 | 0.842144 | 0.553877 |

全部 15 轮均检查了全部 14,432 张训练图的覆盖及每轮 438 次优化器更新，累计 6,570 次更新、216,480 次图像训练曝光；15 轮完整验证集覆盖也已核验。第 15 轮的 AP 最高，APₛ 以第 14 轮最高；最终比较固定使用第 15 轮，不依据中途验证结果更改轮次。第 1 轮另做了独立 COCO 复评，其余轮次核对主评测结果、评测定义及预测文件一致性。最终两份测试 CSV 均已逐掩码解码并独立复评通过。

已核对最终权重中的两路 Large 骨干、全部模型张量和优化器状态，权重张量与第 15 轮两卡同步记录一致；点细化流程绑定同一份 `final.pt`。点细化阶段冻结基座，仅训练普通和深度点细化模块；每卡 batch 4，每轮 1,804 次更新，共 6 轮、10,824 次更新。已依据完整验证集的深度点细化结果锁定第 6 轮权重，基础和完整配置的最终验证与测试已完成。

点细化首轮已覆盖全部 14,432 张训练图，各卡 7,216 张，合并后无重复或遗漏；两卡各 1,804 条更新记录连续、损失有限且汇总损失一致。已检查实际首轮点头张量：融合特征通道为 448，普通分支输入 449 通道、深度分支输入 458 通道，两分支的零初始化输出权重及新增深度通道权重均已学到非零值。首轮优化器状态也已核对：全部参数状态均更新 1,804 次，动量数值有限，保存的点头张量与首轮权重一致。首轮核验只覆盖该轮；完整六轮选头及最终测试证据分别列于后文。

点细化六轮训练和完整验证均已完成并核验：每轮覆盖全部 14,432 张训练图，无重复或遗漏；两卡的更新记录连续、损失有限且汇总一致，实际点头张量及协议绑定正常。六轮累计完成 10,824 次更新、86,592 次训练图像曝光。第 6 轮优化器的全部参数状态均累计更新 10,824 次，动量数值有限，保存的点头张量与第 6 轮权重一致。

已从全部六轮实际验证结果重算选头规则：先比较 APₛ，再比较 AP，最后优先较早轮次，结果确认为第 6 轮，与锁定记录和实际点头文件校验值一致。完成的训练流程还检查了基座训练前后张量不变、没有基座梯度，以及两卡最终点头一致；核验程序复查了冻结源码、全部训练覆盖和实际基座文件校验值。测试集未用于选头。最终评测和 CSV 核验均已完成。

- [第 15 轮训练核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/base15/training_epoch15_audit.json)
- [第 15 轮完整验证核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/base15/val_epoch_15/completion_audit.json)
- [完整基座及固定最终权重核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/base15/base_completion_audit.json)
- [点细化首轮训练覆盖与实际权重核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/training_epoch01_audit.json)
- [点细化第 2 轮训练覆盖与实际权重核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/training_epoch02_audit.json)
- [点细化第 3 轮训练覆盖与实际权重核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/training_epoch03_audit.json)
- [点细化第 4 轮训练覆盖与实际权重核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/training_epoch04_audit.json)
- [点细化第 5 轮训练覆盖与实际权重核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/training_epoch05_audit.json)
- [点细化第 6 轮训练覆盖与实际权重核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/training_epoch06_audit.json)
- [点细化首轮优化器状态核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/optimizer_epoch01_audit.json)
- [点细化第 2 轮优化器状态核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/optimizer_epoch02_audit.json)
- [点细化第 3 轮优化器状态核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/optimizer_epoch03_audit.json)
- [点细化第 4 轮优化器状态核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/optimizer_epoch04_audit.json)
- [点细化第 5 轮优化器状态核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/optimizer_epoch05_audit.json)
- [点细化第 6 轮优化器状态核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/optimizer_epoch06_audit.json)
- [六轮点细化训练与验证选头完整核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/training_completion_audit.json)
- [锁定的点头选择](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/selection.json)
- [第 6 轮学习率阶段切换及优化器延续核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/base15/continuation_checkpoint2200_audit.json)

## Swin-Large 点细化验证结果

以下仍为相同完整验证集 2,584 张图。基座固定使用第 15 轮权重，点细化使用深度分支；尚未启用切片或概率图深度细化。

| 配置 | 验证 Mask AP | 验证 Mask APₛ |
|---|---:|---:|
| 固定第 15 轮基座 | 0.842144 | 0.553877 |
| 点细化第 1 轮 | 0.854561 | 0.533945 |
| 点细化第 2 轮 | 0.865067 | 0.563301 |
| 点细化第 3 轮 | 0.869182 | 0.582060 |
| 点细化第 4 轮 | 0.870224 | 0.585971 |
| 点细化第 5 轮 | 0.871918 | 0.590079 |
| 点细化第 6 轮（选定） | 0.873884 | 0.597225 |

首轮点细化的 APₛ 曾下降；第 6 轮的 AP 和 APₛ 均高于基座，分别增加约 0.03174 和 0.04335，APₛ 比第 5 轮提升约 0.00715。这些是验证集结果。既定六轮已全部完成，并仅依据完整验证集选定第 6 轮点头；基础及完整配置的最终评测也已完成，未依据测试集调整训练或选择权重。

六轮验证各有 129,808 个预测，均已核对图片覆盖、压缩 RLE 格式、每图最多 100 个实例、两卡原始预测与合并结果一致性，以及评测定义和源码与已独立复评的基座验证一致。这六轮没有另做独立 COCO 复评；最终两份测试 CSV 已逐掩码解码并独立复评通过。

- [点细化首轮完整验证核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/val_epoch_01/completion_audit.json)
- [点细化第 2 轮完整验证核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/val_epoch_02/completion_audit.json)
- [点细化第 3 轮完整验证核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/val_epoch_03/completion_audit.json)
- [点细化第 4 轮完整验证核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/val_epoch_04/completion_audit.json)
- [点细化第 5 轮完整验证核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/val_epoch_05/completion_audit.json)
- [点细化第 6 轮完整验证核验](../../reports/course_depth_backbone_v1_gpu02/swinl_xyz/points6/val_epoch_06/completion_audit.json)
