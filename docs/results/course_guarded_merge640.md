# 640×640 切片候选合并对照

> 版本说明：本页及配套 JSON 保留本轮对照完成时的实验快照；下文“当前默认”“不替换默认”均指当时状态。2026-11-06 用户随后指定 `global_guard_edge_consensus` 为当前完整版，更新的主表见[当前累计消融](course_ablation_swinl_current.md)。原始 JSON 的 `default_changed=false` 不回写。

## Material Passport

- 来源：ARS experiment-agent；`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1` 的真实推理、逐图核验与 COCO 评测。
- 核验状态：VERIFIED；协议 SHA-256：`faa7e01fd67b79b8a0715a3ef850690d132b6c5def23bf4f04413540ffca7808`。
- 范围：固定权重、固定规则的探索性测试集对比；不作为验证集选出的最优配置，不替换默认。

## 指标

固定 Swin-Large＋无内参深度，第15轮基座与第6轮点头，完整本地测试集2,112张。保留整图＋四个640×640切片、来源视图点细化及深度引导概率图细化。各组预测分数不重新计算，最终仍按原分数排序，每图最多100个实例，阈值0.01。

| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ | AP50 ↑ | AP75 ↑ | APₘ ↑ | APₗ ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| 原完整 DPSeg（640，当前默认） | 0.525286 | 0.674087 | 0.880289 | 0.962514 | 0.896285 | 0.915591 | 0.975086 |
| 可靠整图候选优先 | 0.524875 | 0.670700 | 0.886913 | 0.953070 | 0.910850 | 0.924365 | 0.982764 |
| 整图优先＋截断处理 | 0.526284 | 0.671324 | 0.887154 | 0.953305 | 0.911077 | 0.924523 | 0.982764 |
| 整图优先＋跨视图一致性 | 0.527541 | 0.674225 | 0.887230 | 0.953083 | 0.910991 | 0.924421 | 0.982764 |
| 整图优先＋截断处理＋一致性 | 0.528710 | 0.674606 | 0.887453 | 0.953313 | 0.911213 | 0.924578 | 0.982764 |
| 整图优先＋仅补充小目标切片候选 | 0.524827 | 0.673759 | 0.895085 | 0.963530 | 0.920376 | 0.934844 | 0.982887 |
| 无切片（参考） | 0.476384 | 0.569641 | 0.886068 | 0.957347 | 0.914049 | 0.934844 | 0.982887 |

相对原640切片，本轮同时提高 AP 与 APₛ 的固定配置为：整图优先＋截断处理、整图优先＋跨视图一致性、整图优先＋截断处理＋一致性。

本轮新规则中，整体AP最高为 **整图优先＋仅补充小目标切片候选**：AP=0.895085，相对原640的 APₛ 变化为 -0.000459。APₛ最高为 **整图优先＋截断处理＋一致性**：APₛ=0.528710，整体AP=0.887453。两个目标应分别比较，不据单一指标称为统一最优。相对无切片参考，同时提高AP和APₛ的配置为：可靠整图候选优先、整图优先＋截断处理、整图优先＋跨视图一致性、整图优先＋截断处理＋一致性、整图优先＋仅补充小目标切片候选。

无切片行复用已完成的同权重结果，作为整体AP与小目标指标的参考；该行不属于本轮新合并规则。分数范围0–1，整体AP并非各尺寸AP的简单平均。

## 规则

- **可靠整图候选优先**：优先保留分数≥0.5、预测面积>1024的整图候选，再进行原跨视图掩码去重
- **整图优先＋截断处理**：保护整图＋降低内部裁剪边缘候选优先级，剔除被可靠整图掩码包含的疑似截断候选
- **整图优先＋跨视图一致性**：保护整图＋按其他高置信度视图的掩码一致性调整选择优先级
- **整图优先＋截断处理＋一致性**：保护整图＋裁剪边缘处理＋跨视图一致性
- **整图优先＋仅补充小目标切片候选**：保护整图，且切片仅提供预测面积≤1024的小目标候选

实际规则及全部常量以配套JSON中的 `guarded_merge_policy` 为准。“可靠”仅指预测分数≥0.5且预测面积>1024的启发式条件，不保证掩码正确；跨视图一致性也不等价于GT准确率。整图保护指冲突选择时优先保留可靠中大目标；最终原分数排序与100实例上限可能使已选保护候选落出输出，相关次数单独记录。边缘与一致性系数只影响候选选择优先级，不修改交付CSV中的置信度。细化与掩码表示沿用原实现，无多边形转换、掩码并集或新增训练。

## 候选变化

| 配置 | 输出实例数 | 整图来源 | 切片来源 | 新增候选 | 移除候选 | 共用候选 | 保护候选落出top100 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 可靠整图候选优先 | 141269 | 79504 | 61765 | 48070 | 47759 | 93199 | 261 |
| 整图优先＋截断处理 | 141051 | 79597 | 61454 | 48832 | 48739 | 92219 | 256 |
| 整图优先＋跨视图一致性 | 141210 | 79018 | 62192 | 49504 | 49252 | 91706 | 259 |
| 整图优先＋截断处理＋一致性 | 140993 | 79117 | 61876 | 50217 | 50182 | 90776 | 254 |
| 整图优先＋仅补充小目标切片候选 | 138042 | 81861 | 56181 | 54102 | 57018 | 83940 | 153 |

新增、移除与共用按同一图像的来源视图/查询编号计算，相对于原640切片候选集合；不是正确检出数、误检数或唯一物体数。相同物体可能在多个视图中产生候选。全部五视图共生成 442,955 条候选，跨变体共享细化的候选并集共 197,187 条。各组边缘过滤、面积过滤、去重及top100截断统计详见JSON。

## 100实例上限诊断

本地测试集每图非crowd GT最多 96 个，平均 35.7301 个；超过100个GT的图像为 0 张。原640切片有 1014 张图输出达到100个实例，平均输出 66.7415 个。因此，在本地测试集上，100上限没有因为真实实例数超过限制而造成必然漏检，但候选排序和重复预测仍可能影响哪些实例进入最终输出。这些计数不能证明多出的候选都是误检。GT仅用于推理后的数量诊断，未用于候选选择；标注分片与参考预测哈希见JSON。

## 核验与范围

- 固定模型与实验源码、权重和协议哈希；RGB、深度、有效性是唯一推理输入，无GT输入。
- 三张验证图仅用于工程门禁；原native、原完整输出以及各变体独立细化与共享缓存细化逐掩码、逐分数完全一致。
- 全部2,112张测试图逐图复现原native及原完整640结果，评分前检查所有分片ID、顺序、实例数及哈希。
- 五份新CSV以及两份参考结果均检查全量RLE解码记录，并重新逐行绑定CSV与实际评分输入。
- 所有组使用相同COCO掩码评测：IoU=0.50:0.05:0.95，maxDets=[1,10,100]，小目标范围[0,1024]，保留原GT面积；预测面积来自独立RLE，不传预测bbox。
- 原640基线复用已独立重评分的核验：全部12项COCO指标完全一致，预测、评分源码与核验文件哈希再次检查。
- 规则在本轮推理前冻结；这是在已查看过的本地测试集上进行的探索性对照，没有根据本轮测试结果继续调阈值或替换默认。没有多种子、显著性或Kaggle分数结论。
- 共享视图特征和候选细化仅用于节省实验时间，本轮耗时不能作为各变体独立推理速度。

## 复现

```bash
.venv/bin/python -u tools/evaluate_course_guarded_merge.py --output reports/course_swinl_guarded_merge640_new --gpus 1,3,4,5,6,7
.venv/bin/python tools/summarize_course_guarded_merge.py --run reports/course_swinl_guarded_merge640_new --docs docs/results
```

输出目录需尚不存在，按实际空闲GPU调整设备列表。入口先检查每张卡至少24GiB空闲显存，并执行工程门禁；不终止其他进程。重评分参考文件默认复用本机 `reports/course_sahi_merge_v1/reference_rescore/verification.json`，它只是一份已有指标核验文件，本轮无SAHI运行依赖。

## CSV 路径

- **原完整 DPSeg（640，当前默认）**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_full.csv`
- **可靠整图候选优先**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard.csv`
- **整图优先＋截断处理**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard_edge.csv`
- **整图优先＋跨视图一致性**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard_consensus.csv`
- **整图优先＋截断处理＋一致性**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard_edge_consensus.csv`
- **整图优先＋仅补充小目标切片候选**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_small_tiles_only.csv`
- **无切片（参考）**：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_without_tiles.csv`
