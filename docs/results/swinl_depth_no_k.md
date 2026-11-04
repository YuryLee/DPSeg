# 当前最佳：Swin-Large＋无内参 RGB-D 实例分割

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent
- Origin Mode: run
- Verification Status: FORMAL_CSV_RESCORED
- 启动日期：2026-11-02（北京时间）
- 完成日期：2026-11-04 05:58（北京时间）

## 任务与设置

补齐四组对照中缺少的 Large＋无内参深度。以已完成的 Large＋XYZ 实验协议为基础，仅将 `model.depth_encoding` 从 `xyz` 改为 `depth`；采用已经验证过的无内参输入实现，不修改模型、训练或评测代码。

- 输入：RGB、按 `depth_scale_m` 转为米的实测深度及有效性；不读取、推断或传入相机内参。深度使用有效像素中位数和绝对偏差第 95 百分位归一化。
- 骨干：RGB 和深度两路 Swin-Large，使用与上一组相同的公开 COCO 初始化，重新开始课程训练。
- 硬件：GPU 0、2，两张 A800 80GB。
- 基座：1024×1024，15 轮，种子 17，每卡 batch 6/5，累积 3 次，有效 batch 33；学习率、优化器与数据增强沿用上一组。
- 点头：冻结本组第 15 轮基座，另训 6 轮；仅依据完整验证集 APₛ、AP、较早轮次依次选择。
- 数据：完整训练集 14,432 张、验证集 2,584 张、测试集 2,112 张。
- 最终评测：分别测试基础配置，以及深度点细化＋切片推理＋概率图深度细化的完整配置，导出两份 CSV，逐掩码解码检查后从 CSV 独立运行 COCO 复评。

## 完整验证与测试结果

15 轮基座训练、6 轮点细化训练及基础和完整配置的最终评测均已完成，所有阶段正常退出。点头仅依据完整验证集的深度点细化 APₛ、AP 和较早轮次选取，最终锁定第 6 轮；测试集未参与选头。基础与完整配置共用固定第 15 轮基座。

| 数据划分与配置 | 图片数 | Mask AP | Mask APₛ |
|---|---:|---:|---:|
| 验证集，基础配置 | 2,584 | 0.852289 | 0.579719 |
| 验证集，整图＋深度点细化 | 2,584 | 0.880978 | 0.607310 |
| 验证集，完整配置 | 2,584 | 0.885717 | 0.702910 |
| 本地测试集，基础配置 | 2,112 | 0.845626 | 0.408797 |
| 本地测试集，完整配置 | 2,112 | **0.880289** | **0.525286** |

完整配置包含深度点细化、整图与四个 640 切片，以及概率图深度细化。本地测试 APₛ 相比 Large＋XYZ 的 0.499089 增加 0.026197，AP 相比其 0.863532 增加 0.016757；在已完成的 Tiny/Large 与 XYZ/无内参四组完整配置中，AP 和 APₛ 均最高。这些是课程归档标注上的本地测试结果，Kaggle 成绩另行报告。

## CSV 与核验

两份 CSV 已完成全部图片顺序、1024×1024 压缩 RLE、单类及每图最多 100 实例的检查，共 257,282 个掩码均已解码。独立复评从 CSV 读取预测，逐图核对评分 JSONL 并重新运行 COCO 评测；AP 和 APₛ 与原结果的差异均为 0。2026-11-04 再次检查文件 SHA-256，与完成时记录一致。

| 配置 | CSV | 字节数 | 实例数 |
|---|---|---:|---:|
| 基础配置 | [dpseg_swinl_depth_no_k_base15_native.csv](../../reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_native.csv) | 40,311,368 | 116,324 |
| 完整配置 | [dpseg_swinl_depth_no_k_base15_full.csv](../../reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_full.csv) | 49,151,193 | 140,958 |

- [两份 CSV 的独立复评](../../reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/independent_audit.json)
- [最终结果与 CSV 校验值](../../reports/course_swinl_depth_no_k_v1/comparison.json)
- [随 Git 分发的结果、架构及权重摘要](swinl_depth_no_k.json)
- [完整验证及测试指标](../../reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/final_evaluation/comparison.json)
- [验证集点头选择](../../reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/points6/selection.json)

## 新比赛 Kaggle 提交

2026-11-04 将上述完整版 CSV 提交至 [RGB-D图像分割-333](https://www.kaggle.com/competitions/rgb-d-333)。提交编号为 **57967712**，状态 **COMPLETE**，公榜 APₛ 为 **0.56785**。查询时队伍“21”位列第 1／3 队；这是查询时排名。

上传前的 SHA-256 与已独立复评的 CSV 一致；官方存储返回的文件大小和 MD5 也与本地一致。文件内容未作修改。新比赛页面的数据说明、指标和提交格式与原任务一致；此分数与本地测试及旧比赛 `rgb-d` 的分数分别列报。[提交记录](kaggle_swinl_depth_no_k_rgb_d_333.json)。

## 运行文件

- [实验计划](../../reports/course_swinl_depth_no_k_v1/plan.json)
- [实时阶段](../../reports/course_swinl_depth_no_k_v1/queue.json)
- [控制日志](../../reports/course_swinl_depth_no_k_v1/controller.log)
- [通过的工程检查记录](../../reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/engineering_gate.json)
- [正式训练协议](../../reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/protocol.json)
- [工程检查协议](../../reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/protocol_engineering.json)
- [四组已完成对比](depth_backbone_comparison.md)

## 比较范围

本组与上一组 Large＋XYZ 的预训练、硬件和训练预算一致，便于比较深度表示；去内参同时从 XYZ 换为归一化 Z，不能把结果差异单独归因于删除一个参数。仍为单种子实验，归档本地测试分数与 Kaggle 分数分别报告。
