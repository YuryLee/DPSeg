# 状态：课程数据两种模型已完成微调与全量比较

更新日期：2026-10-13。沿用 ARS 实验执行与核验流程。课程 `图像分割/blender.zip`（20.55 GB）直接从 ZIP 按需读取，未全量解压。两种模型的公开权重测试与课程训练集微调均已完成；旧 GraspNet/T-LESS 试验数据与权重保留。

## 最新：课程训练集上的两模型微调

2026-10-13 09:22 UTC 启动，分别使用 A800 GPU0/1，两进程均正常退出。各完成 5 个训练轮次，14,432 张/轮，9,020 次更新、72,160 张次曝光；批量 8、AdamW、主学习率 0.0002、骨干学习率 0.1 倍、种子 17。输入 RGB 短边 800，逐实例保留重叠掩码。每轮完整验证 2,584 张，按 APₛ、AP、较早轮次选最佳，两者均选第 5 轮，再测试全部 2,112 张。

| 模型 | 最佳验证 APₛ | 测试 APₛ | 测试 AP | 微调前测试 APₛ |
|---|---:|---:|---:|---:|
| Mask R-CNN R50-FPN v2 | 0.231000 | 0.096327 | 0.590850 | 0.000738 |
| Mask2Former Swin-Tiny | 0.260305 | 0.122838 | 0.639821 | 0.000329 |

全部为 0–1 分数。本次配置下 Mask2Former 的测试 APₛ 高 0.026511。训练/验证/测试的场景、图片 ID、CAD 身份均不交叉，分类仍是 `component=1`。训练 448,469 个实例含 49,476 个 small，验证 73,421 个实例含 10,999 个 small，测试 75,462 个实例含 9,447 个 small。数据与执行相关 37 项单元检查通过；两模型均完成真实梯度与权重保存冒烟检查。Mask2Former 首次冒烟检查的输出字段问题已在正式训练前修正，失败记录保留。

完整记录：[冻结协议](../reports/course_finetune/protocol.json)、[汇总与来源核对](../reports/course_finetune/comparison_summary.json)、[固定案例图库](../reports/course_finetune/gallery/index.html)、[训练损失](../reports/course_finetune/figures/training_loss.png)、[验证曲线](../reports/course_finetune/figures/validation_ap_small.png)。两份提交及最佳权重入口见[项目首页](../README.md)，每份 CSV 均为 2,112 行，压缩 RLE 均已原生解码验证；预测数为 124,754 / 143,973。图库已完成整图/小目标切换及桌面、手机布局验收。

限制：RGB 单输入、单种子、不同骨干、仅 5 轮，不宣称充分收敛或统计显著。两架构损失定义不同，不比较损失数值大小。训练加各轮验证及最终测试的总耗时分别约 2.50 / 5.56 小时，不是单图推理速度。Mask2Former 的微调前后差异还包含插值实现变化；测试集在此前实验中已被查看。产物一致性核对状态为 ANALYZED，不等于独立重训复现。课程 RGB-D 融合、小目标改进和同条件效率评测尚未完成。

共享盘空间曾紧张：仅清理已验证的冒烟断点和旧下载中间缓存；Mask R-CNN 的续训断点曾短暂转存至临时盘，现已按 SHA 核验恢复到项目内的普通文件。两次正式训练均未中断，当前模型产物不依赖临时盘；清理与恢复收据保存在 `reports/course_finetune/`。

## 历史基线：课程数据上的未微调 RGB 测试

- 数据划分为训练 14,432 / 验证 2,584 / 测试 2,112 张；本轮只运行全部测试图。测试包含 75,462 个实例、9,447 个 COCO small，单类 `component=1`。深度为原始 16 位 PNG、比例 0.0001 米，但这两份公开权重只输入 RGB。
- Mask R-CNN R50-FPN v2：Mask APₛ **0.0007375489**，整体 AP **0.0639217644**。
- Mask2Former Swin-Tiny：Mask APₛ **0.0003288826**，整体 AP **0.0795362441**。本节全部指标为 **0–1**；后文历史成绩为 **0–100**。
- 两者均直接使用公开 COCO 权重，将全部前景类别映射为课程零件类；无训练、无测试集调参。测试真值只用于评分与固定案例可视化。Mask2Former 整体 AP 较高，但主指标 APₛ 更低，不能据此判定微调后的架构优劣。
- 两份 `submission.csv` 均完整覆盖 2,112 个官方图片 ID，所有 1024×1024 压缩 RLE 已逐个解码验证；单元检查 19 项通过。可从[项目首页](../README.md)访问两份提交、12 项完整 COCO 指标、数据审计与[固定案例图库](../reports/course_pretrained/gallery/index.html)。
- GPU 推理依次进行，CPU 校验存在重叠，已提前记录[执行调整](../reports/course_pretrained/execution_deviation.json)。耗时只作描述，不作公平速度比较。
- 该次按用户“有公开权重就直接测试”的要求未启动训练；后续课程微调现已完成，结果见上节。

## 历史一轮 v5

- baseline、BCE2、BACE、FreqFusion 各 3 个种子，共 **12 组 × 800 步训练、28 组评测**，完整性审计通过；**125 项单元测试通过**。
- 三种子开发均值：baseline AP/APₛ = **51.56/9.37**，BACE = **52.18/11.31**。BACE APₛ 的实际增量为 **1.936 个百分点，小于 2 点门槛**；三种候选全部未通过冻结门槛，默认模型保持原样。
- T-LESS 单独的 seed17 检查中，BACE 整体 AP 从 baseline 的 **44.99 降到 40.83**。这是已查看数据上的探索性检查，不是新的未见确认集或课程成绩。
- [结果报告](../reports/training_methods_result.md)与[本地离线图库](../reports/v5/gallery/index.html)已生成。图库包含 6 个固定案例、13 个 small，114 张嵌入图片、筛选、缩放、重置、28 份评测凭据导出及手机布局已通过[离线浏览器验收](../reports/v5/browser_offline/browser_test.json)，无 JavaScript 错误；尚无 `/training-methods` 在线路由。
- 首次运行因 ROIAlign 微小浮点越界中止，修复容差后 12 组统一重训；失败记录保留。固定种子不保证 CUDA 逐位复现，材料状态为 ANALYZED，不宣称完整论文复现。
- 本轮仍使用旧公开子集；开发仅 4 个 small、测试仅 1 个 small，重复种子不增加独立数据，不能据此推断统计显著性。

## 历史一轮 v4

- 调研6篇相关论文，核对任务、发表年、方法与代码可访问性。
- 实现full/highres/crop_fuse三个固定权重推理设置，完成30组评测；未重训、未改变权重或数据。
- E2三种子开发APsmall均值：9.96 / 7.29 / 9.53，两种改法均未过冻结工程门槛。T-LESS总体AP上升，但小目标AP也下降。
- 在线Demo继续使用full，新增 [论文改进试验页](http://127.0.0.1:33051/paper-trial)，可查看5个固定案例、全部5个small的局部放大和失败结果。
- 10组full×12项历史COCO指标数值完全一致；30份预测顺序/尺寸/来源校验通过，70项单元测试通过。
- v4 模型评测只使用 GPU4，当轮服务在全部评分完成后重启，未参与计时。
- v4 报告：`reports/paper_trial_result.md`；全部结果：`reports/v4/trial_summary/summary.md`；协议：`docs/paper_trial_protocol.md`；独立审阅：`reports/v4/independent_review.md`。

下列内容保留上一轮数据、训练和稳定性检查。

## 历史 v2/v3 已完成

- 第二轮 GraspNet 数据与权重完整保留：64训练/4开发/24测试帧，small为8/4/1；单类object。
- 追加 seeds23/41，与原seed17形成3组E0/E2配对，各800步，共6组均通过有限loss/权重和来源核验。
- 三种子开发APsmall：RGB 11.25±3.79，E2 9.96±1.94；测试AP：RGB72.91±0.38，E2 72.48±0.11。±为样本SD。不能宣称深度融合稳定改善小目标。
- 已从镜像获得T-LESS单独压力集：100帧/20场景/688可见实例/52small（9场景、12物体身份），下载约828MB。原始720×540，raw depth×0.0001米，精确保留重叠GT；没有用于训练或调参。
- 固定seed17在T-LESS的AP/APsmall：E0 44.05/1.25，E2 42.66/1.14。跨数据集小目标表现仍弱，不与GraspNet合并、不当作BOP官方成绩。
- GraspNet全部5个开发/测试small的三模型逐例分析已完成，15份局部可视化；区分低分候选被显示阈值隐藏与掩码本身不准。
- 新增独立服务管理、固定端口、小目标筛选。50项单元测试、9项完整浏览器验收及追加small-filter真实推理通过；36个新增导出RLE面积一致，手机布局无溢出。
- 新标注格式不改变既有104个GraspNet样本的输入/目标张量和指纹；seed17 E0测试12项COCO精确复现。训练关键代码未改，评测入口变更单独留痕。

## 使用入口

交互Demo：http://127.0.0.1:33051 。通过管理入口以独立进程启动，展示预定seed17权重；不按新测试结果挑权重。

```bash
./start_demo.sh
.venv/bin/python tools/manage_demo.py status
.venv/bin/python tools/manage_demo.py stop
```

当前状态以实时status为准。reports/server.json只是地址记录，不能证明服务仍在运行；reports/demo_manager.json绑定真实进程与身份。远程访问须转发33051，CPU备用见使用说明。

## 核心交付

- [reports/training_methods_result.md](../reports/training_methods_result.md)：最新 v5 完整结果、来源与 11 项解释核对。
- [reports/v5/gallery/index.html](../reports/v5/gallery/index.html)：v5 本地离线图库；没有新增在线入口。
- reports/v5/trial_summary.json、.csv、trial_summary_ap_small.png、.pdf：v5 逐组结果与三种子图。
- reports/v5/training_methods_audit.json：12 组训练、28 组评测完整性审计。
- reports/strengthening_result.md：历史 v3 结论与统计解释核对。
- reports/v3/seed_stability.json、.png、.pdf：逐种子及均值/样本SD。
- reports/v3/small_objects/index.html、summary_zh.md：全部GraspNet小目标诊断。
- reports/v3/tless/{E0,E2}/：全体/逐场景COCO、精确预测RLE与来源。
- reports/v3/tless/gallery/index.html：20个场景各自固定首帧的跨集案例。
- docs/demo_usage.md、course_readiness.md：启动说明与三分钟演示顺序。
- reports/v3/browser/：完整交互及真实小目标筛选验收。
- reports/v3/seed_training_audit.json、multi_mask_compatibility.json、evaluation_regression_check.json：训练和兼容性证据。
- docs/strengthening_protocol.md、reports/v3/tless_confirmation_plan.json：新增实验与选帧规则。

## 保留限制

课程测试类别、深度单位、标注和 small 数量已经核验；实际存在测试真值，与归档 README 的说明冲突。最新结果只说明 COCO RGB 权重的直接迁移表现，尚未在课程数据训练 RGB/RGB-D 模型。旧实验使用公开子集，object单类不等于88身份分类。GraspNet仅2个训练场景，测试只有1个small；三种子不增加测试对象。T-LESS多视角相关，52small中19个是可见比例不足10%的遮挡碎片。整套新旧升级同时改变骨干、预训练、数据、增强与尺寸，不能把收益归因于单一模块。未在指定24GB型号完成独立性能复测。

后续训练应采用课程训练/验证划分，避免基于已查看的测试成绩选参数。v5 已完成控制其他条件的训练方法对照，未获得通过门槛的改进。历史合成演示压缩包保持原样，不包含新结果；以本项目目录与本说明为准。
