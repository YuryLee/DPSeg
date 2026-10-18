# 课程要求覆盖与演示顺序

更新日期：2026-10-15。依据[课程任务书](task_original.txt)及用户最新要求，主指标为 **Mask APₛ@[0.50:0.95]，范围 0–1**，提交 `image_id,PredictionString` 两列的 CSV。

## 当前已完成

在课程 `blender.zip` 的官方划分上，Mask R-CNN R50-FPN v2、Mask2Former Swin-Tiny 和 DA-Fusion 课程适配均完成 **5 轮训练**。前两者使用 RGB，DA-Fusion 使用 RGB 与深度。训练 14,432 张、验证 2,584 张、测试 2,112 张；每轮完整验证，按 APₛ、AP、较早轮次选择权重，三者均选中第 5 轮。训练、验证、测试的场景、图片及 CAD 身份互斥，输出统一课程类别 `component=1`。

| 模型 | 验证 APₛ | 测试 APₛ | 测试 AP |
|---|---:|---:|---:|
| Mask R-CNN | 0.231000 | 0.096327 | 0.590850 |
| Mask2Former · RGB | 0.260305 | 0.122838 | 0.639821 |
| DA-Fusion 课程适配 · RGB-D | **0.392844** | **0.256135** | **0.741837** |

本次设置下，DA-Fusion 相比 RGB Mask2Former 的测试 APₛ 提高 **0.133297**，整体 AP 提高 **0.102016**。这是单种子、固定 5 轮的比较，不证明充分收敛或统计显著。新增深度、第二条骨干、融合参数及微批累积方式同时变化，需消融才能区分各部分贡献。此前已查看 RGB 基线的测试成绩；本轮训练前固定协议，测试不参与轮次选择。

- [三模型结果报告](../reports/course_dafusion/comparison_report.md) · [完整结果与来源核对](../reports/course_dafusion/comparison_summary.json) · [最终独立核验](../reports/course_dafusion/final_independent_review.md)
- [DA-Fusion 提交](../reports/course_dafusion/formal_v1/test/submission.csv) · [最佳权重](../reports/course_dafusion/formal_v1/best.pt) · [方法说明](dafusion_method_contract.md)
- [Mask R-CNN 提交](../reports/course_finetune/maskrcnn/test/submission.csv) · [Mask2Former 提交](../reports/course_finetune/mask2former/test/submission.csv)
- [三模型逐轮验证曲线](../reports/course_dafusion/learning_curves.png) · [RGB 基线训练曲线](../reports/course_finetune/figures/training_loss.png)
- [DA-Fusion 固定案例对照页](../reports/course_dafusion/gallery/index.html) · [页面预览](../reports/course_dafusion/gallery/browser/desktop.png) · [两种 RGB 基线的局部对照页](../reports/course_finetune/gallery/index.html)

三份 CSV 均覆盖全部官方测试 ID；每个实例为类别、置信度和 1024×1024 COCO 压缩 RLE，至多 100 个实例，无预测为 `[]`。所有掩码已原生解码核验；DA-Fusion 提交含 125,425 个实例。DA-Fusion 工程检查共 26 项通过，覆盖深度几何、模型移植和训练器。最终页面已验证 6 个固定案例 × 3 个展示阈值、桌面及 390/320px 手机布局，无 JavaScript 错误和外部请求，详见[验收记录](../reports/course_dafusion/gallery/browser/browser_test.json)。

页面展示已保存的真实预测。将 DA-Fusion 自包含 `index.html` 复制到电脑即可离线打开，无需模型或 GPU；可切换案例及展示阈值。它尚不提供课程新图上传和现场推理。历史 GraspNet 服务仍提供上传推理，但使用的是旧数据与旧模型。

## 课程要求覆盖

| 要求 | 当前证据 | 仍待完成 |
|---|---|---|
| 在课程数据上训练实例分割模型 | 三模型各 5 轮、9,020 次更新；共享图片顺序、有效批量 8 和种子 17，权重与日志已保存 | 多种子和更充分训练可检验稳定性及收敛情况 |
| 分类、定位、独立实例掩码 | 与课程单类 component 标注一致；保留重叠实例掩码，输出压缩 RLE | 不宣称多类别或 CAD 身份识别 |
| RGB 与深度读取、融合、训练 | 已按内参将原始深度转为 XYZ+有效性通道，使用双 Swin-Tiny 与四尺度 DS/DC 完成正式训练 | 深度噪声与缺失条件的稳健性仍可检验 |
| 至少一项面向小目标的算法改进 | DA-Fusion 课程适配提高了测试 APₛ，已形成基线与改进的统一指标对比 | 需要等参数量对照、DS/DC 消融及多种子；尚无显式小目标或边界损失 |
| 以 APₛ 为主，报告整体 AP | 三模型全量测试的 12 项 COCO 指标，保留原始面积，小目标 9,447 个 | 需进一步检验稳定性，不能将一次增益称为统计显著 |
| 模型大小和推理效率 | DA-Fusion 87,517,981 参数；权重、运行环境与单模型耗时有记录 | 尚无三模型同设备同条件的单图速度/显存基准 |
| 漏分割、误分割和边界案例 | 同一组 6 个测试案例，RGB/GT/RGB Mask2Former/DA-Fusion 可切换阈值比较；旧 RGB 图库另有局部放大 | 固定案例不能代替全量 AP，也不能据此归因错误机制 |
| 输入 RGB-D，展示 GT 与改进前后预测 | 课程离线交互页已展示 RGB-D 改进结果；历史 GraspNet 在线 demo 支持上传 | 若课程要求现场处理新图，仍需接入 DA-Fusion 上传与实时推理 |

测试小目标的面积中位数为 **440 像素**，验证为 **714 像素**；small 中面积 ≤256 像素的占比为 **34.14% / 19.87%**。测试场景每图实例也更多。这些是解释验证/测试难度差异的线索，不能证明分差全部由它们造成，也不等于测量了遮挡率。[分布与独立核对](../reports/course_finetune/maskrcnn_independent_audit.json)保留原始计数和来源。

## 三分钟演示顺序

| 时间 | 展示内容 | 说明重点 |
|---|---|---|
| 0:00–0:30 | 课程数据划分和上述结果表 | APₛ 为 0–1；三模型同训练轮数，验证选权重 |
| 0:30–1:15 | 打开 DA-Fusion 固定案例页，切换图像 | 同一 RGB、GT、RGB Mask2Former、DA-Fusion；案例索引提前固定 |
| 1:15–2:00 | 调节展示阈值，观察漏检、重复实例与掩码边缘 | 默认展示 0.5，完整评分 0.01；滑杆只改显示，不改 AP |
| 2:00–2:30 | 展示三模型验证曲线 | 均选第 5 轮；验证曲线没有混入测试分数 |
| 2:30–3:00 | 提交文件、最佳权重和后续计划 | 已完成训练与提交；下一步用消融和多种子区分深度、融合模块与参数量的贡献 |

## 历史公开子集材料

旧 GraspNet/T-LESS 数据、模型和报告均保留，历史指标常用 **0–100** 单位，不能与本页的课程 0–1 分数混用，也不能合并为课程成绩。历史三种子未证实深度融合的稳定优势；v5 的 BCE2、BACE、FreqFusion 均未达到事先冻结的开发门槛。

- [旧 RGB-D demo 使用说明](demo_usage.md) · [历史 v3 结果](../reports/strengthening_result.md)
- [历史 v5 训练改进结果](../reports/training_methods_result.md) · [历史固定案例图库](../reports/v5/gallery/index.html)
