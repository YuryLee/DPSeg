# Swin-Large 基线对比与累计模块消融

> 版本说明：本页保留原分数优先合并的累计消融快照。2026-11-06 起，用户指定整图优先＋截断处理＋一致性为当前完整版；更新后的 S4 及选择依据见[当前累计表](course_ablation_swinl_current.md)。

所有配置均采用 **Swin-Large、1024×1024 输入**，RGB-D 分支使用无需相机内参的深度表示。以下为**完整本地测试集 2,112 张图**的 COCO 掩码评测，数值范围 0–1；各行按表中顺序累计添加模块。加粗标记完整 DPSeg 配置，任务主指标为 APₛ。

| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |
|:---:|:---:|:---:|:---:|
| S0：Mask2Former | 0.205838 | 0.293236 | 0.734611 |
| S1：+ RGB-D 融合（DA-Fusion） | 0.408797 | 0.481296 | 0.845626 |
| S2：+ 深度感知点细化 | 0.403831 | 0.481031 | 0.862922 |
| S3：+ 深度引导概率图细化 | 0.476384 | 0.569641 | 0.886068 |
| **S4：+ 切片增强推理（DPSeg）** | **0.525286** | **0.674087** | **0.880289** |

APₛ、ARₛ 分别表示小目标掩码平均精度和平均召回率，均在 IoU=0.50:0.05:0.95 上平均，最多保留 100 个实例；AP 为所有尺寸的掩码平均精度。小目标使用原始标注面积 [0, 32²] 像素。以上分数与 Kaggle 公榜分开报告。

- **固定权重：** S0 为 RGB Mask2Former 第 15 轮；S1–S4 共用 RGB-D 第 15 轮基座，S2–S4 共用验证集选取的第 6 轮深度点头。RGB 与 RGB-D 均从相同公开 COCO Swin-Large 权重开始训练，使用完整训练集、15 轮和全局 batch 33。
- **累计配置：** S2 为整图点细化；S3 在 S2 后增加概率图细化，不使用切片；S4 保留两种细化并增加整图＋重叠切片候选。表中顺序表示组件加入顺序；完整系统仍先合并候选，再做来源视图点细化和原图概率图细化。
- **比较范围：** RGB-D 增加独立深度编码器和融合模块，点头增加 6 轮训练，切片增加推理计算。结果来自单次 seed=17 训练，显卡与每卡批次分配存在差异。测试标注仅用于评分，不用于训练、选择检查点或本轮调整推理参数。

S2 相对 S1 的 APₛ 变化为 **-0.004966**，整体 AP 变化为 **+0.017296**。因此不能将本表概括为“每个模块均提升小目标 AP”；各行反映既定组合下的增量效果。整体 AP 最高的是 S3（**0.886068**）。完整 S4 的 APₛ 和 ARₛ 最高。

完整 DPSeg 相对 DA-Fusion 适配基座 S1 的 APₛ、ARₛ、AP 绝对增量分别为 **+0.116489、+0.192791、+0.034663**。增量描述本次固定配置的结果，不代表多次训练的统计显著性。

## 结果核验

S0 使用本轮刚完成的 15 轮 RGB 训练及自动测试结果。S1 和 S4 复用已完成并独立从 CSV 复评的 Large＋无内参结果。S2、S3 补跑了全部 2,112 张图；每张图的基座预测均与已有 S1 结果逐掩码核对，概率图细化保持候选数量、类别、分数和顺序。五份 CSV 的全部图像 ID、掩码、类别和分数均与对应评分输入一致，五组 COCO 评测设置及 GT 统计一致。

## 提交文件本机路径

- **S0：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_rgb_v1/submissions/mask2former_swinl_rgb_1024_e15.csv`
- **S1：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_native.csv`
- **S2：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_report_ablation_v1/submissions/dpseg_swinl_depth_no_k_s2_depth_point.csv`
- **S3：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_report_ablation_v1/submissions/dpseg_swinl_depth_no_k_s3_depth_point_probability.csv`
- **S4：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_depth_no_k_v1/submissions/dpseg_swinl_depth_no_k_base15_full.csv`

预测 CSV 和模型权重保留在实验目录，不随本页上传到 GitHub。可分发的数值及校验信息见 [JSON](course_ablation_swinl.json) 和 [CSV](course_ablation_swinl.csv)。

## 补跑复现

```bash
.venv/bin/python -u tools/evaluate_course_swinl_ablation.py \
  --output reports/course_swinl_report_ablation_new --gpus 3,4
```

选择空闲显卡，并按实际位置传入 `--run`、`--archive`。该程序仅补齐 S2/S3，固定原第 15 轮基座、第 6 轮点头和推理参数；不训练、不根据测试成绩选择参数。RGB 基线入口为 [run_course_swinl_rgb.py](../../tools/run_course_swinl_rgb.py)，其最终评测由控制程序自动完成。

## 与历史表的关系

本表替换 README 主表中的旧 Tiny＋XYZ 结果。旧表的 S3 为“点细化＋切片”，本表的 S3 为“点细化＋概率图细化”，二者不应只改标签后混用。[历史累计消融](progressive_ablation.md)保留原数字和顺序。[效率表](course_efficiency_swinl.md)使用相同五种结构；其中 RGB 耗时仍来自此前第 12 轮检查点，精度表使用最终第 15 轮。
