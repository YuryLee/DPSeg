# RGB-D 实例分割补充评测协议

核实日期：2026-10-02。此文记录当前代码定义；这些指标补充 COCO Mask AP，不能替代课程规定的 AP / AP_small。

## 来源及关键区别

- [UOAIS 原论文 §V](https://arxiv.org/html/2109.11103)在 OSD、OCID 上报告区域重叠与边界 P/R/F、F@.75；后者按匹配后 **Overlap F1 > 0.75** 判断。它不是 IoU≥0.75，也不是 COCO AP75。
- [UOIS 作者评测代码](https://github.com/chrisdxie/uois/blob/master/src/evaluation.py)采用 Hungarian 一对一分配，最大化各匹配对的 F1 总和；区域分母包括未匹配对象。边界容差默认为图像对角线的0.003倍、向上取整。其空集合约定也由此核实。
- [UCN 作者评测代码](https://github.com/NVlabs/UnseenObjectClustering/blob/master/lib/utils/evaluation.py)提供相同的匹配及边界方法，0为背景。UOIS 原数据另把1作为桌面；本项目的1是有效物体，绝不删除。
- [UOAIS 作者评测入口](https://github.com/gist-ailab/uoais/blob/main/eval/eval_utils.py)对每张图的指标取算术平均；可见实例需要先成为不重叠标签图。该入口使用后写入遮罩覆盖先写入遮罩；本项目采用下述显式适配规则，不能宣称整个评测流程与作者代码完全相同。

以上是作者发布的论文与实现，不是第三方概述。只核实与当前指标有关的部分；未运行作者旧版完整环境。

## 输入及适配

`instance_metrics(prediction, ground_truth)` 接收等大的非负整数标签图，0为背景，任意正整数为独立实例。只评可见实例，不评完整遮挡后的形状。

`evaluate_extra_predictions(dataset, predictions)` 接受项目原始分辨率数据和 Torchvision 概率遮罩。默认检测分数≥0.5、遮罩概率≥0.5。类别标签不参与匹配。预测发生像素重叠时，分数最高者获得像素，同分时输入顺序优先；空遮罩或被完全覆盖的遮罩在标签图中消失，另报告数量。GT 必须互不重叠。

这是新增适配约定，不是论文的共同标准。COCO AP 仍应使用完整预测列表及原有评价器，不能用此标签图重新计算后冒称未改变协议。

## 每图公式

设真值实例为 G_i、预测实例为 P_j，交集像素数为 I_ij。

- 匹配分数 D_ij = 2I_ij / (|G_i|+|P_j|)。Hungarian 求最大 ΣD_ij 的一对一分配 M。
- Overlap precision = Σ_M I_ij / Σ_j|P_j|；recall = Σ_M I_ij / Σ_i|G_i|；F1 为 P、R 的调和平均。未匹配预测与真值仍进入分母，因此合并、拆分和额外前景会被惩罚。
- 边界使用相邻像素（右、下、右下）标签变化，忽略图像外侧边缘。匹配仍使用区域 F1 的 M。分别计数落在另一边界半径 r 的离散圆盘范围内的预测／真值边界像素；分别除以全部预测／真值实例边界像素数，再取调和平均。默认 r=ceil(0.003×sqrt(H²+W²))，1280×720 为5像素。
- `object_recall_f_gt075` = 匹配中 D_ij>0.75 的对数 / 真值实例数；这对应上述 F@.75 定义，严格大于。
- `object_recall_iou_ge075` 是**另列的诊断项**：独立最大化成对 IoU 总和后，统计 IoU≥0.75 的匹配数 / 真值实例数。它不称 F@.75，也不称 AP75。

## 汇总、空集合及范围

`metrics_macro` 为每图指标的算术平均。`metrics_micro` 使用全数据集累加的交集、前景、边界及对象计数重新计算，属于额外诊断；不能当作论文的逐图均值。`totals` 是整数总数，`per_image` 保存匹配、计数及容差。指标均为0–100百分数，匹配对的 `pair_f1` / `pair_iou` 为0–1。

| 真值 | 预测 | Overlap/Boundary P,R,F | 对象检出比例 |
|---|---|---|---|
| 空 | 空 | 100,100,100 | 100 |
| 非空 | 空 | 100,0,0 | 0 |
| 空 | 非空 | 0,100,0 | 0 |

空数据集的指标为null。完整图像为一个对象时可能没有内部边界；本实现将零分母的边界P或R定义为1，以避免NaN，这是显式扩展。任何坏形状、负标签、非有限概率均报错。

## 使用

```sh
.venv/bin/python tools/evaluate_extra.py --data data/graspnet_subset --checkpoint outputs/graspnet/E2/model.pt --output reports/graspnet_extra_E2.json
```

可用 `--save-predictions PATH` 保存 CPU 张量与图像ID、数据指纹；随后 `--predictions PATH` 重评，必须完全匹配数据指纹和图像顺序。不同阈值属于不同协议，须固定并披露，不能选择最有利的验证结果充当独立测试。

当前 GraspNet 子集与论文 OSD/OCID 数据及尺寸不同，数值不能直接作论文方法排名。当前补充指标也不能证明新类别识别能力。
