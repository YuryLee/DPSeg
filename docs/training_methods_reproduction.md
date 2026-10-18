# BACE / FreqFusion 训练对照的复现入口

本页说明 v5 已固定的运行方式；实测结论以 `reports/training_methods_result.md` 和 `reports/v5/trial_summary.json` 为准。协议在正式训练前保存于 [training_methods_protocol.md](training_methods_protocol.md)。这是一份 GraspNet 小子集上的工程适配，不是两篇论文的完整复现。

## 四组分别改变什么

| 组别 | 训练变化 | 推理变化 |
|---|---|---|
| baseline | E2 原始损失，从本地 COCO 权重重新训练 | 原 RGB-D Mask R-CNN |
| bce2 | 原 mask BCE 再加一次，检查损失权重的影响 | 结构不变 |
| bace | 原 mask BCE 加论文 BACE 辅助项，scale=4、两项权重均为1 | 结构不变，不需要真值 |
| freqfusion | 在 RGB FPN 三个连接加入 ALPF、AHPF 和偏移采样 | 新增314,070个参数与对应计算 |

原 RGB 分支使用0.1倍学习率；原深度/预测头和新增 FreqFusion 参数使用全学习率。共享参数先加载，再初始化新增模块；原 FPN 卷积参数保持对象和名称。完整设置见协议和每组 `run.json`。

BACE 以原始 mask logits 计算辅助项，不先变成概率。它没有新增边界二值标注，也不是仅在边界像素上计算损失。FreqFusion 使用完整核心，CARAFE 用零填充的纯 Torch 分块实现，并在训练反向时重算以控制显存。来源、差异和数学检查见 [BACE说明](../reports/v5/boundary_method_notes.md)与[FreqFusion说明](../reports/v5/freqfusion_implementation_audit.md)。

## 单组训练与评测

以下命令从项目目录执行。`CUDA_VISIBLE_DEVICES=5` 对应本轮记录的物理显卡；在其他机器上应改为实际分配的卡。输出必须是新的或空目录，已有结果会被拒绝覆盖。

```bash
CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 .venv/bin/python tools/run_v5.py train \
  --config configs/GraspNetV2_E2.json --variant bace \
  --init-rgb pretrained/maskrcnn_resnet50_fpn_v2_coco_mirror_state_dict.pt \
  --data data/graspnet_expanded_v2 --steps 800 \
  --output outputs/my_v5_reproduction/seed17/bace \
  --device cuda:0 --threads 4 --cuda-memory-fraction .55

CUDA_VISIBLE_DEVICES=5 OMP_NUM_THREADS=4 .venv/bin/python tools/run_v5.py eval \
  --data data/graspnet_expanded_v2 --split val \
  --checkpoint outputs/my_v5_reproduction/seed17/bace/model.pt \
  --output reports/my_v5_reproduction/seed17/bace/val \
  --device cuda:0 --threads 4 --cuda-memory-fraction .55
```

`--variant` 可选 `baseline`、`bce2`、`bace`、`freqfusion`。seed23/41使用 `configs/seed_replication/seed23_E2.json` 和 `seed41_E2.json`。T-LESS评测改为 `--data data/tless_confirmation --split test`，输入自动使用该数据集的深度单位0.0001米；GraspNet使用0.001米。

正式训练固定800步。其他步数仅标记为工程冒烟，不能混入正式结果。训练状态文件保留优化器、调度器和随机状态用于审计；当前入口没有断点续训功能。每组均从同一份COCO初始化开始。

## 如何检查结果文件

- `model.pt`：模型参数和完整配置；使用 `rgbd_seg.v5_model.load_v5_model` 加载。
- `weights.pt`：纯参数文件，用于公平比较文件大小。
- `run.json`：源码、数据、协议、初始化和权重哈希，以及硬件、参数组、耗时和显存。
- `loss.jsonl`：800步的总损失、各分项、梯度范数及学习率。
- `sampling.jsonl`：样本顺序与增强随机状态。同seed四组应完全一致。
- `evaluation.json`：原图COCO十二项指标、逐帧耗时和来源记录。
- `predictions.jsonl`：保持原图尺寸、样本顺序与全局ID的COCO RLE掩码。

评测会拒绝训练后发生变化的模型源码、协议、原训练数据或检查点。跨数据集输入允许不同的深度单位，但会分别记录。修改模型后需新建一轮训练，不能沿用旧记录并声称是相同实验。

本轮完整顺序在 `tools/run_training_methods_trial.py`，40个进程的实际命令、PID、退出状态和上限时间保存在 `reports/v5/experiment_ledger.json`。调度器不会自动重试失败组，也不会覆盖已有矩阵。它的路径针对本轮固定目录；重做整轮时先改为一套新目录。

固定随机种子用于配对比较，不保证所有CUDA操作逐位确定。三种子标准差反映训练随机性，不能代替更多独立场景。
