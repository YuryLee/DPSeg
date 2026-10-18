# 训练与评测复现

当前入口复现“第15轮基座冻结、重新训练两个点头”的流程。原训练循环、实例匹配、损失、采样、DDP 和评价函数继续使用原源码；路径适配代码在 `rgbd_seg/course_portable.py` 和 `tools/course_train_points.py`。

## 1. 准备环境与文件

使用 README 中的环境和[数据、权重说明](artifacts.md)。以下命令将 `/path/to/` 换成本机文件位置；`--gpus` 是分配给本任务的物理显卡编号。

## 2. 先做工程检查

```bash
python tools/course_train_points.py launch --engineering \
  --archive /path/to/blender.zip \
  --base-weights /path/to/epoch_15.pt \
  --coco-checkpoint /path/to/mask2former_swin_tiny_coco_instance \
  --gpus 0,1,2,3 --batch-size 2 \
  --output outputs/point_engineering
```

默认四卡检查覆盖16张训练图、8张验证图和2次更新。入口保存实际子进程退出记录，并自动检查训练覆盖、冻结基座、点头重载、优化器和各卡随机状态。成功后生成 `outputs/point_engineering/audit.json`。

## 3. 训练六轮点头

```bash
python tools/course_train_points.py launch \
  --archive /path/to/blender.zip \
  --base-weights /path/to/epoch_15.pt \
  --coco-checkpoint /path/to/mask2former_swin_tiny_coco_instance \
  --gpus 0,1,2,3 --batch-size 2 --epochs 6 \
  --engineering-receipt outputs/point_engineering/supervisor.json \
  --output outputs/point_full
```

完整训练要求工程检查的源码、基座、显卡布局和 batch 一致。四卡、每卡2张时，总 batch 为8，每轮1,804次更新，六轮共10,824次。每轮覆盖全部14,432张训练图，并在全部2,584张验证图上评测六组配置。

普通点头与深度点头共享采样和匹配；基座全程冻结。优化器为 AdamW，学习率0.0003、权重衰减0.01，使用预热和余弦衰减。原始源码记录了精确计算规则。最终按验证 APₛ、整体 AP、较早轮次选择，并重载对应点头。

输出目录必须是新目录。默认24小时硬超时；失败会保存日志和状态，入口不自动重试。当前入口从冻结基座新训点头；虽然保存了优化器和随机状态，尚未验收精确断点续训。

## 4. 查看产物

| 路径，相对训练输出目录 | 内容 |
|---|---|
| `protocol.json` | 本次运行的冻结输入、源码摘要和参数 |
| `training.log`、`supervisor.json`、`exit.json` | 日志、实际进程和退出记录 |
| `experiment/heads_epoch_XX.pt` | 每轮点头 |
| `experiment/validation_history.json` | 每轮完整验证指标 |
| `experiment/selection.json` | 验证集选择结果 |
| `audit.json` | 整次运行核验 |
| `model_manifest.json`，完整运行成功后 | 后续推理使用的模型清单 |

推理时用 `--manifest outputs/point_full/model_manifest.json`，并传入该运行选出的点头文件。不同显卡、软件环境、采样随机状态或 batch 布局可能改变结果；这里提供可核对的复现流程，不宣称跨设备逐位一致。

## 5. 完整验证集重新评测

```bash
CUDA_VISIBLE_DEVICES=0 python tools/course_predict.py \
  --archive /path/to/blender.zip \
  --base-weights /path/to/epoch_15.pt \
  --head-weights /path/to/heads_epoch_06.pt \
  --coco-checkpoint /path/to/mask2former_swin_tiny_coco_instance \
  --device cuda:0 --all --evaluate --split val \
  --output outputs/full_validation
```

该命令保存六组完整预测，调用原来的 COCO segm 评价函数，并写入 `evaluation_summary.json`。小规模 Demo 不计算新 AP。测试集推理可显式使用 `--split test --all`；本轮六轮训练及权重选择只使用 train/val，没有新增测试或 Kaggle 结果。
