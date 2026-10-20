# 模型权重下载与消融对应表

下载入口：[DPSeg 权重 Release](https://github.com/YuryLee/DPSeg/releases/tag/weights-v1)。共24份权重，约3.34 GiB；复现主消融 S0–S4 只需前三份，共517.83 MiB。权重通过 Release 附件分发，Git 仓库保存代码和文件清单。

## 主消融 S0–S4

| 实验 | 所需权重 | 推理设置 |
|---|---|---|
| S0：Mask2Former fine-tune | RGB第15轮 | 整图，RGB输入 |
| S1：+ RGB-D融合 | RGB-D第15轮 | 整图，无点头 |
| S2：+ 深度点细化 | RGB-D第15轮 + 配套第6轮点头 | 整图，选择 `depth` 点头 |
| S3：+ 切片推理 | 与S2相同 | 整图加四个640切片，关闭概率细化 |
| S4：+ 概率图深度细化 | 与S2相同 | 切片和概率图深度细化均开启 |

切片、概率图／二值深度后处理及小掩码保护均不增加权重。每份点头文件同时包含普通点头与深度点头，因此也支持整图六组消融。**点头必须与清单指定的基座配对**，第5轮基座的点头不能替换第15轮基座的点头。

| 文件 | 大小 | 对应实验 |
|---|---:|---|
| [dpseg_rgb_swin_t_1024_e15.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_rgb_swin_t_1024_e15.pt) | 181.30 MiB | Mask2Former RGB，1024，15轮；S0 |
| [dpseg_rgbd_swin_t_1024_e15.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_rgbd_swin_t_1024_e15.pt) | 334.80 MiB | DA-Fusion RGB-D，1024，15轮；S1–S4 |
| [dpseg_points_base15_e06.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base15_e06.pt) | 1.72 MiB | 第15轮基座的普通／深度点头，第6轮；S2–S4及整图六组对照 |

主消融结果见[五步消融报告](results/progressive_ablation.md)。完整系统的本地测试APₛ为0.488299，Kaggle公榜为0.52886；两个评分范围不同。

## 历史消融与点头轮次

以下文件覆盖其他已运行课程配置的选定权重。早期实验的训练预算不同，不能直接拼入上面的累计消融表。低学习率分支保留停止前第7轮结果；不代表完成15轮训练。

| 文件 | 大小 | 对应实验 |
|---|---:|---|
| [dpseg_maskrcnn_r50fpnv2_800_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_maskrcnn_r50fpnv2_800_e05.pt) | 175.42 MiB | 早期 Mask R-CNN R50-FPN v2，800，选定第5轮 |
| [dpseg_mask2former_swin_t_800_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_mask2former_swin_t_800_e05.pt) | 181.29 MiB | 早期 Mask2Former RGB，800，选定第5轮 |
| [dpseg_dafusion_swin_t_800_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_dafusion_swin_t_800_e05.pt) | 334.80 MiB | 早期 DA-Fusion RGB-D，800，选定第5轮 |
| [dpseg_dafusion_small_quota_800_e04.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_dafusion_small_quota_800_e04.pt) | 334.80 MiB | 小目标训练点配额，800，五轮中选定第4轮 |
| [dpseg_dafusion_resolution_800_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_dafusion_resolution_800_e05.pt) | 334.80 MiB | 匹配分辨率消融：800，全局batch33，第5轮 |
| [dpseg_rgbd_swin_t_1024_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_rgbd_swin_t_1024_e05.pt) | 334.80 MiB | 匹配分辨率消融：1024，第5轮；旧切片／点头系统基座 |
| [dpseg_dafusion_crop_1024_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_dafusion_crop_1024_e05.pt) | 334.80 MiB | 独立局部裁剪训练，1024，第5轮 |
| [dpseg_dafusion_focal_1024_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_dafusion_focal_1024_e05.pt) | 334.80 MiB | 独立 Focal 掩码损失，1024，第5轮 |
| [dpseg_dafusion_lr2e5_1024_e07.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_dafusion_lr2e5_1024_e07.pt) | 334.80 MiB | 低学习率续训分支，已完成第7轮；后按用户要求停止，非15轮对照 |
| [dpseg_rgb_swin_t_1024_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_rgb_swin_t_1024_e05.pt) | 181.29 MiB | 主RGB控制组前5轮阶段权重 |
| [dpseg_points_base15_e01.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base15_e01.pt) | 1.72 MiB | 第15轮基座的普通／深度点头，第1轮 |
| [dpseg_points_base15_e02.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base15_e02.pt) | 1.72 MiB | 第15轮基座的普通／深度点头，第2轮 |
| [dpseg_points_base15_e03.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base15_e03.pt) | 1.72 MiB | 第15轮基座的普通／深度点头，第3轮 |
| [dpseg_points_base15_e04.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base15_e04.pt) | 1.72 MiB | 第15轮基座的普通／深度点头，第4轮 |
| [dpseg_points_base15_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base15_e05.pt) | 1.72 MiB | 第15轮基座的普通／深度点头，第5轮 |
| [dpseg_points_base05_e01.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base05_e01.pt) | 1.72 MiB | 第5轮基座的普通／深度点头，第1轮 |
| [dpseg_points_base05_e02.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base05_e02.pt) | 1.72 MiB | 第5轮基座的普通／深度点头，第2轮 |
| [dpseg_points_base05_e03.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base05_e03.pt) | 1.72 MiB | 第5轮基座的普通／深度点头，第3轮 |
| [dpseg_points_base05_e04.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base05_e04.pt) | 1.72 MiB | 第5轮基座的普通／深度点头，第4轮 |
| [dpseg_points_base05_e05.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base05_e05.pt) | 1.72 MiB | 第5轮基座的普通／深度点头，第5轮 |
| [dpseg_points_base05_e06.pt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_points_base05_e06.pt) | 1.72 MiB | 第5轮基座的普通／深度点头，第6轮 |

## 下载与校验

在仓库根目录运行，默认只下载主消融三份权重，使用直连并自动核对文件大小和SHA-256：

```bash
python tools/fetch_released_weights.py
# 下载本次发布的全部24份权重
python tools/fetch_released_weights.py --group all
# 离线检查已下载文件
python tools/fetch_released_weights.py --group all --verify-only
```

默认位置为 `pretrained/dpseg_weights/`，可用 `--directory` 指定其他目录。已有同名文件校验失败时脚本会停止，避免覆盖文件。也可以从上方链接手动下载。

- [机器可读清单](../configs/released_weights.json)：记录每份文件的SHA-256、轮次、原始路径、模型张量摘要、协议及点头基座配对。
- [Release清单](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/released_weights.json)。
- [SHA256SUMS.txt](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/SHA256SUMS.txt)。
- [原始协议包](https://github.com/YuryLee/DPSeg/releases/download/weights-v1/dpseg_ablation_protocols.tar.gz)：按协议SHA-256命名，保留历史绝对路径和原始字节；迁移运行时显式提供实际路径。

所有权重保留原始checkpoint字节，只更换发布文件名；未量化、未裁剪、未重新序列化。可用 `torch.load(path, map_location="cpu", weights_only=True)` 读取，基座参数在 `model`，配对点头参数在 `heads`。它们是推理checkpoint，不含优化器和完整随机状态，不用于精确恢复中断训练。

## 运行现有Demo

还需课程数据与公开COCO初始化目录，准备方式见[文件说明](artifacts.md)。以下入口展示第15轮基座的六组整图方法：

```bash
python tools/course_predict.py \
  --archive /path/to/blender.zip \
  --base-weights pretrained/dpseg_weights/dpseg_rgbd_swin_t_1024_e15.pt \
  --head-weights pretrained/dpseg_weights/dpseg_points_base15_e06.pt \
  --coco-checkpoint pretrained/mask2former_swin_tiny_coco_instance \
  --device cuda:0 --indices 0,644,1292,2580 \
  --output outputs/course_demo
```

S0–S4的推理实现见 `rgbd_seg/course_progressive_ablation.py` 和 `tools/run_course_progressive_ablation.py`；历史入口可能包含本机绝对路径，需要按各自协议调整位置。权重发布不包含课程图像、标注、预测CSV、优化器状态、工程冒烟测试权重或重复的中间基座轮次。
