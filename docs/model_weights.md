# 模型权重下载与消融对应表

下载入口：[DPSeg Hugging Face模型仓库](https://huggingface.co/Lemma456/DPSeg)。仅发布主消融 S0–S4 对应的3份权重，共517.83 MiB。权重通过 Hugging Face 模型仓库分发，Git 仓库保存代码和文件清单。

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
| [dpseg_rgb_swin_t_1024_e15.pt](https://huggingface.co/Lemma456/DPSeg/resolve/weights-v1/dpseg_rgb_swin_t_1024_e15.pt) | 181.30 MiB | Mask2Former RGB，1024，15轮；S0 |
| [dpseg_rgbd_swin_t_1024_e15.pt](https://huggingface.co/Lemma456/DPSeg/resolve/weights-v1/dpseg_rgbd_swin_t_1024_e15.pt) | 334.80 MiB | DA-Fusion RGB-D，1024，15轮；S1–S4 |
| [dpseg_points_base15_e06.pt](https://huggingface.co/Lemma456/DPSeg/resolve/weights-v1/dpseg_points_base15_e06.pt) | 1.72 MiB | 第15轮基座的普通／深度点头，第6轮；S2–S4及整图六组对照 |

主消融结果见[五步消融报告](results/progressive_ablation.md)。完整系统的本地测试APₛ为0.488299，Kaggle公榜为0.52886；两个评分范围不同。

## 下载与校验

在仓库根目录运行，默认通过 `hf-mirror.com` 镜像直连下载主消融三份权重，自动核对文件大小和SHA-256：

```bash
python tools/fetch_released_weights.py
# 离线检查已下载文件
python tools/fetch_released_weights.py --verify-only
```

默认位置为 `pretrained/dpseg_weights/`，可用 `--directory` 指定其他目录。新仓库的镜像同步可能延迟，需要官方入口时可传 `--endpoint https://huggingface.co`；两种方式均禁用代理。已有同名文件校验失败时脚本会停止，避免覆盖文件。也可以从上方链接手动下载。

- [机器可读清单](../configs/released_weights.json)：记录每份文件的SHA-256、轮次、模型张量摘要、协议摘要及点头基座配对。
- [Hugging Face清单](https://huggingface.co/Lemma456/DPSeg/resolve/weights-v1/released_weights.json)。
- [SHA256SUMS.txt](https://huggingface.co/Lemma456/DPSeg/resolve/weights-v1/SHA256SUMS.txt)。

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

S0–S4的推理实现见 `rgbd_seg/course_progressive_ablation.py` 和 `tools/run_course_progressive_ablation.py`；历史入口可能包含本机绝对路径，需要按各自协议调整位置。权重发布范围仅为当前五步主消融；不包含历史实验、中间轮次、课程图像、标注、预测CSV和优化器状态。
