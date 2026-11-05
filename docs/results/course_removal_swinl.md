# Swin-Large 逐项去除消融

本表从完整 DPSeg 出发，每次独立去除一项模块，其余组件与权重保持不变。统一使用 **Swin-Large＋无内参深度、完整本地测试集 2,112 张图**，数值范围 0–1。这是固定权重的推理消融，各去除配置独立比较；RGB-D 融合的去除实验暂缓。加粗仅标记完整配置。

| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |
|:---:|:---:|:---:|:---:|
| **完整 DPSeg** | **0.525286** | **0.674087** | **0.880289** |
| 去除深度感知点细化 | 0.516680 | 0.662962 | 0.872696 |
| 去除深度引导概率图细化 | 0.459486 | 0.585932 | 0.860555 |
| 去除切片增强推理 | 0.476384 | 0.569641 | 0.886068 |

APₛ、ARₛ、AP 均采用 COCO 掩码评测，在 IoU=0.50:0.05:0.95 上平均，每图最多 100 个实例；小目标沿用原始标注面积 [0, 32²] 像素。完整本地测试与 Kaggle 公榜分开报告。

## 结果解释

- **点细化：** 完整配置相对去除点细化的 APₛ、ARₛ、AP 分别增加 0.008606、0.011125、0.007594。这是点头在包含切片与概率图细化的组合中的贡献；累计表 S1→S2 的 APₛ 仍下降 0.004966，两项观察对应不同上下文，不能互相替代。
- **概率图细化：** 完整配置相对去除此模块的 APₛ、ARₛ、AP 分别增加 0.065800、0.088155、0.019735。
- **切片：** 完整配置相对无切片的 APₛ、ARₛ 分别增加 0.048902、0.104446，整体 AP 则下降 0.005779。切片改善本任务的小目标指标，但本次结果不支持其同时改善整体 AP。

所有结果来自同一次 seed=17 训练：共用第 15 轮 RGB-D 基座，保留点细化时使用同一验证集选取的第 6 轮点头。没有针对各去除配置重新训练或调参，测试标注只用于评分，结果不表示重新训练后的架构优劣，也不构成统计显著性结论。

## 固定配置与实现

基座第 15 轮、点头第 6 轮，推理为 FP32，关闭 TF32；置信度阈值 0.01、掩码跨视图 NMS 阈值 0.5、每图最多 100 个实例。切片配置为整图加四个 640×640 角点切片，相邻切片重叠 256 像素，切片输入放大到 1024×1024。无切片配置只使用整图候选；去除点细化时仍对原粗概率图做既有深度引导概率细化；去除概率图细化时直接输出点细化掩码。阈值、深度表示、边缘过滤、去重规则和其余权重均沿用冻结完整系统。

四个输出共享各视图的基座前向结果以节省评测计算；无切片输出只取整图候选和整图特征。共享计算的运行耗时不用于逐变体速度对比。

## 核验记录

- 工程检查在 3 张预先指定的验证图上，将四条共享特征路径分别与独立原实现比较，掩码和置信度完全一致。
- 工程检查、5 个推理进程、4 个评分进程全部正常退出。每个变体的 5 个分片合计覆盖全部 2,112 张图，各图恰好出现一次。
- 完整配置逐图精确复现已有累计 S4；无切片配置逐图精确复现上一轮累计 S3，所有 COCO 指标也完全相同。
- 三个含切片配置的候选类别、数量、分数和顺序完全相同。四份 CSV 均已有原生 RLE 解码校验，文件哈希与校验记录一致；发布前再次核对全部 CSV 行与评分预测、分片与合并预测的对应关系。
- 已核对冻结协议、模型源码、权重、进程记录、结果文件、评分设置及 CSV 哈希。数据归档的 SHA-256 沿用训练协议记录，本发布程序核对测试图像 ID 和评分 GT 统计，没有重新计算整个数据压缩包的哈希。

## 提交文件本机路径

- **完整 DPSeg：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_full.csv`
- **去除深度感知点细化：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_without_point.csv`
- **去除深度引导概率图细化：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_without_probability.csv`
- **去除切片增强推理：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_without_tiles.csv`

预测 CSV、权重和完整实验记录保留在本机实验目录；本页配套 [指标 CSV](course_removal_swinl.csv) 和 [核验 JSON](course_removal_swinl.json) 可随代码分发。

## 复现命令

在仓库根目录运行，选择实际空闲显卡，输出目录必须尚不存在：

```bash
.venv/bin/python -u tools/evaluate_course_swinl_removal.py \
  --output reports/course_swinl_removal_new --gpus 3,4,5,6,7 \
  --run reports/course_swinl_depth_no_k_v1/swinl_depth_no_k \
  --archive /data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip \
  --without-tiles-reference reports/course_swinl_report_ablation_v1/S3
.venv/bin/python tools/summarize_course_swinl_removal.py \
  --run reports/course_swinl_removal_new
```

程序需要已存在的冻结基座/点头、原完整 S4 预测、上一轮 S3 预测及累计结果记录。按实际位置调整路径；工程复现检查通过后才开始全量推理，完成后统一重新评分。上述发布命令只更新本页及 README 的逐项去除段落。

## Material Passport

| 字段 | 内容 |
|---|---|
| 来源 | 本地冻结实验 `reports/course_swinl_removal_v1`；academic-research-suite 实验流程 |
| 材料类型 | 实际模型推理、完整测试 COCO 评分与已验证预测 CSV |
| 范围 | 固定权重独立去除三项模块；RGB-D 去除延后 |
| 核验状态 | VERIFIED；四组完整测试、分片与文件哈希、全部 CSV 绑定、两组历史预测精确复现 |
| 协议 SHA-256 | `ae765b94bf2085bed7ccf895289acc25ca4f5d0fc5f1707f8d1d3996d83e62d1` |
| 完成时间（UTC） | 2026-11-05T16:14:13.091507+00:00 |
| 使用边界 | 单次训练、无重训练、无统计显著性检验；本地结果与 Kaggle 分开报告 |
