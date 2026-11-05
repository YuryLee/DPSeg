# DPSeg

**RGB-D Fusion and Depth-Guided Point Refinement for Small Object Instance Segmentation** · 无相机内参 RGB-D 小目标实例分割

课程 `component` 实例分割项目。当前最佳方法采用双路 **Swin-Large**，融合 RGB 与归一化米制深度，通过深度引导点细化、重叠切片推理和概率图深度细化生成实例掩码。训练和推理输入为 RGB、深度及有效性图，**无需读取、估计或提供相机内参**。仓库同时保留 Mask R-CNN、RGB Mask2Former 和原 Tiny＋XYZ 方案的对照代码、结果与 Demo。

## 当前最佳配置（按 Mask APₛ）

**Swin-Large＋无内参 RGB-D 融合＋深度引导点细化＋切片推理＋概率图深度细化**。基座训练 15 轮，冻结后训练点细化头 6 轮，仅依据完整验证集选取第 6 轮点头。

| 评价来源 | Mask AP | Mask APₛ |
|---|---:|---:|
| 完整验证集，2,584 张 | 0.885717 | 0.702910 |
| 完整本地测试集，2,112 张 | **0.880289** | **0.525286** |
| 新 Kaggle 比赛 `rgb-d-333` 公榜 | — | **0.56785** |

本地两份测试 CSV 已逐掩码解码并独立重新评分，AP、APₛ 差异均为 0。Kaggle 提交编号为 **57967712**，状态 COMPLETE；不同评分来源分别列报。[完整实验记录](docs/results/swinl_depth_no_k.md) · [可随代码分发的结果摘要](docs/results/swinl_depth_no_k.json) · [Kaggle 提交记录](docs/results/kaggle_swinl_depth_no_k_rgb_d_333.json)。

## 任务与评价

任务是在合成渲染的电子元器件 RGB-D 图像中，分割单类 `component` 的每一个可见实例。图像与输出掩码尺寸均为1024×1024。

| 划分 | 图像数 | 用途 |
|---|---:|---|
| train | 14,432 | 基座微调与点头训练 |
| val | 2,584 | 选择模型、点头轮次和配置 |
| test | 2,112 | 生成课程竞赛提交 |

主指标是 **COCO Mask APₛ@[0.50:0.95]**，同时报告整体 Mask AP。IoU阈值从0.50到0.95、步长0.05；small按原图标注掩码面积 `[0, 32²]` 像素统计，每图最多100个预测实例。本文所有分数范围均为0–1。Kaggle 公榜按对应比赛的公开评分部分计算，与完整验证集、本地测试及其他比赛成绩分别报告。

评价与提交格式详见[指标说明](docs/course_metric.md)。

## 技术路线

当前最佳配置的路线为：**无内参深度表示 → 双路 Swin-Large 与 DS/DC 融合 → 整图和重叠切片候选 → 深度引导点细化 → 概率图深度细化**。切片参考 [SAHI](https://arxiv.org/abs/2202.06934) 的推理思路，并为多模态输入和来源视图点细化定制实现。

```mermaid
flowchart LR
    Input["RGB + 米制深度 + 有效性<br/>无需相机内参"]
    Views["整图 + 四个640切片<br/>各视图输入1024"]
    Fusion["双路Swin-Large<br/>四尺度DS/DC融合"]
    Merge["Mask2Former粗掩码<br/>原图映射、边缘过滤与去重"]
    Point["来源视图深度点细化<br/>局部深度引导MLP残差"]
    Probability["原图概率图深度细化<br/>1024×1024实例掩码"]
    Input --> Views --> Fusion --> Merge --> Point --> Probability
```

### 1. 无内参深度表示与 RGB-D 融合

根据各图的 `depth_scale_m` 将原始 uint16 深度转换为米，并保留有效性图。每个视图缩放后，在有效深度上减去中位数，再除以绝对偏差的第 95 百分位数加 1e-6；将归一化 Z 重复为三个通道，并附上有效性通道，经 4→3 输入适配器进入深度分支。这一表示不进行 XYZ 反投影，不使用焦距、主点或估计内参。

RGB 与深度使用两套参数独立的 Swin-Large，均由同一公开 COCO 初始化的骨干权重开始课程训练。四个尺度上，DS（可变形自注意力）聚合模态内特征，DC（双向可变形交叉注意力）交换颜色与几何信息；融合结果交给 Mask2Former 的像素和查询解码器。训练保留原生集合匹配、分类、掩码 BCE/Dice 及辅助层监督。

DS/DC 沿用 DA-Fusion 的融合机制；本项目使用无内参深度表示、课程单类解码和对应训练流程。[无内参编码与骨干实现](rgbd_seg/course_depth_backbone.py) · [原论文与实现约定](docs/dafusion_method_contract.md)。

### 2. 深度引导点细化

点细化借鉴 [PointRend](https://arxiv.org/abs/1912.08193) 的不确定点选择思路。冻结本组第 15 轮 RGB-D 基座后，训练配套的残差点细化头：

- **共同输入：** Large 最细融合特征 192 维、像素解码器特征 256 维，以及该实例的原始粗掩码 logit。
- **局部深度输入：** 中心有效性、四邻域有符号深度差和四组配对有效性，共 9 维；有效深度差经 `tanh(差值 / 0.001米)` 压缩。
- **结构：** 三个 256 维隐藏层及一个输出层，各隐藏层后再次拼接粗掩码 logit，输出掩码残差；输出层零初始化。
- **采样与监督：** 每实例 784 点，75% 偏向不确定位置、25% 随机，以点 BCE 监督；普通点头与深度点头共享冻结特征、匹配、采样及监督规则。
- **推理：** 区域由预测掩码确定，粗掩码从 256×256 逐级细化到 512、1024，每级每实例最多修正 784 点；类别、分数及候选顺序保持不变，空掩码按既定规则回退。

当前最佳模型的点头读取 RGB、测量深度及有效性所形成的特征，无需相机内参。Large 的点头特征维度为 448，须与配套基座一起使用；原 Tiny 点头的特征维度为 352。[支持两种骨干的点细化实现](rgbd_seg/course_depth_backbone_points.py)。

### 3. 切片推理：补充小目标候选

保留 1024×1024 整图，并增加四个 640×640 角部切片，相邻切片重叠 256 像素。RGB、米制深度和有效性图同步裁剪，各切片放大到 1024 输入模型，逐视图完成粗预测；每个视图独立进行无内参深度归一化。

预测缩回切片尺寸并映射到原图，过滤触及内部裁剪边缘 2 像素范围的掩码。按原始分数排序，对跨视图掩码 IoU 严格大于 0.5 的重复候选保留高分者，最多输出 100 个实例。候选保留来源视图、查询编号和相应特征，供后续点细化读取。

本流程参考 SAHI 的整图与重叠切片推理思路，RGB-D 对齐、掩码去重和来源特征缓存采用自定义实现。置信度阈值为 0.01；案例页面的 0.50 是展示阈值。[无内参切片与系统衔接](rgbd_seg/course_depth_backbone_system.py)。

### 4. 概率图深度细化：约束最终轮廓

将来源视图点细化后的软概率还原至原图，以空间距离和原始米制深度差决定邻域权重，执行一次联合加权，再以 0.5 阈值生成二值掩码。邻域半径 2 像素、空间标准差 1.5 像素、深度标准差 0.001 米；无效深度、平票与空掩码保留既定回退规则。该步骤不读取相机内参。

概率图细化保持候选、类别、分数和排序，处理的是软掩码概率。它替代历史二值深度平滑。[概率图深度细化实现](rgbd_seg/course_tiled_probability.py)。

### 训练与推理顺序

训练先完成 15 轮基座微调，再冻结基座训练 6 轮点细化头；切片和概率图细化仅用于推理。完整推理顺序为：**五视图粗预测 → 原图映射与筛选 → 候选来源视图点细化 → 原图概率图深度细化 → COCO RLE**。

训练与评测入口见 [基座训练](tools/run_course_depth_backbone.py)、[点头训练与完整系统评测](tools/run_course_depth_backbone_system.py)和 [Large＋无内参实验控制程序](tools/run_course_swinl_depth_no_k.py)。控制程序保留原实验目录和冻结协议约定，重新运行前须按自己的数据与权重位置准备配置。[文件及权重说明](docs/artifacts.md)。

## 消融实验与结果

<!-- progressive-ablation:start -->
### 基线对比与累计模块消融结果

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

完整 DPSeg 相对 S1 的 APₛ、ARₛ 和 AP 分别提高 **+0.116489、+0.192791、+0.034663**。S2、S3 已补齐完整测试推理；五组 CSV 与各自的评分预测逐行一致。

[完整报告与提交文件路径](docs/results/course_ablation_swinl.md) · [结果 CSV](docs/results/course_ablation_swinl.csv) · [结果 JSON](docs/results/course_ablation_swinl.json)。[历史 Tiny＋XYZ 累计消融](docs/results/progressive_ablation.md)保留旧数值及原模块顺序；本表 S3 为新增的“点细化＋概率图细化”结果。
<!-- progressive-ablation:end -->

<!-- module-removal:start -->
### 逐项去除消融

本表从完整 DPSeg 出发，每次独立去除一项模块，其余组件与权重保持不变。统一使用 **Swin-Large＋无内参深度、完整本地测试集 2,112 张图**，数值范围 0–1。这是固定权重的推理消融，各去除配置独立比较；RGB-D 融合的去除实验暂缓。加粗仅标记完整配置。

| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |
|:---:|:---:|:---:|:---:|
| **完整 DPSeg** | **0.525286** | **0.674087** | **0.880289** |
| 去除深度感知点细化 | 0.516680 | 0.662962 | 0.872696 |
| 去除深度引导概率图细化 | 0.459486 | 0.585932 | 0.860555 |
| 去除切片增强推理 | 0.476384 | 0.569641 | 0.886068 |

- **点细化：** 完整配置相对去除点细化的 APₛ、ARₛ、AP 分别增加 0.008606、0.011125、0.007594。这是点头在包含切片与概率图细化的组合中的贡献；累计表 S1→S2 的 APₛ 仍下降 0.004966，两项观察对应不同上下文，不能互相替代。
- **概率图细化：** 完整配置相对去除此模块的 APₛ、ARₛ、AP 分别增加 0.065800、0.088155、0.019735。
- **切片：** 完整配置相对无切片的 APₛ、ARₛ 分别增加 0.048902、0.104446，整体 AP 则下降 0.005779。切片改善本任务的小目标指标，但本次结果不支持其同时改善整体 AP。

所有结果来自同一次 seed=17 训练：共用第 15 轮 RGB-D 基座，保留点细化时使用同一验证集选取的第 6 轮点头。没有针对各去除配置重新训练或调参，测试标注只用于评分，结果不表示重新训练后的架构优劣，也不构成统计显著性结论。完整配置与无切片配置均逐图精确复现已有 S4、S3，四组 CSV 与评分预测完全一致。

[完整报告、提交文件路径与复现命令](docs/results/course_removal_swinl.md) · [指标 CSV](docs/results/course_removal_swinl.csv) · [核验 JSON](docs/results/course_removal_swinl.json)。
<!-- module-removal:end -->

<!-- tile-size:start -->
### 切片尺寸敏感性分析

固定完整 DPSeg 的 Swin-Large＋无内参深度、第 15 轮基座与第 6 轮点头，保留两种掩码细化。在完整本地测试集 **2,112 张图**上比较切片尺寸，数值范围 0–1。512、768 重新推理；640 的窗口和完整流程与已完成的默认配置完全一致，复用其全量结果并通过样例复现核验。

| 切片尺寸 | 每图视图数（含整图） | APₛ ↑ | ARₛ ↑ | AP ↑ |
|:---:|:---:|:---:|:---:|:---:|
| 512×512 | 10 | 0.490499 | 0.675241 | 0.852428 |
| 640×640（当前默认） | 5 | 0.525286 | 0.674087 | 0.880289 |
| 768×768 | 5 | 0.518685 | 0.647804 | 0.888220 |

三组中 APₛ 最高的是 **640**，ARₛ 最高的是 **512**，整体 AP 最高的是 **768**。不同尺寸的优势取决于评价指标。512 虽然有更多切片并将目标放得更大，APₛ 仍低于 640；切片更小不保证本任务的分割精度更高。

- 512 相对 640：APₛ **-0.034787**，ARₛ **+0.001154**，AP **-0.027861**。
- 768 相对 640：APₛ **-0.006601**，ARₛ **-0.026283**，AP **+0.007931**。

本轮固定名义重叠率 20%，采用滑窗步长并将末端窗口移至图像边界。**实际重叠和视图数量会随尺寸变化，结果同时反映尺度、覆盖和候选数量的影响，不能当作同计算量的纯尺度比较。**这是在已查看过结果的本地测试集上追加的敏感性分析，没有依据这张表更换默认配置；若要选择新的切片参数，应另在验证集确认。分数与 Kaggle 公榜分开报告。

[窗口几何、核验和 CSV 路径](docs/results/course_tile_size_swinl.md) · [结果 CSV](docs/results/course_tile_size_swinl.csv) · [结果 JSON](docs/results/course_tile_size_swinl.json)。
<!-- tile-size:end -->

### 去内参与 Swin-Large 补充对比

三组补充实验均已完成 15 轮基座训练、6 轮点细化训练，以及完整验证和测试；点头仅依据验证集选择。下表统一使用课程归档中的本地测试标注（2,112 张图），分数范围 0–1。

| 骨干与深度表示 | 推理配置 | Mask AP | Mask APₛ |
|---|---|---:|---:|
| 原 Swin-Tiny，内参反投影 XYZ | 基础配置 | 0.817895 | 0.368086 |
| 新 Swin-Tiny，无内参深度 | 基础配置 | 0.820201 | 0.362218 |
| 新 Swin-Large，内参反投影 XYZ | 基础配置 | 0.835743 | 0.393056 |
| 新 Swin-Large，无内参深度 | 基础配置 | 0.845626 | 0.408797 |
| 原 Swin-Tiny，内参反投影 XYZ | 完整配置 | 0.868243 | 0.488299 |
| 新 Swin-Tiny，无内参深度 | 完整配置 | 0.872224 | 0.488260 |
| 新 Swin-Large，内参反投影 XYZ | 完整配置 | 0.863532 | 0.499089 |
| 新 Swin-Large，无内参深度 | 完整配置 | **0.880289** | **0.525286** |

完整配置包含深度点细化、切片推理和概率图深度细化。Large＋无内参在上表完整配置中的 AP 和 APₛ 均最高；相对 Large＋XYZ 分别增加 0.016757 和 0.026197，相对原 Tiny＋XYZ 分别增加 0.012046 和 0.036986。三组补充实验的六份 CSV 均已逐掩码解码并从 CSV 独立运行 COCO 复评，AP 和 APₛ 差异均为 0。这些是本地测试结果，Kaggle 成绩另行报告。本轮为单种子实验，预训练、深度表示和微批布局等比较限制及 CSV 路径见[补充对比记录](docs/results/depth_backbone_comparison.md)和[Large＋无内参结果](docs/results/swinl_depth_no_k.md)。下方效率表已更新为当前 Swin-Large＋无内参配置。

Large＋无内参完整版已于 2026-11-04 提交到新比赛 [RGB-D图像分割-333](https://www.kaggle.com/competitions/rgb-d-333)，公榜 APₛ 为 **0.56785**，提交编号 **57967712**，状态 COMPLETE。查询时队伍“21”位列第 1／3 队；排名会随其他提交变化。该成绩属于新比赛，与上方本地测试及旧比赛 `rgb-d` 的成绩分别报告。[提交与文件校验记录](docs/results/kaggle_swinl_depth_no_k_rgb_d_333.json)。

<!-- inference-efficiency:start -->
### 模型规模与推理开销对比

统一在 **NVIDIA A100-PCIE-40GB 单卡、FP32、batch=1、1024×1024 输入** 下测量。从完整验证集等间隔选取 32 张图，预热 5 张后重复 3 次，每组共 96 次计时。各组使用相同图片与每轮顺序，按表中顺序累计添加模块。

| 配置 | 参数量（M） | 推理权重（MiB） | 平均耗时（ms/图） | FPS |
|:---:|:---:|:---:|:---:|:---:|
| Mask2Former | 215.502 | 826.22 | 531.2 | 1.88 |
| + RGB-D 融合（DA-Fusion） | 460.955 | 1766.71 | 944.0 | 1.06 |
| + 深度感知点细化 | 461.205 | 1767.67 | 1137.9 | 0.88 |
| + 深度引导概率图细化 | 461.205 | 1767.67 | 1303.2 | 0.77 |
| **+ 切片增强推理（DPSeg）** | 461.205 | 1767.67 | 4794.0 | 0.21 |

RGB 基线本次使用已保存的第 12 轮检查点；测速时 15 轮训练尚未完成，其最终权重的耗时仍需复核。RGB-D 各组共用第 15 轮基座，含点细化的各组共用第 6 轮深度点头。

参数量包含冻结骨干；权重文件只计推理使用的基座和深度点头。概率图细化与切片不增加参数。计时包含预处理、模型、对应的细化/切片及 CPU COCO RLE 编码，不含磁盘读取、图像解码与模型加载。本次不统计 GPU 峰值显存。

测量口径、共享服务器限制及逐轮结果见[当前效率报告](docs/results/course_efficiency_swinl.md)，另提供[CSV](docs/results/course_efficiency_swinl.csv)和[JSON](docs/results/course_efficiency_swinl.json)。[历史 Tiny＋XYZ 效率报告](docs/results/course_efficiency.md)保留原始测量，硬件和模块顺序与本表不同。
<!-- inference-efficiency:end -->

点头类型、点头轮次、早期Mask R-CNN/Mask2Former/DA-Fusion基线，以及旧权重的切片实验，统一移至[补充实验](docs/results/ablation_background.md)。

## 安装

已验证环境为 Linux、Python 3.13.5、PyTorch 2.7.1、Transformers 4.53.2。原 Tiny 单卡推理与四卡点头入口、当前 Large 无内参双卡训练与完整评测均已在真实数据上运行。

```bash
git clone https://github.com/YuryLee/DPSeg.git
cd DPSeg

python3.13 -m venv .venv
source .venv/bin/activate

env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
    -u http_proxy -u https_proxy -u all_proxy \
    python -m pip install -r requirements.txt \
    -i https://pypi.tuna.tsinghua.edu.cn/simple
```

安装使用[TUNA PyPI镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)，命令取消当前代理环境变量。下载公开初始化权重的可选依赖在 `requirements-download.txt`，浏览器验收依赖在 `requirements-dev.txt`。补充 UOIS/UOAIS 评测与相关检查需要 `requirements-benchmark.txt` 中的图像处理依赖，可在上述安装命令中替换依赖文件名。

## 模型权重

消融实验权重见 [Hugging Face](https://huggingface.co/Lemma456/DPSeg)。原 Tiny＋XYZ 主消融 S0–S4 共用三份权重，合计约518 MiB：第15轮RGB基线、第15轮RGB-D基座、配套第6轮普通／深度点头。切片和概率图细化不增加权重。

Hugging Face 上现有的 `weights-v1` 对应原 Tiny＋XYZ 五步消融。当前最佳 Large＋无内参基座与点头保存在本机实验目录，尚未发布到该权重版本；文件位置与校验值见[模型权重说明](docs/model_weights.md)。

```bash
python tools/fetch_released_weights.py # 主消融三份权重
```

## 运行 Demo

### GT与五阶段消融案例展台

这一图库展示原 Tiny＋XYZ 的 S0–S4 固定预测。用于课堂展示的 12 个精选案例包含3个分割良好的案例、4个漏检改善案例、2个误检案例、2个边界改善案例，以及1个仍未解决的漏检案例。
每例提供RGB、原始深度、GT、S0–S4预测、独立二值掩码和可用于PPT的对比图。网页支持局部放大、整图对照、纯掩码、轮廓和误差着色。

本地入口为 `outputs/course_showcase_test_v1/index.html`；打包文件为 `outputs/DPSeg_case_demo_v1.zip`。解压后双击 `index.html` 即可离线演示。
查看[案例清单与解释](docs/results/showcase_cases.md)和[演示说明](docs/demo_usage.md)。
案例GT来自课程下载归档中已有的test标注；这些定性案例不能替代完整测试集AP或Kaggle评分。

### 从原 Tiny 模型生成预测演示

准备课程 `blender.zip`、第15轮基座、第6轮点头和公开 COCO 初始化目录，详见[文件说明](docs/artifacts.md)。将下方路径换成各自文件的位置：

```bash
CUDA_VISIBLE_DEVICES=0 python tools/course_predict.py \
  --archive /path/to/blender.zip \
  --base-weights /path/to/epoch_15.pt \
  --head-weights /path/to/heads_epoch_06.pt \
  --coco-checkpoint /path/to/mask2former_swin_tiny_coco_instance \
  --device cuda:0 --indices 0,644,1292,2580 \
  --output outputs/course_demo
```

完成后打开 `outputs/course_demo/index.html`。页面自包含，可复制到其他电脑离线查看；浏览时无需 Python 或 GPU。支持六组方法对比、置信度和透明度调节、同步缩放、JSON与叠加图下载。

![真实预测生成的 Demo](docs/images/course_demo.png)

模型清单位于 [configs/course_e15_point.json](configs/course_e15_point.json)。加载器会验证基座、点头、冻结协议、源码及张量摘要，并把历史路径映射到当前项目和已安装的固定版本库。

## 训练与重新评测

[复现说明](docs/reproduction.md)提供以下流程：

1. 使用新训练入口做四卡工程检查：16张训练、8张验证、2次更新。
2. 将通过的工程收据传给完整六轮训练。
3. 用完整验证集选择点头，核对实际退出状态并审计全部训练和评分产物。
4. 使用 `tools/course_predict.py --all --evaluate --split val` 重新计算完整 COCO Mask AP/APₛ。

训练入口沿用原训练循环，仅适配文件位置。历史协议原始字节保留在 [artifacts/protocols](artifacts/protocols)，数据与权重通过显式参数提供。

## 已完成的软件检查

- 五项加载边界检查，包括源码变化、库文件迁移、错误基座配对和推理输入字段。
- 将源码复制到新目录后，8张验证图、6组方法共48份预测与原评分记录完全一致。
- 从实际 Git 提交检出到新目录后，386个文件字节一致；单张验证图的六组预测与原评分记录完全一致。
- 复制目录中的四卡工程运行通过完整审计。
- 离线浏览器检查通过：8个样例、6组方法、2,832个RLE掩码、单像素坐标、下载和手机布局。

详见[软件验收记录](docs/results/software_checks.json)。模型单元检查可运行：

```bash
python -m unittest tests.test_course_portable -v
```

## 项目目录

| 目录/文件 | 内容 |
|---|---|
| `rgbd_seg/course_portable.py` | 当前权重加载与六组推理路径 |
| `tools/course_predict.py` | 样例Demo、完整划分推理和验证评测 |
| `tools/course_train_points.py` | 工程检查与点头训练入口 |
| `web/course.html` | 离线交互界面 |
| `configs/course_e15_point.json` | 固定模型与指标清单 |
| `artifacts/protocols/` | 保留原始字节的训练协议 |
| `docs/results/` | 核验后的指标、报告和曲线 |
| `rgbd_seg/`、`tools/`、`tests/` | 各阶段模型、实验、校验及测试代码 |
| `third_party/` | 固定版本的第三方代码、来源记录与原许可证 |

本机原始数据、权重、大型逐图预测与历史运行目录由 Git 忽略。旧实验脚本和文档保留原有环境路径，用于追溯；当前跨目录运行入口见本页与复现说明。

## 结论范围

这是一次单种子、固定训练预算的课程实验。第6轮在本轮候选中最好，不据此宣称已收敛或达到统计显著。五步累计消融已完成完整验证与本地测试；结果见主表。Kaggle公榜成绩须单独通过竞赛提交获得。

公开实现的课程适配与原论文设置存在差异，见[方法约定](docs/dafusion_method_contract.md)。第三方代码保留各自许可证；课程数据按课程渠道取得，已训练权重见[权重说明](docs/model_weights.md)。
