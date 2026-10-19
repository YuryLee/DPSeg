# DPSeg

**Depth-guided Point Refinement for Segmentation** · RGB-D 小目标实例分割与深度边界细化

课程 `component` 实例分割项目，包含 Mask R-CNN、Mask2Former、DA-Fusion 课程适配、深度点头和深度引导的掩码细化。提供训练与推理入口、完整验证结果、源码核验记录和离线交互 Demo。

## 本轮完成结果

冻结 DA-Fusion 第15轮基座，训练普通点头与深度点头各6轮。每轮覆盖14,432张训练图，并对全部2,584张验证图评测六组配置。四卡、全局 batch=8，累计10,824次更新；整次训练和验证核验通过。

以下为**第6轮、1024×1024整图、完整验证集**结果，分数范围0–1：

| 配置 | Mask AP | Mask APₛ |
|---|---:|---:|
| DA-Fusion 原始输出 | 0.814871 | 0.503742 |
| ＋概率图深度细化 | 0.851316 | 0.566674 |
| ＋普通点头 | 0.813697 | 0.488593 |
| ＋普通点头＋概率图深度细化 | 0.852257 | 0.561337 |
| ＋深度点头 | 0.846843 | 0.547258 |
| **＋深度点头＋概率图深度细化** | **0.864680** | **0.586192** |

最终按验证 APₛ、AP、较早轮次选择第6轮。相对同一基座的概率细化，深度点头组合的 APₛ 增加 **0.019518**。本轮只完成整图验证，历史切片、测试与 Kaggle 成绩按来源单独列报。

- [最终实验报告](docs/results/point_e15.md) · [逐轮结构化指标](docs/results/point_e15_results.json) · [整次核验](docs/results/point_e15_audit.json)
- [评价指标与 CSV 格式](docs/course_metric.md) · [结构和方法区别](docs/methods.md) · [历史实验](docs/historical_results.md)

![完整验证学习曲线](docs/results/point_learning_curves.png)

## 安装

已验证环境为 Linux、Python 3.13.5、PyTorch 2.7.1、Transformers 4.53.2。单卡推理和四卡点头训练入口均已用真实数据检查。

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

安装使用[TUNA PyPI镜像](https://mirrors.tuna.tsinghua.edu.cn/help/pypi/)，命令取消当前代理环境变量。下载公开初始化权重的可选依赖在 `requirements-download.txt`，浏览器验收依赖在 `requirements-dev.txt`。

## 运行 Demo

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

这是一次单种子、固定训练预算的课程实验。第6轮在本轮候选中最好，不据此宣称已收敛或达到统计显著。本轮没有新增测试 CSV、Kaggle 提交或新权重的完整切片评价。

公开实现的课程适配与原论文设置存在差异，见[方法约定](docs/dafusion_method_contract.md)。第三方代码保留各自许可证；数据与课程训练权重由项目组按课程渠道共享。
