# 数据与模型文件

代码使用显式文件路径，数据和训练权重可以放在项目目录外。加载时会核对文件、源码、模型结构和张量摘要。

| 文件 | 用途 | 准备方式 |
|---|---|---|
| 课程 `blender.zip` | RGB、uint16 深度、内参及 COCO 标注 | 使用课程提供的完整归档 |
| COCO Mask2Former Swin-Tiny 初始化目录 | 重建原始模型结构并核对初始化来源 | 使用本机缓存，或运行下方镜像脚本 |
| `epoch_15.pt` | 本轮冻结的 DA-Fusion 基座 | 项目组共享的已训练基座 |
| `heads_epoch_06.pt` | 本轮选出的普通/深度两个点头 | 项目组共享的已训练点头 |

模型清单位于 [configs/course_e15_point.json](../configs/course_e15_point.json)，包含基座、点头、协议和张量的 SHA-256。协议原始字节保存在 [artifacts/protocols](../artifacts/protocols)，其中历史路径是实验记录；运行位置由命令行参数提供。

课程归档为 20,554,250,971 字节，SHA-256：

```text
931708a0bbfbdc0908c3118098897fccf9a05e4f6d768758ba7328136898c94e
```

数据包含 train 14,432 张、val 2,584 张、test 2,112 张，原图和掩码为 1024×1024，前景类别为 `component`。深度必须使用原始 uint16 文件及逐图 `depth_scale_m`，并保留相机内参；伪彩色深度图只用于展示。

## 公开初始化权重

已有初始化目录时可直接传给 `--coco-checkpoint`。需要下载时：

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m pip install -r requirements-download.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python tools/fetch_course_mask2former.py
```

脚本固定 COCO 模型版本，通过 `hf-mirror.com` 入口下载并禁用代理，生成 `pretrained/mask2former_swin_tiny_coco_instance/`。它下载的是公开初始化权重；课程训练后的基座与点头由项目组单独共享。

`data/`、`pretrained/`、`outputs/`、`reports/` 和权重文件由 Git 忽略。仓库保留用于复现的源码、冻结协议、模型清单和经过核验的指标。

## 库源码核对

本次训练所用的Mask2Former实现文件已与[Transformers v4.53.2官方源码](https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/mask2former/modeling_mask2former.py)逐字节核对一致。加载器从当前安装目录找到该文件并校验摘要，保留模型原有的源码一致性检查。
