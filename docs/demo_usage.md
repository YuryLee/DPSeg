# 课程 Demo 使用说明

## 离线演示

当前入口展示第15轮 DA-Fusion 基座与第6轮点头的六组真实预测。生成方法见[README](../README.md)，输出为 `outputs/course_demo/index.html`。将 HTML 复制到电脑后直接用浏览器打开即可；页面不需要网络、Python或GPU。

工作区已经生成的八样例版本位于 `outputs/course_e15_point_e06_demo/index.html`，可分发的压缩包为 `outputs/rgbd_demo_e15_point_e06.zip`。这些运行产物保留本机，源码仓库中的入口可以重新生成。

## 页面操作

1. 选择样例，对照原始RGB、深度视图与掩码叠加。
2. 切换六种方法，观察细小边界和分离情况。颜色在相同候选之间保持对应。
3. 调整展示置信度与透明度，放大后可同步拖动三个视图。
4. 下载当前显示实例的COCO压缩RLE JSON，或下载1024×1024叠加PNG用于汇报。

页面的AP/APₛ来自完整2,584张验证图；本图显示数量和阈值不改变这些指标。页面操作使用已保存预测，不重新运行模型。深度颜色是逐图归一化的展示，不能代替原始uint16深度输入。

## 浏览器检查

```bash
env -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY -u http_proxy -u https_proxy -u all_proxy \
  python -m pip install -r requirements-dev.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python tools/check_course_demo.py \
  --demo outputs/course_demo/index.html \
  --output outputs/course_demo_browser_check
```

检查使用本机Chrome或Chromium，离线核对RLE、单像素坐标、方法切换、阈值、空结果、下载和手机布局。检查结果与截图写入指定的新目录。

## 远程服务器

可先下载生成的单个HTML或压缩包。需要在服务器上浏览时，可启动仅监听本机的静态服务：

```bash
python -m http.server 8000 --bind 127.0.0.1 --directory outputs/course_demo
```

在个人电脑建立SSH转发后访问本机8000端口：

```bash
ssh -N -L 8000:127.0.0.1:8000 用户名@服务器地址
```

## 历史演示

`start_demo.sh` 和 `tools/manage_demo.py` 是旧 GraspNet 实时服务入口，需要对应旧数据、权重和运行记录。课程原始基线、切片及点头的历史图库保留在本机 `reports/` 中；当前演示使用上面的课程入口。
