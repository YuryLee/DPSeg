# 课程 Demo 使用说明

本页现有图库和模型演示使用原 Tiny＋XYZ 的固定预测及权重。当前最佳方法已更新为 Large＋无内参 RGB-D 融合、深度点细化、切片和概率图细化，结果见[最新实验记录](results/swinl_depth_no_k.md)；现有案例不能作为新 Large 模型的可视化预测。

## 推荐：带GT的案例展台

打开 `outputs/course_showcase_test_v1/index.html`，或下载 `outputs/DPSeg_case_demo_v1.zip`，解压后双击 `index.html`。网页自包含，离线即可使用。

展示12个精选案例，包含分割良好的器件、小目标漏检、背景误检、边界偏差，以及完整方案仍未解决的漏检。详情见[案例清单](results/showcase_cases.md)。

1. 左侧按成功、漏检、误检、边界、局限筛选；左右箭头切换案例。
2. 对照RGB、GT和两个可选消融阶段；可以将任意S0–S4设为比较对象。
3. 切换局部/整图、关注实例/全部实例、纯掩码/轮廓/漏分多分像素；RGB栏可切换深度。
4. 导出当前对比PNG或本例所有阶段的原始掩码JSON。`figures/`已提供12张七栏局部图，可直接用于PPT。
5. 包内`cases/`按案例和原图ID分目录，保存原始RGB、uint16深度、原始COCO标注、所有预测和逐实例1024×1024二值PNG。文件对应关系见包内README。

默认展示置信度为0.50；原始JSON和独立PNG保留全部已保存候选。页面IoU是指定实例的最佳掩码重合度，不是AP。误检例另用“原误检区域的预测残留比例”描述变化。案例来自本地课程test归档，GT不是Kaggle隐藏评分标注。

生成与检查入口如下，依赖本工作区已有固定预测和课程数据，浏览打包结果时不需要这些依赖：

```bash
python tools/select_course_showcase.py --split test \
  --output reports/course_showcase/new_selection
python tools/build_course_showcase.py \
  --selection reports/course_showcase/new_selection/candidates.json \
  --choices configs/showcase_cases.json --output outputs/new_showcase
python tools/check_course_showcase.py \
  --demo outputs/new_showcase/index.html --output outputs/new_showcase/browser
python tools/package_course_showcase.py \
  --demo-dir outputs/new_showcase --output outputs/new_showcase.zip
```

图表依赖已列于`requirements.txt`；浏览器检查另依赖`requirements-dev.txt`。脚本检查固定预测的SHA256，检查时还逐张核对原始图、GT、预测及二值PNG，并核对浏览器解码后的掩码像素。

## 离线演示

模型推理入口展示第15轮 DA-Fusion 基座与第6轮点头的六组真实预测。生成方法见[README](../README.md)，输出为 `outputs/course_demo/index.html`。将 HTML 复制到电脑后直接用浏览器打开即可；页面不需要网络、Python或GPU。

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
