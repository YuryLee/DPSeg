# DPSeg 与官方 SAHI 后处理对比

> 版本说明：本页保留旧分数优先掩码合并与官方 SAHI 后处理的实验快照。下文“当前默认”指实验完成时的状态；2026-11-06 用户已指定整图优先＋截断处理＋一致性为当前完整版，见[当前结果](course_ablation_swinl_current.md)。

固定 Swin-Large＋无内参深度、第 15 轮基座与第 6 轮点头，在完整本地测试集 **2,112 张图**上比较官方 SAHI 后处理与对照配置。各组均保留整图和四个 640×640 角点切片、来源视图点细化及深度引导概率图细化。分数范围 0–1，加粗标记原完整配置。

| 方法 | APₛ ↑ | ARₛ ↑ | AP ↑ |
|:---:|:---:|:---:|:---:|
| **原完整 DPSeg（当前默认）** | **0.525286** | **0.674087** | **0.880289** |
| 提前 SAHI NMS（框 IoU） | 0.488231 | 0.651149 | 0.861288 |
| 延后掩码 NMS（顺序对照） | 0.525101 | 0.672700 | 0.879204 |
| 延后掩码 NMS＋多边形转换（表示对照） | 0.524663 | 0.670710 | 0.878570 |
| 延后 SAHI NMS（框 IoS） | 0.253917 | 0.285276 | 0.736970 |
| 延后 SAHI GreedyNMM（框 IoS） | 0.157502 | 0.186821 | 0.557821 |

本表六组中，APₛ 最高为 **原完整 DPSeg（当前默认）**，ARₛ 最高为 **原完整 DPSeg（当前默认）**，整体 AP 最高为 **原完整 DPSeg（当前默认）**。

## 对照设计

| 配置 | 操作顺序与对照目的 |
|---|---|
| 原完整 DPSeg | 粗候选跨视图掩码 IoU NMS → 仅对保留候选做点细化和概率图细化 |
| 提前 SAHI NMS（框 IoU） | 只把提前选择改为官方框 IoU NMS；保留原 RLE 和来源特征，不做多边形转换 |
| 延后掩码 NMS | 先细化全部候选，再用原跨视图掩码 IoU NMS；检验顺序影响 |
| 延后掩码 NMS＋多边形转换 | 相同延后流程，加入官方 bool-mask→COCO polygon→mask 转换后再做原掩码 NMS；检验表示变化 |
| 延后 SAHI NMS（框 IoS） | 对同一批转换后的 ObjectPrediction 执行官方框 IoS NMS |
| 延后 SAHI GreedyNMM（框 IoS） | 对相同 ObjectPrediction 执行官方框 IoS GreedyNMM；使用官方多边形并集并保留组内最高分 |

- **提前官方 NMS：** 提前 SAHI NMS（框 IoU） 相对 原完整 DPSeg（当前默认），APₛ -0.037055、ARₛ -0.022938、AP -0.019002。
- **合并顺序：** 延后掩码 NMS（顺序对照） 相对 原完整 DPSeg（当前默认），APₛ -0.000184、ARₛ -0.001387、AP -0.001085。
- **多边形表示：** 延后掩码 NMS＋多边形转换（表示对照） 相对 延后掩码 NMS（顺序对照），APₛ -0.000439、ARₛ -0.001990、AP -0.000634。
- **官方框 IoS NMS 规则：** 延后 SAHI NMS（框 IoS） 相对 延后掩码 NMS＋多边形转换（表示对照），APₛ -0.270746、ARₛ -0.385435、AP -0.141600。
- **GreedyNMM 相对 NMS：** 延后 SAHI GreedyNMM（框 IoS） 相对 延后 SAHI NMS（框 IoS），APₛ -0.096415、ARₛ -0.098455、AP -0.179149。

提前 NMS 是候选选择的直接替换。延后组在各候选来源视图中完成点细化，再用原图深度做概率图细化，最后才合并。掩码 NMS 只抑制跨视图重复；官方后处理同时允许同类别、同视图候选互相抑制。框 IoS 为交集面积除以较小框面积，与框 IoU、掩码 IoU 均不同。因此“官方框 IoS NMS 规则”这一对比同时改变重叠定义、同视图抑制行为和框的几何约定，不能解释为单独 IoS 的贡献。

提前官方 NMS 的框由 COCO RLE 像素边界得到，延后官方组的框由 SAHI 多边形轮廓得到；两种坐标约定在重叠阈值附近可能产生不同选择。补充合成检查中，转换前后像素掩码均不变，但像素边界框 IoU 为 0.5，轮廓框 IoU 为 0.481481；同一官方 NMS、阈值 0.5 分别保留 1、2 个候选。因而提前/延后官方组不能仅归因于细化顺序或 IoU/IoS。两个延后官方组共享相同框和预测对象，可在这一共同表示下比较。

两个延后官方方法使用同一批转换后的预测对象，每次调用使用新的列表容器。合成检查已验证 GreedyNMM 不修改该批输入对象，避免影响 NMS 对照。合成例子也检验了框重叠而掩码不相交、同视图重复、IoU 与 IoS 的区别，以及矩形候选并集和最高分规则。

GreedyNMM 使用官方多边形并集，不能一般性等同于二值掩码逐像素 OR：并集及再次栅格化可能填充孔洞等改变拓扑。补充环形例子中，合并前表示往返均无变化，但官方并集面积为 580 像素，逐像素 OR 为 480 像素，多出 100 像素。因此 NMS 与 GreedyNMM 的差异包含“选择或合并”以及并集后的表示变化，不把它解释为纯分组算法的贡献。该例子用于解释几何行为，不代表测试集的总体新增像素量。

## 固定参数与范围

所有组置信度阈值为 0.01、合并匹配阈值为 0.5、每图最多 100 个实例。模型输入统一为 1024×1024，四个 640×640 切片相邻重叠 256 像素，内部裁剪边缘过滤为 2 像素。RGB、米制深度和有效性同步裁剪，无相机内参；FP32 推理，关闭 TF32，seed=17。

所有合并方法接收的候选都经过 DPSeg 的切片边缘过滤。对照结论限定在这批候选和固定阈值下；完整 SAHI 流程、关闭边缘过滤或另行验证选参后的效果需单独评估。

SAHI 0.14.0 的 predict.py 在模型置信度阈值低于 0.1 时会自动使用 NMS/IOU，除非 force_postprocess_type=True。本轮阈值为 0.01，因此 IOS 与 GreedyNMM 组是显式调用所选官方类的对照，不能称为该阈值下的自动默认行为。

本轮调用 **SAHI 0.14.0 的官方 NMSPostprocess、GreedyNMMPostprocess 类**，RGB-D 输入和切片前向仍由 DPSeg 负责，未调用完整 get_sliced_prediction 接口。这是在已查看过结果的本地测试集上追加的固定权重比较；没有根据本表调阈值、选择检查点或替换默认配置。单种子结果不代表统计显著性，Kaggle 成绩另行报告。共享特征的实验运行时间不用于速度比较。

## 多边形转换影响

五视图合计产生 **442,955** 个候选记录；同一物体在不同视图中可能重复出现。对全部细化候选进行官方多边形转换后，**2,911** 个候选因无法形成多边形而丢弃，**111,846** 个仍保留候选的掩码发生变化。转换累计增加 **0** 个掩码像素、移除 **429,160** 个掩码像素；这些是合并前的表示转换统计，不包含 GreedyNMM 多边形并集后的拓扑或栅格化变化。移除量包含丢弃候选的像素，各候选分别计数，不是唯一物体数或图像覆盖面积。

合成单像素掩码无法形成官方多边形，已明确丢弃并记录，没有用边界框代替其掩码。合成环形掩码在该版本中的像素往返没有变化。不据此概括所有带孔或细小连通域都能无损转换。三组包含转换的延后方法共享这批转换结果；表示对照用于区分转换损失和后处理规则的影响。

## 官方实现依据

版本与发布文件见 [SAHI 0.14.0](https://pypi.org/project/sahi/0.14.0/)。后处理类见[官方 combine.py](https://github.com/obss/sahi/blob/main/sahi/postprocess/combine.py)，掩码与分数的合并见[官方 utils.py](https://github.com/obss/sahi/blob/main/sahi/postprocess/utils.py)。GitHub 主分支可能继续更新；本次实验实际运行上述固定 wheel，具体源码哈希已保存在配套 JSON 中。

## 核验

- 冻结实验协议、旧模型源码、新集成源码、基座/点头文件，以及 SAHI、Shapely、OpenCV 文件哈希和版本。
- 3 张指定验证图上，独立原实现与共享特征入口的 legacy full/native 输出逐掩码、逐分数一致；合成行为检查全部通过。
- 独立 CPU 几何审计另记录了带孔多边形并集与框坐标约定的差异，源码哈希及完整 receipt 均纳入配套 JSON；没有更改已冻结推理。
- 全部 2,112 张测试图在运行时逐图严格核对原 S1 整图预测和原 S4 完整预测。发布程序检查对应日志的覆盖、哈希和成功标记。
- 五组分片实际 ID、顺序、实例数及哈希均与记录一致；每条合并评分预测与来源分片再次逐图绑定。
- 六份结果（原完整配置加五组新结果）的预测 CSV 均通过全量原生 RLE 解码。发布时核对原解码记录与文件哈希，并重新逐行核对 CSV 的 ID、类别、置信度和掩码与评分输入一致。
- 六组均使用相同 COCO 掩码设置和 GT 统计。APₛ、ARₛ 对原始标注面积 [0, 32²] 像素计算，AP 对所有尺寸计算；IoU=0.50:0.05:0.95，每图最多 100 个实例。
- 原完整配置另外使用当前课程归档重新评分，全部 12 项 COCO 指标与原记录完全一致，核验记录纳入配套 JSON。

## 依赖与复现

SAHI 固定为 **0.14.0**，Shapely 固定为 **2.1.2**；本次运行 OpenCV 为 5.0.0、NumPy 为 2.3.1，后处理使用 NumPy backend。SAHI wheel SHA-256 为 `7fb73078e82fb34fb170276ae6a017e9250c4b64842926c332a8b683aa7a8d81`，与官方发布文件哈希一致；全部运行文件的哈希见配套 JSON。

依赖来自清华 PyPI 镜像 `https://pypi.tuna.tsinghua.edu.cn/simple`；另以显式清除代理变量的独立下载核对 SAHI 和 Shapely wheel，其全部 payload 与冻结安装目录逐文件一致，receipt 见配套 JSON。依赖隔离存放在 `reports/course_sahi_merge_deps/site`。下列复现命令用于新建环境，不覆盖运行中实验的冻结 site；`--no-deps` 避免更换现有 NumPy、PyTorch 等依赖，其余包需与项目环境预先兼容。

```bash
mkdir -p reports/course_sahi_merge_deps/wheels
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  .venv/bin/python -m pip download --no-deps \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  --dest reports/course_sahi_merge_deps/wheels sahi==0.14.0 shapely==2.1.2
env -u http_proxy -u https_proxy -u all_proxy -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  .venv/bin/python -m pip install --no-deps \
  --target reports/course_sahi_merge_deps/site \
  reports/course_sahi_merge_deps/wheels/sahi-0.14.0-py3-none-any.whl \
  reports/course_sahi_merge_deps/wheels/shapely-2.1.2-*.whl
.venv/bin/python -u tools/evaluate_course_sahi_merge.py \
  --output reports/course_sahi_merge_new --gpus 0,2,3,4,5,6,7 \
  --reference-run reports/course_swinl_removal_v1
.venv/bin/python tools/check_course_sahi_reference.py --run reports/course_sahi_merge_new
.venv/bin/python tools/summarize_course_sahi_merge.py \
  --run reports/course_sahi_merge_new \
  --semantics-audit reports/course_sahi_merge_v1/supplementary_semantics.json
```

按实际空闲显卡调整 --gpus，输出实验目录须尚不存在。参考实验目录需包含已完成的完整预测、数据与权重路径；入口先冻结依赖 receipt，再通过工程检查后执行全量推理和评分。--semantics-audit 指向本轮独立 CPU 几何审计，发布时要求其源码与当前依赖一致；可运行 tools/check_course_sahi_semantics.py 重现该合成检查。五组共享视图特征只用于节省实验时间，不据总运行时间计算各方法 FPS。

## CSV 本机路径

- **原完整 DPSeg（当前默认）：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_full.csv`
- **提前 SAHI NMS（框 IoU）：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_sahi_merge_v1/submissions/dpseg_swinl_depth_no_k_sahi_nms_iou_early.csv`
- **延后掩码 NMS（顺序对照）：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_sahi_merge_v1/submissions/dpseg_swinl_depth_no_k_mask_nms_late.csv`
- **延后掩码 NMS＋多边形转换（表示对照）：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_sahi_merge_v1/submissions/dpseg_swinl_depth_no_k_sahi_polygon_mask_nms_late.csv`
- **延后 SAHI NMS（框 IoS）：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_sahi_merge_v1/submissions/dpseg_swinl_depth_no_k_sahi_nms_ios_late.csv`
- **延后 SAHI GreedyNMM（框 IoS）：** `/data/yinli/workspace/rgbd_instance_seg/reports/course_sahi_merge_v1/submissions/dpseg_swinl_depth_no_k_sahi_greedynmm_ios_late.csv`

原始预测 CSV、权重和运行日志保留在本机；可分发的指标与核验索引见 [CSV](course_sahi_merge.csv) 和 [JSON](course_sahi_merge.json)。

## Material Passport

- 来源：ARS experiment-agent run；`reports/course_sahi_merge_v1` 的真实推理和 COCO 评分产物。
- 核验状态：VERIFIED；协议 SHA-256：`7e4f13e2bf5dfd9a74926d2583d32538d5df3c0377d3ccab07d31c2295afe5c3`。
- 范围：固定权重的官方 SAHI 后处理、顺序与表示对照；完整本地测试集，不作为验证选参或 Kaggle 分数。
- 限制：单次训练；各路径的合并顺序、表示及规则按表分别变化；无统计显著性或速度结论。RGB-D 融合去除实验仍待后续训练。
