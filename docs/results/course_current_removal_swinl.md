# 当前 DPSeg 的独立掩码细化消融

| 配置 | APₛ | ARₛ | AP | AP50 | AP75 | APₘ | APₗ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前完整 DPSeg（guarded640） | 0.528710 | 0.674606 | 0.887453 | 0.953313 | 0.911213 | 0.924578 | 0.982764 |
| w/o 深度感知点细化 | 0.519366 | 0.663216 | 0.873441 | 0.953163 | 0.909445 | 0.910148 | 0.980677 |
| w/o 深度引导概率图细化 | 0.462478 | 0.586535 | 0.865874 | 0.952851 | 0.900361 | 0.908704 | 0.977642 |
| w/o 切片增强推理（已有结果） | 0.476384 | 0.569641 | 0.886068 | 0.957347 | 0.914049 | 0.934844 | 0.982887 |

本表从当前完整 DPSeg 分别去除一个模块。完整方法使用 Swin-Large、无内参深度输入、640×640 四切片与整图，以及 `global_guard_edge_consensus` 候选合并。基座为第15轮、点头为第6轮固定权重，两项去除均未重新训练。

去除点细化时，选中的粗掩码直接进入深度引导概率图细化；去除概率图细化时，直接输出点细化后的掩码。两项去除与当前完整方法使用完全相同的候选身份、类别、置信度、排序和每图最多100实例限制，仅掩码细化路径不同。共享前向与参考核验用于节省实验计算，本运行不能作为独立方法的速度测量。

完整方法及无切片行复用已有完整测试集结果。所有2112张均逐图重新精确复现当前完整方法、无切片S3与native S1，掩码、分数和次序一致；三张验证图的独立实现工程核验也全部通过。新推理与评测进程实际退出均为0。

## 评测与范围

完整本地 test=2112；使用统一 COCO 分割 AP，IoU=0.50:0.05:0.95，maxDets=[1,10,100]，小目标面积范围为[0,1024]。GT仅用于评分，预测只接收RGB、米制深度和有效性图。每份CSV均完成2112行覆盖、类别、尺寸、压缩RLE完整解码及与评分预测逐项绑定校验。

这是已多次查看过的本地测试集上的固定协议独立消融，用于分析既定配置下的模块影响；不表示在独立验证集上选出的最优方案，也不代表新增Kaggle评分。当前默认方法未改变。

## 提交文件

- 当前完整 DPSeg（guarded640）：CSV（本机路径：`reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard_edge_consensus.csv`）；140,993 个预测实例。
- w/o 深度感知点细化：CSV（本机路径：`reports/course_swinl_current_removal_v1/submissions/dpseg_swinl_depth_no_k_guarded640_without_point.csv`）；140,993 个预测实例。
- w/o 深度引导概率图细化：CSV（本机路径：`reports/course_swinl_current_removal_v1/submissions/dpseg_swinl_depth_no_k_guarded640_without_probability.csv`）；140,993 个预测实例。
- w/o 切片增强推理（已有结果）：CSV（本机路径：`reports/course_swinl_removal_v1/submissions/dpseg_swinl_depth_no_k_removal_without_tiles.csv`）；116,324 个预测实例。

## 可复核来源

- 本轮协议（本机路径：`reports/course_swinl_current_removal_v1/protocol.json`），SHA256：`76e6aecb1f13ee0fe28dfddf80ae6477ff439b09d6201aeb7ece3a01a91df84a`。
- 工程核验（本机路径：`reports/course_swinl_current_removal_v1/engineering_gate.json`）。
- 当前完整版来源协议（本机路径：`reports/course_swinl_guarded_merge640_v1/protocol.json`）。
- 本轮输出目录：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_current_removal_v1`。
