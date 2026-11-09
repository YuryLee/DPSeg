# Swin-Large 当前完整流程的点头类型消融

统一采用Swin-Large＋无内参深度、第15轮冻结基座、1024×1024输入、整图＋四个640×640切片（40%重叠）、整图优先＋截断处理＋跨视图一致性合并，以及深度感知概率图细化。

以下为完整本地测试集2,112张图的COCO掩码评测，分数范围0–1，与Kaggle分数分别报告。

| 点头配置 | APₛ ↑ | ARₛ ↑ | AP ↑ |
|---|---:|---:|---:|
| 无点头 | 0.519366 | 0.663216 | 0.873441 |
| 普通点头 | 0.520640 | 0.664380 | 0.876872 |
| 深度点头 | 0.528710 | 0.674606 | 0.887453 |

普通与深度点头均来自同一成对训练的第6轮检查点。两者共享冻结的448维RGB-D特征、粗掩码logit、MLP隐藏宽度256及三层结构，深度点头额外接收9维显式局部深度线索。普通点头也使用RGB-D融合特征，不是纯RGB模型或原版PointRend的直接复现。

三组使用相同候选身份、类别、原始置信度、排序和最多100实例限制。各点头使用相同的选点算法与设置；后续轮次的具体不确定点位置可随各自掩码变化。

本次只新增普通点头的完整测试推理与评测；无点头及深度点头逐图重新推理并与既有预测逐掩码、逐分数、逐次序精确核对，表中两行沿用已核验的原指标。普通点头统一固定第6轮，未在测试集选轮，也不宣称其为普通点头的验证最优轮次。

该测试集此前已被查看，本表为固定协议、单次训练种子的描述性消融；没有新增训练、参数搜索、统计显著性或Kaggle结果。共享前向用于节省实验计算，运行总耗时不能作为各方法独立FPS。

## 文件与核验

- 本轮协议与结果：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_plain_point_v1`。
- 普通点头CSV全量压缩RLE解码，且逐行绑定COCO评分所用JSONL；原图尺寸1024×1024、category_id=1、每图最多100实例。
- 无点头 CSV：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_current_removal_v1/submissions/dpseg_swinl_depth_no_k_guarded640_without_point.csv`。
- 普通点头 CSV：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_plain_point_v1/submissions/dpseg_swinl_depth_no_k_guarded640_plain_point_e06.csv`。
- 深度点头 CSV：`/data/yinli/workspace/rgbd_instance_seg/reports/course_swinl_guarded_merge640_v1/submissions/dpseg_swinl_depth_no_k_guarded640_global_guard_edge_consensus.csv`。

来源：ARS experiment-agent/run；协议SHA-256：`4f8306bc6aa498a251557fb31a7349874f3c939a183c0d5f22048c64192ae5b6`；状态：COMPLETED。
