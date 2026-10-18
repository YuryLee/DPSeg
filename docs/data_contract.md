# 数据契约

`manifest.json` 示例（JSON 中不写注释）：

```json
{
  "provenance": "course_data",
  "depth_scale": 0.001,
  "classes": ["cup", "bottle"],
  "samples": [
    {
      "id": "scene01_frame0001",
      "split": "train",
      "rgb": "rgb/scene01_frame0001.png",
      "depth": "depth/scene01_frame0001.png",
      "instances": "instances/scene01_frame0001.png",
      "labels": {"1": 2, "2": 1}
    }
  ]
}
```

文件路径必须位于清单目录内，使用相对路径。`classes` 顺序定义 1 开始的类别 ID，0 保留为背景。`labels` 将实例 ID 映射到类别 ID；同一类别可以有多个不同实例 ID。

- RGB：Pillow 可读取的图片，转为 RGB，不按 BGR 读取。
- 深度：原始 16 位单通道 PNG，或单通道数值 NPY（禁止 pickle）。`raw × depth_scale` 为米。NaN/Inf/零/负值全部视为无效，不采用 GT 判断有效性。数据提供方若有其他无效编码，先在转换时置零并记录。
- 实例：单通道非负整数 ID PNG/NPY；0 背景。每个非零 ID 在 `labels` 中有类别。一个像素实例也保留，框坐标为左闭右开。
- 训练必须有实例标注；全零的已标注图是合法负样本。无 GT 的验证图可用于 demo，但评测会拒绝。演示上传 GT 仅用于可视化，不需要类别映射。
- RGB、深度、GT 必须事先空间对齐且尺寸一致；相同尺寸本身不能证明标定对齐，需由真实数据来源说明及人工检查确认。
- 划分名称限定 `train`、`val`、`test`。图片 ID 不可重复；同一路径不可跨样本重复。审计还报告跨集合相同 RGB/RGB-D 内容，但不能据此排除相邻帧泄漏。

合成样例为 48 张训练、12 张验证、8 张保留测试图。仅验证集被评测；保留测试图未用于训练、选模或展示。所有合成实例都在 small 面积组，因此 medium/large AP 是 N/A。合成场景不含真实纹理、严重遮挡、复杂类别或传感器标定误差，不能用于证明真实泛化性能。
