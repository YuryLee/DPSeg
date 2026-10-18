# 可用于真实 RGB-D 开发的数据集候选

检查日期：2026-10-02。用户说明课程数据尚不清楚，并要求下载时使用镜像、不要开启代理。本次只核查公开资料，**没有下载公开数据集**。网页中的镜像入口存在不等于已经实测下载可用。

## 建议

优先考虑 **GraspNet-1Billion 的场景子集**，因为作者列出了百度与交大 Jbox 镜像；如果必须输出明确的日常语义类别，HouseCat6D 的类别标注更直接，但国内镜像尚未核实。公开数据适合预先开发，是否可用于最终课程成绩仍取决于老师要求。

|候选|已核实内容|适用性及限制|镜像/规模|
|---|---|---|---|
|GraspNet-1Billion|190场景，97,280幅RGB-D，88个物体；RGB、深度、物体掩码|遮罩1–88为物体身份标签，不能称为88种日常语义类别；可做物体身份分类或建立有依据的语义映射|官方百度/Jbox；训练分包20、20、20、6.3 GB|
|OCID|2,346样本、96条堆叠序列、89物体；RGB、16位毫米深度、实例掩码|较小，适合检查遮挡和实例分割；完整场景的语义映射尚未核实，语义裁切为另一个包|官方DOI下载；本次未确认国内镜像及包大小|
|HouseCat6D|194物体、10语义类、41场景；RGB、深度、实例掩码与class ID|多类实例分割契约最直接；有透明/反光物体，可检查深度失效场景|官方train约30+60+120 GB，val约15 GB、test约30 GB；国内镜像未核实|
|UOAIS-Sim|50,000合成RGB-D，1,000场景、375模型，可见/完整遮罩|用于类别无关物体分割预训练；不是实拍数据，多语义分类需另核验标签，课程可见分割应使用visible masks|作者提供Google Drive；国内镜像未核实|

以上规模来自作者资料，不代表本机已经取得这些文件。GraspNet最小公开训练分包6.3 GB的内容、独立解压能力尚未检查。当前项目所在磁盘仅约30 GB剩余空间，整套数据与解压临时文件不适合直接放到此盘。

## 划分与小目标

- GraspNet官方train为scene 0–99，test seen 100–129、similar 130–159、novel 160–189。预先开发可只选一个相机，在train场景内留出独立验证场景；不把同场景相邻帧随机混入不同集合。
- HouseCat6D官方34 train / 2 val / 5 test场景。OCID应按整条序列分组；本次未找到统一官方训练划分，不编造一个。
- 桌面杂乱不代表COCO意义下的小目标充足。接入后应先按原始分辨率的可见实例面积统计small（32²附近边界遵循COCO实现）数量；不能靠修改评测图尺寸或面积来制造APₛ提升。
- 对真实传感器数据先确认RGB/深度标定、对齐、单位与无效编码，再转换清单；不将着色深度当原始深度。

## 来源

1. [GraspNet官方数据格式及镜像下载](https://graspnet.net/datasets.html)：RGB/depth/label组织、1–88物体标签、各分包大小。
2. [GraspNet官方首页](https://graspnet.net/)：190场景、97,280影像、88物体。
3. [GraspNet API官方划分实现](https://graspnetapi.readthedocs.io/en/latest/_modules/graspnetAPI/graspnet.html)：训练和三组测试场景。
4. [OCID作者数据页](https://www.acin.tuwien.ac.at/en/vision-for-robotics/software-tools/object-clutter-indoor-dataset/)：采集规模、RGB-D格式、语义裁切与下载入口。
5. [HouseCat6D作者数据页](https://sites.google.com/view/housecat6d/dataset)及[论文](https://arxiv.org/html/2212.10428v3)：类别、实例标签、分包与划分。
6. [UOAIS作者仓库](https://github.com/gist-ailab/uoais#dataset-preparation)及[论文](https://arxiv.org/html/2109.11103)：合成数据、可见/完整掩码与下载方式。

补充：搜索发现Hugging Face上的第三方GraspNet重包，但其scenes.tar约142 GB，未经作者认证为完整镜像，不作为当前下载方案。也未把普通RGB-D数据误当成具备实例监督的数据。

## 下载验证状态

百度/Jbox入口已由官方页面核实，实际下载连通性、速度和文件完整性未验证。浏览工具打开镜像时返回错误；本地一次明确 `ProxyHandler({})` 的直连页面请求因受限环境DNS失败。未因访问失败启用代理，也未改为非镜像下载。后续只对选定的镜像执行显式关闭代理的下载，保存来源、文件哈希与使用说明。
