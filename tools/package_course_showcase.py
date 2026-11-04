#!/usr/bin/env python3
"""Package a checked offline showcase with original data and independent masks."""
from __future__ import annotations

import argparse
import csv
from io import StringIO
import json
from pathlib import Path
import sys
from zipfile import ZipFile, ZIP_DEFLATED

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image, ImageDraw
from tools.build_course_showcase import text
from tools.select_course_showcase import sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--demo-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    folder=args.demo_dir
    payload=json.loads((folder/'cases.json').read_text())
    check=json.loads((folder/'browser'/'check.json').read_text())
    assert check['status']=='passed' and check['cases']==len(payload['cases'])
    stages=list(payload['labels'])
    table=['| 案例 | 类型 | image_id | GT面积 | 默认对比 | 单实例IoU / 原误检区域残留 |',
           '|---|---|---:|---:|---|---|']
    index=StringIO()
    writer=csv.writer(index)
    writer.writerow(['case_id','type','image_id','annotation_id','gt_area','before','after',*stages,'note'])
    for c in payload['cases']:
        if c['kind']=='false_positive':
            metric=f"100.0% → {c['remaining_coverage'][c['after']]*100:.1f}%"
        else:
            metric=f"{c['ious'][c['before']]:.4f} → {c['ious'][c['after']]:.4f}"
        table.append(f"| {c['id']} | {c['title']} | {c['image_id']} | {c.get('gt_area','—')} | {c['before']} → {c['after']} | {metric} |")
        writer.writerow([c['id'],c['group'],c['image_id'],c.get('annotation_id',''),c.get('gt_area',''),
                         c['before'],c['after'],*[c.get('ious',{}).get(s,'') for s in stages],c['note']])
    readme='''# DPSeg 实例分割案例包

解压后双击 `index.html`。页面包含全部数据，离线运行，不需要Python、GPU或网络。
可切换五个消融阶段、整图/局部、关注实例/全部实例、RGB/深度，以及掩码、轮廓、像素误差。
可在页面导出当前四栏对比图及原始COCO压缩RLE JSON。

## 内容

- `index.html`：交互演示入口。
- `figures/`：RGB、GT和五个阶段的局部误差对比图，可直接插入PPT。
- `overview.png`：12例局部缩略图索引。
- `case_index.csv`：案例ID、原图ID、关注GT、阶段IoU与说明。
- `cases/case-XX_IMAGE_ID/`：每个案例的原始素材。
  - `rgb.png`、`depth_raw.png`：从课程ZIP逐字节复制的原图、uint16深度图。
  - `depth_preview.png`：逐图2%–98%分位归一化的深度伪彩图，仅用于显示。
  - `image_metadata.json`：原图元数据；深度米数=原始整数×`depth_scale_m`，无效值见元数据。
  - `gt_coco.json`：本图全部原始COCO标注；包括不可见的零面积标注。
  - `gt_masks/ann_ID.png`：每个GT实例独立的1024×1024二值掩码，0为背景、255为实例。
  - `gt_overlay.png`、`gt_instances_color.png`：所有可见GT的叠加图与纯色实例图。
  - `predictions/S0...S4/predictions.json`：原预测列表，保留类别、分数、压缩RLE与列表顺序。
  - `predictions/S*/masks/pred_NNN.png`：对应JSON列表0起始下标的独立二值掩码，保留全部原始候选。
  - `predictions/S*/overlay.png`、`instances_color.png`：置信度≥0.50的预测叠加图与纯色图。
  - `focus/`：固定观察掩码、各阶段最佳重合候选掩码，以及选择信息。误检例的观察掩码是S0误检区域。
- `browser/check.json`：素材一致性、浏览器RLE解码与交互检查结果。
- `provenance.json`、`manifest.json`：来源、固定预测摘要与包内文件SHA256。

## 方法

S0：课程微调Mask2Former；S1：+RGB-D融合；S2：+深度点细化；
S3：+切片推理；S4：+概率图深度细化。
所有图片均为已有固定实验输出，没有为展示重新推理或修改预测。

## 阅读说明

- 这是课程合成渲染数据的本地test归档。GT来自下载ZIP内已有的test标注，不是Kaggle隐藏评分标注。
- 案例在实验结束后按现象挑选，只用于定性展示。单例IoU及挑选出的成功比例不代表完整测试集AP或Kaggle得分。
- 表中IoU是展示置信度≥0.50时，与指定GT重合度最高的单个预测的全图Mask IoU，不是COCO一对一AP匹配。
- 小目标指GT可见掩码面积≤1024像素，可能只是较大物体被遮挡后露出的部分。
- 误差图：青色=GT与预测重合，红色=GT中有而预测缺失，橙色=预测多出的像素。
- 误检例只在固定S0误检区域内计算误差着色；残留比例统计所有可见预测对该区域的覆盖。
- 全实例彩色图按GT重合关系配色，颜色会循环；重叠预测在彩色图中按列表顺序覆盖。独立二值PNG和RLE保留完整实例、重叠与身份，请用它们做分析。
- 展示阈值只控制显示，不裁掉原始JSON或独立PNG。合并显示全部实例时，像素误差不反映相邻实例之间的身份混淆。

## 案例清单

'''+ '\n'.join(table)+'\n\n## 逐例观察\n\n'+ '\n\n'.join(f"### {c['id']} · {c['title']}\n\n{c['note']}" for c in payload['cases'])+'\n'
    (folder/'README.md').write_text(readme)
    (folder/'case_index.csv').write_text(index.getvalue(),encoding='utf-8-sig')
    # Four columns of aligned local RGB/GT/S0/S4, with case IDs for slide selection.
    width=1200;tile=300;row_h=360
    overview=Image.new('RGB',(width,row_h*len(payload['cases'])),'white')
    draw=ImageDraw.Draw(overview)
    for n,c in enumerate(payload['cases']):
        base=folder/'cases'/f"{c['id']}_{c['image_id']}"
        text(draw,(10,n*row_h+5),f"{c['id']} · {c['title']} · image {c['image_id']}",18)
        files=['rgb.png','gt_overlay.png','predictions/S0/overlay.png','predictions/S4/overlay.png']
        for j,(file,label) in enumerate(zip(files,['RGB','GT · 全实例','S0 · 全实例','S4 · 全实例'])):
            text(draw,(j*tile+10,n*row_h+32),label,15)
            image=Image.open(base/file).crop(c['bounds']).resize((tile,tile),Image.Resampling.NEAREST)
            overview.paste(image,(j*tile,n*row_h+60))
    overview.save(folder/'overview.png')
    # A contact sheet plus individual figures does not replace independent raw masks.
    excluded={'cases.json','manifest.json'}  # cases.json duplicates the embedded HTML payload.
    included=[p for p in sorted(folder.rglob('*')) if p.is_file() and p.name not in excluded]
    manifest={str(p.relative_to(folder)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in included}
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    included.append(folder/'manifest.json')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.output,'x',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for p in included:
            archive.write(p,arcname='DPSeg_case_demo/'+str(p.relative_to(folder)))
    with ZipFile(args.output) as archive:
        assert archive.testzip() is None
    result=dict(zip=str(args.output.resolve()),files=len(included),cases=len(payload['cases']),
                unpacked_bytes=sum(p.stat().st_size for p in included),
                zip_bytes=args.output.stat().st_size,sha256=sha(args.output))
    args.output.with_suffix('.manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
