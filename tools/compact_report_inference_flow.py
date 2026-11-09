#!/usr/bin/env python3
"""Reflow the editable inference diagram into two readable rows."""
from __future__ import annotations
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import export_report_figures_pptx as ppt
from pptx import Presentation
import matplotlib.pyplot as plt
from matplotlib import patches
import numpy as np

src = ppt.source
PREV = ROOT/'outputs/report_figures_pptx_v5'
OUT = ROOT/'outputs/report_figures_pptx_v6'
OUT.mkdir(exist_ok=True)


def diagram(cases):
    c = cases['teaser_boundary']; im = c['image']
    trace = src.OUT/'framework_trace'
    actual = {s: json.loads((trace/f'{s}.json').read_text()) for s in ('merged','point','final')}
    anns = json.loads((src.OUT/'source_assets'/str(c['image_id'])/'ground_truth.json').read_text())['annotations']
    merged = src.overlay_all(im['rgb'], anns, actual['merged'])
    focus = {s: src.cut(src.overlay(im['rgb'], [src.decode(items[72]['segmentation'])], [src.PRED], .4), c['bounds'])
             for s, items in actual.items()}
    prob = src.cut(np.load(trace/'point_probability_72.npy'), c['bounds'])
    probability = np.uint8(plt.get_cmap('viridis')(prob)[..., :3]*255)
    W, H = 1500, 900
    fig = plt.figure(figsize=(15, 9))
    ax = fig.add_axes([0,0,1,1]); ax.set(xlim=(0,W), ylim=(H,0)); ax.axis('off')

    def text(x,y,t,size=14,bold=False,color=src.INK):
        ax.text(x,y,t,ha='center',va='center',fontsize=size,color=color,
                fontweight='bold' if bold else 'normal',zorder=5)
    def box(x,y,w,h,t='',fc='white',size=14,dashed=False):
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,
            boxstyle='round,pad=0,rounding_size=5',fc=fc,ec='#666666',lw=1.0,
            linestyle=(0,(4,3)) if dashed else '-',zorder=1))
        if t: text(x+w/2,y+h/2,t,size)
    def arrow(points,dashed=False,color=src.INK,lw=1.6):
        for p,q in zip(points[:-2],points[1:-1]):
            ax.plot([p[0],q[0]],[p[1],q[1]],color=color,lw=lw,
                    ls='--' if dashed else '-',zorder=2)
        ax.add_patch(patches.FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',
            mutation_scale=13,lw=lw,color=color,linestyle='--' if dashed else '-',zorder=2))
    def photo(x,y,side,img,label=None,fs=12.5):
        a = src.image_ax(fig,[x/W,1-(y+side)/H,side/W,side/H],img)
        if label: text(x+side/2,y-17,label,fs)
        return a

    # Row 1 proceeds left-to-right. Cards are large enough to contain examples.
    for x,title in [(20,'1. RGB-D 输入'),(390,'2. 切片增强推理'),
                    (760,'3. RGB-D 分割'),(1130,'4. 候选合并')]:
        box(x,45,330,330,dashed=True)
        text(x+165,76,title,18,True)
    photo(40,128,135,im['rgb'],'RGB',14)
    photo(195,128,135,im['depth'],'深度 Z',14)
    text(185,300,'颜色、深度与有效性',15)
    text(185,342,'无需相机内参',14)

    whole = photo(410,130,140,im['rgb'],'整图',14)
    for j,(x,y) in enumerate([(0,0),(384,0),(0,384),(384,384)]):
        whole.add_patch(patches.Rectangle((x,y),640,640,fill=False,ec=src.COLORS[j],lw=.9))
        photo(573+(j%2)*68,130+(j//2)*72,62,im['rgb'][y:y+640,x:x+640])
    text(638,113,'4 个重叠切片',13)
    text(555,295,'RGB / 深度同步裁剪',14)
    text(555,324,'每个视图缩放至 1024²',14)
    text(555,351,'逐视图深度归一化与适配',13)

    box(782,125,135,64,'RGB\nSwin-Large',src.PEACH,13.5)
    box(932,125,135,64,'深度\nSwin-Large',src.PEACH,13.5)
    arrow([(849,190),(849,208),(875,208),(875,228)],lw=1.1)
    arrow([(1000,190),(1000,208),(975,208),(975,228)],lw=1.1)
    box(800,228,250,64,'多尺度 RGB-D 融合\n实例掩码解码',src.PEACH,14)
    box(800,322,250,36,'缓存来源视图细粒度特征',src.BLUE,12)
    arrow([(925,293),(925,321)],True,color='#587282',lw=1.1)

    box(1152,120,286,87,'坐标恢复 · 边缘过滤\n跨视图掩码去重',src.PURPLE,14)
    photo(1156,233,121,merged)
    box(1295,237,142,39,'来源视图',src.BLUE,13)
    box(1295,287,142,39,'候选掩码',src.PURPLE,13)
    text(1365,350,'保留对应关系',12)
    for x in (350,720,1090):
        arrow([(x+1,212),(x+39,212)])

    # Row 2 folds back right-to-left: point -> probability -> output.
    for x,w,title in [(940,520,'5. 深度感知点细化'),(480,400,'6. 深度引导概率图细化'),
                      (20,400,'7. 实例掩码输出')]:
        box(x,500,w,350,dashed=True)
        text(x+w/2,529,title,17.5,True)
    arrow([(1295,376),(1295,452),(1330,452),(1330,499)])
    arrow([(925,359),(925,429),(1030,429),(1030,562)],True,color='#587282',lw=1.3)
    text(800,407,'来源视图细粒度特征',13,color='#587282')

    box(960,563,480,55,'不确定点采样 · 细粒度特征 · 局部深度\n点 MLP 预测掩码 logit 残差',src.GREEN,13.5)
    photo(1240,650,180,focus['merged'],'粗掩码',14)
    photo(980,650,180,focus['point'],'点细化后',14)
    arrow([(1236,740),(1164,740)],lw=1.4)
    text(1200,716,'点细化',12)

    box(502,563,356,55,'原图坐标 · 5 × 5 深度加权\n更新实例掩码边界',src.YELLOW,13.5)
    photo(752,660,105,src.cut(im['depth'],c['bounds']),'局部深度',11.5)
    photo(626,660,105,probability,'点细化概率',11.5)
    photo(500,660,105,focus['final'],'细化后掩码',11.5)
    ax.plot([804,804,679],[767,795,795],color=src.INK,lw=1.0,zorder=2)
    arrow([(679,767),(679,795),(552,795),(552,767)],lw=1.1)
    text(680,825,'深度约束概率更新',12)

    photo(39,580,245,im['visuals']['M4'],'整图预测',13)
    text(347,553,'独立掩码',12.5)
    for y,key in zip([568,660,752],['teaser_miss','teaser_boundary','good']):
        photo(307,y,80,src.focus(cases[key],'M4','binary'))
    arrow([(939,681),(881,681)])
    arrow([(479,681),(421,681)])
    text(750,882,'上行从左向右，下行从右向左；虚线表示来源视图特征的传递。',12.5,color=src.GRAY)
    return fig


def main():
    source = next(PREV.glob('*.pptx'))
    prs = Presentation(source)
    slide = prs.slides[7]
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    fig = diagram(src.cached())
    stats = ppt.Converter(slide,fig,title='总体框架：完整推理流程').draw()
    plt.close(fig)
    slide.notes_slide.notes_text_frame.text = (
        '推理流程按 1—7 编号，上行从左向右、下行从右向左。\n'
        'RGB-D 同步切片 → 共享分割基座 → 坐标恢复与候选过滤去重 → 来源视图点细化 → 原图概率图深度细化 → 掩码输出。\n'
        '基座为双分支 Swin-Large，深度 Z 与有效性输入，无需相机内参。\n'
        '虚线表示缓存的来源视图细粒度特征，直接送到点细化模块。\n'
        '所有例图均沿用真实数据、冻结预测和原框架流水线。粗掩码、点细化结果、概率图细化结果跟踪同一候选。\n'
        '原图输出与独立掩码来自既有 M4 预测；示意图不使用 GT 参与推理。')
    prs.core_properties.title = 'DPSeg 报告插图：双行推理流程'
    path = OUT/'DPSeg_报告插图_双行推理流程.pptx'
    prs.save(path); ppt.set_theme_fonts(path)
    manifest = json.loads((PREV/'slides_manifest.json').read_text())
    manifest.update(file=str(path),previous_version=str(source),
                    changed_slides=[8],inference_layout='two rows, left-to-right then right-to-left',
                    previous_flow_aspect_ratio=1790/414,flow_aspect_ratio=1500/900,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),bytes=path.stat().st_size)
    manifest['slides'][7].update(stats)
    sizes = [r.font.size.pt for s in prs.slides for sh in s.shapes if sh.has_text_frame
             for p in sh.text_frame.paragraphs for r in p.runs if r.font.size]
    manifest.update(minimum_font_pt=min(sizes),maximum_font_pt=max(sizes))
    (OUT/'slides_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    shutil.copy2(PREV/'provenance.json',OUT/'provenance.json')
    trace = src.OUT/'framework_trace'
    evidence = {n:dict(path=str(trace/n),sha256=hashlib.sha256((trace/n).read_bytes()).hexdigest())
                for n in ['merged.json','point.json','final.json','point_probability_72.npy']}
    (OUT/'flow_sources.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    repro = OUT/'reproduction'; repro.mkdir(exist_ok=True)
    for f in ['compact_report_inference_flow.py','export_report_figures_pptx.py','build_report_figures_v2.py']:
        shutil.copy2(ROOT/'tools'/f,repro/f)
    (OUT/'使用说明.md').write_text('''# DPSeg 报告插图：双行推理流程

- 第 8 页“总体框架：完整推理流程”改为两行折返布局，图形宽高比由约 4.3:1 改为 1.67:1，增加上下空间并放大模块与例图。
- 上行从左向右：RGB-D 输入、切片增强推理、RGB-D 分割、候选合并。
- 下行从右向左：深度感知点细化、深度引导概率图细化、实例掩码输出。编号与箭头标明执行顺序。
- 保留来源视图特征到点细化模块的虚线连接；点细化后再进行原图坐标下的概率图深度细化。
- 例图采用既有真实 RGB、深度、候选、点细化结果及概率图。预测与评价结果保持原样，来源见 provenance.json 与 flow_sources.json。
- 其余九页保留此前的原版配色、紧凑 teaser、整图与局部对比、居中标签。中文宋体，字母与数字 Times New Roman，文字和图形可编辑。
- preview.pdf / previews 使用服务器可用的宋体风格和衬线替代字体；PPT 指定原字体，未嵌入字体。
- 上一版保留在 report_figures_pptx_v5。
''')
    print(json.dumps(dict(file=str(path),slides=len(prs.slides),changed_slide=8,stats=stats,
                          MiB=round(path.stat().st_size/1024**2,2)),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
