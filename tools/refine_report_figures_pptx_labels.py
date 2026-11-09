#!/usr/bin/env python3
"""Tighten the example-only teaser and center editable image labels."""
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
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.dml import MSO_COLOR_TYPE, MSO_FILL_TYPE
from pptx.util import Inches
import matplotlib.pyplot as plt

src = ppt.source
PREV = ROOT/'outputs/report_figures_pptx_v4'
OUT = ROOT/'outputs/report_figures_pptx_v5'
OUT.mkdir(exist_ok=True)


def teaser(cases):
    a, b = cases['teaser_miss'], cases['teaser_boundary']
    W, H = 18., 8.4
    fig = plt.figure(figsize=(W, H))
    fig.text(.235/W, 8.01/H, 'DPSeg：检出小目标，细化实例边界',
             fontsize=30.7, fontweight='bold', va='center')
    left, full, small, gap = .235, 6.9, 3.4, .10
    bottom, grid = .30, left+full+.14
    ax = src.image_ax(fig, [left/W, bottom/H, full/W, full/H], a['image']['rgb'])
    src.roi(ax, a['bounds'], 'A', src.ORANGE)
    src.roi(ax, b['bounds'], 'B', src.PRED)
    for t in ax.texts:
        t.set_fontsize(17.6)
    fig.text((left+full/2)/W, 7.43/H, 'RGB 原图', ha='center', va='center', fontsize=20)
    for j, (name, stage) in enumerate([('真实掩码（GT）', 'GT'), ('RGB-D 融合基线', 'M1'), ('DPSeg', 'M4')]):
        x = grid+j*(small+gap)
        fig.text((x+small/2)/W, 7.43/H, name, ha='center', va='center', fontsize=20, fontweight='bold')
        for y, case, caption in [(bottom+small+gap, a, 'A  漏检改善'), (bottom, b, 'B  边界改善')]:
            pane = src.image_ax(fig, [x/W, y/H, small/W, small/H], src.focus(case, stage))
            note = caption if stage == 'GT' else src.iou(case, stage)
            pane.text(.035, .045, note, transform=pane.transAxes, fontsize=16.1,
                      color=src.INK, bbox=dict(fc='white', ec='none', alpha=.91, pad=2))
    return fig


def center_badges(prs):
    edits = []
    for page, slide in enumerate(prs.slides, 1):
        shapes = list(slide.shapes)
        for i, text in enumerate(shapes):
            if not text.has_text_frame:
                continue
            value = text.text.strip()
            is_roi = value in ('A', 'B')
            if not (is_roi or value == '未匹配' or value.startswith('IoU ') or value in ('A  漏检改善', 'B  边界改善')):
                continue
            assert i > 0
            backing = shapes[i-1]
            assert backing.name == '可编辑图形', (page, value, backing.name)
            assert str(backing.fill.fore_color.rgb) == 'FFFFFF'
            assert backing.height < Inches(.6)
            roi_id = None
            if is_roi:
                # Axes draw all ROI outlines before any text. Match the colored
                # outline by color and proximity, not by z-order adjacency.
                color = text.text_frame.paragraphs[0].runs[0].font.color.rgb
                candidates = [q for q in shapes[:i-1] if q.name == '可编辑图形'
                              and q.fill.type != MSO_FILL_TYPE.SOLID
                              and q.line.color.type == MSO_COLOR_TYPE.RGB
                              and q.line.color.rgb == color]
                assert candidates, (page, value)
                roi = min(candidates, key=lambda q: abs(q.left-backing.left)+abs(q.top-backing.top-backing.height))
                roi_id = roi.shape_id
                backing.left = int(roi.left+roi.width/2-backing.width/2)
            text.left, text.top = backing.left, backing.top
            text.width, text.height = backing.width, backing.height
            tf = text.text_frame
            tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
            tf.vertical_anchor = MSO_ANCHOR.MIDDLE
            for paragraph in tf.paragraphs:
                paragraph.alignment = PP_ALIGN.CENTER
            edits.append(dict(slide=page, label=value, horizontal='center', vertical='middle',
                              centered_above_roi=is_roi, roi_shape_id=roi_id,
                              text_shape_id=text.shape_id, backing_shape_id=backing.shape_id))
    return edits


def main():
    source = next(PREV.glob('*.pptx'))
    prs = Presentation(source)
    slide = prs.slides[0]
    for shape in list(slide.shapes):
        shape._element.getparent().remove(shape._element)
    fig = teaser(src.cached())
    counts = ppt.Converter(slide, fig).draw()
    plt.close(fig)
    slide.notes_slide.notes_text_frame.text = (
        'DPSeg teaser：整图与两组真实局部对比；去除底部三个方法模块框。\n'
        '图片间距为 0.10 英寸，整图与局部网格间距为 0.14 英寸（导出前布局）。\n'
        '原始配色、宋体及 Times New Roman；真实 RGB、GT、冻结预测与 IoU 沿用原版。\n'
        'A/B 在框选区域上方居中；所有白色图内标签均水平、垂直居中。\n'
        '局部最大整图 IoU < 0.5 标为未匹配，不显示该实例预测。')
    centered = center_badges(prs)
    prs.core_properties.title = 'DPSeg 报告插图：紧凑 teaser 与居中图内标注'
    path = OUT/'DPSeg_报告插图_紧凑Teaser_标注居中.pptx'
    prs.save(path); ppt.set_theme_fonts(path)
    manifest = json.loads((PREV/'slides_manifest.json').read_text())
    manifest.update(file=str(path), previous_version=str(source), teaser_method_boxes_removed=True,
                    centered_labels=centered, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), bytes=path.stat().st_size)
    manifest['slides'][0].update(counts)
    sizes = [r.font.size.pt for s in prs.slides for sh in s.shapes if sh.has_text_frame
             for p in sh.text_frame.paragraphs for r in p.runs if r.font.size]
    manifest.update(minimum_font_pt=min(sizes), maximum_font_pt=max(sizes))
    (OUT/'slides_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    shutil.copy2(PREV/'provenance.json', OUT/'provenance.json')
    repro = OUT/'reproduction'; repro.mkdir(exist_ok=True)
    for f in ['refine_report_figures_pptx_labels.py', 'export_report_figures_pptx.py', 'build_report_figures_v2.py']:
        shutil.copy2(ROOT/'tools'/f, repro/f)
    (OUT/'使用说明.md').write_text('''# DPSeg 报告插图：紧凑 teaser 与居中标注

- 第 1 页 teaser 删除底部三个方法框，仅保留整图、两组局部真实对比、标题与必要说明；缩小图片之间的横向和纵向间距。
- 所有整图中的 A、B 标签在对应框选区域上方居中；标签内部文字也水平、垂直居中。
- 图片左下角的 IoU、未匹配和案例说明在对应白色标签框内水平、垂直居中。
- 保留原版配色、宋体与 Times New Roman、整图及局部对比。文字、标签和图形均可编辑。最小字号 10.5 pt。
- 原始 RGB、GT、掩码预测、IoU 和局部裁剪范围保持原样。逐实例来源记录见 provenance.json。
- RGB Swin-Large 最终预测列沿用此前显式占位。
- preview.pdf / previews 使用服务器的宋体风格及衬线替代字体；PPT 仍指定宋体和 Times New Roman，未嵌入字体。
- 上一版保留在 report_figures_pptx_v4。
''')
    print(json.dumps(dict(file=str(path), slides=len(prs.slides), labels_centered=len(centered),
                          teaser=counts, MiB=round(path.stat().st_size/1024**2,2)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
