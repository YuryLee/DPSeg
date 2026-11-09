#!/usr/bin/env python3
"""Original editable-deck palette, whole-image comparisons, larger type."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import export_report_figures_pptx_vivid as base
from matplotlib.text import Text
from matplotlib.transforms import Bbox
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.util import Inches, Pt

src, ppt = base.src, base.ppt
OUT = ROOT/'outputs/report_figures_pptx_v4'
OUT.mkdir(exist_ok=True)


def enlarge(fig, stem):
    # Recompute artist bounds before export, so editable text and its backing
    # boxes grow together. The 18-inch overview needs at least 10.5-pt labels.
    for artist in fig.findobj(match=Text):
        if not artist.get_text().strip():
            continue
        artist.set_fontsize(max(11.0, artist.get_fontsize()*1.12))
        artist.set_text(artist.get_text().replace('预测为亮蓝色', '预测为蓝色'))
    if stem == 'fig02_framework':
        # Let the dense overview's two long footnotes use the full group width.
        for artist in fig.axes[0].texts:
            if artist.get_text() == '多尺度 RGB-D 融合\n与实例掩码解码':
                artist.set_text('多尺度 RGB-D 融合\n与实例掩码解码')
                artist.set_fontsize(13)
            elif artist.get_text() == '坐标恢复与边缘过滤\n跨视图掩码去重':
                artist.set_text('坐标恢复\n边缘过滤\n跨视图掩码去重')
            elif artist.get_text() == '逐视图深度归一化与适配':
                artist.set_text('逐视图深度归一化\n与输入适配')
            elif artist.get_text().startswith('DS：'):
                artist.set_fontsize(11)
            elif artist.get_text().startswith('两路 Swin'):
                artist.set_fontsize(11)


def main():
    base.OUT = OUT
    # No vivid_style/reference_style: these are precisely the original source
    # colors and original 0.30 / 0.40 mask overlay strengths.
    assert src.PEACH == '#F6E6D7' and src.PRED == '#0077BB'
    def mark(ax, bounds, label, colour=None):
        src.roi(ax, bounds, label, src.PRED if label == 'B' else src.ORANGE)
    base.mark = mark
    cases = src.cached(); base.add_context_cases(cases); base.recolor(cases)
    src.save = lambda fig, stem: base.FIGURES.__setitem__(stem, fig)
    src.teaser(cases); src.challenges(cases); src.framework(cases)
    base.global_comparison(cases, ['teaser_miss', 'teaser_boundary'], 'fig03_ablation_global_black')
    base.global_comparison(cases, ['green_miss', 'green_boundary'], 'fig03b_ablation_global_green')
    base.contextual_examples(cases, ['white_miss', 'teaser_boundary', 'good'],
                            ['小目标漏检改善', '掩码边界改善', '分割良好案例'], 'fig04_improvement_context')
    base.contextual_examples(cases, ['failure', 'regression'],
                            ['遮挡下仍未匹配', '原有匹配丢失'], 'fig05_failure_context')
    for stem, fig in base.FIGURES.items():
        enlarge(fig, stem)
    items = [('fig00_teaser', 'DPSeg：检出小目标，细化实例边界'),
        ('fig01_challenges', '密集小目标实例分割的典型挑战'), ('fig02_framework', 'DPSeg 总体框架'),
        ('fig03_ablation_global_black', '基线与累计模块消融：整图与局部对比（一）'),
        ('fig03b_ablation_global_green', '基线与累计模块消融：整图与局部对比（二）'),
        ('fig04_improvement_context', '典型改善案例：整图定位与局部对比'),
        ('fig05_failure_context', '失败案例：整图定位与局部对比')]
    prs = Presentation(); prs.slide_width = Inches(ppt.SW); prs.slide_height = Inches(ppt.SH)
    prs.core_properties.title = 'DPSeg 报告插图：原版配色、整图对比与放大字号'
    prs.core_properties.author = 'DPSeg'
    prs.core_properties.subject = '中文宋体；英文数字 Times New Roman；普通标注至少 10.5 pt'
    stats = []
    for stem, title in items:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        counts = ppt.Converter(slide, base.FIGURES[stem],
            title=None if stem in ('fig00_teaser', 'fig02_framework') else title).draw()
        stats.append(dict(slide=len(prs.slides), stem=stem, title=title, **counts))
        slide.notes_slide.notes_text_frame.text = (title+'\n中文宋体；英文和数字 Times New Roman。\n'
            '恢复 DPSeg_报告插图_可编辑_宋体_TimesNewRoman 的配色、灰度深度展示和掩码叠加强度。\n'
            '普通标注至少 10.5 pt（五号），已有较大文字不缩小。保留整图和对应区域的局部放大。\n'
            '沿用真实 GT、冻结预测、案例及匹配指标；本次不改变训练或推理结果。\n'
            '整图显示 score ≥ 0.50 的全部预测；局部最大整图 IoU < 0.5 标为未匹配。\n'
            '局部 GT 为绿色，预测为蓝色；整图实例色仅辅助目视对应。\n'
            'RGB Swin-Large 最终预测栏保留之前的显式占位。来源见本目录 provenance.json。')
    framework = base.FIGURES['fig02_framework']; framework.canvas.draw()
    height = framework.canvas.get_width_height()[1]
    for title, (l,t,r,b) in [('总体框架：完整推理流程', (5,48,1795,462)),
        ('模块展开：多尺度 RGB-D 特征融合', (15,473,1159,992)),
        ('模块展开：深度感知掩码细化', (1170,473,1788,992))]:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        counts = ppt.Converter(slide, framework, Bbox.from_extents(l,height-b,r,height-t), title).draw()
        stats.append(dict(slide=len(prs.slides), title=title, **counts))
        slide.notes_slide.notes_text_frame.text = '第 3 页对应模块展开；同一真实例图。文字、模块、箭头可编辑。'
    font_sizes = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if not shape.has_text_frame:
                continue
            for para in shape.text_frame.paragraphs:
                for run in para.runs:
                    if run.font.size is not None:
                        # The only smaller runs after artist scaling are the
                        # mathematical subscripts; keep those legible as well.
                        if run.font.size.pt < 10.5:
                            run.font.size = Pt(10.5)
                        font_sizes.append(run.font.size.pt)
    path = OUT/'DPSeg_报告插图_原版配色_整图对比_字号加大.pptx'
    prs.save(path); ppt.set_theme_fonts(path)
    for fig in base.FIGURES.values():
        plt.close(fig)
    manifest = dict(file=str(path), slides=stats,
        original_palette_source=str(ROOT/'outputs/report_figures_pptx_v1/DPSeg_报告插图_可编辑_宋体_TimesNewRoman.pptx'),
        palette=dict(peach=src.PEACH, yellow=src.YELLOW, green=src.GREEN, purple=src.PURPLE,
                     blue=src.BLUE, ink=src.INK, gt=src.TEAL, prediction=src.PRED),
        instance_palette=src.COLORS, global_overlay_alpha=.30, focus_overlay_alpha=.40,
        fonts={'east_asian':'宋体', 'latin':'Times New Roman', 'embedded':False},
        minimum_font_pt=min(font_sizes), maximum_font_pt=max(font_sizes),
        text_scale_vs_previous=1.12, font_note='Five-size text = 10.5 pt; bigger labels/titles retained.',
        depth_display='Original grayscale preview', training_or_inference_changes=False,
        sha256=base.sha(path), bytes=path.stat().st_size)
    (OUT/'slides_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(original_figures=str(src.OUT), masks=base.AUDIT,
        roi_bounds={k:dict(image_id=c['image_id'], annotation_id=c['annotation_id'], bounds=c['bounds']) for k,c in cases.items()},
        note='Original v1 colors; identical frozen masks and IoUs to v2/v3; whole-image comparisons and larger type.'),
        ensure_ascii=False, indent=2)+'\n')
    (OUT/'使用说明.md').write_text('''# DPSeg 报告插图：原版配色、整图对比与放大字号

- 共 10 页，恢复最初 `DPSeg_报告插图_可编辑_宋体_TimesNewRoman.pptx` 的低饱和浅色模块、黑灰文字与箭头，以及原始掩码配色和叠加强度。
- 第 4、5 页：每列为同一种方法，首行为整图结果，后两行为 A/B 对应区域的局部放大。
- 第 6、7 页：每个典型案例包含整图定位、局部 RGB、GT、RGB-D 融合基线与 DPSeg 结果。
- 文字相较原排版整体放大约 12%，所有字号至少 10.5 pt（五号），原有大字不缩小；标题保持更大字号。插入报告后如整体缩放，字也会按比例缩放。
- 中文宋体，英文数字 Times New Roman；文字、模块、箭头及框选区域可编辑，照片和掩码为独立图片。字体未嵌入。
- 真实 RGB、GT、预测、IoU 和案例位置均保持原样。深度恢复原始灰度展示，方法仍为 depth Z + 有效性、无需内参。
- RGB Swin-Large 最终预测栏保留之前的“最终结果待补”占位。本轮只改显示与排版。
- preview.pdf 和 previews 使用服务器可用的宋体风格和衬线替代字体；PPT 内指定的是宋体和 Times New Roman。
- 旧版文件保留在各自目录。来源及逐实例对应见 provenance.json。
''')
    print(json.dumps(dict(file=str(path), slides=len(prs.slides), min_font_pt=min(font_sizes),
                         MiB=round(path.stat().st_size/1024**2,2)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
