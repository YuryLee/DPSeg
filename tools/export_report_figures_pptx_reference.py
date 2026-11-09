#!/usr/bin/env python3
"""Apply the user's blue/orange/lavender/teal reference palette to editable slides."""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import export_report_figures_pptx_vivid as base
from matplotlib import patches
from matplotlib.text import Text
from matplotlib.transforms import Bbox
from pptx import Presentation
from pptx.util import Inches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

src, ppt = base.src, base.ppt
OUT = ROOT / 'outputs/report_figures_pptx_v3'
OUT.mkdir(exist_ok=True)
COLORS = dict(
    ink='#12334B', muted='#687E90', border='#CEDBE5', card='#FAFCFE',
    blue='#2F80ED', blue_fill='#EAF3FD', blue_edge='#72B2F8',
    orange='#F19A58', orange_fill='#FFF3E7', orange_edge='#F3A066',
    purple='#7953A7', purple_fill='#F1EAF8', purple_edge='#AB83CB',
    teal='#12A89D', teal_fill='#E9F6F2', teal_edge='#5BB9AC',
    gray_fill='#EFF4F7', gray_edge='#A1B3C1')
INSTANCE_COLORS = ['#2F80ED', '#F19A58', '#12A89D', '#9872C4', '#59B4D6',
                   '#E6BD52', '#67AC84', '#D88DAB', '#6B91B0', '#C49564',
                   '#7ABBAF', '#AF9AD0']


def style():
    src.INK = COLORS['ink']; src.GRAY = COLORS['muted']
    src.PEACH = COLORS['orange_fill']; src.BLUE = COLORS['blue_fill']
    src.PURPLE = COLORS['purple_fill']; src.GREEN = COLORS['teal_fill']
    src.YELLOW = COLORS['teal_fill']
    src.TEAL = COLORS['teal']; src.PRED = COLORS['blue']; src.ORANGE = COLORS['orange']
    src.COLORS = INSTANCE_COLORS
    plt.rcParams['text.color'] = COLORS['ink']
    original = src.overlay
    def overlay(rgb, masks, colors, alpha=.30):
        return original(rgb, masks, colors, alpha=alpha+.16)
    src.overlay = overlay
    original_text = ppt.add_text
    def add_text(*args, **kwargs):
        kwargs.setdefault('color', COLORS['ink'])
        return original_text(*args, **kwargs)
    ppt.add_text = add_text


def recolor_depth(cases):
    """Color-map the existing normalized display image, never the model input."""
    seen = set()
    for c in cases.values():
        if c['image_id'] in seen:
            continue
        seen.add(c['image_id'])
        gray = c['image']['depth'][..., 0].astype(float)/255.
        rgb = np.uint8(plt.get_cmap('viridis')(gray)[..., :3]*255)
        c['image']['depth'] = rgb
        Image.fromarray(rgb).save(OUT/'assets'/str(c['image_id'])/'depth_display.png')


def role(patch, name):
    patch.set_facecolor(COLORS[name+'_fill'])
    patch.set_edgecolor(COLORS[name+'_edge'])
    patch.set_linewidth(1.05)


def restyle(fig, stem):
    # Preserve annotation colors while making ordinary labels dark navy.
    for text in fig.findobj(match=Text):
        if not text.get_text():
            continue
        if text.get_text() in ('A', 'B'):
            text.set_color(COLORS['orange'] if text.get_text() == 'A' else COLORS['purple'])
        else:
            text.set_color(COLORS['ink'])
        text.set_text(text.get_text().replace('GT 为绿色，预测为亮蓝色', 'GT 为青绿色，预测为蓝色')
                      .replace('绿色轮廓', '青绿色轮廓'))
    for ax in fig.axes:
        for sp in ax.spines.values():
            sp.set_edgecolor(COLORS['border'])
        for p in ax.patches:
            if isinstance(p, patches.Rectangle) and not p.get_fill():
                old = p.get_edgecolor()
                if old[:3] != (1., 1., 1.):
                    p.set_edgecolor(COLORS['purple'] if old[2] > old[1]*1.2 else COLORS['orange'])
    if stem == 'fig00_teaser':
        for p, r in zip(fig.patches, ['purple', 'teal', 'blue']):
            role(p, r)
    if stem != 'fig02_framework':
        return
    ax = fig.axes[0]
    for p in ax.patches:
        if isinstance(p, patches.FancyArrowPatch):
            points = p._posA_posB
            x, y = np.mean(points, axis=0)
            p.set_color(COLORS['teal'] if (x > 1175 and y > 475) or (825 < x < 1309 and y < 410)
                        else COLORS['blue'] if y < 475 else COLORS['muted'])
        elif isinstance(p, patches.FancyBboxPatch):
            x, y, w, h = p.get_x(), p.get_y(), p.get_width(), p.get_height()
            if h > 250:
                p.set_facecolor(COLORS['card']); p.set_edgecolor(COLORS['border'])
                p.set_linestyle('-')
            elif y == 545:
                role(p, 'blue')
            elif y == 800:
                role(p, 'orange')
            elif y in (625, 715):
                role(p, 'purple')
            elif y == 606:
                role(p, 'gray')
            elif (x, y) in ((400, 160), (1192, 564)):
                role(p, 'purple')
            elif (x, y) in ((420, 304), (1192, 615)):
                role(p, 'blue')
            elif (x, y) == (622, 160):
                role(p, 'gray')
            else:
                role(p, 'teal')
        elif isinstance(p, patches.Circle):
            role(p, 'blue' if p.center[1] == 773 else 'gray')
    for line in ax.lines:
        x, y = np.mean(line.get_xdata()), np.mean(line.get_ydata())
        if len(line.get_xdata()) == 2 and y == 761:
            line.set_color(COLORS['border'])
        else:
            line.set_color(COLORS['teal'] if x > 1175 and y > 475 else COLORS['muted'])
    for text in ax.texts:
        if text.get_text() == 'RGB':
            text.set_color(COLORS['blue'])
        elif text.get_text() in ('深度', '深度 Z + 有效性'):
            text.set_color(COLORS['orange'])
        elif text.get_text() in ('DS', 'DC'):
            text.set_color(COLORS['purple'])


def main():
    base.OUT = OUT
    style()
    cases = src.cached(); base.add_context_cases(cases); base.recolor(cases)
    recolor_depth(cases)
    def capture(fig, stem):
        base.FIGURES[stem] = fig
    src.save = capture
    src.teaser(cases); src.challenges(cases); src.framework(cases)
    base.global_comparison(cases, ['teaser_miss', 'teaser_boundary'], 'fig03_ablation_global_black')
    base.global_comparison(cases, ['green_miss', 'green_boundary'], 'fig03b_ablation_global_green')
    base.contextual_examples(cases, ['white_miss', 'teaser_boundary', 'good'],
                            ['小目标漏检改善', '掩码边界改善', '分割良好案例'], 'fig04_improvement_context')
    base.contextual_examples(cases, ['failure', 'regression'], ['遮挡下仍未匹配', '原有匹配丢失'], 'fig05_failure_context')
    for stem, fig in base.FIGURES.items():
        restyle(fig, stem)
    items = [('fig00_teaser', 'DPSeg：检出小目标，细化实例边界'),
        ('fig01_challenges', '密集小目标实例分割的典型挑战'), ('fig02_framework', 'DPSeg 总体框架'),
        ('fig03_ablation_global_black', '基线与累计模块消融：整图与局部对比（一）'),
        ('fig03b_ablation_global_green', '基线与累计模块消融：整图与局部对比（二）'),
        ('fig04_improvement_context', '典型改善案例：整图定位与局部对比'),
        ('fig05_failure_context', '失败案例：整图定位与局部对比')]
    prs = Presentation(); prs.slide_width = Inches(ppt.SW); prs.slide_height = Inches(ppt.SH)
    prs.core_properties.title = 'DPSeg 报告插图：参考配色版'
    prs.core_properties.author = 'DPSeg'
    prs.core_properties.subject = '可编辑；中文宋体；英文数字 Times New Roman'
    stats = []
    for stem, title in items:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        counts = ppt.Converter(slide, base.FIGURES[stem], title=None if stem in ('fig00_teaser', 'fig02_framework') else title).draw()
        stats.append(dict(slide=len(prs.slides), stem=stem, title=title, **counts))
        slide.notes_slide.notes_text_frame.text = (title+'\n中文宋体；英文和数字 Times New Roman。\n'
            '参考用户图片配色：浅蓝、浅橙、淡紫和青绿，深蓝文字。\n'
            '沿用上一版真实 GT、冻结预测及整图/局部排版，仅调整显示颜色。\n'
            '深度伪彩色来自原有归一化展示图，不改变网络输入。当前方法为 depth Z + 有效性，无需相机内参。\n'
            '整图显示 score ≥ 0.50 的全部预测；局部最大整图 IoU < 0.5 标为未匹配。\n'
            '局部 GT 为青绿色，预测为蓝色；整图实例色仅辅助目视对应。\n'
            'RGB Swin-Large 最终结果保留上一版显式占位。来源详见本目录 provenance.json。')
    framework = base.FIGURES['fig02_framework']; framework.canvas.draw()
    height = framework.canvas.get_width_height()[1]
    for title, (l,t,r,b) in [('总体框架：完整推理流程', (5,48,1795,462)),
        ('模块展开：多尺度 RGB-D 特征融合', (15,473,1159,992)),
        ('模块展开：深度感知掩码细化', (1170,473,1788,992))]:
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        counts = ppt.Converter(slide, framework, Bbox.from_extents(l,height-b,r,height-t), title).draw()
        stats.append(dict(slide=len(prs.slides), title=title, **counts))
        slide.notes_slide.notes_text_frame.text = '第 3 页对应模块展开；文字、模块及箭头可编辑，使用相同真实例图。'
    path = OUT/'DPSeg_报告插图_参考配色_含整图对比.pptx'
    prs.save(path); ppt.set_theme_fonts(path)
    for f in base.FIGURES.values():
        plt.close(f)
    manifest = dict(file=str(path), slides=stats, palette=COLORS, instance_palette=INSTANCE_COLORS,
        fonts={'east_asian':'宋体', 'latin':'Times New Roman', 'embedded':False},
        global_overlay_alpha=.46, focus_overlay_alpha=.56, depth_display='viridis of previous normalized display image',
        previous_version=str(ROOT/'outputs/report_figures_pptx_v2'),
        reference='/data/yinli/.codex/attachments/c1fa2aaf-9ab4-4647-8265-c1facf347870/codex-clipboard-dafcb479-333c-4440-8678-4678c3f14a6c.png',
        training_or_inference_changes=False, sha256=base.sha(path), bytes=path.stat().st_size)
    (OUT/'slides_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(original_figures=str(src.OUT), masks=base.AUDIT,
        roi_bounds={k:dict(image_id=c['image_id'], annotation_id=c['annotation_id'], bounds=c['bounds']) for k,c in cases.items()},
        note='Same frozen predictions, masks, IoUs and layout as v2; display palette only.'), ensure_ascii=False, indent=2)+'\n')
    (OUT/'使用说明.md').write_text('''# DPSeg 插图 PPT：参考配色版

- 共 10 页，保留整图与局部定性对比、teaser 和总体框架的模块展开。
- 配色参考用户提供图片：白底和浅灰蓝分组；RGB 分支浅蓝，深度分支浅橙，DS/DC 淡紫，掩码细化青绿；文字深蓝，主箭头蓝色，细化箭头青绿。
- 原始 RGB、GT、预测掩码、IoU 和案例位置均沿用上一版。深度图改为 viridis 伪彩色展示，模型仍使用无需内参的深度 Z 与有效性信息。
- 局部 GT 为青绿色，预测为蓝色；整图各实例使用一致配色辅助对应。图中不显示样本 ID，来源保存在 provenance.json。
- 中文宋体，英文数字 Times New Roman；文字、模块、连线、框选区域可编辑，照片和掩码为独立图片。字体未嵌入。
- preview.pdf / previews 使用服务器可用的宋体风格和衬线替代字体，PPT 内指定的字体仍为宋体和 Times New Roman。
- RGB Swin-Large 最终预测栏保留上一版“最终结果待补”占位。本轮仅改配色。
- 上一版保留在 report_figures_pptx_v2。
''')
    print(json.dumps({'file':str(path), 'slides':len(prs.slides), 'MiB':round(path.stat().st_size/1024**2, 2)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
