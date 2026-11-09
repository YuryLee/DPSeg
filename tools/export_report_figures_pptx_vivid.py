#!/usr/bin/env python3
"""More vivid, traceable report slides with whole-image qualitative panels."""
from __future__ import annotations
import os
import sys
import json
import hashlib
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/dpseg-report-matplotlib')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
if Path('/tmp/dpseg_ppt_deps').exists():sys.path.insert(0,'/tmp/dpseg_ppt_deps')
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from pptx import Presentation
from pptx.util import Inches
from tools import build_report_figures_v2 as src
from tools import export_report_figures_pptx as ppt

OUT=ROOT/'outputs/report_figures_pptx_v2'
OUT.mkdir(exist_ok=True)
FIGURES={}
PALETTE=['#00B8FF','#FF623D','#00DF9A','#FF46B8','#8B6CFF','#FFD13B',
         '#45D94E','#FF8F26','#28E0E8','#E84C4C','#DA68F8','#72A5FF']
STAGES=['RGB','GT','M0','M1','M2','M3','M4']
LABELS=['RGB 原图','真实掩码（GT）','RGB Mask2Former','＋RGB-D 融合','＋点细化','＋概率图细化','＋切片增强推理']
AUDIT=[]

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def vivid_style():
    src.PEACH='#FFBD82';src.YELLOW='#FFE36C';src.GREEN='#A7E68C'
    src.PURPLE='#C6A0F1';src.BLUE='#93C7F2'
    src.TEAL='#00D995';src.PRED='#00B8FF';src.ORANGE='#F08700';src.COLORS=PALETTE
    original=src.overlay
    def overlay(rgb,masks,colors,alpha=.30):
        return original(rgb,masks,colors,alpha=min(.64,alpha+.22))
    src.overlay=overlay

def add_context_cases(cases):
    """Use existing frozen outputs; no new inference or changes to masks."""
    pool=json.loads((src.OUT/'candidate_cases.json').read_text())['all_candidates']['recovered']
    row=max((r for r in pool if r['image_id']==202615007),key=lambda r:r['priority'])
    image=cases['green_boundary']['image'];folder=src.OUT/'source_assets'/str(row['image_id'])
    anns=json.loads((folder/'ground_truth.json').read_text())['annotations']
    ann=next(a for a in anns if a['id']==row['annotation_id'])
    c=dict(row,case_key='green_miss',image=image,gt=src.decode(ann['segmentation']),stages={})
    c['bounds']=src.tight_bounds(row['bbox'])
    for s in src.STAGES:
        index,iou,mask=src.best(image['predictions'][s],ann)
        c['stages'][s]=dict(index=index,iou=iou,mask=mask,
                          display=mask if iou>=.5 else np.zeros_like(mask))
    cases['green_miss']=c

def recolor(cases):
    seen=set();assetroot=OUT/'assets';assetroot.mkdir(exist_ok=True)
    for c in cases.values():
        iid=c['image_id']
        if iid in seen:continue
        seen.add(iid);image=c['image'];folder=src.OUT/'source_assets'/str(iid)
        annotations=json.loads((folder/'ground_truth.json').read_text())['annotations']
        image['visuals']['GT']=src.overlay_all(image['rgb'],annotations)
        for s in src.STAGES:image['visuals'][s]=src.overlay_all(image['rgb'],annotations,image['predictions'][s])
        target=assetroot/str(iid);target.mkdir(exist_ok=True)
        for s,im in image['visuals'].items():Image.fromarray(im).save(target/f'{s}_overlay.png')
    for key,c in cases.items():
        folder=src.OUT/'source_assets'/str(c['image_id'])
        for s in src.STAGES:
            AUDIT.append(dict(case=key,image_id=c['image_id'],annotation_id=c['annotation_id'],stage=s,
                iou=c['stages'][s]['iou'],source_prediction=str(folder/f'{s}_predictions.json'),
                source_sha256=sha(folder/f'{s}_predictions.json'),
                raw_mask_sha256=hashlib.sha256(c['stages'][s]['mask'].tobytes()).hexdigest(),
                display_mask_sha256=hashlib.sha256(c['stages'][s]['display'].tobytes()).hexdigest()))

def mark(ax,box,label,colour='#F08700'):
    # White under-stroke keeps the ROI distinct from multicoloured instances.
    src.roi(ax,box,label,colour)

def placeholder(fig,rect):
    ax=fig.add_axes(rect);ax.set_facecolor('#F5F5F5');ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_color('#CCC');sp.set_linewidth(.6)
    ax.text(.5,.5,'Swin-Large\n最终结果待补',ha='center',va='center',transform=ax.transAxes,fontsize=13,color='#555')
    return ax

def label(ax,text):
    ax.text(.03,.04,text,transform=ax.transAxes,fontsize=11,color='#1D1D1D',
            bbox=dict(fc='white',ec='none',alpha=.94,pad=2))

def global_comparison(cases,keys,stem):
    a,b=(cases[k] for k in keys)
    assert a['image_id']==b['image_id']
    fig=plt.figure(figsize=(18,9.6));xs=[.049+i*.135 for i in range(7)];w=.131;h=w*18/9.6
    for x,name in zip(xs,LABELS):fig.text(x+w/2,.963,name,ha='center',fontsize=13)
    row_y=[.675,.385,.095]
    for row,(y,focus) in enumerate(zip(row_y,[None,a,b])):
        rowlabel=['整图结果','A  小目标漏检','B  边界细节'][row]
        fig.text(.018,y+h/2,rowlabel,ha='center',va='center',rotation=90,fontsize=13)
        for x,s in zip(xs,STAGES):
            rect=[x,y,w,h]
            if s=='M0':placeholder(fig,rect);continue
            if focus is None:
                im=a['image']['rgb'] if s=='RGB' else a['image']['visuals'][s]
                ax=src.image_ax(fig,rect,im)
                mark(ax,a['bounds'],'A');mark(ax,b['bounds'],'B','#DE259A')
            else:
                im=src.cut(focus['image']['rgb'],focus['bounds']) if s=='RGB' else src.focus(focus,s)
                ax=src.image_ax(fig,rect,im)
                if s in src.STAGES:label(ax,src.iou(focus,s))
    fig.text(.049,.035,'整图显示全部实例；A、B 为对应放大区域。局部图突出所关注目标：GT 为绿色，预测为亮蓝色。',fontsize=12,color='#444')
    FIGURES[stem]=fig

def contextual_examples(cases,keys,rowlabels,stem):
    """A full RGB context column accompanies every local before/after case."""
    rows=len(keys);width=14.1;height=8.8 if rows==3 else 6.2
    fig=plt.figure(figsize=(width,height));xs=[.045+i*.19 for i in range(5)];w=.181;h=w*width/height
    for x,title in zip(xs,['整图定位','局部 RGB','真实掩码（GT）','RGB-D 融合基线','DPSeg']):
        fig.text(x+w/2,.972,title,ha='center',fontsize=13)
    ys=[.65,.345,.04] if rows==3 else [.515,.075]
    for key,rowlabel,y in zip(keys,rowlabels,ys):
        c=cases[key];fig.text(.018,y+h/2,rowlabel,ha='center',va='center',rotation=90,fontsize=12)
        for j,s in enumerate(['FULL','RGB','GT','M1','M4']):
            if s=='FULL':im=c['image']['rgb']
            elif s=='RGB':im=src.cut(c['image']['rgb'],c['bounds'])
            else:im=src.focus(c,s)
            ax=src.image_ax(fig,[xs[j],y,w,h],im)
            if s=='FULL':mark(ax,c['bounds'],'')
            elif s in src.STAGES:label(ax,src.iou(c,s))
    FIGURES[stem]=fig

def main():
    vivid_style();cases=src.cached();add_context_cases(cases);recolor(cases)
    src.save=lambda fig,stem:FIGURES.__setitem__(stem,fig)
    src.teaser(cases);src.challenges(cases);src.framework(cases)
    global_comparison(cases,['teaser_miss','teaser_boundary'],'fig03_ablation_global_black')
    global_comparison(cases,['green_miss','green_boundary'],'fig03b_ablation_global_green')
    contextual_examples(cases,['white_miss','teaser_boundary','good'],
        ['小目标漏检改善','掩码边界改善','分割良好案例'],'fig04_improvement_context')
    contextual_examples(cases,['failure','regression'],['遮挡下仍未匹配','原有匹配丢失'],'fig05_failure_context')
    items=[('fig00_teaser','DPSeg：检出小目标，细化实例边界'),
        ('fig01_challenges','密集小目标实例分割的典型挑战'),('fig02_framework','DPSeg 总体框架'),
        ('fig03_ablation_global_black','基线与累计模块消融：整图与局部对比（一）'),
        ('fig03b_ablation_global_green','基线与累计模块消融：整图与局部对比（二）'),
        ('fig04_improvement_context','典型改善案例：整图定位与局部对比'),
        ('fig05_failure_context','失败案例：整图定位与局部对比')]
    prs=Presentation();prs.slide_width=Inches(ppt.SW);prs.slide_height=Inches(ppt.SH)
    prs.core_properties.title='DPSeg 报告插图：鲜明配色与整图对比'
    prs.core_properties.author='DPSeg';prs.core_properties.subject='可编辑；中文宋体；英文数字 Times New Roman'
    stats=[]
    for stem,title in items:
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        cv=ppt.Converter(slide,FIGURES[stem],title=None if stem in ('fig00_teaser','fig02_framework') else title)
        counts=cv.draw();stats.append(dict(slide=len(prs.slides),stem=stem,title=title,**counts))
        slide.notes_slide.notes_text_frame.text=(title+'\n中文宋体；英文和数字 Times New Roman。\n'+
            '本版仅调整显示配色、掩码叠加透明度和排版。沿用原始 GT 与冻结预测，无训练或推理结果变更。\n'+
            '整图显示 score ≥ 0.50 的全部预测；局部图仅显示所关注目标的对应掩码，最大整图 IoU < 0.5 标为未匹配。\n'+
            '绿色为局部 GT，亮蓝色为局部预测。整图的实例配色按与 GT 的重叠关系统一，仅用于辅助目视对应。\n'+
            ('RGB Swin-Large 最终输出未完成，保留显式占位。\n' if 'ablation' in stem else '')+
            '来源与对应关系：outputs/report_figures_pptx_v2/provenance.json')
    framework=FIGURES['fig02_framework'];framework.canvas.draw();height=framework.canvas.get_width_height()[1]
    for title,(l,t,r,b) in [('总体框架：完整推理流程',(5,48,1795,462)),
        ('模块展开：多尺度 RGB-D 特征融合',(15,473,1159,992)),
        ('模块展开：深度感知掩码细化',(1170,473,1788,992))]:
        slide=prs.slides.add_slide(prs.slide_layouts[6])
        cv=ppt.Converter(slide,framework,Bbox.from_extents(l,height-b,r,height-t),title)
        counts=cv.draw();stats.append(dict(slide=len(prs.slides),title=title,**counts))
        slide.notes_slide.notes_text_frame.text='第 3 页的对应模块展开。文字、模块和箭头可编辑；例图来自同一冻结流水线。'
    path=OUT/'DPSeg_报告插图_鲜明配色_含整图对比.pptx'
    prs.save(path);ppt.set_theme_fonts(path)
    for f in FIGURES.values():plt.close(f)
    manifest=dict(file=str(path),slides=stats,palette=PALETTE,
        fonts={'east_asian':'宋体','latin':'Times New Roman','embedded':False},
        global_overlay_alpha=.52,focus_overlay_alpha=.62,
        previous_version=str(ROOT/'outputs/report_figures_pptx_v1'),
        rgb_mask2former_final='pending, visible placeholder',training_or_inference_changes=False,
        sha256=sha(path),bytes=path.stat().st_size)
    (OUT/'slides_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(original_figures=str(src.OUT),
        masks=AUDIT,roi_bounds={k:dict(image_id=c['image_id'],annotation_id=c['annotation_id'],bounds=c['bounds']) for k,c in cases.items()},
        note='Posthoc illustrative examples, no tuning. Green scene additional focus uses already stored predictions.'),ensure_ascii=False,indent=2)+'\n')
    (OUT/'使用说明.md').write_text('''# DPSeg 插图 PPT：鲜明配色与整图对比

- 共 10 页：原有插图内容、两页整图与局部累计消融对比，以及三页框架展开。
- 掩码改用更鲜明的实例色和更高叠加不透明度；网络模块同步提高颜色饱和度。原始 RGB、GT、预测几何与 IoU 保持原样。
- 第 4、5 页：每列为同一方法，首行为同一场景的整图，后两行为 A/B 区域。整图包含全部实例；局部图只突出当前目标。绿色为局部 GT，亮蓝色为局部预测。
- 第 6、7 页：每行新增整图定位列，用框选对应的局部展示区域。
- RGB Swin-Large 最终结果仍待补，保留显式占位，不用中间模型代替。
- 中文宋体，英文数字 Times New Roman；文字、模块、连线、框选区域可编辑，照片和掩码为独立图片。字体未嵌入，打开的电脑需安装这两种字体。
- preview.pdf / 预览图片使用服务器可用的宋体风格与衬线替代字体，仅用于查看颜色与版式；PPT 中指定的是宋体和 Times New Roman。
- provenance.json 保存真实样本、预测文件哈希和逐实例对应关系，图中不显示样本 ID。
''')
    print(json.dumps(dict(file=str(path),slides=len(prs.slides),MiB=round(path.stat().st_size/1024**2,2)),ensure_ascii=False))

if __name__=='__main__':main()
