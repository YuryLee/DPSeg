#!/usr/bin/env python3
"""Build the five report figures from real, frozen Swin-Large predictions.

No training, synthetic masks, probability reconstruction, or pixel retouching.
The unfinished RGB baseline is visibly reserved rather than substituted.
"""
from __future__ import annotations
import csv
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from collections import defaultdict
from zipfile import ZipFile, ZIP_DEFLATED

os.environ.setdefault('MPLCONFIGDIR', '/tmp/dpseg-report-matplotlib')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager, patches
import numpy as np
from PIL import Image
from pycocotools import mask as cm
from scipy.ndimage import binary_erosion
from rgbd_seg.course_archive import CourseArchive
from tools.select_report_figure_cases import ARCHIVE, OUT, RUN, SOURCES, THRESHOLD, digest, load_records, write

font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],
    'font.size':18, 'axes.unicode_minus':False, 'pdf.fonttype':42,
    'svg.fonttype':'none', 'savefig.facecolor':'white'})
PALETTE = np.array(['#0077BB','#EE7733','#009988','#CC79A7','#56B4E9','#E6AB02','#7655A3','#607D8B'])
INK='#233044'; GREY='#5E6A79'; BLUE='#2474A3'; GREEN='#27866D'; PURPLE='#7755A4'
GT_COLOR=np.array([0,153,136]); PRED_COLOR=np.array([0,119,187])
MISS=np.array([230,145,35]); EXTRA=np.array([75,115,205]); HIT=np.array([165,170,178])
FIGURES = [
 ('fig01_challenges','密集小目标实例分割的典型挑战'),
 ('fig02_framework','DPSeg 总体框架'),
 ('fig03_ablation_draft','基线与消融定性对比（基线待补）'),
 ('fig04_improvement_cases','小目标漏检与边界改善、分割良好案例'),
 ('fig05_failure_cases','失败案例及误差分析')]

def decode(rle):
    return cm.decode(rle).astype(bool)

def rgb_color(color):
    return np.array(matplotlib.colors.to_rgb(color))*255

def overlay(rgb, masks, colors, alpha=.38):
    out=np.array(rgb,dtype=np.float32).copy()
    # Smaller masks are painted last to keep their visible contours legible.
    for k in sorted(range(len(masks)), key=lambda i:-int(masks[i].sum())):
        m=masks[k]; color=np.array(colors[k])
        out[m]=out[m]*(1-alpha)+color*alpha
        edge=m & ~binary_erosion(m)
        out[edge]=color
    return np.uint8(np.clip(out,0,255))

def error_map(rgb, gt, pred):
    # Four disjoint sets. These are mask pixels, not instance FP/FN counts.
    gray=np.array(rgb,dtype=np.float32).mean(2,keepdims=True)
    out=np.repeat(gray,3,axis=2)*.32+255*.68
    out[gt & pred]=HIT
    out[gt & ~pred]=MISS
    out[pred & ~gt]=EXTRA
    return out.astype(np.uint8)

def panel(ax, image, box=None):
    ax.imshow(image,interpolation='nearest')
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color('#DDE2E8'); sp.set_linewidth(.7)
    if box:
        l,t,r,b=box
        ax.add_patch(patches.Rectangle((l,t),r-l,b-t,fill=False,edgecolor='white',linewidth=3.1))
        ax.add_patch(patches.Rectangle((l,t),r-l,b-t,fill=False,edgecolor='#D88715',linewidth=1.7))

def crop(im, bounds):
    l,t,r,b=bounds
    return np.array(im)[t:b,l:r]

def scalebar(ax, side, n=20):
    x,y=side*.07,side*.91
    ax.plot([x,x+n],[y,y],color='white',linewidth=5,solid_capstyle='butt')
    ax.plot([x,x+n],[y,y],color='#263347',linewidth=2,solid_capstyle='butt')
    ax.text(x,y-side*.035,f'{n} px',fontsize=13,color=INK,
            bbox=dict(facecolor='white',edgecolor='none',alpha=.88,pad=1))

def save(fig, stem, vector=False):
    folder=OUT/'figures';folder.mkdir(exist_ok=True)
    fig.savefig(folder/f'{stem}.png',dpi=300)
    fig.savefig(folder/f'{stem}.pdf',dpi=300)
    if vector:
        fig.savefig(folder/f'{stem}.svg')
    fig.savefig(folder/f'{stem}_preview.jpg',dpi=110)
    plt.close(fig)

def best(preds, annotation):
    visible=[i for i,p in enumerate(preds) if p['score']>=THRESHOLD]
    ious=cm.iou([preds[i]['segmentation'] for i in visible],[annotation['segmentation']],[0])
    if not visible or not np.max(ious)>0:
        return None,0.,np.zeros((1024,1024),bool)
    local=int(np.argmax(ious[:,0])); index=visible[local]
    return index,float(ious[local,0]),decode(preds[index]['segmentation'])

def displayed_mask(record):
    # A neighbouring true object may have a tiny nonzero overlap with the
    # target. Do not depict it as a failed mask prediction of this target.
    return record['mask'] if record['iou'] >= .5 else np.zeros_like(record['mask'])

def iou_label(record):
    return (f"IoU = {record['iou']:.3f}" if record['iou'] >= .5 else
            f"未匹配（IoU < 0.5）\n最大 IoU = {record['iou']:.3f}")

def all_overlay(rgb, annotations, preds=None):
    if preds is None:
        return overlay(rgb,[decode(a['segmentation']) for a in annotations],
                       [rgb_color(PALETTE[i%len(PALETTE)]) for i in range(len(annotations))],alpha=.30)
    visible=[p for p in preds if p['score']>=THRESHOLD]
    if not visible:
        return rgb
    matrix=cm.iou([p['segmentation'] for p in visible],
                 [a['segmentation'] for a in annotations],[0]*len(annotations))
    colors=[rgb_color(PALETTE[int(row.argmax())%len(PALETTE)]) if row.max()>=.1
            else np.array([235,180,35]) for row in matrix]
    return overlay(rgb,[decode(p['segmentation']) for p in visible],colors,alpha=.30)

def prepare():
    rows=json.loads((OUT/'selected_cases.json').read_text())
    ids={r['image_id'] for r in rows}
    stages={**SOURCES,**{s:OUT/'supplemental_predictions'/f'{s}.jsonl' for s in ('M2','M3')}}
    records={s:load_records(p,ids) for s,p in stages.items()}
    dataset=CourseArchive(ARCHIVE,split='test')
    gt=dataset.coco_ground_truth(); anns=defaultdict(list)
    for a in gt['annotations']:
        if a['image_id'] in ids:
            anns[a['image_id']].append(a)
    cases={}
    for row in rows:
        iid=row['image_id'];truth=anns[iid]
        annotation=next(a for a in truth if a['id']==row['annotation_id'])
        rgb=np.array(dataset.load_rgb(row['image_index']))
        target=decode(annotation['segmentation'])
        info=dict(row,annotation=annotation,rgb=rgb,gt=target,all_gt=truth,
                  stages={},overlays={'GT':all_overlay(rgb,truth)})
        folder=OUT/'source_assets'/str(iid);folder.mkdir(parents=True,exist_ok=True)
        meta=dataset.images[row['image_index']]
        with ZipFile(ARCHIVE) as archive:
            (folder/'rgb.png').write_bytes(archive.read(meta['file_name']))
            (folder/'depth_raw.png').write_bytes(archive.read(meta['depth_file_name']))
        depth,valid=dataset.load_depth(row['image_index'])
        info['depth']=depth;info['valid']=valid
        lo,hi=np.quantile(depth[valid],[.01,.99])
        preview=plt.get_cmap('viridis')(np.clip((depth-lo)/(hi-lo+1e-8),0,1))[:,:,:3]
        preview[~valid]=.85
        Image.fromarray((preview*255).astype(np.uint8)).save(folder/'depth_preview.png')
        info['depth_preview']=(preview*255).astype(np.uint8)
        Image.fromarray(target.astype(np.uint8)*255).save(folder/'focus_gt_mask.png')
        Image.fromarray(info['overlays']['GT']).save(folder/'gt_overlay.png')
        write(folder/'gt_coco.json',dict(images=[meta],annotations=truth,categories=dataset.categories))
        for stage in ('M1','M2','M3','M4'):
            preds=records[stage][iid]
            idx,iou,mask=best(preds,annotation)
            info['stages'][stage]=dict(index=idx,iou=iou,mask=mask,predictions=preds)
            info['overlays'][stage]=all_overlay(rgb,truth,preds)
            Image.fromarray(mask.astype(np.uint8)*255).save(folder/f'focus_{stage}_mask.png')
            Image.fromarray(displayed_mask(info['stages'][stage]).astype(np.uint8)*255).save(folder/f'focus_{stage}_display_mask.png')
            Image.fromarray(info['overlays'][stage]).save(folder/f'{stage}_overlay.png')
            write(folder/f'{stage}_predictions.json',dict(image_id=iid,predictions=preds))
        write(folder/'focus.json',dict(row,stages={s:dict(prediction_index=v['index'],mask_iou=v['iou'])
                                                 for s,v in info['stages'].items()},
            score_threshold=THRESHOLD, gt_mask_area_pixels=int(target.sum()),
            rgb_source_sha256=digest(folder/'rgb.png'),depth_source_sha256=digest(folder/'depth_raw.png')))
        cases[iid]=info
    dataset.close()
    return cases,stages

def challenges(cases):
    specs=[(202553004,'(a) 遮挡后仅剩小面积可见区域'),
           (202612002,'(b) 相近外观与相邻实例边界'),
           (202358007,'(c) 密集堆叠中的细长结构')]
    fig,axes=plt.subplots(3,3,figsize=(12.6,12.6))
    fig.subplots_adjust(left=.065,right=.99,bottom=.055,top=.945,wspace=.045,hspace=.13)
    for j,(iid,title) in enumerate(specs):
        c=cases[iid];box=c['bounds']; side=box[2]-box[0]
        panel(axes[0,j],c['rgb'],box);axes[0,j].set_title(title,fontsize=17,pad=12)
        panel(axes[1,j],crop(c['rgb'],box));scalebar(axes[1,j],side)
        gt_view=overlay(c['rgb'],[c['gt']],[GT_COLOR],alpha=.48)
        panel(axes[2,j],crop(gt_view,box))
        axes[2,j].set_xlabel(f"ID {iid}\n目标可见掩码面积：{int(c['gt'].sum())} px",fontsize=16,labelpad=9)
    for i,label in enumerate(('整图与局部位置','RGB 局部','目标实例 GT')):
        axes[i,0].set_ylabel(label,fontsize=17,labelpad=13)
    save(fig,'fig01_challenges')

def box(ax,x,y,w,h,lines,fill='#FFFFFF',edge='#C9D2DD',size=18,lw=1.1):
    p=patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.01,rounding_size=9',
                            facecolor=fill,edgecolor=edge,linewidth=lw)
    ax.add_patch(p)
    ax.text(x+w/2,y+h/2,lines,ha='center',va='center',fontsize=size,color=INK,linespacing=1.45)
    return p

def arrow(ax,start,end,color=GREY,dashed=False,connection='arc3,rad=0'):
    ax.add_patch(patches.FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=13,
        linewidth=1.35,color=color,linestyle='--' if dashed else '-',
        connectionstyle=connection,shrinkA=2,shrinkB=2))

def framework(cases):
    fig=plt.figure(figsize=(15,10.1));ax=fig.add_axes([.015,.018,.97,.965])
    ax.set_xlim(0,1500);ax.set_ylim(0,1010);ax.axis('off')
    ax.text(20,985,'推理流程',fontsize=22,weight='bold',color=INK)
    items=[('对齐 RGB-D\n1024 × 1024',None),('整图与\n重叠切片',GREEN),
           ('多尺度\nRGB-D 融合',BLUE),('候选映射\n筛选与去重',GREEN),
           ('深度感知\n点细化',PURPLE),('深度引导\n概率图细化',PURPLE),('实例掩码\n与置信度',None)]
    xs=[20,232,444,656,868,1080,1292]
    for i,((label,color),x) in enumerate(zip(items,xs)):
        box(ax,x,872,184,83,label,fill={GREEN:'#EAF5F0',BLUE:'#EAF3FA',PURPLE:'#F1EDF7',None:'#F3F5F8'}[color],
            edge=color or '#C2CCD7',size=18)
        if i:
            arrow(ax,(xs[i-1]+184,913),(x,913))
    # Saved source-view features reach point refinement without re-extracting
    # an unrelated whole-image feature map.
    ax.plot([536,536,960,960],[872,842,842,872],color=BLUE,ls='--',lw=1.1)
    ax.text(748,823,'保留来源视图、查询编号与细粒度特征',ha='center',fontsize=15,color=BLUE)
    cards=[(20,BLUE,'① 多尺度 RGB-D 特征融合'),(518,PURPLE,'② 深度感知掩码细化'),
           (1016,GREEN,'③ 切片增强推理')]
    for x,color,title in cards:
        box(ax,x,85,464,700,'',fill='#FCFDFE',edge='#CFD8E2',lw=1.2)
        ax.text(x+22,748,title,fontsize=21,color=color,weight='bold')
        ax.plot([x+22,x+442],[726,726],color=color,lw=2)
    # Fusion: encoders run independently, four same-resolution feature pairs
    # feed DS/DC; fusion outputs do not feed later backbone stages.
    box(ax,48,651,170,48,'RGB',fill='#EAF3FA',edge=BLUE,size=18)
    box(ax,270,640,183,67,'深度 Z + 有效性\n逐视图归一化',fill='#F0F6FA',size=17)
    arrow(ax,(133,650),(133,604),BLUE)
    arrow(ax,(361,639),(361,604),BLUE)
    box(ax,48,532,170,71,'Swin-Large\nRGB 分支',fill='#EAF3FA',edge=BLUE,size=17)
    box(ax,270,518,183,85,'输入适配器\nSwin-Large\n深度分支',fill='#EAF3FA',edge=BLUE,size=15)
    arrow(ax,(133,531),(133,487),BLUE)
    arrow(ax,(361,517),(361,487),BLUE)
    ax.plot([93,405],[487,487],color=BLUE,lw=1.1)
    for k,stride in enumerate(('1/4','1/8','1/16','1/32')):
        arrow(ax,(93+k*104,487),(93+k*104,468),BLUE)
        box(ax,46+k*104,382,94,85,f'{stride}\nDS → DC',fill='#EDF4FA',edge=BLUE,size=13)
    ax.text(250,360,'DS：模态内交互   DC：跨模态交互',ha='center',fontsize=15,color=GREY)
    arrow(ax,(251,344),(251,310),BLUE)
    box(ax,66,235,366,75,'像素解码器 + 查询掩码解码器\nMask2Former',fill='#EAF3FA',edge=BLUE,size=17)
    arrow(ax,(251,234),(251,192),BLUE)
    box(ax,66,117,366,74,'粗掩码概率、类别分数\n与供点细化使用的细粒度特征',size=17)
    # Refinement details: metric depth is a separate input from normalized
    # encoder channels. Probability filtering acts over the full map.
    box(ax,549,653,402,48,'原始米制深度 Z + 有效性图',fill='#F1EDF7',edge=PURPLE,size=18)
    arrow(ax,(750,652),(750,619),PURPLE)
    ax.plot([549,533,533],[676,676,346],color=PURPLE,lw=1.05)
    arrow(ax,(533,346),(549,346),PURPLE)
    box(ax,549,539,402,80,'不确定点选择与特征采样\n细粒度特征 · 粗掩码 · 局部深度差',fill='#F8F5FB',edge=PURPLE,size=17)
    arrow(ax,(750,538),(750,502),PURPLE)
    box(ax,549,430,402,72,'点 MLP 预测掩码残差\n逐级细化：256 → 512 → 1024',fill='#F1EDF7',edge=PURPLE,size=17)
    arrow(ax,(750,429),(750,390),PURPLE)
    box(ax,549,303,402,87,'深度引导概率图细化\n5 × 5 邻域：空间与深度联合加权',fill='#F1EDF7',edge=PURPLE,size=17)
    arrow(ax,(750,302),(750,260),PURPLE)
    box(ax,549,195,402,65,'阈值 0.5 → 实例二值掩码',size=17)
    ax.text(750,151,'点头可学习；概率图细化无可学习参数',ha='center',fontsize=15,color=GREY)
    ax.text(750,119,'两步保留既有候选类别与置信度',ha='center',fontsize=15,color=GREY)
    # Actual crop geometry, with one 1024 full view and four 640 corner views.
    c=cases[202553004];rgb=c['rgb']
    ax.imshow(rgb,extent=(1050,1240,510,700),zorder=1)
    colors=['#0077BB','#EE7733','#009988','#CC79A7']
    for (l,t),color in zip(((0,0),(384,0),(0,384),(384,384)),colors):
        ax.add_patch(patches.Rectangle((1050+l/1024*190,700-(t+640)/1024*190),
            640/1024*190,640/1024*190,fill=False,edgecolor=color,linewidth=1.9,zorder=2))
    ax.text(1350,664,'整图 × 1\n640² 切片 × 4\n相邻重叠 40%',ha='center',va='center',fontsize=17,color=INK,linespacing=1.6)
    ax.text(1265,482,'每个视图缩放至 1024 × 1024',ha='center',fontsize=16,color=GREEN)
    arrow(ax,(1250,463),(1250,428),GREEN)
    box(ax,1045,351,404,77,'预测坐标恢复\n过滤接触切片内部边缘的候选',fill='#EAF5F0',edge=GREEN,size=17)
    arrow(ax,(1250,350),(1250,313),GREEN)
    box(ax,1045,238,404,75,'跨视图掩码去重\nmask IoU > 0.5；最多 100 个实例',fill='#EAF5F0',edge=GREEN,size=17)
    arrow(ax,(1250,237),(1250,199),GREEN)
    box(ax,1045,117,404,82,'保留来源视图信息\n利用对应特征进行后续点细化',fill='#F6FAF8',edge=GREEN,size=17)
    ax.text(750,46,'训练：分割基座 15 轮 → 冻结基座后训练点头 6 轮；切片与概率图细化仅用于推理。',
            ha='center',fontsize=17,color=INK)
    ax.text(750,14,'深度输入无需相机内参，不构建 XYZ；上方箭头表示实际执行顺序。',ha='center',fontsize=16,color=GREY)
    save(fig,'fig02_framework',vector=True)

def ablation(cases):
    chosen=[cases[202553004],cases[202609003]]
    titles=['RGB','GT','M0\nRGB 基线','M1\nRGB-D 融合','M2\n+ 点细化','M3\n+ 概率图细化','M4\n+ 切片推理']
    fig,axes=plt.subplots(4,7,figsize=(15.4,11.2))
    fig.subplots_adjust(left=.045,right=.995,top=.88,bottom=.13,wspace=.035,hspace=.38)
    for col,title in enumerate(titles):
        axes[0,col].set_title(title,fontsize=16,pad=12)
    for block,c in enumerate(chosen):
        row=block*2; b=c['bounds']
        values={'RGB':c['rgb'],'GT':c['overlays']['GT'],**c['overlays']}
        for col,stage in enumerate(('RGB','GT','M0','M1','M2','M3','M4')):
            if stage=='M0':
                for r in (row,row+1):
                    ax=axes[r,col];ax.set_facecolor('#F1F3F6')
                    ax.set_box_aspect(1)
                    ax.set_xticks([]);ax.set_yticks([])
                    ax.text(.5,.5,'Swin-Large\n训练中\n待最终权重预测',ha='center',va='center',
                            transform=ax.transAxes,fontsize=16,color=GREY,linespacing=1.65)
                    for sp in ax.spines.values(): sp.set_color('#D3D9E1')
                continue
            panel(axes[row,col],values[stage],b)
            if stage=='RGB':
                local=crop(c['rgb'],b)
            elif stage=='GT':
                local=crop(overlay(c['rgb'],[c['gt']],[GT_COLOR]),b)
            else:
                v=c['stages'][stage]
                local=crop(overlay(c['rgb'],[displayed_mask(v)],[PRED_COLOR]),b)
                axes[row+1,col].set_xlabel(iou_label(v),fontsize=13,labelpad=5)
            panel(axes[row+1,col],local)
        axes[row,0].set_ylabel(f"整图 {c['image_id']}",fontsize=14,labelpad=8)
        axes[row+1,0].set_ylabel(f"局部 · GT {int(c['gt'].sum())} px",fontsize=14,labelpad=8)
    fig.text(.52,.985,'统一 Swin-Large；RGB-D 分支使用无内参深度输入',ha='center',va='top',fontsize=19,color=INK)
    fig.text(.5,.045,'M1 → M2 → M3 → M4 为累计加入模块；所有预测展示阈值为 0.50。',ha='center',fontsize=16,color=GREY)
    fig.text(.5,.015,'图 3 为待补基线的工作稿；M2、M3 为固定权重的选定图像推理，不代表完整测试集消融已完成。',ha='center',fontsize=14,color='#9A6523')
    save(fig,'fig03_ablation_draft')

def cases_grid(cases, specs, stem):
    n=len(specs)
    height=n*3.05+1.35
    fig,axes=plt.subplots(n,6,figsize=(14.4,height),squeeze=False)
    fig.subplots_adjust(left=.025,right=.995,top=1-.75/height,bottom=1.05/height,
                        wspace=.035,hspace=.42)
    titles=['RGB 局部','目标实例 GT','M1：RGB-D 融合','M1 掩码误差','M4：完整 DPSeg','M4 掩码误差']
    for j,title in enumerate(titles):
        axes[0,j].set_title(title,fontsize=16,pad=27)
    for i,(iid,label) in enumerate(specs):
        c=cases[iid];gt=c['gt'];rgb=c['rgb'];b=c['bounds']
        p=c['stages']['M1'];q=c['stages']['M4']
        pm,qm=displayed_mask(p),displayed_mask(q)
        ims=[rgb,overlay(rgb,[gt],[GT_COLOR]),overlay(rgb,[pm],[PRED_COLOR]),
             error_map(rgb,gt,pm),overlay(rgb,[qm],[PRED_COLOR]),error_map(rgb,gt,qm)]
        for j,im in enumerate(ims): panel(axes[i,j],crop(im,b))
        scalebar(axes[i,0],b[2]-b[0],n=20)
        pos=axes[i,0].get_position()
        fig.text(pos.x0,pos.y1+.008,f"({chr(97+i)}) {label}    ID {iid}    GT 面积 {int(gt.sum())} px",fontsize=15,color=INK)
        for j,record in ((2,p),(4,q)):
            axes[i,j].set_xlabel(iou_label(record),fontsize=14,labelpad=5)
    handles=[patches.Patch(color=HIT/255,label='交集（正确覆盖）'),
             patches.Patch(color=MISS/255,label='漏分像素（GT − 预测）'),
             patches.Patch(color=EXTRA/255,label='多分像素（预测 − GT）')]
    fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,fontsize=15,bbox_to_anchor=(.5,.01))
    save(fig,stem)

def documentation(cases, stages):
    subprocess.run(['pdfunite',*[str(OUT/'figures'/f'{stem}.pdf') for stem,_ in FIGURES],
                    str(OUT/'DPSeg_报告插图.pdf')],check=True)
    metrics=[]
    for c in cases.values():
        for s,v in c['stages'].items():
            metrics.append(dict(image_id=c['image_id'],annotation_id=c['annotation_id'],kind=c['kind'],
                gt_area=int(c['gt'].sum()),stage=s,mask_iou=v['iou'],display_correspondence=v['iou']>=.5,prediction_index=v['index'],
                score=v['predictions'][v['index']]['score'] if v['index'] is not None else None,
                crop_bounds_xyxy=' '.join(map(str,c['bounds'])),score_threshold=THRESHOLD))
    with (OUT/'case_metrics.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(metrics[0]));writer.writeheader();writer.writerows(metrics)
    captions='''# DPSeg 报告图与图注

本组图片采用 **Swin-Large + 无内参深度表示**。源图为课程 ZIP 的本地 test 划分；GT 来自该 ZIP 的 COCO 标注，不是 Kaggle 私有测试集标注。所有实例预测均来自真实保存结果或固定权重推理，没有重画、修补或生成预测掩码。

## 图 1 密集小目标实例分割的典型挑战

每列依次给出整图、框选区域的 RGB 放大图及目标实例的真实掩码（GT）。案例展示遮挡后可见面积较小、相近外观下的实例边界，以及密集堆叠中的细长结构。面积按原图中目标实例的可见掩码像素数计算；比例尺单位为原图像素，不表示物理长度。

## 图 2 DPSeg 总体框架

输入为对齐的 RGB 图像、米制深度及有效性图。整图与四个重叠切片分别经过双分支 Swin-Large、多尺度 DS/DC 交互与 Mask2Former 掩码解码。候选恢复至原图坐标后，经内部切片边缘过滤及跨视图掩码去重，保留来源视图的查询编号和细粒度特征供深度感知点细化使用。随后在原图坐标下进行深度引导概率图细化，生成最终实例掩码。深度编码不依赖相机内参、不构建 XYZ；点细化和概率图细化另使用原始米制深度。上方为实际推理顺序，下方为三个模块的展开。多尺度融合与解码基于 DA-Fusion / Mask2Former 的实现，点细化借鉴 PointRend，切片策略参考 SAHI 思路并按 RGB-D 任务实现。

## 图 3 基线与消融定性对比（待补 RGB 基线的工作稿）

各列为 RGB、GT，以及 M0—M4 的预测。M0 为 RGB Mask2Former Swin-Large；M1 为多尺度 RGB-D 融合；M2 在 M1 上加入深度感知点细化；M3 在 M2 上加入深度引导概率图细化；M4 在 M3 上加入切片增强推理。每个场景同时给出整图和对应局部。整图中的实例色彩按与 GT 的重合关系尽量对齐，仅用于展示；局部显示所选目标 GT 及其对应预测：最大 IoU ≥ 0.5 时显示该预测，否则标为未匹配并仅报告真实最大 IoU，避免将邻近实例误当作该目标的预测。当前 M0 仍在训练，灰色区域是明确标注的占位，不能将本图当作完整消融的最终图。M2、M3 在同一冻结基座和已锁定的第 6 轮点头上补做了选定图像推理，未重新训练，也没有完整测试集的 M3 指标。

## 图 4 小目标漏检与边界改善、分割良好案例

从左至右为 RGB 局部、目标 GT、M1 预测及其误差图、完整 DPSeg（M4）预测及其误差图。案例包含小目标预测改善、两类局部轮廓改善和原本已分割较好的目标。灰色表示交集，橙色表示漏分像素，蓝色表示多分像素。此处的“多分像素”不等同于一个误检实例。案例按既有结果事后选择，用于解释现象，不能用于估计总体提升或将改善归因于某个单独模块。

## 图 5 失败案例及误差分析

图 (a) 展示密集细长结构中两种配置均未达到 IoU 0.5 对应标准的目标；图 (b) 展示完整方案的局部结果差于融合基线。配色和列含义与图 4 相同。由图像可以观察到遮挡、可见区域狭窄等困难，但仅凭这些图片不能确定退化发生于候选去重、点细化或概率图细化中的哪一步；原因分析应表述为待验证的解释，不作确定因果结论。

## 统一展示与解释规则

- 展示置信度阈值固定为 0.50，掩码阈值沿用各配置实际推理实现；完整 COCO 评测使用的是保存的低阈值候选，不以这些展示图片重算 AP。
- 局部 IoU 是所选 GT 与所有展示阈值以上预测的最大**完整 1024×1024 掩码 IoU**。它不是局部裁剪 IoU、边界 IoU，也不是 COCO 一对一匹配下的 AP / APₛ。
- 局部图采用 IoU ≥ 0.5 的实例对应显示规则。未匹配时，该目标的显示掩码置空，标注真实的最大 IoU；这不表示整张图没有任何预测，也不将相邻实例当作该目标的多分像素。原始预测始终保存在 RLE 和独立源文件中。
- 同一行所有配置使用同一 RGB、同一 GT、同一裁剪坐标。RGB 与二值掩码放大均使用最近邻显示，未修改边界。
- 各场景只有一个前景类别 component。不同实例的颜色是可视化标识，不代表不同类别。
- 图 4、5 仅比较 M1 与 M4；不能据此宣称中间每一步都提升。完整方案存在退化案例。
- 所有阶段使用同一份 15 轮 RGB-D 基座；M2—M4 使用同一份 6 轮点头。累计消融的展示顺序与完整模型的执行顺序并不相同：切片先生成候选，再执行两种细化。

## 使用方式

- `figures/*.png`：300 dpi，适合插入 Word / PPT。
- `figures/*.pdf`：适合排版。案例图保留栅格像素，文字为矢量。
- `DPSeg_报告插图.pdf`：五张图合并的预览文件，可直接逐页查看。
- `figures/fig02_framework.svg`：可编辑矢量框架图；中文字体为 Droid Sans Fallback，跨设备建议使用 PDF 或 PNG 保持外观。
- `source_assets/`：原始 RGB、原始 16 位深度、GT COCO、各配置压缩 RLE、整图叠加与选定实例的独立二值掩码。
- `case_metrics.csv`：选定实例 ID、掩码面积、各配置 IoU、预测索引、置信度和裁剪坐标。
- `manifest.json`：预测、权重、制图代码和输出文件的校验记录。
- `verification.json`：二值掩码、实例 IoU、图片分辨率与压缩包的检查结果。

## 复现

在仓库根目录依次运行（数据与模型使用本地现有文件）：

```bash
.venv/bin/python tools/select_report_figure_cases.py
# selected_cases.json 记录人工审阅后选择的图例；不要重新选择来追求更高分数。
CUDA_VISIBLE_DEVICES=4 .venv/bin/python tools/infer_report_figure_cases.py
.venv/bin/python tools/build_report_figures.py
```

RGB 基线完成后，应读取最终权重的对应图像预测更新图 3；不能用训练中间轮次或旧 Swin-Tiny 输出填补。图 1、2、4、5 不依赖该列。
'''
    (OUT/'图注与使用说明.md').write_text(captions)
    (OUT/'README.md').write_text(captions)
    # A portable, offline review page; it uses no servers, CDN, or scripts.
    cards=''.join(f'<section><h2>{i}. {title}</h2><a href="figures/{stem}.png"><img src="figures/{stem}_preview.jpg" alt="{title}"></a><p><a href="figures/{stem}.png">300 dpi PNG</a> · <a href="figures/{stem}.pdf">PDF</a>'+(' · <a href="figures/fig02_framework.svg">可编辑 SVG</a>' if i==2 else '')+'</p></section>' for i,(stem,title) in enumerate(FIGURES,1))
    (OUT/'index.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>DPSeg 报告图</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#243044;background:#f3f5f8;margin:0}main{max-width:1250px;margin:auto;padding:30px}section{background:white;border:1px solid #dce2e9;border-radius:10px;margin:28px 0;padding:20px}img{width:100%;height:auto}a{color:#176a9b}h1,h2{line-height:1.3}.note{padding:16px;background:#fff5de;border-left:4px solid #ca9138}</style><main><h1>DPSeg 报告图</h1><p>Swin-Large · 无内参深度输入 · 真实预测与 GT</p><p class="note">图 1、2、4、5 已完成。图 3 的 RGB Swin-Large 基线仍在训练，待补的一列已明确标注。M2、M3 仅补做了选定图像推理。</p><p><a href="图注与使用说明.md">图注与使用说明</a> · <a href="case_metrics.csv">局部实例指标</a> · <a href="manifest.json">来源与校验记录</a></p>'+cards+'</main></html>')
    combined=OUT/'DPSeg_报告插图.pdf'
    artifacts={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=digest(p))
               for p in [*(OUT/'figures').iterdir(),combined] if p.is_file()}
    source_trace={s:dict(path=str(p),sha256=digest(p)) for s,p in stages.items()}
    supplemental=json.loads((OUT/'supplemental_predictions/run.json').read_text())
    manifest=dict(figure_titles=dict(FIGURES),scope='Swin-Large + depth without camera intrinsics',
        split='historical local course test ZIP',archive_path=str(ARCHIVE),
        no_kaggle_hidden_labels=True,posthoc_case_selection=True,
        source_predictions=source_trace,supplemental_inference=supplemental,
        selected_cases=json.loads((OUT/'selected_cases.json').read_text()),
        generation_scripts={str(p.relative_to(ROOT)):digest(p) for p in
            (Path(__file__),ROOT/'tools/select_report_figure_cases.py',ROOT/'tools/infer_report_figure_cases.py',
             ROOT/'tools/check_report_figures.py')},
        source_asset_hashes={str(p.relative_to(OUT)):digest(p)
                             for p in (OUT/'source_assets').rglob('*') if p.is_file()},
        figures=artifacts,
        status={'fig01':'complete','fig02':'complete','fig03':'draft_missing_final_M0',
                'fig04':'complete','fig05':'complete'},
        limitations=['M0 final RGB Swin-Large training is unfinished',
                    'M2/M3 predictions here cover selected examples only',
                    'Qualitative posthoc cases do not estimate aggregate AP or prove causality',
                    'No model weights, training data beyond selected images, or credentials packaged'])
    write(OUT/'manifest.json',manifest)
    source_dir=OUT/'reproduction';source_dir.mkdir(exist_ok=True)
    for path in manifest['generation_scripts']:
        shutil.copy2(ROOT/path,source_dir/Path(path).name)
    zip_path=OUT.parent/'DPSeg_report_figures_SwinL_v1.zip'
    with ZipFile(zip_path,'w',compression=ZIP_DEFLATED,compresslevel=6) as archive:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and not p.name.startswith('selection_'):
                archive.write(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)))
    print(f'Package: {zip_path} ({zip_path.stat().st_size/2**20:.1f} MiB)',flush=True)

def finalize_exports():
    """Repackage verified existing exports without repainting every source mask."""
    cases={}
    rows=json.loads((OUT/'selected_cases.json').read_text())
    stages={**SOURCES,**{s:OUT/'supplemental_predictions'/f'{s}.jsonl' for s in ('M2','M3')}}
    for row in rows:
        folder=OUT/'source_assets'/str(row['image_id'])
        focus=json.loads((folder/'focus.json').read_text())
        stages_info={}
        for stage,values in focus['stages'].items():
            stages_info[stage]=dict(index=values['prediction_index'],iou=values['mask_iou'],
                predictions=json.loads((folder/f'{stage}_predictions.json').read_text())['predictions'])
        cases[row['image_id']]=dict(row,gt=np.array(Image.open(folder/'focus_gt_mask.png'))>0,
                                   stages=stages_info)
    documentation(cases,stages)

def main():
    cases,stages=prepare()
    print('Exported source assets',flush=True)
    challenges(cases); print('Figure 1 done',flush=True)
    framework(cases); print('Figure 2 done',flush=True)
    ablation(cases); print('Figure 3 draft done',flush=True)
    cases_grid(cases,[(202553004,'小目标预测补回'),(202612002,'局部轮廓改善'),
                      (202609003,'窄可见区域的边界改善'),(202340002,'分割良好的小目标')],
               'fig04_improvement_cases'); print('Figure 4 done',flush=True)
    cases_grid(cases,[(202358007,'密集细长结构仍未准确分割'),(202553006,'完整方案出现局部退化')],
               'fig05_failure_cases'); print('Figure 5 done',flush=True)
    documentation(cases,stages)

if __name__=='__main__':
    if '--finalize' in sys.argv:
        finalize_exports()
    else:
        main()
