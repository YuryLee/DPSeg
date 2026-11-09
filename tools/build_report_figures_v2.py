#!/usr/bin/env python3
"""Compact paper figures with genuine near-overhead RGB-D examples.

Layout only: no training, result retouching, or invented mask probabilities.
"""
from __future__ import annotations
import csv
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
os.environ.setdefault('MPLCONFIGDIR', '/tmp/dpseg-report-matplotlib')
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import patches, font_manager
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion
from pycocotools import mask as cm
from rgbd_seg.course_archive import CourseArchive
from tools.select_report_figure_cases import ARCHIVE, RUN, SOURCES, load_records, digest, write
from tools.build_report_figures import best, decode, error_map

OUT=ROOT/'outputs/report_figures_swinl_v2'
FIG=OUT/'figures'
font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams.update({'font.family':['DejaVu Sans','Droid Sans Fallback'],'font.size':13,
 'axes.unicode_minus':False,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
PEACH='#F6E6D7';YELLOW='#FFF2C5';GREEN='#E3EFD8';PURPLE='#DCCFE7';BLUE='#C6D5E0'
INK='#202020';GRAY='#5B6065';TEAL='#009988';PRED='#0077BB';ORANGE='#D8841F'
COLORS=['#0077BB','#EE7733','#009988','#CC79A7','#56B4E9','#E6AB02','#7655A3','#607D8B']
STAGES=('M1','M2','M3','M4')
NAMES={'M0':'RGB Mask2Former','M1':'＋RGB-D 融合','M2':'＋深度感知点细化',
       'M3':'＋深度引导概率图细化','M4':'＋切片增强推理'}
TITLES=[('fig00_teaser','DPSeg：检出小目标，细化实例边界'),
 ('fig01_challenges','密集小目标实例分割的典型挑战'),('fig02_framework','DPSeg 总体框架'),
 ('fig03_ablation_draft','基线与累计模块消融定性对比（RGB 基线待补）'),
 ('fig04_improvement_cases','小目标漏检与边界改善、分割良好案例'),
 ('fig05_failure_cases','失败案例及误差分析')]

def color(c):return np.asarray(matplotlib.colors.to_rgb(c))*255

def overlay(rgb,masks,colors,alpha=.30):
    """Identical mask pixels; alpha and contour are illustration styling only."""
    out=np.array(rgb,dtype=np.float32).copy()
    for j in sorted(range(len(masks)),key=lambda i:-int(masks[i].sum())):
        m=masks[j];ys,xs=np.where(m)
        if not len(xs):continue
        l,r=max(0,xs.min()-1),min(m.shape[1],xs.max()+2)
        t,b=max(0,ys.min()-1),min(m.shape[0],ys.max()+2)
        local=m[t:b,l:r];view=out[t:b,l:r];c=color(colors[j])
        view[local]=view[local]*(1-alpha)+c*alpha
        edge=local & ~binary_erosion(local)
        view[edge]=c
    return np.uint8(np.clip(out,0,255))

def overlay_all(rgb,anns,preds=None):
    if preds is None:return overlay(rgb,[decode(a['segmentation']) for a in anns],
                                   [COLORS[j%len(COLORS)] for j in range(len(anns))])
    pred=[p for p in preds if p['score']>=.5]
    if not pred:return rgb
    mat=cm.iou([p['segmentation'] for p in pred],[a['segmentation'] for a in anns],[0]*len(anns))
    colors=[COLORS[int(r.argmax())%len(COLORS)] if r.max()>=.1 else '#E6AB02' for r in mat]
    return overlay(rgb,[decode(p['segmentation']) for p in pred],colors)

def tight_bounds(bbox):
    x,y,w,h=bbox;side=max(72,int(max(w,h)*1.45));cx=x+w/2;cy=y+h/2
    l=int(max(0,min(1024-side,cx-side/2)));t=int(max(0,min(1024-side,cy-side/2)))
    return [l,t,l+side,t+side]

def prepare():
    rows=json.loads((OUT/'selected_cases.json').read_text());ids={r['image_id'] for r in rows}
    sources={**SOURCES,**{s:OUT/'supplemental_predictions'/f'{s}.jsonl' for s in ('M2','M3')}}
    records={s:load_records(p,ids) for s,p in sources.items()}
    metrics=[];images={};cases={}
    with CourseArchive(ARCHIVE,'test') as data,ZipFile(ARCHIVE) as archive:
        anns=defaultdict(list)
        for a in data.coco_ground_truth()['annotations']:
            if a['image_id'] in ids:anns[a['image_id']].append(a)
        for iid in sorted(ids):
            idx=data.image_ids.index(iid);meta=data.images[idx]
            folder=OUT/'source_assets'/str(iid);folder.mkdir(parents=True,exist_ok=True)
            rgb=np.array(data.load_rgb(idx));depth,valid=data.load_depth(idx)
            lo,hi=np.quantile(depth[valid],[.01,.99]);gray=np.clip((depth-lo)/(hi-lo+1e-8),0,1)
            gray[~valid]=0;depth_preview=np.repeat(np.uint8(gray*255)[...,None],3,2)
            for src,dst in [('file_name','rgb.png'),('depth_file_name','depth_raw.png')]:
                (folder/dst).write_bytes(archive.read(meta[src]))
            Image.fromarray(depth_preview).save(folder/'depth_preview.png')
            write(folder/'ground_truth.json',{'image_id':iid,'annotations':anns[iid]})
            write(folder/'metadata.json',{k:meta[k] for k in ['id','file_name','depth_file_name','depth_scale_m','view','camera']})
            visuals={'GT':overlay_all(rgb,anns[iid])}
            for s in STAGES:
                write(folder/f'{s}_predictions.json',records[s][iid])
                visuals[s]=overlay_all(rgb,anns[iid],records[s][iid])
            for s,im in visuals.items():Image.fromarray(im).save(folder/f'{s}_overlay.png')
            images[iid]=dict(rgb=rgb,depth=depth_preview,visuals=visuals,anns=anns[iid],predictions={s:records[s][iid] for s in STAGES})
            print('prepared image',iid,flush=True)
        for row in rows:
            iid=row['image_id'];ann=next(a for a in anns[iid] if a['id']==row['annotation_id'])
            folder=OUT/'focus_assets'/row['case_key'];folder.mkdir(parents=True,exist_ok=True)
            gt=decode(ann['segmentation']);rgb=images[iid]['rgb'];row['bounds']=tight_bounds(row['bbox'])
            info=dict(row,image=images[iid],gt=gt,stages={})
            Image.fromarray(gt.astype('uint8')*255).save(folder/'GT.png')
            for s in STAGES:
                index,iou,mask=best(records[s][iid],ann);display=mask if iou>=.5 else np.zeros_like(mask)
                info['stages'][s]=dict(index=index,iou=iou,mask=mask,display=display)
                Image.fromarray(mask.astype('uint8')*255).save(folder/f'{s}_best.png')
                Image.fromarray(display.astype('uint8')*255).save(folder/f'{s}_display.png')
                metrics.append(dict(case_key=row['case_key'],image_id=iid,annotation_id=row['annotation_id'],
                    stage=s,gt_area=int(gt.sum()),best_iou=iou,prediction_index=index,matched=iou>=.5))
            write(folder/'case.json',row)
            cases[row['case_key']]=info
    with (OUT/'case_metrics.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
    write(OUT/'selected_cases.json',rows)
    write(OUT/'sources.json',{s:dict(path=str(p),sha256=digest(p)) for s,p in sources.items()})
    return cases

def cached():
    images={};cases={}
    for row in json.loads((OUT/'selected_cases.json').read_text()):
        iid=row['image_id'];folder=OUT/'source_assets'/str(iid)
        if iid not in images:
            images[iid]=dict(rgb=np.array(Image.open(folder/'rgb.png')),depth=np.array(Image.open(folder/'depth_preview.png')),
                visuals={s:np.array(Image.open(folder/f'{s}_overlay.png')) for s in ('GT',*STAGES)},
                predictions={s:json.loads((folder/f'{s}_predictions.json').read_text()) for s in STAGES})
        ff=OUT/'focus_assets'/row['case_key'];metrics=list(csv.DictReader((OUT/'case_metrics.csv').open()))
        rec=dict(row,image=images[iid],gt=np.array(Image.open(ff/'GT.png'))>0,stages={})
        for s in STAGES:
            m=next(r for r in metrics if r['case_key']==row['case_key'] and r['stage']==s)
            rec['stages'][s]=dict(iou=float(m['best_iou']),mask=np.array(Image.open(ff/f'{s}_best.png'))>0,
                                  display=np.array(Image.open(ff/f'{s}_display.png'))>0)
        cases[row['case_key']]=rec
    return cases

def cut(im,bounds):
    l,t,r,b=bounds;return np.array(im)[t:b,l:r]

def focus(c,stage='GT',mode='overlay'):
    rgb=c['image']['rgb'];mask=c['gt'] if stage=='GT' else c['stages'][stage]['display']
    if mode=='error':im=error_map(rgb,c['gt'],mask)
    elif mode=='binary':im=np.repeat((mask.astype('uint8')*255)[...,None],3,2)
    else:im=overlay(rgb,[mask],[TEAL if stage=='GT' else PRED],.40)
    return cut(im,c['bounds'])

def iou(c,s):
    r=c['stages'][s];return f"IoU {r['iou']:.3f}" if r['iou']>=.5 else '未匹配'

def image_ax(fig,rect,im,title=None,fs=13):
    ax=fig.add_axes(rect);ax.imshow(im,interpolation='nearest');ax.set_xticks([]);ax.set_yticks([])
    for sp in ax.spines.values():sp.set_color('#C3C3C3');sp.set_linewidth(.6)
    if title:ax.set_title(title,fontsize=fs,pad=5)
    return ax

def roi(ax,box,label,c=ORANGE):
    l,t,r,b=box
    ax.add_patch(patches.Rectangle((l,t),r-l,b-t,fill=False,edgecolor='white',lw=2.7))
    ax.add_patch(patches.Rectangle((l,t),r-l,b-t,fill=False,edgecolor=c,lw=1.5))
    ax.text(l,t-8,label,color=c,fontsize=12,fontweight='bold',bbox=dict(fc='white',ec='none',pad=.8))

def save(fig,stem):
    FIG.mkdir(exist_ok=True)
    for ext in ('png','pdf','svg'):fig.savefig(FIG/f'{stem}.{ext}',dpi=300)
    fig.savefig(FIG/f'{stem}_preview.jpg',dpi=110)
    plt.close(fig)

def teaser(cases):
    a,b=cases['teaser_miss'],cases['teaser_boundary']
    fig=plt.figure(figsize=(13.8,6.8))
    fig.text(.02,.963,'DPSeg：检出小目标，细化实例边界',fontsize=21,fontweight='bold')
    ax=image_ax(fig,[.015,.215,.345,.700],a['image']['rgb'])
    roi(ax,a['bounds'],'A');roi(ax,b['bounds'],'B',PRED)
    fig.text(.018,.177,'合成元器件场景（接近俯视）',fontsize=13)
    for j,(name,s) in enumerate([('真实掩码（GT）','GT'),('RGB-D 融合基线','M1'),('DPSeg','M4')]):
        x=.391+j*.201
        fig.text(x+.091,.912,name,ha='center',fontsize=14,fontweight='bold')
        for y,c,label in [(.536,a,'A  漏检改善'),(.153,b,'B  边界改善')]:
            axt=image_ax(fig,[x,y,.178,.361],focus(c,s))
            note=label if s=='GT' else iou(c,s)
            axt.text(.035,.045,note,transform=axt.transAxes,fontsize=11,color=INK,
                     bbox=dict(fc='white',ec='none',alpha=.91,pad=2))
    for x,w,label,fc in [(.02,.30,'多尺度 RGB-D 特征融合',PEACH),(.35,.30,'深度感知掩码细化',GREEN),(.68,.30,'切片增强推理',PURPLE)]:
        fig.patches.append(patches.FancyBboxPatch((x,.023),w,.066,boxstyle='round,pad=0.003,rounding_size=0.008',
                 transform=fig.transFigure,fc=fc,ec='#8E8E8E',lw=.8))
        fig.text(x+w/2,.055,label,ha='center',va='center',fontsize=13)
    save(fig,'fig00_teaser')

def challenges(cases):
    fig=plt.figure(figsize=(13.5,5.2));keys=['teaser_miss','white_miss','green_boundary']
    titles=['可见像素少，目标容易漏检','外观相近，邻接实例难以区分','遮挡与细长结构增加边界误差']
    for j,key in enumerate(keys):
        c=cases[key];x=.012+j*.332
        ax=image_ax(fig,[x,.20,.321,.795],c['image']['rgb']);roi(ax,c['bounds'],'')
        image_ax(fig,[x+.185,.205,.13,.337],focus(c,'GT'))
        fig.text(x+.16,.12,titles[j],ha='center',fontsize=13)
    fig.text(.015,.035,'框选区域为局部放大；绿色轮廓为所关注实例的真实掩码。',fontsize=12,color=GRAY)
    save(fig,'fig01_challenges')

def framework(cases):
    c=cases['teaser_boundary'];im=c['image'];W,H=1800,1010
    trace=OUT/'framework_trace'
    actual={s:json.loads((trace/f'{s}.json').read_text()) for s in ('merged','point','final')}
    anns=json.loads((OUT/'source_assets'/str(c['image_id'])/'ground_truth.json').read_text())['annotations']
    merged_overlay=overlay_all(im['rgb'],anns,actual['merged'])
    actual_focus={s:cut(overlay(im['rgb'],[decode(records[72]['segmentation'])],[PRED],.40),c['bounds'])
                  for s,records in actual.items()}
    prob=np.load(trace/'point_probability_72.npy')
    prob_image=(plt.get_cmap('viridis')(cut(prob,c['bounds']))[:,:,:3]*255).astype('uint8')
    fig=plt.figure(figsize=(18,10.1));ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
    def text(x,y,t,size=13,**kw):ax.text(x,y,t,ha='center',va='center',fontsize=size,color=INK,**kw)
    def box(x,y,w,h,t='',fc='white',size=12,dashed=False):
        ax.add_patch(patches.FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=4',
            fc=fc,ec='#555',lw=1,linestyle=(0,(4,3)) if dashed else '-',zorder=1))
        if t:text(x+w/2,y+h/2,t,size)
    def arrow(points,dashed=False,c=INK,lw=1.2):
        for p,q in zip(points[:-2],points[1:-1]):ax.plot([p[0],q[0]],[p[1],q[1]],color=c,lw=lw,ls='--' if dashed else '-',zorder=2)
        ax.add_patch(patches.FancyArrowPatch(points[-2],points[-1],arrowstyle='-|>',mutation_scale=11,
                lw=lw,color=c,linestyle='--' if dashed else '-',zorder=2))
    def circle(x,y,r,t,fc=BLUE,size=12):
        ax.add_patch(patches.Circle((x,y),r,fc=fc,ec='#555',lw=.9,zorder=3));text(x,y,t,size)
    def photo(x,y,w,h,img,label=None,fs=11):
        a=image_ax(fig,[x/W,1-(y+h)/H,w/W,h/H],img)
        if label:text(x+w/2,y-13,label,fs)
        return a

    text(900,24,'DPSeg 总体框架',20,fontweight='bold')
    photo(18,83,125,125,im['rgb'],'RGB')
    photo(18,246,125,125,im['depth'],'深度 Z + 有效性',11)
    text(80,393,'无需相机内参',10)
    arrow([(148,145),(171,145),(171,202),(194,202)])
    arrow([(148,308),(171,308),(171,202)])
    box(195,62,178,341,dashed=True)
    text(284,83,'切片增强推理',13,fontweight='bold')
    v=photo(215,108,138,138,im['rgb'])
    for j,(x,y) in enumerate([(0,0),(384,0),(0,384),(384,384)]):
        v.add_patch(patches.Rectangle((x,y),640,640,fill=False,ec=COLORS[j],lw=.9))
    text(284,264,'整图 + 4 个重叠切片',10)
    for j,(x,y) in enumerate([(0,0),(384,0),(0,384),(384,384)]):
        photo(212+j*38,285,34,34,im['rgb'][y:y+640,x:x+640])
    text(284,342,'RGB / 深度同步裁剪\n每个视图缩放至 1024²',10)
    text(284,381,'逐视图深度归一化与适配',9)
    arrow([(374,207),(400,207)])
    box(400,160,182,94,'多尺度 RGB-D 融合\n与实例掩码解码',PEACH,13)
    text(491,280,'双分支 Swin-Large',11)
    box(420,304,142,40,'缓存来源视图特征',BLUE,10)
    arrow([(410,254),(410,324),(420,324)],True)
    arrow([(582,207),(622,207)])
    box(622,160,166,94,'坐标恢复与边缘过滤\n跨视图掩码去重',PURPLE,12)
    photo(660,280,90,90,merged_overlay)
    text(705,389,'合并后的候选',10)
    arrow([(788,207),(842,207)])
    box(825,111,484,298,dashed=True)
    text(1067,131,'深度感知掩码细化',13,fontweight='bold')
    box(842,167,202,80,'深度感知点细化\n不确定点残差修正',GREEN,12)
    box(1093,167,198,80,'深度引导概率图细化\n局部深度加权',YELLOW,12)
    arrow([(1044,207),(1093,207)])
    for x,s,label in [(846,'merged','粗掩码'),(1004,'point','点细化'),(1162,'final','概率图细化')]:
        photo(x,282,105,105,actual_focus[s],label)
    text(1124,425,'同一候选的真实局部结果',10)
    arrow([(562,324),(600,324),(600,449),(977,449),(977,249)],True,c='#587282')
    text(774,441,'来源视图细粒度特征',10)
    arrow([(1292,207),(1342,207)])
    photo(1345,107,236,236,im['visuals']['M4'],'最终实例预测',13)
    text(1463,366,'原图坐标下的实例掩码',11)
    for j,key in enumerate(['teaser_miss','teaser_boundary','good']):
        photo(1643,83+j*102,86,86,focus(cases[key],'M4','binary'))
    text(1686,397,'独立掩码（局部）',11)
    arrow([(1582,221),(1636,221)])

    # Four independent Swin stages, with per-scale fusion outputs to the decoder.
    box(18,476,1137,513,dashed=True)
    text(586,498,'多尺度 RGB-D 融合与解码（各视图共享参数）',14,fontweight='bold')
    text(49,565,'RGB',10);text(49,820,'深度',10)
    for i in range(4):
        x=111+205*i
        box(x-37,545,74,40,f'Stage {i+1}',PEACH,11)
        box(x-37,800,74,40,f'Stage {i+1}',PEACH,11)
        if i<3:
            arrow([(x+37,565),(x+168,565)])
            arrow([(x+37,820),(x+168,820)])
        box(x-22,625,46,30,'DS',YELLOW,11);box(x-22,715,46,30,'DS',YELLOW,11)
        box(x+57,625,46,30,'DC',GREEN,11);box(x+57,715,46,30,'DC',GREEN,11)
        arrow([(x,585),(x,625)]);arrow([(x,800),(x,745)])
        arrow([(x+24,640),(x+57,640)]);arrow([(x+24,730),(x+57,730)])
        arrow([(x+24,647),(x+57,721)]);arrow([(x+24,723),(x+57,649)])
        circle(x+133,685,12,'C','#EAEAEA',10)
        arrow([(x+103,640),(x+133,640),(x+133,672)])
        arrow([(x+103,730),(x+133,730),(x+133,698)])
        circle(x+133,773,14,f'$f_{i+1}$',BLUE,12)
        arrow([(x+145,685),(x+157,685),(x+157,773),(x+147,773)])
        # Features go to the pixel decoder, not back into the encoder streams.
        yy=862+i*15;xx=946+i*5
        arrow([(x+133,787),(x+133,yy),(xx,yy),(xx,757),(989+i*7,757),(989+i*7,737)])
        text(x+25,605,f'1/{4*2**i}',9)
    box(970,606,64,130,'Pixel\nDecoder',PURPLE,10)
    box(1064,606,65,130,'Mask\nDecoder',BLUE,10)
    arrow([(1035,671),(1064,671)]);arrow([(1096,555),(1096,605)])
    text(1096,541,'Queries',11);arrow([(1130,671),(1147,671)])
    text(587,937,'DS：可变形自注意力    DC：双向可变形交叉注意力    C：拼接、投影与残差融合',10)
    text(587,965,'两路 Swin 编码器分别提取四级特征；各级融合结果送入 Mask2Former 解码器。',10)

    box(1175,476,607,513,dashed=True)
    text(1478,498,'深度感知掩码细化',14,fontweight='bold')
    text(1469,533,'不确定点采样 → 点级残差预测',11)
    for y,t,fc in [(564,'粗掩码 logit',PURPLE),(615,'细粒度特征',BLUE),(666,'局部深度与有效性',GREEN)]:
        box(1192,y,153,36,t,fc,10)
        arrow([(1345,y+18),(1371,y+18),(1371,635),(1391,635)])
    box(1392,579,84,112,'点 MLP',GREEN,12)
    circle(1521,635,14,'+',YELLOW,15)
    arrow([(1477,635),(1506,635)])
    arrow([(1266,564),(1266,548),(1521,548),(1521,621)])
    arrow([(1536,635),(1582,635)])
    photo(1584,571,141,141,actual_focus['point'],'点细化结果',11)
    text(1478,737,'逐级提高掩码分辨率：256 → 512 → 1024',11)
    ax.plot([1194,1762],[761,761],color='#BBBBBB',lw=.8)
    text(1478,785,'在原图坐标下进行概率图深度细化',11)
    photo(1193,833,95,95,cut(im['depth'],c['bounds']),'局部深度',10)
    photo(1305,833,95,95,prob_image,'点细化概率图',10)
    arrow([(1401,880),(1434,880)])
    box(1435,838,132,84,'5 × 5 邻域\n深度加权平均',YELLOW,10)
    arrow([(1240,929),(1240,950),(1501,950),(1501,924)])
    arrow([(1568,880),(1600,880)])
    photo(1601,833,95,95,actual_focus['final'],'细化后掩码',10)
    text(1478,973,'概率图：深紫为 0，亮黄为 1；输出以二值轮廓显示。',10)
    save(fig,'fig02_framework')

def ablation(cases):
    keys=['teaser_miss','teaser_boundary','green_boundary'];labels=['漏检改善','边界改善','细长结构']
    fig=plt.figure(figsize=(15.4,7.7));xs=[.048,.205,.362,.519,.676,.833];w=.150;h=w*15.4/7.7
    names=['GT','RGB Mask2Former','＋RGB-D 融合','＋点细化','＋概率图细化','＋切片增强推理']
    for j,name in enumerate(names):fig.text(xs[j]+w/2,.973,name,ha='center',fontsize=12)
    for i,key in enumerate(keys):
        c=cases[key];y=.645-i*.312
        fig.text(.02,y+h/2,labels[i],rotation=90,va='center',ha='center',fontsize=12)
        for j,s in enumerate(['GT','M0',*STAGES]):
            if s=='M0':
                ax=fig.add_axes([xs[j],y,w,h]);ax.set_facecolor('#F4F4F4');ax.set_xticks([]);ax.set_yticks([])
                for sp in ax.spines.values():sp.set_color('#CCC')
                ax.text(.5,.5,'Swin-Large\n训练中\n完成后补入',ha='center',va='center',transform=ax.transAxes,fontsize=12,color=GRAY)
            else:
                ax=image_ax(fig,[xs[j],y,w,h],focus(c,s))
                if s!='GT':ax.text(.03,.04,iou(c,s),transform=ax.transAxes,fontsize=10,
                                     bbox=dict(fc='white',ec='none',alpha=.9,pad=2))
    save(fig,'fig03_ablation_draft')

def improvement(cases):
    keys=['white_miss','teaser_boundary','good'];labels=['小目标漏检改善','掩码边界改善','分割良好案例']
    fig=plt.figure(figsize=(10.9,8.1));xs=[.055,.29,.525,.76];w=.227;h=w*10.9/8.1
    for j,label in enumerate(['局部 RGB','真实掩码（GT）','RGB-D 融合基线','DPSeg']):fig.text(xs[j]+w/2,.97,label,ha='center',fontsize=13)
    for i,key in enumerate(keys):
        c=cases[key];y=.641-i*.315
        # Fixed tight square crops; row height fits without artificial image gutters.
        hh=.298;ww=hh*8.1/10.9;shift=(w-ww)/2
        fig.text(.02,y+hh/2,labels[i],rotation=90,va='center',ha='center',fontsize=12)
        for j,s in enumerate(['RGB','GT','M1','M4']):
            im=cut(c['image']['rgb'],c['bounds']) if s=='RGB' else focus(c,s)
            ax=image_ax(fig,[xs[j]+shift,y,ww,hh],im)
            if s in STAGES:ax.text(.03,.04,iou(c,s),transform=ax.transAxes,fontsize=10,bbox=dict(fc='white',ec='none',alpha=.9,pad=2))
    save(fig,'fig04_improvement_cases')

def failures(cases):
    fig=plt.figure(figsize=(10.1,5.6));keys=['failure','regression'];xs=[.052,.289,.526,.763]
    for j,label in enumerate(['局部 RGB','真实掩码（GT）','RGB-D 融合基线','DPSeg']):fig.text(xs[j]+.11,.972,label,ha='center',fontsize=13)
    for i,key in enumerate(keys):
        c=cases[key];y=.51-i*.452;h=.414;w=h*5.6/10.1
        fig.text(.021,y+h/2,'遮挡下仍未检出' if i==0 else '原有匹配丢失',rotation=90,ha='center',va='center',fontsize=12)
        for j,s in enumerate(['RGB','GT','M1','M4']):
            ax=image_ax(fig,[xs[j],y,w,h],cut(c['image']['rgb'],c['bounds']) if s=='RGB' else focus(c,s))
            if s in STAGES:ax.text(.03,.04,iou(c,s),transform=ax.transAxes,fontsize=10,bbox=dict(fc='white',ec='none',alpha=.9,pad=2))
    save(fig,'fig05_failure_cases')

def package():
    repro=OUT/'reproduction';repro.mkdir(exist_ok=True)
    scripts=['build_report_figures_v2.py','select_report_figures_v2.py','infer_report_figures_v2.py','trace_report_framework_v2.py','check_report_figures_v2.py',
             'build_report_figures.py','select_report_figure_cases.py','infer_report_figure_cases.py']
    for name in scripts:shutil.copyfile(ROOT/'tools'/name,repro/name)
    subprocess.run(['pdfunite',*[str(FIG/f'{stem}.pdf') for stem,_ in TITLES],str(OUT/'DPSeg_报告插图_v2.pdf')],check=True)
    captions='''# DPSeg 报告插图 v2

本版采用白底、浅色模块、黑色箭头和虚线分组；新增 teaser，使用接近俯视的真实案例。图中不显示样本 ID，来源见 selected_cases.json 与 source_assets。保留上一版，不覆盖原文件。

## 图注

**Teaser：DPSeg：检出小目标，细化实例边界。** 左侧为密集元器件场景，框选区域对应右侧局部放大。A 展示小目标由未匹配到成功分割，B 展示掩码轮廓改善。RGB-D 融合基线与 DPSeg 使用相同的 Swin-Large 基座。绿色为 GT，蓝色为所关注目标的预测。例图为真实结果，无人工修改掩码。

**图 1：密集小目标实例分割的典型挑战。** 展示目标可见像素少、外观相近和细长结构等情况。插入图叠加对应 GT，用于定位所讨论的目标。

**图 2：DPSeg 总体框架。** 上方依次为同步 RGB-D 裁剪、各视图共享的 RGB-D 基座、坐标恢复与候选过滤去重、来源视图中的点细化，以及原图坐标中的概率图细化。左下展开四级双分支 Swin-Large、DS/DC 融合与 Mask2Former 解码器；右下展开点级残差预测和局部深度加权。编码器各自逐级提取特征，融合输出送入解码器，不反馈至下一编码阶段。网络结构为示意图；RGB、深度、候选掩码和细化小图均来自同一场景的真实数据及完整流水线。细化插图跟踪同一候选，依次展示合并后粗掩码、点细化结果、概率图细化结果；概率热图来自实际浮点预测。最终结果已与原 M4 实验逐像素核对一致。框架图不使用 GT 辅助推理。深度仅用 Z 与有效性，不用内参或 XYZ；相机姿态只用于筛选俯视展示视角。深度图亮度为显示归一化，不等同于网络输入归一化。

**图 3：基线与累计模块消融定性对比（草稿）。** 按 GT、RGB Mask2Former、+RGB-D 融合、+点细化、+概率图细化、+切片增强推理的顺序排列。RGB Swin-Large 仍在训练，保留显式占位。M2/M3 是冻结权重对选定案例的补充推理，不是新训练或完整测试集评估。

**图 4：小目标漏检与边界改善、分割良好案例。** 分别展示新增目标匹配、轮廓改善及两种方法均能较好分割的案例，避免只展示困难案例。

**图 5：失败案例及误差分析。** 展示完整方法仍未能匹配目标，以及基线原本匹配但完整方法未匹配的情况。遮挡和细长可见区域可从图中观察；具体误差归因仍需逐阶段分析，不能仅凭最终图片认定某个模块导致失败。

## 使用与真实性说明

- 当前配置：Swin-Large + depth（无内参），基座 15 轮，冻结基座后点头 6 轮。
- 例图来自课程历史本地 ZIP 的 test 标注，不代表获得 Kaggle 隐藏评分标签。
- 全部选例视角约 85°，属于接近俯视。可视化后验选例用于解释现象，不用于估计总体收益。
- 图中显示阈值为 score ≥ 0.50；正式评测的预测保留阈值为 0.01，两者不同。
- 局部 IoU 为整张 1024×1024 掩码相对一个选定 GT 的最大 IoU，不是裁剪 IoU，也不是 COCO AP 的一对一匹配。
- 最大 IoU < 0.5 时标为“未匹配”，对应局部图不叠加该实例的预测。完整实例预测和实际最大重叠掩码仍保存在 source_assets 与 focus_assets，避免将另一邻接实例错误画成当前目标的预测。
- 框架图的切片为四个 640×640 角点视图，重叠 256 像素，并保留整图；去重使用跨视图 mask IoU > 0.5，最多 100 个候选。
- 点细化使用不确定性采样、细粒度特征、局部深度差和有效性，MLP 预测 logit 残差。概率图细化使用一次 5×5 深度加权，不额外叠加旧二值形态处理。
- PNG 为 300 dpi；PDF 可直接插入报告；SVG 保留文字、模块和箭头的可编辑性，实际照片和掩码以嵌入位图保存。
- 本版图 4 未声称“误检数量降低”：图示是漏检、轮廓和分割良好案例。需要实例误检分析时，应依据一对一 GT 匹配单独统计。
'''
    (OUT/'图注与使用说明.md').write_text(captions)
    (OUT/'README.md').write_text('# DPSeg 报告插图 v2\n\n打开 `index.html` 浏览全部插图，或使用 `DPSeg_报告插图_v2.pdf`。\n\n'+
        '\n'.join(f'- `{stem}`：{title}' for stem,title in TITLES)+'\n\nPNG / PDF / 可编辑 SVG 均在 figures。真实源图与原始深度、GT、预测 RLE 见 source_assets。详细图注和使用限制见 图注与使用说明.md。\n\n'+
        '## 复现\n\n在原仓库根目录执行 `.venv/bin/python tools/build_report_figures_v2.py --cached` 可使用已保存的源图和预测重新排版，无需 GPU。执行 `.venv/bin/python tools/check_report_figures_v2.py` 核验图中掩码和对应 IoU。reproduction 保存源代码快照；重新推理还需要原始课程 ZIP 和既有模型权重，压缩包不包含完整数据集或模型。\n')
    (OUT/'index.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>DPSeg 报告插图 v2</title><style>body{max-width:1500px;margin:20px auto;font:16px sans-serif;color:#222}img{width:100%;height:auto}section{margin-bottom:28px}a{color:#176da0}</style><h1>DPSeg 报告插图 v2</h1><p>真实俯视案例 · 紧凑排版 · PNG / PDF / SVG</p>'+''.join(f'<section><h2>{title}</h2><p><a href="figures/{stem}.png">PNG</a> · <a href="figures/{stem}.pdf">PDF</a> · <a href="figures/{stem}.svg">SVG</a></p><img src="figures/{stem}_preview.jpg"></section>' for stem,title in TITLES)+'</html>')
    manifest=dict(configuration='Swin-Large + depth, no camera intrinsics',qualitative_only=True,
                  missing='M0 RGB Swin-Large final predictions',figures=TITLES,
                  files={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=digest(p))
                         for p in OUT.rglob('*') if p.is_file() and p.name not in ('manifest.json','verification.json') and not p.name.startswith('contact_')})
    write(OUT/'manifest.json',manifest)
    archive=ROOT/'outputs/DPSeg_report_figures_SwinL_v2.zip'
    with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and not p.name.startswith('contact_'):
                z.write(p,Path(OUT.name)/p.relative_to(OUT))
    print('archive MiB',round(archive.stat().st_size/1024**2,1),flush=True)

def main():
    if '--package-only' in sys.argv:package();return
    cases=cached() if '--cached' in sys.argv else prepare()
    functions=[teaser,challenges,framework,ablation,improvement,failures]
    for fn in functions:
        fn(cases);print('rendered',fn.__name__,flush=True)
    package()

if __name__=='__main__':main()
