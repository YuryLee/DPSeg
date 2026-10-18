#!/usr/bin/env python3
"""Export every validation case and transparent mask error diagnostics."""
import argparse
import base64
import html
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from demo import Demo


def diagnose(gt,pred):
    """Greedy same-class mask IoU matching at .5; diagnostic, not COCO AP."""
    masks=pred['masks'][:,0]>=.5; target=gt['masks'].bool()
    candidates=[]
    overlaps=np.zeros((len(masks),len(target)))
    for i,m in enumerate(masks):
        for j,t in enumerate(target):
            if pred['labels'][i]!=gt['labels'][j]: continue
            union=(m|t).sum().item()
            overlaps[i,j]=(m&t).sum().item()/union if union else 0
            candidates.append((overlaps[i,j],i,j))
    usedp=set(); usedg=set(); matched=[]
    for iou,i,j in sorted(candidates,reverse=True):
        if iou<.5: break
        if i not in usedp and j not in usedg:
            usedp.add(i); usedg.add(j); matched.append(iou)
    return {'gt':len(target),'predictions':len(masks),'matched':len(matched),
        'missed':len(target)-len(usedg),'false_positive':len(masks)-len(usedp),
        'unmatched_gt_with_partial_overlap_0.1_to_0.5':int(sum(0.1<=overlaps[:,j].max()<.5 for j in range(len(target)) if j not in usedg)) if len(masks) else 0,
        'matched_mean_iou':float(np.mean(matched)) if matched else None}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data/synthetic'); p.add_argument('--runs',default='outputs/demo')
    p.add_argument('--output',default='outputs/gallery');p.add_argument('--device',default='cpu')
    p.add_argument('--split',choices=['val','test'],default='val');args=p.parse_args()
    torch.set_num_threads(2); demo=Demo(args.data,args.runs,args.device,args.split); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    cards=[]; rows=[]
    count=len(demo.dataset.samples)
    if demo.provenance=='public_graspnet_subset':
        scope='GraspNet 公开子集探索实验 · 仅覆盖少量场景与帧，保持训练/评测场景隔离，不能代表完整 GraspNet 或老师统一数据集的成绩。'
    elif demo.provenance=='synthetic_smoke':
        scope='合成数据工程演示 · 不代表课程数据成绩。'
    else:
        scope='课程数据验证结果。'
    area_note='COCO 尺度分组按原始分辨率的掩码像素面积计算，small 使用 32² 像素阈值；无对应实例时 AP small 不可用。'
    split_name='开发验证' if args.split=='val' else '独立测试'
    banner=f'{scope} {count} 个{split_name}样例全部展示，未筛选好例。{area_note}'
    classes=demo.dataset.manifest.get('class_scheme')
    if classes=='object_identity':
        banner+=' 类别为 GraspNet 的 88 个物体身份 ID；同一帧内同身份重复物体无法由该标注区分。'
    elif classes=='category_agnostic':
        banner+=' 类别为统一 object，进行类别无关实例分割。'
    for i,record in enumerate(demo.dataset.samples):
        result=demo.infer({'sample_id':record['id'],'threshold':.5})
        # Files include the complete result and actual rendered predictions for inspection.
        (out/(record['id']+'.json')).write_text(json.dumps(result,ensure_ascii=False,indent=2))
        sample,target=demo.dataset[i]; row={'sample_id':record['id']}
        with torch.inference_mode():
            for mode,model in demo.models.items():
                pred=model([{k:v.to(args.device) for k,v in sample.items()}])[0]
                pred={k:v[pred['scores']>=.5].cpu() for k,v in pred.items()}
                row[mode]=diagnose(target,pred)
        rows.append(row)
        panels=[]
        for key,uri in result['panels'].items():
            if uri is None: continue
            filename=record['id']+'_'+key+'.png'; (out/filename).write_bytes(base64.b64decode(uri.split(',',1)[1]))
            panels.append(f'<figure><img src="{filename}" alt="{key}" width="256"><figcaption>{key}</figcaption></figure>')
        cards.append('<section><h2>'+html.escape(record['id'])+'</h2><div class="grid">'+''.join(panels)+'</div></section>')
    report={'provenance':demo.provenance,'split':args.split,'display_threshold':.5,'mask_threshold':.5,
            'scope':scope,'evaluation_area_reference':'original-resolution mask pixel area; COCO small threshold 32^2',
            'class_scheme':demo.dataset.manifest.get('class_scheme'),'validation_cases':count,
            'matching':'greedy same-class mask IoU >= 0.5; diagnostic only, not COCO AP; partial overlaps suggest boundary/localization issues but do not prove cause',
            'cases':rows}
    (out/'error_analysis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    (out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>RGB-D evaluation gallery</title><style>body{background:#101722;color:#e5ebf4;font:16px system-ui;margin:30px}.grid{display:flex;flex-wrap:wrap}figure{margin:8px}img{image-rendering:pixelated}section{border-top:1px solid #46505f}p{color:#ebc998}</style><h1>RGB-D：全部'+split_name+'样例</h1><p>'+html.escape(banner)+'</p>'+''.join(cards))
    print(json.dumps({'cases':len(rows),'output':str(out)}))


if __name__=='__main__': main()
