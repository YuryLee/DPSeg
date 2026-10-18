#!/usr/bin/env python3
"""Render preselected validation cases from already-scored point-head outputs."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pycocotools import mask as mask_utils

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha

ARMS=('native','point','point_depth','point_depth_probability_depth')
TITLES=('Native','Point head','Depth point head','Depth point + depth filter')


def read(path):return json.loads(Path(path).read_text())


def render_cases(cases,arms,titles,output,stem,epoch):
    fig,axes=plt.subplots(len(cases),len(arms)+2,figsize=(2.65*(len(arms)+2),2.9*len(cases)),squeeze=False)
    for row,case in enumerate(cases):
        left,top,right,bottom=case['bounds'];crop=(slice(top,bottom),slice(left,right))
        image=case['rgb'][crop];target=case['gt'][crop]
        for col,ax in enumerate(axes[row]):
            ax.set_xticks([]);ax.set_yticks([])
            for spine in ax.spines.values():spine.set_visible(False)
            if col==0:
                ax.imshow(image)
                ax.set_ylabel(f"image {case['image_id']}\nGT area {case['gt_area']} px",fontsize=9)
                ax.set_title('RGB crop',fontsize=10)
            elif col==1:
                ax.imshow(target,cmap='gray',vmin=0,vmax=1)
                ax.set_title('Selected GT instance',fontsize=10)
            else:
                arm=arms[col-2];pred=case['masks'][arm][crop]
                ax.imshow(image)
                if target.any() and not target.all():ax.contour(target,levels=[.5],colors=['#ff4f69'],linewidths=1.1)
                if pred.any() and not pred.all():ax.contour(pred,levels=[.5],colors=['#00f0a8'],linewidths=1.1)
                ax.set_title(titles[col-2]+f"\nmask IoU {case['ious'][arm]:.3f}",fontsize=9)
    fig.suptitle(f"Preselected validation examples | head epoch {epoch}\nRed: GT boundary   Green: predicted boundary",fontsize=12)
    fig.tight_layout(rect=(0,0,1,.95))
    fig.savefig(output/f'{stem}.png',dpi=180,bbox_inches='tight')
    fig.savefig(output/f'{stem}.pdf',bbox_inches='tight')
    plt.close(fig)


def plot(protocol,experiment,output,guard_validation=None):
    final=read(experiment/'comparison.json')
    if final['status']!='completed' or final['scope']!='full_experiment':
        raise ValueError('Requires completed full experiment')
    choice=final['selection']['selected'];folder=experiment/f"val_epoch_{choice['epoch']:02d}"
    # These rank positions were fixed before any formal validation scores existed.
    rank=protocol['gpus'].index(5)
    rank_ids=read(folder/f'rank{rank}.json')['ordered_ids']
    ids=[rank_ids[position] for position in (0,97,251,501)]
    records={};bindings={};guard_receipt=None
    paths={arm:folder/arm/'predictions.jsonl' for arm in ARMS}
    if guard_validation is not None:
        guard_receipt=read(guard_validation/'receipt.json')
        if (guard_receipt['status']!='completed' or guard_receipt['split']!='val'
                or guard_receipt['checkpoint']!=choice['checkpoint']):
            raise ValueError('Guard illustration does not use the same selected validation checkpoint')
        paths['native_probability_depth']=folder/'native_probability_depth/predictions.jsonl'
        paths['guard']=guard_validation/'predictions.jsonl'
        if file_sha(paths['guard'])!=guard_receipt['prediction_sha256']:
            raise ValueError('Guard illustration prediction bytes changed')
    for arm,path in paths.items():
        bindings[str(path.resolve())]=file_sha(path)
        records[arm]={}
        with path.open() as source:
            for line in source:
                record=json.loads(line)
                if record['image_id'] in ids:records[arm][record['image_id']]=record['predictions']
        if set(records[arm])!=set(ids):raise ValueError('Illustration references incomplete')
    cases=[]
    with CourseArchive(protocol['archive'],'val') as data:
        truth=data.coco_ground_truth()
        for image_id in ids:
            annotations=sorted((a for a in truth['annotations'] if a['image_id']==image_id and not a['iscrowd']
                                and 0<a['area']<=1024),key=lambda a:a['id'])
            if not annotations:continue
            # First eligible annotation ID; no selection by before/after performance.
            annotation=annotations[0]
            gt=mask_utils.decode(annotation['segmentation']).astype(bool)
            native=records['native'][image_id]
            if not native:continue
            overlaps=mask_utils.iou([annotation['segmentation']],[v['segmentation'] for v in native],[0]*len(native))[0]
            index=int(np.argmax(overlaps))
            ys,xs=np.nonzero(gt)
            width=max(64,min(192,int(max(xs.max()-xs.min()+1,ys.max()-ys.min()+1)+32)))
            cx,cy=int((xs.min()+xs.max())/2),int((ys.min()+ys.max())/2)
            left=max(0,min(1024-width,cx-width//2));top=max(0,min(1024-width,cy-width//2))
            rgb=np.asarray(data.load_rgb(data.image_ids.index(image_id)))
            masks={};scores={};ious={}
            for arm in paths:
                values=records[arm][image_id]
                if len(values)!=len(native) or values[index]['score']!=native[index]['score']:
                    raise ValueError('Illustration query identity/scores differ')
                masks[arm]=mask_utils.decode(values[index]['segmentation']).astype(bool)
                scores[arm]=values[index]['score']
                ious[arm]=float(mask_utils.iou([annotation['segmentation']],[values[index]['segmentation']],[0])[0,0])
            cases.append({'image_id':image_id,'annotation_id':annotation['id'],'gt_area':annotation['area'],
                'native_prediction_index':index,'bounds':[left,top,left+width,top+width],
                'rgb':rgb,'gt':gt,'masks':masks,'scores':scores,'ious':ious})
    if not cases:raise ValueError('No eligible small instances in preselected validation examples')
    output.mkdir(parents=True,exist_ok=False)
    render_cases(cases,ARMS,TITLES,output,'point_boundary_examples',choice['epoch'])
    figure_names=['point_boundary_examples.png','point_boundary_examples.pdf']
    if guard_receipt is not None:
        render_cases(cases,('native_probability_depth','point_depth_probability_depth','guard'),
            ('Native + depth filter','Depth point + depth filter','Small-mask guard'),
            output,'guard_boundary_examples',choice['epoch'])
        figure_names+=['guard_boundary_examples.png','guard_boundary_examples.pdf']
    receipt={'status':'completed','scope':'validation_examples_only','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'selected_checkpoint':choice['checkpoint'],'preselected_validation_ids':ids,
        'selection_rule':'fixed GPU5 rank positions 0,97,251,501; first positive-area noncrowd annotation <=1024; native max-IoU prediction index shared by all arms',
        'cases':[{k:v for k,v in case.items() if k not in ('rgb','gt','masks')} for case in cases],
        'predictions_sha256':bindings,'images_sha256':{name:file_sha(output/name) for name in figure_names},
        'guard_validation_receipt':None if guard_receipt is None else {
            'path':str((guard_validation/'receipt.json').resolve()),'sha256':file_sha(guard_validation/'receipt.json')},
        'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_sha(__file__)},
        'limits':['Qualitative small-instance crops; no full-dataset effectiveness claim from these cases',
                  'The selected native prediction index is kept across methods; this is not a fresh GT match for each method']}
    write_json(output/'examples_receipt.json',receipt)
    print(json.dumps({'status':'completed','cases':len(cases),'output':str(output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--guard-validation',type=Path)
    args=parser.parse_args()
    plot(read(args.protocol),args.experiment,args.output,args.guard_validation)


if __name__=='__main__':main()
