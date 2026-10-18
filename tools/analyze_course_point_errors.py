#!/usr/bin/env python3
"""Geometry-only paired validation diagnosis, separate from official COCO AP."""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from pycocotools import mask as mask_utils
from scipy.optimize import linear_sum_assignment

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha

PAIRS={'plain_point_vs_native':('native','point'),
       'depth_point_filter_vs_native_filter':('native_probability_depth','point_depth_probability_depth')}
ARMS=tuple(dict.fromkeys(arm for pair in PAIRS.values() for arm in pair))


def read(path):return json.loads(Path(path).read_text())


def stratum(area):
    if area<=64:return 'small_le64'
    if area<=256:return 'small_65_256'
    if area<=1024:return 'small_257_1024'
    if area<=9216:return 'medium'
    return 'large'


def summary(values):
    if not values:return {'instances':0}
    original=np.array([v['baseline_iou'] for v in values])
    refined=np.array([v['refined_iou'] for v in values])
    delta=refined-original
    area_ratio=np.array([v['refined_area']/max(1,v['baseline_area']) for v in values])
    return {'instances':len(values),'baseline_iou_mean':float(original.mean()),
        'refined_iou_mean':float(refined.mean()),'iou_delta_mean':float(delta.mean()),
        'iou_delta_median':float(np.median(delta)),'improved':int((delta>1e-6).sum()),
        'worsened':int((delta< -1e-6).sum()),'unchanged':int((np.abs(delta)<=1e-6).sum()),
        'median_area_ratio':float(np.median(area_ratio)),
        'shrank_over_5percent':int((area_ratio<.95).sum()),
        'expanded_over_5percent':int((area_ratio>1.05).sum()),
        'threshold_transitions':{str(threshold):{
            'baseline_above':int((original>=threshold).sum()),
            'refined_above':int((refined>=threshold).sum()),
            'gained':int(((original<threshold)&(refined>=threshold)).sum()),
            'lost':int(((original>=threshold)&(refined<threshold)).sum())}
            for threshold in (.5,.75,.9,.95)}}


def analyze(protocol,folder,output):
    if output.exists():raise FileExistsError('Diagnosis output must be new')
    comparison=read(folder/'comparison.json')
    if comparison['images']!=2584 or comparison['split']!='val':
        raise ValueError('This diagnosis is for complete validation only')
    bindings={}
    for arm in ARMS:
        path=folder/arm/'predictions.jsonl';receipt=read(folder/arm/'evaluation.json')
        digest=file_sha(path)
        if receipt['predictions_sha256']!=digest or receipt['images']!=2584:
            raise ValueError('Prediction/evaluation binding differs')
        bindings[arm]={'path':str(path.resolve()),'sha256':digest}
    output.mkdir(parents=True)
    collected={pair:[] for pair in PAIRS}
    total_gt=defaultdict(int)
    area_counts={pair:defaultdict(int) for pair in PAIRS}
    began=time.monotonic()
    with CourseArchive(protocol['archive'],'val') as archive,ExitStack() as stack:
        truth=archive.coco_ground_truth()
        annotations=defaultdict(list)
        for ann in truth['annotations']:
            if ann['area']>0 and not ann['iscrowd']:
                annotations[ann['image_id']].append(ann)
                total_gt[stratum(ann['area'])]+=1
        annotation_digest=hashlib.sha256()
        with ZipFile(protocol['archive']) as source:
            for name in archive.annotation_members:annotation_digest.update(name.encode()+b'\0'+source.read(name))
        streams={arm:stack.enter_context((folder/arm/'predictions.jsonl').open()) for arm in ARMS}
        details={pair:stack.enter_context((output/f'{pair}_instances.jsonl').open('x')) for pair in PAIRS}
        for index,image_id in enumerate(archive.image_ids):
            records={arm:json.loads(next(stream)) for arm,stream in streams.items()}
            if any(record['image_id']!=image_id for record in records.values()):
                raise ValueError('Validation IDs/order differ')
            gt=annotations[image_id]
            gt_rles=[a['segmentation'] for a in gt]
            values={arm:record['predictions'] for arm,record in records.items()}
            identities=[(v['category_id'],v['score']) for v in values['native']]
            if any([(v['category_id'],v['score']) for v in items]!=identities for items in values.values()):
                raise ValueError('Paired prediction identities/scores differ')
            rles={arm:[v['segmentation'] for v in items] for arm,items in values.items()}
            areas={arm:np.array(mask_utils.area(items),dtype=np.int64) if items else np.empty(0,dtype=np.int64)
                   for arm,items in rles.items()}
            matrices={arm:mask_utils.iou(items,gt_rles,[0]*len(gt)).T if items and gt
                      else np.zeros((len(gt),len(items))) for arm,items in rles.items()}
            for pair,(baseline,refined) in PAIRS.items():
                stats=area_counts[pair]
                stats['baseline_predicted_small']+=int((areas[baseline]<=1024).sum())
                stats['refined_predicted_small']+=int((areas[refined]<=1024).sum())
                stats['changed_medium_or_large_to_small']+=int(((areas[baseline]>1024)&(areas[refined]<=1024)).sum())
                stats['changed_small_to_medium_or_large']+=int(((areas[baseline]<=1024)&(areas[refined]>1024)).sum())
                stats['all_predictions']+=len(areas[baseline])
                gi,pi=linear_sum_assignment(-matrices[baseline])
                for g,p in zip(gi,pi):
                    ann=gt[g]
                    row={'image_id':image_id,'annotation_id':ann['id'],'gt_area':ann['area'],
                        'stratum':stratum(ann['area']),'native_prediction_index':int(p),
                        'score':values[baseline][p]['score'],
                        'baseline_iou':float(matrices[baseline][g,p]),
                        'refined_iou':float(matrices[refined][g,p]),
                        'baseline_area':int(areas[baseline][p]),'refined_area':int(areas[refined][p])}
                    collected[pair].append(row)
                    details[pair].write(json.dumps(row)+'\n')
            if index==0 or (index+1)%100==0 or index+1==len(archive):
                for stream in details.values():stream.flush()
                write_json(output/'progress.json',{'phase':'analyzing','images':index+1,'expected':2584,
                    'elapsed_seconds':time.monotonic()-began})
        if any(next(stream,None) is not None for stream in streams.values()):
            raise ValueError('Extra validation prediction rows')
    pair_summaries={}
    for pair,values in collected.items():
        groups={}
        for group in ('all','small','small_le64','small_65_256','small_257_1024','medium','large'):
            subset=[v for v in values if group=='all' or (group=='small' and v['gt_area']<=1024) or v['stratum']==group]
            groups[group]={'all_assigned':summary(subset),
                           'baseline_iou_ge_0p1':summary([v for v in subset if v['baseline_iou']>=.1])}
        pair_summaries[pair]={'groups':groups,'predicted_area_transitions':dict(area_counts[pair])}
    report={'status':'completed','scope':'epoch1_complete_validation_geometry_diagnosis',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'images':2584,
        'annotation_member_name_and_bytes_sha256':annotation_digest.hexdigest(),
        'positive_noncrowd_gt_strata':dict(total_gt),'prediction_bindings':bindings,
        'pairs':pair_summaries,'elapsed_seconds':time.monotonic()-began,
        'formal_AP_metrics':comparison['metrics'],'test_accessed':False,
        'method':'Hungarian maximum-mask-IoU assignment on each baseline; keep the same prediction index for its paired refined mask',
        'limits':['Geometry diagnosis only; assignment ignores score ordering and does not reproduce COCO AP matching',
                  'Zero-area and crowd GT excluded from diagnosis, preserved in original formal COCO evaluation',
                  'GT diagnostic strata are nonoverlapping; original COCO area ranges retain their boundary conventions',
                  'Changing predicted small-mask counts is not itself evidence of true/false positive changes'],
        'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_sha(__file__)}}
    write_json(output/'diagnosis.json',report)
    lines=['# 第一轮点头验证集几何诊断','','本文件解释同一预测掩码的变化，不替代正式 COCO AP。训练参数保持固定，未访问测试集。','',
           '| 配对 | GT 面积分组 | 实例数 | 基线平均 IoU | 点头平均 IoU | 改善数 | 变差数 | 面积比中位数 |',
           '|---|---|---:|---:|---:|---:|---:|---:|']
    for pair,stats in pair_summaries.items():
        for group,value in stats['groups'].items():
            if group=='all':continue
            s=value['baseline_iou_ge_0p1']
            if not s['instances']:continue
            lines.append(f"| {pair} | {group} | {s['instances']} | {s['baseline_iou_mean']:.4f} | {s['refined_iou_mean']:.4f} | {s['improved']} | {s['worsened']} | {s['median_area_ratio']:.3f} |")
    lines+=['','表中仅统计基线 IoU≥0.1 的可定位配对；JSON 同时保留全部 Hungarian 配对的结果。',
        '每对方法固定基线的预测实例索引，反映掩码变化；正式 COCO 评分使用原来的独立匹配规则。',
        '该诊断能支持特定面积组上的几何观察，不能单独证明训练区域、损失函数或网络结构是差异的原因。']
    (output/'diagnosis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'completed','images':2584,'seconds':report['elapsed_seconds'],'output':str(output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--validation',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    analyze(read(args.protocol),args.validation,args.output)


if __name__=='__main__':main()
