#!/usr/bin/env python3
"""Apply and independently score one preregistered mask protection rule."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_dafusion_eval import score_predictions
from rgbd_seg.course_finetune_eval import validate_csv_fast
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_point_guard import POLICY,guard_predictions
from rgbd_seg.course_submission import SubmissionWriter
from tools.audit_course_point_refinement import compare_csv_predictions


def read(path):return json.loads(Path(path).read_text())


def run(protocol,path,input_folder,output):
    if protocol['status']!='frozen' or protocol['guard_policy']!=POLICY:
        raise ValueError('Guard protocol/policy differs')
    for name,digest in protocol['source_sha256'].items():
        if file_sha(ROOT/name)!=digest:raise ValueError(f'Guard source changed: {name}')
    if file_sha(protocol['point_protocol']['path'])!=protocol['point_protocol']['sha256']:
        raise ValueError('Point-head protocol changed')
    if file_sha(protocol['design_evidence']['path'])!=protocol['design_evidence']['sha256']:
        raise ValueError('Guard design evidence changed')
    point_protocol=read(protocol['point_protocol']['path'])
    main=read(input_folder/'comparison.json')
    split=main['split'];epoch=main['epoch']
    if split not in ('val','test'):raise ValueError('Guard accepts complete val/test only')
    expected=2584 if split=='val' else 2112
    if main['images']!=expected:raise ValueError('Guard requires a complete official split')
    input_receipts={}
    for arm in (POLICY['baseline'],POLICY['candidate']):
        prediction_path=input_folder/arm/'predictions.jsonl'
        evaluation=read(input_folder/arm/'evaluation.json')
        if evaluation['images']!=expected or file_sha(prediction_path)!=evaluation['predictions_sha256']:
            raise ValueError('Guard input prediction/evaluation binding differs')
        input_receipts[arm]={'path':str(prediction_path.resolve()),'sha256':file_sha(prediction_path)}
    checkpoint=read(input_folder/'head_checkpoint.json')
    if file_sha(checkpoint['path'])!=checkpoint['sha256']:
        raise ValueError('Guard input head checkpoint changed')
    saved=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
    if (saved['protocol_sha256']!=protocol['point_protocol']['sha256']
            or saved['base_checkpoint_sha256']!=point_protocol['weights']['sha256']):
        raise ValueError('Guard head checkpoint belongs to a different model/protocol')
    if split=='test':
        selection=read(input_folder.parent/'selection.json')['selected']
        if selection['checkpoint']!=checkpoint:
            raise ValueError('Test input does not use original validation-selected point checkpoint')
        gate_selection=read(output.parent/'selection.json')
        if gate_selection['checkpoint']!=checkpoint or gate_selection['selection_split']!='val':
            raise ValueError('Guard validation selection must be frozen before test processing')
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();total={};guard_times=[]
    with CourseArchive(point_protocol['archive'],split) as archive:
        ids=archive.image_ids
        if len(ids)!=expected or read(input_folder/'image_ids.json')!=ids:
            raise ValueError('Guard official split IDs differ')
        with Path(input_receipts[POLICY['baseline']]['path']).open() as base_stream,\
             Path(input_receipts[POLICY['candidate']]['path']).open() as candidate_stream,\
             (output/'predictions.jsonl').open('x') as target:
            for position,image_id in enumerate(ids):
                old,new=json.loads(next(base_stream)),json.loads(next(candidate_stream))
                if old['image_id']!=image_id or new['image_id']!=image_id:
                    raise ValueError('Guard prediction ID/order differs')
                began=time.perf_counter()
                values,statistics=guard_predictions(old['predictions'],new['predictions'])
                guard_times.append(time.perf_counter()-began)
                for key,value in statistics.items():total[key]=total.get(key,0)+value
                target.write(json.dumps({'image_id':image_id,'predictions':values},allow_nan=False)+'\n')
                if position==0 or (position+1)%100==0 or position+1==len(ids):
                    target.flush()
                    write_json(output/'progress.json',{'phase':'guarding','images':position+1,'expected':len(ids)})
            if next(base_stream,None) is not None or next(candidate_stream,None) is not None:
                raise ValueError('Extra guard input images')
        write_json(output/'progress.json',{'phase':'scoring','images':len(ids)})
        result=score_predictions(archive,output/'predictions.jsonl',output)
        baseline=main['metrics'][POLICY['baseline']]
        gain=result['metrics']['mask_ap_small']-baseline['mask_ap_small']
        drop=baseline['mask_ap']-result['metrics']['mask_ap']
        receipt={'status':'completed','split':split,'images':len(ids),'head_epoch':epoch,'checkpoint':checkpoint,
            'guard_protocol_sha256':file_sha(path),'point_protocol_sha256':protocol['point_protocol']['sha256'],
            'input_predictions':input_receipts,'guard_statistics':total,
            'metrics':result['metrics'],'baseline_metrics':baseline,'candidate_metrics':main['metrics'][POLICY['candidate']],
            'APsmall_gain_vs_native_probability_depth':gain,'AP_drop_vs_native_probability_depth':drop,
            'validation_eligible':(gain>=.005 and drop<=.005) if split=='val' else None,
            'observed_gain_passes_numeric_threshold':gain>=.005 and drop<=.005,
            'guard_extra_mean_ms':1000*sum(guard_times)/len(ids),
            'pipeline_mean_ms_attributed':main['timing_mean_ms'][POLICY['candidate']]
                +main['timing_mean_ms'][POLICY['baseline']]-main['timing_mean_ms']['native']
                +1000*sum(guard_times)/len(ids),
            'timing_scope':'Original mixed-GPU stage timing attribution plus measured CPU guard; native model forward shared; full original probability filter retained',
            'prediction_sha256':file_sha(output/'predictions.jsonl'),
            'evaluation_sha256':file_sha(output/'evaluation.json'),
            'ground_truth_in_guard':False,'parameters_chosen_on':'first-epoch validation diagnosis, before this guard score or any point-head test score',
            'elapsed_seconds':time.monotonic()-started,'created_at_utc':datetime.now(timezone.utc).isoformat()}
        if split=='test':
            csv=output/f'rgbd_dafusion_swin_t_1024_e05_point_depth_probdepth_guard_s1024_iou080_head_e{epoch:02d}.csv'
            with SubmissionWriter(csv,ids) as writer,(output/'predictions.jsonl').open() as source:
                for line in source:
                    record=json.loads(line);writer.write(record['image_id'],record['predictions'])
            validation=validate_csv_fast(csv,ids)
            compare_csv_predictions(csv,output/'predictions.jsonl',ids)
            write_json(output/'submission_validation.json',validation)
            receipt['submission']={'path':str(csv.resolve()),'sha256':validation['sha256'],'rows':validation['rows']}
    write_json(output/'receipt.json',receipt)
    write_json(output/'progress.json',{'phase':'completed','images':expected})
    print(json.dumps({'status':'completed','split':split,'AP':receipt['metrics']['mask_ap'],
        'APs':receipt['metrics']['mask_ap_small'],'validation_eligible':receipt['validation_eligible'],'output':str(output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();torch.set_num_threads(2)
    run(read(args.protocol),args.protocol,args.input,args.output)


if __name__=='__main__':main()
