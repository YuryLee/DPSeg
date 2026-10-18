#!/usr/bin/env python3
"""Replay the fixed guard on every selected validation/test prediction."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_point_guard import POLICY,guard_predictions
from tools.audit_course_point_refinement import compare_csv_predictions


def read(path):return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def replay(folder,protocol,artifacts):
    receipt=read(folder/'receipt.json')
    expected=2584 if receipt['split']=='val' else 2112
    require(receipt['status']=='completed' and receipt['images']==expected,'Guard result is incomplete')
    point=read(protocol['point_protocol']['path'])
    total={};predicted=0
    with CourseArchive(point['archive'],receipt['split']) as archive,ExitStack() as stack:
        ids=archive.image_ids
        require(len(ids)==expected,'Guard official split coverage differs')
        streams={name:stack.enter_context(Path(data['path']).open()) for name,data in receipt['input_predictions'].items()}
        for name,data in receipt['input_predictions'].items():
            require(file_sha(data['path'])==data['sha256'],f'Guard input changed: {name}')
            artifacts[data['path']]=data['sha256']
        actual=stack.enter_context((folder/'predictions.jsonl').open())
        for image_id in ids:
            records={name:json.loads(next(stream)) for name,stream in streams.items()}
            guarded=json.loads(next(actual))
            require(guarded['image_id']==image_id and all(v['image_id']==image_id for v in records.values()),
                    'Guard image IDs/order differ')
            values,statistics=guard_predictions(records[POLICY['baseline']]['predictions'],records[POLICY['candidate']]['predictions'])
            require(values==guarded['predictions'],'Guard masks/scores differ from exact fixed-policy replay')
            predicted+=len(values)
            for key,value in statistics.items():total[key]=total.get(key,0)+value
        require(next(actual,None) is None and all(next(stream,None) is None for stream in streams.values()),'Extra replay image rows')
    require(total==receipt['guard_statistics'],'Guard fallback statistics differ')
    evaluation=read(folder/'evaluation.json')
    require(file_sha(folder/'predictions.jsonl')==receipt['prediction_sha256']==evaluation['predictions_sha256'],
            'Guard prediction/evaluation digest differs')
    require(file_sha(folder/'evaluation.json')==receipt['evaluation_sha256'],'Guard evaluator output changed')
    require(evaluation['metrics']==receipt['metrics'] and evaluation['predicted_instances']==predicted,
            'Guard metrics/count differ from scorer')
    require(evaluation['iou_type']=='segm' and evaluation['area_ranges'][1]==[0,1024]
            and evaluation['max_detections']==[1,10,100],'Guard metric semantics differ')
    if receipt['split']=='test':
        validation=read(folder/'submission_validation.json');csv=receipt['submission']
        require(validation['rows']==2112 and validation['rle_native_decode_verified'] and validation['rle_runs_verified']
                and validation['id_order_verified'] and validation['image_size']==[1024,1024]
                and validation['predictions']==predicted,'Guard CSV native validation incomplete')
        require(file_sha(csv['path'])==csv['sha256']==validation['sha256'],'Guard CSV bytes changed')
        compare_csv_predictions(csv['path'],folder/'predictions.jsonl',ids)
        artifacts[csv['path']]=csv['sha256']
    for path in (folder/'receipt.json',folder/'evaluation.json',folder/'predictions.jsonl'):
        artifacts[str(path.resolve())]=file_sha(path)
    return {'split':receipt['split'],'images':len(ids),'predictions':predicted,
            'fallback_instances':total['fallback_instances'],'exact_guard_replay':True}


def audit(root,original_audit):
    supervisor=read(root/'supervisor.json');protocol=read(root/'protocol.json');selection=read(root/'selection.json')
    require(supervisor['status']=='completed' and supervisor['exit_code']==0
            and read(root/'exit.json')=={'status':'completed','exit_code':0},'Guard supervisor did not complete successfully')
    require(protocol['status']=='frozen' and protocol['guard_policy']==POLICY,'Guard frozen policy differs')
    for name,digest in {**protocol['source_sha256'],**supervisor['source_sha256']}.items():
        require(file_sha(ROOT/name)==digest,f'Guard source changed: {name}')
    for field in ('point_protocol','design_evidence','unit_tests'):
        require(file_sha(protocol[field]['path'])==protocol[field]['sha256'],f'Guard protocol input changed: {field}')
    core=read(original_audit)
    require(core['status']=='passed' and core['full_experiment_verified'],'Original point-head completion audit missing')
    main_supervisor=Path(supervisor['original_supervisor']);experiment=main_supervisor.parent/'experiment'
    point_selection=read(experiment/'selection.json')
    require(selection['status']=='frozen' and selection['selection_split']=='val' and not selection['test_used_for_selection']
            and not selection['guard_thresholds_searched'],'Guard selection did not use the fixed validation policy')
    require(selection['checkpoint']==point_selection['selected']['checkpoint']
            and selection['original_selection_sha256']==file_sha(experiment/'selection.json'),
            'Guard changed the original selected point checkpoint')
    require(selection['guard_protocol_sha256']==file_sha(root/'protocol.json'),'Guard selection protocol digest differs')
    validation=read(selection['validation_receipt']['path'])
    require(file_sha(selection['validation_receipt']['path'])==selection['validation_receipt']['sha256'],
            'Guard selected validation receipt changed')
    require(validation['checkpoint']==selection['checkpoint'] and validation['split']=='val','Guard validation checkpoint/split differs')
    gain=validation['metrics']['mask_ap_small']-validation['baseline_metrics']['mask_ap_small']
    drop=validation['baseline_metrics']['mask_ap']-validation['metrics']['mask_ap']
    require(selection['validation_eligible']==validation['validation_eligible']==(gain>=.005 and drop<=.005),
            'Guard validation eligibility differs')
    test=read(root/'test/receipt.json')
    require(test['checkpoint']==selection['checkpoint'] and test['guard_protocol_sha256']==selection['guard_protocol_sha256']
            and test['validation_eligible'] is None,'Test input identity/selection semantics differ')
    require(selection['created_at_utc']<test['created_at_utc'],'Guard validation selection was not frozen before test processing')
    artifacts={str(original_audit.resolve()):file_sha(original_audit)}
    for name,digest in supervisor['artifact_sha256'].items():
        require(file_sha(name)==digest,f'Guard supervisor artifact changed: {name}')
        artifacts[name]=digest
    for path in (root/'supervisor.json',root/'selection.json',root/'protocol.json'):
        artifacts[str(path.resolve())]=file_sha(path)
    validation_check=replay(Path(selection['validation_receipt']['path']).parent,protocol,artifacts)
    test_check=replay(root/'test',protocol,artifacts)
    return {'status':'passed','scope':'fixed_guard_extension_complete','audited_at_utc':datetime.now(timezone.utc).isoformat(),
        'original_point_experiment_verified':True,'guard_selection_validation_only':True,
        'original_point_checkpoint_preserved':True,'validation_eligible':selection['validation_eligible'],
        'validation_replay':validation_check,'test_replay':test_check,
        'csv_exactly_matches_scored_predictions':True,'submission':test['submission'],
        'artifact_sha256':artifacts,'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_sha(__file__)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--original-audit',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Guard audit receipts are immutable')
    receipt=audit(args.root,args.original_audit);write_json(args.output,receipt)
    print(json.dumps({'status':'passed','test_images':receipt['test_replay']['images'],'validation_eligible':receipt['validation_eligible']}))


if __name__=='__main__':main()
