#!/usr/bin/env python3
"""Verify complete point-head experiment artifacts against their actual sources."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime,timezone
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha,tensor_digest
from rgbd_seg import course_submission as submission
from tools.run_course_point_refinement import METHODS,VARIANTS,checked,select_validation


def read(path):return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:raise ValueError(message)


def compare_csv_predictions(csv_path,prediction_path,ids):
    """Bind every actual submitted mask/score to the independently scored JSONL."""
    spec=submission._Spec(ids,(1,),(1024,1024),100)
    previous_limit=csv.field_size_limit()
    count=0
    try:
        csv.field_size_limit(max(previous_limit,spec.max_json_chars))
        with Path(csv_path).open(newline='',encoding='utf-8-sig') as values,Path(prediction_path).open() as predictions:
            reader=csv.reader(values,strict=True)
            require(next(reader,None)==['image_id','PredictionString'],'CSV header differs')
            for row,line in zip_longest(reader,predictions):
                require(row is not None and line is not None,'CSV and scored prediction row counts differ')
                item=json.loads(line)
                require(len(row)==2 and row[0]==str(item['image_id'])==str(ids[count]),'CSV image identity differs')
                require(json.loads(row[1])==item['predictions'],'CSV masks/scores differ from scored predictions')
                count+=1
        require(count==len(ids),'CSV binding coverage incomplete')
    finally:csv.field_size_limit(previous_limit)
    return count


def inspect_rows(folder,ids,artifacts):
    reference=None
    for variant in VARIANTS:
        path=folder/variant/'predictions.jsonl'
        digest=file_sha(path)
        evaluated=read(folder/variant/'evaluation.json')
        scored=read(folder/variant/'scored.json')
        require(evaluated['status']=='completed' and evaluated['images']==len(ids),'Evaluation count/status differs')
        require(evaluated['predictions_sha256']==digest==scored['prediction_sha256'],'Prediction digest differs')
        require(file_sha(folder/variant/'evaluation.json')==scored['evaluation_sha256'],'Evaluation receipt changed')
        require(evaluated['iou_type']=='segm' and evaluated['max_detections']==[1,10,100], 'Incorrect metric type/cap')
        require(evaluated['area_ranges'][1]==[0,1024] and evaluated['ground_truth_area']=='original annotation area preserved',
                'Small-object area semantics changed')
        require(len(evaluated['iou_thresholds'])==10 and abs(evaluated['iou_thresholds'][0]-.5)<1e-12
                and abs(evaluated['iou_thresholds'][-1]-.95)<1e-12,'IoU averaging changed')
        rows,identities,count=[],[],0
        with path.open() as stream:
            for line in stream:
                item=json.loads(line);rows.append(item['image_id'])
                values=item['predictions'];require(len(values)<=100,'Too many predictions')
                require(all(v['category_id']==1 and 0<=v['score']<=1 and v['segmentation']['size']==[1024,1024]
                            and isinstance(v['segmentation']['counts'],str) for v in values),'Prediction schema differs')
                identities.append([(v['category_id'],v['score']) for v in values]);count+=len(values)
        require(rows==ids and count==evaluated['predicted_instances'],'Merged coverage/count differs')
        if reference is None:reference=identities
        else:require(identities==reference,'Refinement changed identities/scores/count')
        artifacts[str(path.resolve())]=digest
        artifacts[str((folder/variant/'evaluation.json').resolve())]=scored['evaluation_sha256']
    children=read(folder/'scorer_processes.json')
    require(len(children)==6 and {c['variant'] for c in children}==set(VARIANTS)
            and all(c['exit_code']==0 for c in children),'Scorer processes incomplete/failed')


def audit(protocol,supervisor_path,diagnostic_path,allow_engineering=False):
    supervisor=read(supervisor_path);root=supervisor_path.parent/'experiment'
    formal=protocol['scope']=='full_experiment'
    require(formal or allow_engineering,'Engineering output cannot prove a full experiment')
    require(supervisor['status']=='completed' and supervisor['exit_code']==0,'Supervisor did not finish successfully')
    require(read(supervisor_path.parent/'exit.json')=={'status':'completed','exit_code':0},'Authoritative exit receipt differs')
    require(read(root/'run.json')['status']=='completed','Training/evaluation run incomplete')
    final=read(root/'comparison.json')
    require(final['status']=='completed' and final['scope']==protocol['scope']
            and final['protocol_sha256']==protocol['_sha256'],'Final report identity differs')
    artifacts={str(Path(protocol['_path'])):protocol['_sha256']}
    for name,digest in supervisor['artifact_sha256'].items():
        require(file_sha(name)==digest,f'Supervisor artifact changed: {name}')
        artifacts[name]=digest
    world=len(protocol['gpus']);epochs=protocol['epochs']
    training_ids=[]
    with CourseArchive(protocol['archive'],'train') as data:
        require(len(data)==14432,'Official train split differs')
        indices=protocol.get('training_indices',list(range(len(data))))
        training_ids=[data.image_ids[index] for index in indices]
    expected_steps=epochs*protocol['training_images']//protocol['global_batch_size']
    base=read(root/'base_reload.json')
    rank_receipts=[read(root/f'completed_rank{rank}.json') for rank in range(world)]
    for rank,receipt in enumerate(rank_receipts):
        require(receipt['rank']==rank and receipt['status']=='completed' and receipt['base_unchanged']
                and receipt['base_gradients_absent'] and receipt['global_steps']==expected_steps,'Rank completion/training steps differ')
        require(receipt['base_tensor_sha256_before_and_after']==base['model_sha256'],'Frozen base tensor identity differs')
    require(len({r['head_tensor_sha256'] for r in rank_receipts})==1,'Final ranks disagree on head weights')
    histories=read(root/'validation_history.json')
    require(len(histories)==epochs and final['validation_history']==histories,'Validation history coverage differs')
    coverage=[]
    for epoch in range(1,epochs+1):
        ids=[image_id for rank in range(world) for image_id in read(root/f'coverage_epoch{epoch:02d}_rank{rank}.json')]
        require(len(ids)==len(set(ids))==len(training_ids) and set(ids)==set(training_ids),'Training coverage incomplete/duplicated')
        checkpoint=read(root/f'checkpoint_epoch_{epoch:02d}.json')
        require(checkpoint['epoch']==epoch and file_sha(checkpoint['path'])==checkpoint['sha256'],'Checkpoint bytes/epoch differ')
        values=torch.load(checkpoint['path'],map_location='cpu',weights_only=True)
        require(values['protocol_sha256']==protocol['_sha256'] and values['source_sha256']==protocol['source_sha256'],
                'Checkpoint source/protocol differs')
        require(tensor_digest(values['heads'])==checkpoint['head_tensor_sha256'],'Checkpoint tensor identity differs')
        artifacts[checkpoint['path']]=checkpoint['sha256']
        folder=root/f'val_epoch_{epoch:02d}'
        with CourseArchive(protocol['archive'],'val') as data:
            expected_ids=data.image_ids if formal else data.image_ids[:2*world]
        require(read(folder/'image_ids.json')==expected_ids,'Validation official IDs differ')
        inspect_rows(folder,expected_ids,artifacts)
        require(histories[epoch-1]==read(folder/'comparison.json'),'Validation report binding differs')
        for variant in VARIANTS:
            require(histories[epoch-1]['metrics'][variant]==read(folder/variant/'evaluation.json')['metrics'],
                    'Validation table differs from evaluator')
        coverage.append({'epoch':epoch,'training_images':len(ids),'validation_images':len(expected_ids)})
    selection=select_validation(histories)
    require(selection==final['selection']==read(root/'selection.json'),'Selection differs from validation-only policy')
    chosen=selection['selected']['checkpoint']
    require(rank_receipts[0]['head_tensor_sha256']==chosen['head_tensor_sha256'],'Final head is not the selected checkpoint')
    csvs=[]
    if formal:
        folder=root/'test'
        with CourseArchive(protocol['archive'],'test') as data:ids=data.image_ids
        require(len(ids)==2112 and read(folder/'image_ids.json')==ids,'Complete test IDs differ')
        inspect_rows(folder,ids,artifacts)
        require(read(folder/'head_checkpoint.json')==chosen,'Test used a different checkpoint')
        require(final['test']==read(folder/'comparison.json'),'Final test report binding differs')
        for variant in VARIANTS:
            require(final['test']['metrics'][variant]==read(folder/variant/'evaluation.json')['metrics'],
                    'Test table differs from evaluator')
        for variant in METHODS:
            receipt=read(folder/variant/'submission_validation.json')
            require(receipt['rows']==2112 and receipt['image_size']==[1024,1024]
                    and receipt['id_order_verified'] and receipt['rle_runs_verified']
                    and receipt['rle_native_decode_verified'] and receipt['max_detections']==100,'CSV validation incomplete')
            require(file_sha(receipt['path'])==receipt['sha256'],'CSV bytes differ from native validation receipt')
            require(receipt['predictions']==read(folder/variant/'evaluation.json')['predicted_instances'],
                    'CSV and evaluated instance counts differ')
            compare_csv_predictions(receipt['path'],folder/variant/'predictions.jsonl',ids)
            artifacts[receipt['path']]=receipt['sha256'];csvs.append({'variant':variant,'path':receipt['path'],'sha256':receipt['sha256']})
        require(diagnostic_path is not None,'Independent trained-head reload/geometry diagnostic missing')
        diagnostic=read(diagnostic_path)
        require(diagnostic['status']=='completed' and diagnostic['protocol_sha256']==protocol['_sha256']
                and diagnostic['selected_checkpoint']==chosen and diagnostic['head_reload_exact']
                and diagnostic['validation_predictions_reloaded_exact'] and diagnostic['base_unchanged']
                and not diagnostic['ground_truth_used'] and not diagnostic['test_accessed'],'Independent reload diagnostic incomplete')
        require(diagnostic['ordinary_predictor_weight_l2']>0 and diagnostic['depth_predictor_weight_l2']>0,
                'Trained point predictors remain at zero initialization')
        artifacts[str(diagnostic_path.resolve())]=file_sha(diagnostic_path)
    return {'status':'passed','scope':protocol['scope'],'full_experiment_verified':formal,
        'audited_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol['_sha256'],
        'requirements':{'complete_training_coverage':coverage,'frozen_base_all_ranks':True,
            'rank_head_weights_identical':True,'validation_only_selection_reproduced':True,
            'all_prediction_ids_counts_and_scores_checked':True,'six_arms_all_evaluations_bound':True,
            'test_images':2112 if formal else 0,'validated_csvs':len(csvs),
            'all_csv_masks_and_scores_equal_scored_predictions':formal,
            'independent_reload_verified':formal},
        'selected':selection['selected'],'submissions':csvs,'artifact_sha256':artifacts,
        'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_sha(__file__)}}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--supervisor',type=Path,required=True)
    parser.add_argument('--diagnostic',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--allow-engineering',action='store_true')
    args=parser.parse_args();torch.set_num_threads(2)
    if args.output.exists():raise FileExistsError('Audit receipts are immutable')
    receipt=audit(checked(args.protocol),args.supervisor,args.diagnostic,args.allow_engineering)
    write_json(args.output,receipt)
    print(json.dumps({'status':receipt['status'],'scope':receipt['scope'],
                      'full_experiment_verified':receipt['full_experiment_verified'],'csvs':len(receipt['submissions'])}))


if __name__=='__main__':main()
