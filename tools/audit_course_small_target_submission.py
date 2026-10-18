#!/usr/bin/env python3
"""Independent full native RLE and CSV/JSONL equality audit after formal completion."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[name]='4'

import argparse
import csv
from datetime import datetime,timezone
import hashlib
import io
import json
from pathlib import Path
import sys
import time
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rgbd_seg.course_finetune_eval import validate_csv_fast
from rgbd_seg.course_submission import _Spec, _check_id, _unique_object


def digest(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,default=ROOT/'reports/course_small_target/formal_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/course_small_target/submission_independent_check.json')
    args=parser.parse_args();folder=args.run.resolve();output=args.output.resolve()
    run=json.loads((folder/'run.json').read_text())
    if run.get('status')!='completed' or run.get('scope')!='full_train_val_test':
        raise ValueError('Formal full-split run incomplete; no audit output generated')
    if output.exists():raise FileExistsError(output)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    started=time.monotonic();test=folder/'test'
    receipt=json.loads((test/'evaluation_run.json').read_text())
    original=json.loads((test/'submission_validation.json').read_text())
    paths={name:test/name for name in ('submission.csv','predictions.jsonl','evaluation_run.json',
                                     'evaluation.json','submission_validation.json')}
    before={name:digest(path) for name,path in paths.items()}
    if before['evaluation_run.json']!=run['test_evaluation_run_sha256']:
        raise ValueError('Final test receipt binding differs')
    for name,key in [('submission.csv','submission_sha256'),('predictions.jsonl','predictions_sha256'),
                     ('evaluation.json','evaluation_sha256'),('submission_validation.json','submission_validation_sha256')]:
        if before[name]!=receipt[key]:raise ValueError(f'Hash binding differs: {name}')
    protocol_path=Path(run['arguments']['protocol']);protocol=json.loads(protocol_path.read_text())
    if digest(protocol_path)!=run['protocol_sha256']:raise ValueError('Frozen protocol differs')
    for name,expected in run['source_sha256'].items():
        if digest(ROOT/name)!=expected:raise ValueError(f'Frozen source differs: {name}')
    with ZipFile(protocol['archive']) as archive:template=archive.read('submission.csv')
    reader=csv.DictReader(io.StringIO(template.decode('utf-8-sig')))
    if reader.fieldnames!=['image_id','PredictionString']:raise ValueError('Official template header differs')
    ids=[int(row['image_id']) for row in reader]
    if len(ids)!=2112 or len(set(ids))!=2112:raise ValueError('Official image IDs differ')
    result={'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'scope':'Full CSV native decode and semantic equality to saved scored predictions',
        'script_sha256':digest(__file__),'gpu_used':False,'new_inference':False,'new_scoring':False,
        'evidence':{name:{'path':str(path),'sha256':before[name]} for name,path in paths.items()},
        'template_sha256':hashlib.sha256(template).hexdigest(),'protocol_sha256':digest(protocol_path)}
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    try:
        native=validate_csv_fast(test/'submission.csv',ids)
        for key in ('rows','predictions','empty_images','summed_instance_area','image_size','category_ids',
                    'max_detections','id_order_verified','rle_runs_verified','rle_native_decode_verified'):
            if native[key]!=original[key]:raise ValueError(f'Native validation receipt differs: {key}')
        spec=_Spec(ids,(1,),(1024,1024),100);rows=0;count=0
        with (test/'submission.csv').open(newline='') as stream,(test/'predictions.jsonl').open() as raw:
            csv_rows=csv.DictReader(stream)
            if csv_rows.fieldnames!=['image_id','PredictionString']:raise ValueError('CSV columns differ')
            for index,row in enumerate(csv_rows):
                record=json.loads(next(raw),object_pairs_hook=_unique_object)
                _check_id(record['image_id'],index,spec)
                if int(row['image_id'])!=record['image_id']:raise ValueError('CSV/JSONL image ID differs')
                predictions=json.loads(row['PredictionString'],object_pairs_hook=_unique_object)
                if predictions!=record['predictions']:raise ValueError('CSV/JSONL predictions differ')
                rows+=1;count+=len(predictions)
            if next(raw,None) is not None:raise ValueError('Extra prediction row')
        if rows!=2112 or count!=native['predictions']:raise ValueError('Full prediction coverage differs')
        if any(digest(path)!=before[name] for name,path in paths.items()):raise ValueError('Inputs changed during audit')
        if torch.cuda.is_initialized():raise ValueError('CPU audit unexpectedly initialized CUDA')
        result.update(status='pass',native_validation=native,
            semantic_comparison={'rows':rows,'predictions':count,'all_rows_equal':True},
            completed_at_utc=datetime.now(timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started)
    except BaseException as error:
        result.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'pass','rows':rows,'predictions':count,'elapsed_seconds':result['elapsed_seconds']}))


if __name__=='__main__':main()
