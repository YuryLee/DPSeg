#!/usr/bin/env python3
"""Export fixed completed ablations; overlap GPU inference with CPU CSV validation."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_inference_sample
from rgbd_seg.course_depth_refinement import refine_predictions
from rgbd_seg.course_optimized import load_for_inference
from rgbd_seg.course_distributed_eval import merge_shards,write_json
from rgbd_seg.course_finetune_eval import validate_csv_fast
from rgbd_seg.course_submission import SubmissionWriter
from tools.check_course_prediction_reload import compare_predictions
from tools.select_course_resolution import read,sha,require


def checked(path):
    p=read(path)
    require(p['status']=='frozen' and p['gpus']==[2,3,5,6,7], 'Unexpected export protocol')
    for name,digest in p['source_sha256'].items():
        require(sha(ROOT/name)==digest,f'Frozen dependency changed: {name}')
    for arm in p['arms']:
        for key in ['weights','training_protocol','training_run','selection','validation_predictions']:
            require(sha(arm[key]['path'])==arm[key]['sha256'],f'Arm identity changed: {arm["name"]}/{key}')
        run=read(arm['training_run']['path']); selection=read(arm['selection']['path'])
        require(run['status']=='completed' and not run['test_used']
                and run['optimizer_steps']==2190 and run['training_image_exposures']==72160,
                'Training incomplete')
        require(selection['best_epoch']==arm['epoch'] and run['best_checkpoint_sha256']==arm['weights']['sha256'],
                'Not the completed validation-best checkpoint')
        require(read(arm['training_protocol']['path'])['archive']==p['archive'],'Dataset differs')
    return p


def worker(p,arm,output,rank):
    gpus=p['gpus']; require(os.environ.get('CUDA_VISIBLE_DEVICES')==str(gpus[rank])
                            and torch.cuda.device_count()==1,'Wrong GPU mapping')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    model,reload=load_for_inference(arm['weights']['path'],arm['training_protocol']['path'],device='cuda:0')
    require(reload['epoch']==arm['epoch'] and reload['method']==arm['method']
            and reload['resolution']==arm['resolution'],'Reconstructed model differs')
    old_rank=p['training_gpus'].index(gpus[rank])
    reference_folder=Path(arm['validation_predictions']['path']).parent
    ids=read(reference_folder/f'rank{old_rank}.json')['ordered_ids'][:2]
    references={}
    with Path(arm['validation_predictions']['path']).open() as stream:
        for line in stream:
            item=json.loads(line)
            if item['image_id'] in ids:references[item['image_id']]=item['predictions']
    require(set(references)==set(ids),'Missing validation reference')
    with torch.inference_mode(),CourseArchive(p['archive'],'val') as data:
        first=load_inference_sample(data,ids[0])
        for _ in range(3):model.predict_sample(first,score_threshold=.01,max_detections=100)
        checks=[{'image_id':key,'max_score_error':compare_predictions(references[key],
                model.predict_sample(load_inference_sample(data,key),score_threshold=.01,max_detections=100))}
                for key in ids]
        write_json(output/f'reload_rank{rank}.json',{'status':'passed','rank':rank,'physical_gpu':gpus[rank],
                   'checks':checks,'reload':reload})
    started=time.monotonic(); counts={v['name']:0 for v in arm['variants']}
    with torch.inference_mode(),CourseArchive(p['archive'],'test') as data,ExitStack() as stack:
        require(data.image_ids==read(output/'image_ids.json'),'Official test order differs')
        indices=list(range(rank,len(data),len(gpus)))
        streams={v['name']:stack.enter_context((output/v['name']/f'predictions_rank{rank}.jsonl').open('x'))
                 for v in arm['variants']}
        for position,index in enumerate(indices):
            record=data.images[index]; sample=load_inference_sample(data,record['id'])
            require(set(sample)=={'image','depth_m','depth_valid','camera_K'},'Unexpected inference inputs')
            torch.cuda.synchronize();begin=time.perf_counter()
            original=model.predict_sample(sample,score_threshold=.01,max_detections=100)
            torch.cuda.synchronize();inference_seconds=time.perf_counter()-begin
            for variant in arm['variants']:
                mode=variant['postprocessing']; values=original; refinement_seconds=0.
                if mode!='native':
                    begin=time.perf_counter()
                    values,_=refine_predictions(original,sample['depth_m'][0],sample['depth_valid'][0],
                                               mode=mode,device='cuda:0')
                    torch.cuda.synchronize();refinement_seconds=time.perf_counter()-begin
                    require([(x['category_id'],x['score']) for x in values]==
                            [(x['category_id'],x['score']) for x in original],'Postprocessing changed scores/count')
                streams[variant['name']].write(json.dumps({'image_id':record['id'],'file_name':record['file_name'],
                    'predictions':values,'inference_seconds':inference_seconds,
                    'refinement_seconds':refinement_seconds},allow_nan=False)+'\n')
                counts[variant['name']]+=len(values)
            if position==0 or (position+1)%25==0 or position+1==len(indices):
                for stream in streams.values():stream.flush()
                write_json(output/f'progress_rank{rank}.json',{'phase':'inferring','images':position+1,
                    'expected':len(indices),'instances':counts,'elapsed_seconds':time.monotonic()-started})
        write_json(output/f'rank{rank}.json',{'status':'completed','rank':rank,'physical_gpu':gpus[rank],
            'ordered_ids':[data.image_ids[i] for i in indices],'images':len(indices),'instances':counts,
            'predictions_sha256':{v['name']:sha(output/v['name']/f'predictions_rank{rank}.jsonl') for v in arm['variants']},
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started})


def finish(p,arm,output,submissions):
    ids=read(output/'image_ids.json'); receipts={}
    require(len(ids)==2112 and len(set(ids))==2112,'Incomplete test metadata')
    for variant in arm['variants']:
        name=variant['name'];folder=output/name
        paths=[folder/f'predictions_rank{rank}.jsonl' for rank in range(len(p['gpus']))]
        for rank,path in enumerate(paths):
            r=read(output/f'rank{rank}.json')
            require(r['status']=='completed' and r['ordered_ids']==ids[rank::len(p['gpus'])]
                    and r['predictions_sha256'][name]==sha(path),'Shard identity/coverage differs')
            require(read(output/f'reload_rank{rank}.json')['status']=='passed','Reload checks missing')
        count=merge_shards(paths,ids,folder/'predictions.jsonl')
        csv_path=submissions/variant['filename']
        write_json(output/'finish_progress.json',{'phase':'writing_csv','variant':name})
        with SubmissionWriter(csv_path,ids) as writer,(folder/'predictions.jsonl').open() as stream:
            for line in stream:
                item=json.loads(line);writer.write(item['image_id'],item['predictions'])
        write_json(output/'finish_progress.json',{'phase':'native_rle_validation','variant':name})
        validation=validate_csv_fast(csv_path,ids)
        require(validation['rows']==2112 and validation['predictions']==count,'CSV coverage differs')
        write_json(folder/'submission_validation.json',validation)
        receipt={'status':'completed','images':2112,'instances':count,'csv_path':str(csv_path),
            'csv_sha256':sha(csv_path),'predictions_path':str(folder/'predictions.jsonl'),
            'predictions_sha256':sha(folder/'predictions.jsonl'),'test_ground_truth_used':False,
            'test_scored_locally':False,'variant':variant,'validation_path':str(folder/'submission_validation.json')}
        write_json(folder/'export.json',receipt);receipts[name]=receipt
    write_json(output/'finish.json',{'status':'completed','variants':receipts})


def parent(p,path,output):
    require(not output.exists(),'Output must be new'); output.mkdir(parents=True)
    submissions=Path(p['submissions_dir']);submissions.mkdir(parents=True,exist_ok=True)
    run={'status':'running','pid':os.getpid(),'started_at_utc':datetime.now(timezone.utc).isoformat(),
         'protocol_sha256':sha(path),'source_sha256':p['source_sha256'],'gpus':p['gpus'],
         'arms':{},'active_arm':None,'test_ground_truth_used':False,'test_scored_locally':False}
    write_json(output/'run.json',run);children=[];finishers=[]
    def spawn(command,log,gpu=None):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES='' if gpu is None else str(gpu),
                 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
        child=subprocess.Popen(command,env=env,stdout=log,stderr=subprocess.STDOUT);children.append(child);return child
    try:
        with CourseArchive(p['archive'],'test') as data:
            ids=data.image_ids;require(len(ids)==2112 and data.submission_rows is not None,'Missing official test template')
        with ExitStack() as stack:
            for arm in p['arms']:
                folder=output/arm['name'];folder.mkdir();write_json(folder/'image_ids.json',ids)
                for variant in arm['variants']:(folder/variant['name']).mkdir()
                workers=[]
                for rank,gpu in enumerate(p['gpus']):
                    log=stack.enter_context((folder/f'worker{rank}.log').open('x'))
                    cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--protocol',str(path),'--output',str(output),
                         '--arm',arm['name'],'--rank',str(rank)]
                    workers.append(spawn(cmd,log,gpu))
                run['active_arm']=arm['name'];run['arms'][arm['name']]={'phase':'inferring',
                    'workers':[{'rank':i,'physical_gpu':p['gpus'][i],'pid':x.pid} for i,x in enumerate(workers)]}
                write_json(output/'run.json',run);began=time.monotonic()
                while any(x.poll() is None for x in workers):
                    require(all(x.poll() in (None,0) for x in children),'An export process failed; inspect logs')
                    require(time.monotonic()-began<p['worker_timeout_seconds'],'Inference deadline exceeded');time.sleep(1)
                require(all(x.returncode==0 for x in workers),'Inference worker failed')
                log=stack.enter_context((folder/'finish.log').open('x'))
                cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'--protocol',str(path),
                     '--output',str(output),'--arm',arm['name'],'--finish']
                finisher=spawn(cmd,log);finishers.append(finisher)
                run['arms'][arm['name']].update(phase='writing_and_validating_csv',finisher_pid=finisher.pid,
                                                worker_exit_codes=[x.returncode for x in workers])
                write_json(output/'run.json',run)
            run['active_arm']=None;write_json(output/'run.json',run);began=time.monotonic()
            while any(x.poll() is None for x in finishers):
                require(all(x.poll() in (None,0) for x in children),'CSV validation process failed')
                require(time.monotonic()-began<p['finisher_timeout_seconds'],'CSV validation deadline exceeded');time.sleep(1)
            require(all(x.returncode==0 for x in finishers),'CSV validation process failed')
            for arm in p['arms']:
                finished=read(output/arm['name']/'finish.json');require(finished['status']=='completed','Missing CSV receipt')
                run['arms'][arm['name']].update(phase='completed',results=finished['variants'])
        run.update(status='completed',exit_code=0,finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as e:
        run.update(status='failed',exit_code=1,error_type=type(e).__name__,error=str(e))
        for child in children:
            if child.poll() is None:child.terminate()
        for child in children:
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait()
        raise
    finally:
        write_json(output/'run.json',run)
        write_json(output/'exit.json',{'status':run['status'],'exit_code':run.get('exit_code',1),
                                     'process_exit_codes':[x.returncode for x in children]})
    print(json.dumps({'status':run['status'],'arms':list(run['arms'])}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--arm');parser.add_argument('--rank',type=int);parser.add_argument('--finish',action='store_true')
    args=parser.parse_args();p=checked(args.protocol)
    if args.arm:
        arm=next(x for x in p['arms'] if x['name']==args.arm);folder=args.output.resolve()/arm['name']
        if args.finish:finish(p,arm,folder,Path(p['submissions_dir']))
        else:
            require(args.rank is not None and 0<=args.rank<len(p['gpus']),'Invalid rank');worker(p,arm,folder,args.rank)
    else:parent(p,args.protocol.resolve(),args.output.resolve())


if __name__=='__main__':main()
