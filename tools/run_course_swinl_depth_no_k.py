#!/usr/bin/env python3
"""Train the missing Large + uncalibrated-depth arm with the frozen prior budget.

Uses the existing model/train/eval implementations unchanged. Engineering is a
separate run; formal base training starts afresh from public COCO initialization.
No automatic crash retries, uploads, or changes to completed experiments.
"""
from datetime import datetime, timezone
import copy
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.queue_course_depth_backbone import read, write, sha, command, system_protocol, completed_base, check_results

OUT = ROOT/'reports/course_swinl_depth_no_k_v1'
PRIOR = ROOT/'reports/course_depth_backbone_v1_gpu02'
ARM = 'swinl_depth_no_k'
LEDGER = Path('/data/yinli/.codex/skills/academic-research-suite/ars/scripts/run_ledger.py')

def now():
    return datetime.now(timezone.utc).isoformat()

def append(entry):
    path = OUT/'ledger_entry_pending.json'
    write(path,entry)
    subprocess.run([sys.executable,str(LEDGER),'append','--passport-path',str(OUT/'plan.json'),
                    '--entry-file',str(path)],check=True,cwd=ROOT,stdout=subprocess.DEVNULL)
    path.unlink()

def reference(path,role):
    append({'kind':'file_reference','path':str(path),'role':role})

def check_sources(expected):
    for name,digest in expected.items():
        path=Path(name)
        if not path.is_absolute(): path=ROOT/path
        if sha(path)!=digest: raise ValueError('Frozen source changed: '+name)

def check_architecture(path):
    import torch
    import re
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    metadata=saved['model_metadata']
    if metadata['backbone']!='large' or metadata['depth_encoding']!='depth' or metadata['camera_intrinsics_used']:
        raise ValueError('Actual checkpoint must be Large with no intrinsics')
    for branch in ['swin_rgb','swin_d']:
        prefix='model.model.pixel_level_module.encoder.'+branch+'.'
        if tuple(saved['model'][prefix+'embeddings.patch_embeddings.projection.weight'].shape)!=(192,3,4,4):
            raise ValueError('Actual patch embedding is not Large')
        blocks={int(m.group(1)) for k in saved['model'] if (m:=re.match(re.escape(prefix)+r'encoder\.layers\.2\.blocks\.(\d+)\.',k))}
        if blocks!=set(range(18)): raise ValueError('Actual third stage is not Large')
    return {'backbone':'large','depth_encoding':'depth','camera_intrinsics_used':False,
            'embedding_channels':192,'third_stage_blocks':18,'epoch':saved['epoch'],'parameters':metadata['parameters']}

def main():
    from tools.run_course_depth_backbone import sources
    from tools.run_course_depth_backbone_system import sources as system_sources
    if (OUT/'queue.json').exists(): raise FileExistsError('Existing run; inspect before taking further action')
    if shutil.disk_usage(ROOT).free < 50*2**30: raise OSError('Need at least 50 GiB free before this Large run')
    prior=read(PRIOR/'swinl_xyz/protocol.json')
    if prior['source_sha256']!=sources() or prior['frozen_system_source_sha256']!=system_sources():
        raise ValueError('Prior frozen model or training source changed')
    check_sources(prior['source_sha256']); check_sources(prior['frozen_system_source_sha256'])
    folder=OUT/ARM
    folder.mkdir(parents=True,exist_ok=False)
    protocol=copy.deepcopy(prior)
    protocol.pop('engineering_validation',None)
    protocol.pop('hardware_amendment',None)
    protocol['status']='engineering'
    protocol['model']['depth_encoding']='depth'
    protocol['hardware']['physical_gpus']=['0','2']
    write(folder/'protocol_engineering.json',protocol)
    shutil.copyfile(PRIOR/'test_images.json',OUT/'test_images.json')
    plan={'schema':'course_swinl_depth_no_k_v1','status':'engineering',
        'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','mode':'run','verification_status':'ENGINEERING_PENDING'},
        'user_objective':'那就训练一下large+无内参',
        'authorization':'User authorized Large with no intrinsics; same full training/evaluation budget as the previous comparison.',
        'arm':{'id':ARM,'backbone':'large','depth_encoding':'depth','camera_intrinsics_used':False},
        'training':{'base_epochs':15,'point_epochs':6,'resolution':1024,'seed':17,'effective_batch':33,
                    'base_initialization':'Same public COCO Swin-L as previous XYZ Large; fresh course training',
                    'point_base':'Frozen fixed epoch15; fresh point heads; full-validation-only selection'},
        'evaluation':{'train_images':14432,'val_images':2584,'test_images':2112,
                    'variants':['native','full'],'main_metric':'COCO mask APsmall @[.50:.95]',
                    'full':'Depth point head + whole image and four640 crops + depth probability refinement',
                    'test_used_for_selection':False,'outputs':['native and full test CSV','independent CSV COCO AP and APsmall']},
        'hardware':protocol['hardware'],'reference_protocol':str(PRIOR/'swinl_xyz/protocol.json'),
        'reference_protocol_sha256':sha(PRIOR/'swinl_xyz/protocol.json'),
        'comparison_scope':'Against completed Large XYZ and Tiny depth; same data and frozen computation sources',
        'limitations':['Single seed17','No-intrinsics encoding replaces XYZ with normalized measured Z plus validity',
                       'Local archive test scores are separate from Kaggle public scores'],
        'created_at_utc':now()}
    write(OUT/'plan.json',plan)
    append({'kind':'initial_instructions','user_words':'那就训练一下large+无内参'})
    reference(folder/'protocol_engineering.json','large_no_intrinsics_engineering_protocol')
    reference(Path(__file__),'large_no_intrinsics_controller_source')
    state={'status':'engineering','pid':os.getpid(),'started_at_utc':now(),'stages':[],
           'controller_sha256':sha(__file__),'physical_gpus':[0,2],'arm':ARM}
    write(OUT/'queue.json',state)
    env=dict(os.environ)
    env.update(CUDA_VISIBLE_DEVICES='0,2',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',
               TOKENIZERS_PARALLELISM='false',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')

    def execute(name,argv,output,timeout=605000):
        stage={'name':name,'command':argv,'output':str(output),'status':'starting','started_at_utc':now(),
               'hard_timeout_seconds':timeout}
        state['stages'].append(stage);state['current_stage']=name
        write(OUT/'queue.json',state)
        logpath=OUT/(name+'.log')
        with logpath.open('x') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            stage.update(status='running',pid=child.pid,start_ticks=Path(f'/proc/{child.pid}/stat').read_text().split(') ')[1].split()[19],log=str(logpath))
            began=time.monotonic()
            write(OUT/'queue.json',state)
            while child.poll() is None:
                elapsed=time.monotonic()-began
                if elapsed>timeout:
                    os.killpg(child.pid,signal.SIGTERM)
                    try: child.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(child.pid,signal.SIGKILL);child.wait()
                    raise TimeoutError('Declared hard timeout: '+name)
                stage.update(last_observed_alive_at_utc=now(),elapsed_seconds=elapsed,log_bytes=logpath.stat().st_size)
                write(OUT/'queue.json',state)
                time.sleep(30)
            stage.update(status='completed' if child.returncode==0 else 'failed',exit_code=child.returncode,
                         finished_at_utc=now(),elapsed_seconds=time.monotonic()-began)
            write(OUT/'queue.json',state)
            append({'kind':'tool_receipt','step':name,'command':' '.join(argv),'status':'passed' if child.returncode==0 else 'failed',
                    'exit_status':child.returncode,'retries_used':0})
            if child.returncode: raise RuntimeError(name+' failed; no automatic retry')
        receipt=read(output/'run.json')
        if receipt['status']!='completed':raise ValueError('Missing completed receipt: '+name)
        reference(output/'run.json',name+'_completed_run')

    try:
        # Exercise dense normal batches and the actual uneven tail in one run.
        engineering=folder/'engineering_dense_tail'
        execute('engineering_dense_tail',command('run_course_depth_backbone.py','--protocol',folder/'protocol_engineering.json',
            '--output',engineering,'--resolution',1024,'--smoke-steps',2,'--smoke-tail-images',2,'--smoke-dense'),engineering,1800)
        erun=read(engineering/'run.json')
        arch=check_architecture(engineering/'final.pt')
        if erun['optimizer_steps']!=2 or erun['training_image_exposures']!=35:
            raise ValueError('Engineering did not cover normal and tail updates')
        if erun['training_data']['camera_intrinsics_read_by_input_loader']:
            raise ValueError('No-intrinsics input loader used camera parameters')
        eng_system=system_protocol(protocol,engineering,scope='engineering_smoke')
        eng_system['limits']={**eng_system['limits'],'per_trial_seconds':1800}
        write(folder/'protocol_system_engineering.json',eng_system)
        ep=folder/'engineering_points'
        execute('engineering_points',command('run_course_depth_backbone_system.py','--protocol',folder/'protocol_system_engineering.json',
            '--base',engineering/'final.pt','--output',ep,'--mode','points','--smoke'),ep,1800)
        chosen=read(ep/'selection.json')
        if not all(chosen[k] for k in ['base_unchanged','base_gradients_absent','head_replicas_equal']):
            raise ValueError('Engineering point training changed frozen base or diverged')
        ee=folder/'engineering_system'
        execute('engineering_system',command('run_course_depth_backbone_system.py','--protocol',folder/'protocol_system_engineering.json',
            '--base',engineering/'final.pt','--heads',chosen['selected']['checkpoint'],'--output',ee,'--mode','evaluate','--smoke'),ee,1800)
        if set(read(ee/'comparison.json')['val'])!={'native','full'}:raise ValueError('Engineering inference variants incomplete')
        paths=[engineering/'run.json',engineering/'model_sync_epoch_01.json',engineering/'final.pt',ep/'run.json',ep/'selection.json',ee/'run.json',ee/'comparison.json']
        gate={'status':'passed','scope':'engineering_only','architecture':arch,'dense_and_tail_updates':2,'images':35,
              'source_sha256':sources(),'system_source_sha256':system_sources(),'peak_allocated_bytes':erun['rank_peak_allocated_bytes'],
              'evidence_sha256':{str(p):sha(p) for p in paths},'finished_at_utc':now()}
        write(folder/'engineering_gate.json',gate)
        reference(folder/'engineering_gate.json','large_no_intrinsics_actual_gpu_engineering_gate')
        protocol.update(status='frozen',engineering_validation={'path':str(folder/'engineering_gate.json'),'sha256':sha(folder/'engineering_gate.json')})
        write(folder/'protocol.json',protocol)
        reference(folder/'protocol.json','frozen_formal_large_no_intrinsics_protocol')
        plan.update(status='running',material_passport={**plan['material_passport'],'verification_status':'ENGINEERING_PASSED_FORMAL_RUNNING'},
                    formal_protocol_sha256=sha(folder/'protocol.json'))
        write(OUT/'plan.json',plan)
        state['status']='running';write(OUT/'queue.json',state)
        base=folder/'base15'
        execute('base15',command('run_course_depth_backbone.py','--protocol',folder/'protocol.json','--output',base,'--resolution',1024),base)
        completed_base(base,protocol)
        if check_architecture(base/'final.pt')['epoch']!=15:raise ValueError('Expected fixed base epoch15')
        system=system_protocol(protocol,base)
        system.update(official_test_metadata=str(OUT/'test_images.json'),head_selection=str(folder/'points6/selection.json'))
        write(folder/'protocol_system.json',system)
        points=folder/'points6'
        execute('points6',command('run_course_depth_backbone_system.py','--protocol',folder/'protocol_system.json','--base',base/'final.pt',
            '--output',points,'--mode','points'),points)
        selected=read(points/'selection.json')
        if (selected['status']!='locked' or selected['epochs']!=6 or selected['training_images']!=14432
                or selected['global_steps']!=10824 or selected['test_used']
                or not all(selected[k] for k in ['base_unchanged','base_gradients_absent','head_replicas_equal'])):
            raise ValueError('Point training or validation-only selection incomplete')
        evaluation=folder/'final_evaluation'
        execute('final_evaluation',command('run_course_depth_backbone_system.py','--protocol',folder/'protocol_system.json',
            '--base',base/'final.pt','--heads',selected['selected']['checkpoint'],'--output',evaluation,'--mode','evaluate'),evaluation)
        result=check_results(evaluation)
        # Fresh CPU COCO scoring from the actual exported CSVs, reusing the
        # previously tested parser and scorer but no hard-coded previous arm ID.
        import torch
        from tools.audit_course_depth_backbone import csv_detections,rescore
        from rgbd_seg.course_archive import CourseArchive
        from pycocotools.coco import COCO
        torch.set_num_threads(4)
        with CourseArchive(protocol['archive'],'test') as dataset:
            ids=dataset.image_ids
            if ids!=[v['id'] for v in read(OUT/'test_images.json')['images']]:raise ValueError('Official test IDs/order changed')
            coco=COCO();coco.dataset=dataset.coco_ground_truth();coco.createIndex()
        submissions=OUT/'submissions';submissions.mkdir(exist_ok=False)
        rows=[]
        for variant in ['native','full']:
            path=evaluation/f'test_{variant}'
            before=sha(path/'submission.csv')
            detections=csv_detections(path/'submission.csv',path/'predictions.jsonl',ids)
            metrics=rescore(coco,detections,ids,folder/f'independent_csv_{variant}_coco.log')
            differences={k:abs(v-result['test'][variant]['metrics'][k]) for k,v in metrics.items()}
            if max(differences.values())>1e-12 or sha(path/'submission.csv')!=before:raise ValueError('Independent CSV score differs')
            target=submissions/f'dpseg_{ARM}_base15_{variant}.csv';os.link(path/'submission.csv',target)
            row={'arm':ARM,'variant':variant,'csv':str(target),'sha256':before,'instances':len(detections),
                 'metrics':metrics,'absolute_differences':differences}
            rows.append(row)
            reference(target,ARM+'_'+variant+'_final_submission')
        audit={'status':'passed','scope':'full_test_2112_csv_rescoring','architecture':check_architecture(base/'final.pt'),
            'selected_head_epoch':selected['selected']['epoch'],'test_used_for_selection':False,'rows':rows,'finished_at_utc':now(),
            'controller_sha256':sha(__file__),'plan_sha256':sha(OUT/'plan.json')}
        write(folder/'independent_audit.json',audit)
        reference(folder/'independent_audit.json','large_no_intrinsics_two_csv_independent_coco_rescore')
        write(OUT/'comparison.json',{'status':'completed','rows':rows,'test_used_for_selection':False,'finished_at_utc':now(),
             'reference_verified_comparison_sha256':sha(PRIOR/'verified_comparison.json'),'limitations':plan['limitations']})
        lines=['# Swin-Large＋无内参深度实验','', '## Material Passport','',
               '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run','- Verification Status: FORMAL_CSV_RESCORED','',
               '完整本地测试集2,112张，单种子17；此处为本地分数，不是Kaggle分数。','',
               '| 配置 | AP | APₛ |','|---|---:|---:|']
        for row in rows:lines.append(f"| {row['variant']} | {row['metrics']['mask_ap']:.6f} | {row['metrics']['mask_ap_small']:.6f} |")
        (OUT/'comparison.md').write_text('\n'.join(lines)+'\n')
        reference(OUT/'comparison.json','large_no_intrinsics_final_comparison')
        append({'kind':'progress','stage':ARM,'counters':{'base_epochs_completed':15,'point_epochs_completed':6,'independently_scored_test_csvs':2}})
        state.update(status='completed',finished_at_utc=now(),comparison_sha256=sha(OUT/'comparison.json'))
        write(OUT/'queue.json',state)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error),failed_at_utc=now())
        write(OUT/'queue.json',state)
        raise

if __name__=='__main__':main()
