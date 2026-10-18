#!/usr/bin/env python3
"""Frozen-base, paired point-head training and complete val/test evaluation."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timedelta, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.distributed as dist
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Subset
from transformers.models.mask2former.modeling_mask2former import Mask2FormerHungarianMatcher

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import CourseDepthTrainingDataset, load_inference_sample
from rgbd_seg.course_dafusion_eval import score_predictions
from rgbd_seg.course_distributed_eval import merge_shards, write_json
from rgbd_seg.course_finetune_eval import validate_csv_fast
from rgbd_seg.course_optimized import file_sha, load_for_inference, tensor_digest
from rgbd_seg.course_point_refinement import POLICY, PairedPointHeads, predict_paired, training_batch
from rgbd_seg.course_submission import SubmissionWriter
from rgbd_seg.course_tiled_probability import refined_public
from tools.check_course_prediction_reload import compare_predictions

VARIANTS=('native','native_probability_depth','point','point_probability_depth',
          'point_depth','point_depth_probability_depth')
METHODS=VARIANTS[2:]


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def require(condition,message):
    if not condition:
        raise ValueError(message)


def checked(path):
    protocol=read(path)
    require(protocol['status']=='frozen','Protocol must be frozen')
    require(protocol['scope'] in ('engineering_smoke','full_experiment'),'Invalid experiment scope')
    require(protocol['point_policy']==POLICY,'Point-head policy changed')
    archive_stat=Path(protocol['archive']).stat()
    require(protocol['archive_stat']=={'bytes':archive_stat.st_size,'mtime_ns':archive_stat.st_mtime_ns},
            'Dataset archive identity changed')
    for name,digest in protocol['source_sha256'].items():
        require(file_sha(ROOT/name)==digest,f'Frozen source changed: {name}')
    for key in ('weights','training_protocol'):
        require(file_sha(protocol[key]['path'])==protocol[key]['sha256'],f'Frozen input changed: {key}')
    protocol['_path']=str(Path(path).resolve())
    protocol['_sha256']=file_sha(path)
    return protocol


def collate(items):
    return [item[0] for item in items],[item[1] for item in items]


def save_heads(path,heads,epoch,protocol,base_receipt):
    value={'heads':{key:tensor.detach().cpu() for key,tensor in heads.state_dict().items()},
           'epoch':epoch,'protocol_sha256':protocol['_sha256'],
           'source_sha256':protocol['source_sha256'],
           'base_checkpoint_sha256':base_receipt['checkpoint_sha256'],
           'base_model_tensor_sha256':base_receipt['model_sha256'],
           'point_policy':POLICY}
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)
    return {'path':str(path.resolve()),'sha256':file_sha(path),
            'head_tensor_sha256':tensor_digest(value['heads']),'epoch':epoch}


def load_heads(path,heads,protocol):
    value=torch.load(path,map_location='cpu',weights_only=True)
    require(value['protocol_sha256']==protocol['_sha256'] and value['source_sha256']==protocol['source_sha256'],
            'Point checkpoint provenance differs')
    require(value['point_policy']==POLICY and value['base_checkpoint_sha256']==protocol['weights']['sha256'],
            'Point checkpoint base/policy differs')
    heads.load_state_dict(value['heads'],strict=True)
    require(tensor_digest(heads.state_dict())==tensor_digest(value['heads']),'Point checkpoint reload differs')
    return value


def score_arm(protocol,folder,split,variant):
    ids=read(folder/'image_ids.json')
    paths=[folder/variant/f'predictions_rank{rank}.jsonl' for rank in range(len(protocol['gpus']))]
    for rank,path in enumerate(paths):
        receipt=read(folder/f'rank{rank}.json')
        require(receipt['status']=='completed' and receipt['ordered_ids']==ids[rank::len(paths)]
                and receipt['prediction_sha256'][variant]==file_sha(path),'Prediction shard changed/incomplete')
    count=merge_shards(paths,ids,folder/variant/'predictions.jsonl')
    with CourseArchive(protocol['archive'],split) as data:
        result=score_predictions(data,folder/variant/'predictions.jsonl',folder/variant,
            image_ids=ids if protocol['scope']=='engineering_smoke' else None)
        require(result['images']==len(ids) and result['predicted_instances']==count,'Scoring coverage differs')
    result.update(scope=protocol['scope'],split=split,variant=variant)
    write_json(folder/variant/'evaluation.json',result)
    if split=='test' and variant in METHODS:
        epoch=read(folder/'head_checkpoint.json')['epoch']
        names={'point':'point','point_probability_depth':'point_probdepth',
               'point_depth':'point_depth','point_depth_probability_depth':'point_depth_probdepth'}
        path=folder/variant/f'rgbd_dafusion_swin_t_1024_e05_{names[variant]}_head_e{epoch:02d}.csv'
        with SubmissionWriter(path,ids) as writer,(folder/variant/'predictions.jsonl').open() as source:
            for line in source:
                item=json.loads(line)
                writer.write(item['image_id'],item['predictions'])
        receipt=validate_csv_fast(path,ids)
        write_json(folder/variant/'submission_validation.json',receipt)
    write_json(folder/variant/'scored.json',{'status':'completed','images':len(ids),
        'prediction_sha256':file_sha(folder/variant/'predictions.jsonl'),
        'evaluation_sha256':file_sha(folder/variant/'evaluation.json')})


def run_scorers(protocol,folder,split):
    children=[]
    for offset in range(0,len(VARIANTS),3):
        batch=[]
        with ExitStack() as stack:
            for variant in VARIANTS[offset:offset+3]:
                log=stack.enter_context((folder/f'score_{variant}.log').open('x'))
                command=[sys.executable,str(Path(__file__).resolve()),'score','--protocol',protocol['_path'],
                         '--output',str(folder),'--split',split,'--variant',variant]
                env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'}
                proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
                batch.append((variant,proc))
            for variant,proc in batch:
                code=proc.wait()
                children.append({'variant':variant,'pid':proc.pid,'exit_code':code})
        write_json(folder/'scorer_processes.json',children)
        require(all(child['exit_code']==0 for child in children),'A scorer failed; see its log')


def evaluate(protocol,adapter,heads,folder,split,rank,world,checkpoint):
    started=time.monotonic()
    if rank==0:
        folder.mkdir(parents=True,exist_ok=False)
        for variant in VARIANTS:
            (folder/variant).mkdir()
        write_json(folder/'head_checkpoint.json',checkpoint)
        with CourseArchive(protocol['archive'],split) as data:
            ids=data.image_ids
            expected=2584 if split=='val' else 2112
            require(len(ids)==expected,'Official split coverage differs')
            if protocol['scope']=='engineering_smoke':
                ids=ids[:2*world]
            write_json(folder/'image_ids.json',ids)
    dist.barrier()
    ids=read(folder/'image_ids.json')
    local_ids=ids[rank::world]
    heads.eval()
    timing={'native_seconds':0.,'point_extra_seconds':0.,'point_depth_extra_seconds':0.,
            'probability_refinement_seconds':{key:0. for key in VARIANTS if 'probability' in key}}
    totals={key:0 for key in VARIANTS}
    with CourseArchive(protocol['archive'],split) as data,ExitStack() as stack,torch.no_grad():
        streams={variant:stack.enter_context((folder/variant/f'predictions_rank{rank}.jsonl').open('x'))
                 for variant in VARIANTS}
        first=load_inference_sample(data,local_ids[0])
        for _ in range(3):
            adapter.predict_sample(first)
        for index,image_id in enumerate(local_ids):
            sample=load_inference_sample(data,image_id)
            paired,times=predict_paired(adapter,heads,sample,return_timing=True)
            for key,value in times.items():
                timing[key]=timing.get(key,0.)+value
            if index==0:
                compare_predictions(adapter.predict_sample(sample),[c.public() for c in paired['native']])
            outputs={key:[c.public() for c in values] for key,values in paired.items()}
            for key,values in paired.items():
                torch.cuda.synchronize(adapter.device)
                began=time.perf_counter()
                name='native_probability_depth' if key=='native' else key+'_probability_depth'
                outputs[name],_=refined_public(values,sample,'probability',adapter.device)
                torch.cuda.synchronize(adapter.device)
                timing['probability_refinement_seconds'][name]+=time.perf_counter()-began
            identity=[(v['category_id'],v['score']) for v in outputs['native']]
            for variant,values in outputs.items():
                require([(v['category_id'],v['score']) for v in values]==identity,
                        'Refinement changed instance count/order/scores')
                totals[variant]+=len(values)
                streams[variant].write(json.dumps({'image_id':image_id,'predictions':values},allow_nan=False)+'\n')
            if index==0 or (index+1)%10==0 or index+1==len(local_ids):
                for stream in streams.values():stream.flush()
                write_json(folder/f'progress_rank{rank}.json',{'phase':'inferring','split':split,'rank':rank,
                    'images':index+1,'expected':len(local_ids),'elapsed_seconds':time.monotonic()-started})
    write_json(folder/f'rank{rank}.json',{'status':'completed','rank':rank,'pid':os.getpid(),
        'ordered_ids':local_ids,'images':len(local_ids),'instances':totals,'timing':timing,
        'prediction_sha256':{key:file_sha(folder/key/f'predictions_rank{rank}.jsonl') for key in VARIANTS},
        'gpu_name':torch.cuda.get_device_name(adapter.device),
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'elapsed_seconds':time.monotonic()-started,'ground_truth_in_prediction':False})
    dist.barrier()
    result=None
    if rank==0:
        write_json(folder/'progress.json',{'phase':'scoring','split':split,'images':len(ids)})
        run_scorers(protocol,folder,split)
        result={'epoch':checkpoint['epoch'],'checkpoint':checkpoint,'split':split,'images':len(ids),
                'metrics':{key:read(folder/key/'evaluation.json')['metrics'] for key in VARIANTS},
                'folder':str(folder.resolve()),'elapsed_seconds':time.monotonic()-started}
        rank_times=[read(folder/f'rank{r}.json')['timing'] for r in range(world)]
        base=sum(row['native_seconds'] for row in rank_times)
        roi=sum(row.get('roi_preparation_seconds',0.) for row in rank_times)
        cues=sum(row.get('depth_cue_seconds',0.) for row in rank_times)
        totals={'native':base,'point':base+roi+sum(row['point_extra_seconds'] for row in rank_times),
                'point_depth':base+roi+cues+sum(row['point_depth_extra_seconds'] for row in rank_times)}
        for key in ('native','point','point_depth'):
            name='native_probability_depth' if key=='native' else key+'_probability_depth'
            totals[name]=totals[key]+sum(row['probability_refinement_seconds'][name] for row in rank_times)
        result['timing_mean_ms']={key:1000*value/len(ids) for key,value in totals.items()}
        result['timing_scope']='batch1 preprocessing/model/point head/CPU RLE/postprocessing, excludes ZIP I/O; shared stages attributed to each arm; mixed A800/A100 devices'
        write_json(folder/'comparison.json',result)
        write_json(folder/'progress.json',{'phase':'completed','split':split,'images':len(ids)})
    payload=[result]
    dist.broadcast_object_list(payload,src=0)
    return payload[0]


def select_validation(history):
    candidates=[]
    for result in history:
        baseline=result['metrics']['native_probability_depth']
        for variant in METHODS:
            metrics=result['metrics'][variant]
            gain=metrics['mask_ap_small']-baseline['mask_ap_small']
            drop=baseline['mask_ap']-metrics['mask_ap']
            candidates.append({'epoch':result['epoch'],'variant':variant,'mask_ap_small':metrics['mask_ap_small'],
                'mask_ap':metrics['mask_ap'],'gain_vs_native_probability_depth':gain,
                'AP_drop_vs_native_probability_depth':drop,'eligible':gain>=.005 and drop<=.005,
                'checkpoint':result['checkpoint']})
    eligible=[item for item in candidates if item['eligible']]
    selected=max(eligible or candidates,key=lambda item:(item['mask_ap_small'],item['mask_ap'],-item['epoch']))
    return {'selected':selected,'candidates':candidates,'selection_split':'val',
            'test_labels_used_for_selection':False,'recommended_over_existing_native_depth':bool(eligible)}


def finish_report(protocol,output,history,selection,test,rank_receipts):
    names={'native':'原模型','native_probability_depth':'原模型＋概率深度细化',
        'point':'普通点细化头','point_probability_depth':'普通点头＋概率深度细化',
        'point_depth':'深度点细化头','point_depth_probability_depth':'深度点头＋概率深度细化'}
    result={'status':'completed','scope':protocol['scope'],
        'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','mode':'run',
                             'verification_status':'VERIFIED_COMPLETE_SPLITS' if test else 'ENGINEERING_ONLY'},
        'protocol_sha256':protocol['_sha256'],'validation_history':history,'selection':selection,
        'test':test,'rank_receipts':rank_receipts,'finished_at_utc':now(),
        'limits':['One seed and two head-training epochs; no significance claim',
                  'PointRend-inspired residual head, GT training ROIs and prediction-only inference ROIs',
                  'Plain head already receives fused RGB-D features; depth arm adds explicit metric local cues',
                  'Fixed scores isolate mask geometry; no new queries/detections',
                  'Training uses whole images; this experiment does not include tile inference']}
    write_json(output/'comparison.json',result)
    lines=['# DA-Fusion 点细化头实验','','## Material Passport','',
           f"- 状态：{result['material_passport']['verification_status']}",
           '- 训练：冻结 DA-Fusion 1024 e05；只训练两个点头。',
           f"- 范围：{protocol['scope']}；训练 {protocol['training_images']} 张、{protocol['epochs']} 轮。",'',
           '## 验证集','', '| 点头轮次 | 配置 | Mask AP | Mask APₛ |','|---:|---|---:|---:|']
    for item in history:
        for key,metrics in item['metrics'].items():
            lines.append(f"| {item['epoch']} | {names[key]} | {metrics['mask_ap']:.6f} | {metrics['mask_ap_small']:.6f} |")
    choice=selection['selected']
    lines+=['',f"按验证集选择：第 {choice['epoch']} 轮，{names[choice['variant']]}。",
        f"是否达到相对现有整图概率深度版的预设提升门槛：{selection['recommended_over_existing_native_depth']}。",'',
        '门槛：APₛ 至少提高 0.005，整体 AP 下降不超过 0.005。未达门槛也保留完整结果。']
    if test:
        lines+=['','## 完整测试集','', '| 配置 | Mask AP | Mask APₛ | 平均推理 ms | CSV |','|---|---:|---:|---:|---|']
        for key,metrics in test['metrics'].items():
            csvs=list((Path(test['folder'])/key).glob('*.csv'))
            link=f'[{csvs[0].name}]({csvs[0].resolve()})' if csvs else '历史基线'
            lines.append(f"| {names[key]} | {metrics['mask_ap']:.6f} | {metrics['mask_ap_small']:.6f} | {test['timing_mean_ms'][key]:.1f} | {link} |")
        lines+=['','课程下载包含测试标注；仅最终独立评测使用，未参与点头训练或选择。',
                '本地完整测试集 APₛ 与 Kaggle 约 30% 公榜分数不同，本报告不是 Kaggle 得分。',
                '此前切片＋概率深度版完整测试集 AP=0.850086、APₛ=0.436858；本轮是整图点头实验。']
    lines+=['','## 实现与限制','',*['- '+item for item in result['limits']], '',
        '- 点头：3×256 MLP，输入 stride4 融合特征和像素解码特征，以及该实例粗掩码 logit。',
        '- 深度版额外输入 9 维局部深度差与有效性；新增深度通道初始化为零，其余初始化对齐。',
        '- 每实例训练 784 个点；推理在 512 和 1024 两级分别选择至多 784 个不确定点。',
        '- 保存主模型前后权重摘要、冻结梯度检查、训练图像覆盖、各子进程退出码、CSV 校验。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


def train(protocol,output):
    local_rank=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    require(os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,protocol['gpus']))
            and world==len(protocol['gpus']),'Unexpected visible devices/world size')
    torch.cuda.set_device(local_rank)
    dist.init_process_group('nccl',timeout=timedelta(minutes=90),device_id=torch.device(f'cuda:{local_rank}'))
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    if rank==0:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'run.json',{'status':'starting','pid':os.getpid(),'started_at_utc':now(),
            'protocol_sha256':protocol['_sha256'],'gpus':protocol['gpus'],'scope':protocol['scope']})
    dist.barrier()
    started=time.monotonic()
    device=f'cuda:{local_rank}'
    adapter,base_receipt=load_for_inference(protocol['weights']['path'],protocol['training_protocol']['path'],device=device)
    base_digest=tensor_digest(adapter.state_dict())
    torch.manual_seed(protocol['seed'])
    heads=PairedPointHeads().to(device)
    ddp=DistributedDataParallel(heads,device_ids=[local_rank],broadcast_buffers=False)
    optimizer=torch.optim.AdamW(ddp.parameters(),lr=protocol['learning_rate'],weight_decay=.01)
    torch.manual_seed(protocol['seed']+10000*rank)
    cfg=adapter.model.config
    matcher=Mask2FormerHungarianMatcher(cost_class=cfg.class_weight,cost_mask=cfg.mask_weight,
        cost_dice=cfg.dice_weight,num_points=cfg.train_num_points)
    history=[];global_step=0
    try:
        with CourseDepthTrainingDataset(protocol['archive'],'train',augment=True,seed=protocol['seed']) as data:
            require(len(data)==14432,'Course train split differs')
            indices=protocol.get('training_indices',list(range(len(data))))
            require(len(indices)==protocol['training_images'] and len(set(indices))==len(indices),'Invalid training coverage')
            view=Subset(data,indices)
            sampler=DistributedSampler(view,num_replicas=world,rank=rank,shuffle=True,seed=protocol['seed'],drop_last=False)
            require(len(indices)%(world*protocol['batch_size'])==0,'Run requires exact divisible coverage')
            steps_per_epoch=len(sampler)//protocol['batch_size']
            total_steps=steps_per_epoch*protocol['epochs']
            if rank==0:
                write_json(output/'training_data.json',data.summary())
                write_json(output/'base_reload.json',base_receipt)
            for epoch in range(1,protocol['epochs']+1):
                data.set_epoch(epoch-1);sampler.set_epoch(epoch-1);heads.train()
                loader=DataLoader(view,batch_size=protocol['batch_size'],sampler=sampler,
                    num_workers=2,collate_fn=collate,pin_memory=False,persistent_workers=False,prefetch_factor=2)
                coverage=[];epoch_started=time.monotonic()
                log=(output/f'train_epoch{epoch:02d}_rank{rank}.jsonl').open('x')
                for step,(samples,targets) in enumerate(loader):
                    began=time.monotonic()
                    coverage.extend(int(target['image_id'][0]) for target in targets)
                    values,counts=training_batch(adapter,samples,targets,matcher)
                    require(values is not None,'Unexpected batch without matched instances')
                    fine,coarse,depth,truth=values
                    del samples,targets,values
                    predictions=ddp(fine,coarse,depth)
                    sums={name:F.binary_cross_entropy_with_logits(coarse+delta,truth,reduction='none').mean((1,2)).sum()
                          for name,delta in predictions.items()}
                    stats=torch.stack((sums['point'].detach(),sums['point_depth'].detach(),
                                       torch.tensor(float(len(fine)),device=device)))
                    dist.all_reduce(stats)
                    require(stats[2]>0 and torch.isfinite(stats).all(),'Non-finite loss/count')
                    loss=sum(sums.values())*world/stats[2]
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    for head in (heads.plain,heads.depth):
                        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
                    warmup=min(100,max(1,total_steps//10))
                    factor=(global_step+1)/warmup if global_step<warmup else .1+.9*.5*(1+math.cos(math.pi*(global_step-warmup)/max(1,total_steps-warmup-1)))
                    for group in optimizer.param_groups:group['lr']=protocol['learning_rate']*factor
                    optimizer.step();global_step+=1
                    torch.cuda.synchronize()
                    row={'epoch':epoch,'step':step+1,'global_step':global_step,
                        'local_instances':sum(counts),'global_instances':int(stats[2]),
                        'point_loss':float(stats[0]/stats[2]),'depth_point_loss':float(stats[1]/stats[2]),
                        'lr':optimizer.param_groups[0]['lr'],'seconds':time.monotonic()-began}
                    log.write(json.dumps(row,allow_nan=False)+'\n')
                    if step==0 or (step+1)%10==0 or step+1==steps_per_epoch:
                        log.flush()
                        write_json(output/f'progress_rank{rank}.json',{'phase':'training','rank':rank,
                            'epoch':epoch,'epochs':protocol['epochs'],'images':len(coverage),'expected_images':len(sampler),
                            'global_step':global_step,'steps_per_epoch':steps_per_epoch,
                            'epoch_elapsed_seconds':time.monotonic()-epoch_started,**row})
                        if rank==0:print(json.dumps(row),flush=True)
                    del fine,coarse,depth,truth,predictions,sums,loss
                log.close()
                write_json(output/f'coverage_epoch{epoch:02d}_rank{rank}.json',coverage)
                dist.barrier()
                checkpoint=None
                if rank==0:
                    all_ids=[image_id for r in range(world) for image_id in read(output/f'coverage_epoch{epoch:02d}_rank{r}.json')]
                    require(len(all_ids)==len(set(all_ids))==len(indices)
                            and set(all_ids)=={data.image_ids[i] for i in indices},'Training coverage is incomplete/duplicated')
                    checkpoint=save_heads(output/f'heads_epoch_{epoch:02d}.pt',heads,epoch,protocol,base_receipt)
                    write_json(output/f'checkpoint_epoch_{epoch:02d}.json',checkpoint)
                payload=[checkpoint];dist.broadcast_object_list(payload,src=0);checkpoint=payload[0]
                history.append(evaluate(protocol,adapter,heads,output/f'val_epoch_{epoch:02d}','val',rank,world,checkpoint))
                if rank==0:write_json(output/'validation_history.json',history)
        selection=select_validation(history)
        load_heads(selection['selected']['checkpoint']['path'],heads,protocol)
        if rank==0:write_json(output/'selection.json',selection)
        test=None
        if protocol['scope']=='full_experiment':
            test=evaluate(protocol,adapter,heads,output/'test','test',rank,world,selection['selected']['checkpoint'])
        require(tensor_digest(adapter.state_dict())==base_digest,'Frozen model weights changed')
        require(all(param.grad is None for param in adapter.parameters()),'Frozen model received gradients')
        receipt={'status':'completed','rank':rank,'pid':os.getpid(),'base_unchanged':True,
            'base_gradients_absent':True,'base_tensor_sha256_before_and_after':base_digest,
            'head_tensor_sha256':tensor_digest(heads.state_dict()),'global_steps':global_step,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started}
        write_json(output/f'completed_rank{rank}.json',receipt)
        dist.barrier()
        if rank==0:
            ranks=[read(output/f'completed_rank{r}.json') for r in range(world)]
            require(len({r['head_tensor_sha256'] for r in ranks})==1,'Rank point head tensors disagree')
            finish_report(protocol,output,history,selection,test,ranks)
            write_json(output/'run.json',{'status':'completed','scope':protocol['scope'],'finished_at_utc':now(),
                'protocol_sha256':protocol['_sha256'],'elapsed_seconds':time.monotonic()-started})
    except BaseException as error:
        write_json(output/f'failure_rank{rank}.json',{'status':'failed','type':type(error).__name__,
            'error':str(error),'global_step':global_step,'at_utc':now()})
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['train','score'])
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--split',choices=['val','test'])
    parser.add_argument('--variant',choices=VARIANTS)
    args=parser.parse_args()
    protocol=checked(args.protocol)
    if args.mode=='train':train(protocol,args.output)
    else:
        torch.set_num_threads(2)
        require(args.split is not None and args.variant is not None,'Scoring requires split/variant')
        score_arm(protocol,args.output,args.split,args.variant)


if __name__=='__main__':main()
