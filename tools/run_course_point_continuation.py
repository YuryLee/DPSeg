#!/usr/bin/env python3
"""Continue the existing paired heads on the frozen original base; validation only."""
from __future__ import annotations

import argparse
from datetime import timedelta
import json
import math
import os
from pathlib import Path
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

from rgbd_seg.course_depth_data import CourseDepthTrainingDataset
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha, load_for_inference, tensor_digest
from rgbd_seg.course_point_refinement import PairedPointHeads, training_batch
from rgbd_seg.course_longer_training import continuation_lr

from tools.run_course_point_refinement import (VARIANTS, METHODS, checked, collate,
    save_heads, load_heads, evaluate, select_validation as legacy_selection, read, require, now)
from tools.run_course_dafusion import save_torch, rng_state


def select_validation(history):
    result=legacy_selection(history)
    result['selected']=max(result['candidates'],
        key=lambda row:(row['mask_ap_small'],row['mask_ap'],-row['epoch']))
    result['selection_rule']='Maximum full validation APs, then AP, then earlier epoch'
    result['recommended_over_existing_native_depth']=result['selected']['eligible']
    result['legacy_gain_gate']='Descriptive only; it does not override the primary APs ranking'
    return result


def parent_inputs(protocol):
    initial=protocol['initial_head']
    for key in ('checkpoint','protocol','validation'):
        require(file_sha(initial[key]['path'])==initial[key]['sha256'],f'Parent point input changed: {key}')
    parent_protocol=checked(initial['protocol']['path'])
    require(parent_protocol['weights']==protocol['weights'] and parent_protocol['point_policy']==protocol['point_policy'],
            'Parent head is bound to a different base or policy')
    validation=read(initial['validation']['path'])
    require(validation['epoch']==initial['checkpoint']['epoch'] and validation['split']=='val'
            and validation['images']==2584 and validation['checkpoint']['sha256']==initial['checkpoint']['sha256'],
            'Parent validation does not bind the starting head')
    return parent_protocol,validation


def selection_history(protocol, history):
    _,parent_validation=parent_inputs(protocol)
    return history if protocol['scope']=='engineering_smoke' else [parent_validation,*history]


def finish_validation_report(protocol, output, history, selection, ranks):
    result = {'status': 'completed', 'scope': protocol['scope'],
        'material_passport': {'origin_skill': 'academic-research-suite/experiment-agent',
                             'mode': 'run', 'verification_status': 'FULL_VALIDATION_ONLY'
                             if protocol['scope']=='full_experiment' else 'ENGINEERING_ONLY'},
        'protocol_sha256': protocol['_sha256'], 'validation_history': history,
        'selection': selection, 'rank_receipts': ranks, 'finished_at_utc': now(),
        'test_used': False, 'base_weights': protocol['weights'],
        'base_training_protocol': protocol['training_protocol'],
        'epochs': protocol['epochs'], 'parent_head_epoch': protocol['initial_head']['checkpoint']['epoch'],
        'initial_head': protocol['initial_head'],
        'optimizer_initialization': 'Fresh AdamW; original two-epoch run did not save optimizer moments',
        'remaining': ['Full validation with tiled inference and fixed probability-depth refinement',
                      'Lock the final method, then evaluate test and export CSV'],
        'limitations': ['Single seed; no significance claim',
                        'Head weights retained, optimizer moments reset; this is not exact optimizer-state resumption',
                        'Global batch remains eight; rank layout and point-sampling RNG differ from the original four-GPU run',
                        'Training uses whole images; crop performance requires separate validation',
                        'Fixed instance scores and count; point heads only change mask geometry']}
    write_json(output/'comparison.json', result)
    lines = ['# 延长训练后的点头验证', '',
             f"从原点头第 {protocol['initial_head']['checkpoint']['epoch']} 轮继续训练 {protocol['epochs']} 轮。原主模型冻结；重新初始化 AdamW，仅使用验证集选择，尚未评测测试集。", '',
             '| 点头轮次 | 配置 | Mask AP | Mask APs |', '|---:|---|---:|---:|']
    for entry in history:
        for variant, metrics in entry['metrics'].items():
            lines.append(f"| {entry['epoch']} | {variant} | {metrics['mask_ap']:.6f} | {metrics['mask_ap_small']:.6f} |")
    chosen = selection['selected']
    lines += ['', f"验证集候选：第 {chosen['epoch']} 轮，{chosen['variant']}。",
              '完整切片组合仍需验证，最终选择完成后再评测测试集。']
    (output/'comparison.md').write_text('\n'.join(lines)+'\n')


def train(protocol,output):
    require(protocol.get('evaluation_policy') == 'validation_only', 'Only validation selection is allowed')
    parent_protocol,parent_validation=parent_inputs(protocol)
    parent_epoch=protocol['initial_head']['checkpoint']['epoch']
    if protocol['scope']=='full_experiment':
        gate=protocol['engineering']
        require(file_sha(gate['path'])==gate['sha256'],'Engineering receipt changed')
        receipt=read(gate['path'])
        require(receipt['status']=='completed' and receipt['scope']=='engineering_smoke',
                'Engineering run is required')
        engineer_protocol=read(receipt['protocol'])
        require(engineer_protocol['source_sha256']==protocol['source_sha256']
                and engineer_protocol['weights']==protocol['weights']
                and engineer_protocol['initial_head']==protocol['initial_head']
                and engineer_protocol['batch_size']==protocol['batch_size']
                and engineer_protocol['gpus']==protocol['gpus'], 'Engineering inputs differ')
        audit_gate=protocol['engineering_audit']
        require(file_sha(audit_gate['path'])==audit_gate['sha256'],'Engineering audit changed')
        engineering_audit=read(audit_gate['path'])
        require(engineering_audit['status']=='passed' and engineering_audit['protocol_sha256']==file_sha(receipt['protocol']),
                'Engineering artifacts were not audited')
    local_rank=int(os.environ['LOCAL_RANK']);rank=int(os.environ['RANK']);world=int(os.environ['WORLD_SIZE'])
    require(os.environ['CUDA_VISIBLE_DEVICES']==','.join(map(str,protocol['gpus']))
            and world==len(protocol['gpus']),'Unexpected visible devices/world size')
    torch.cuda.set_device(local_rank)
    torch.cuda.set_per_process_memory_fraction(.9,local_rank)
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
    parent_value=load_heads(protocol['initial_head']['checkpoint']['path'],heads,parent_protocol)
    initial_digest=tensor_digest(heads.state_dict())
    require(parent_value['epoch']==parent_epoch and initial_digest==parent_validation['checkpoint']['head_tensor_sha256'],
            'Starting head tensors do not match the validated parent')
    require('optimizer' not in parent_value,'Unexpected parent optimizer state; revisit initialization policy')
    if rank==0:
        write_json(output/'initial_head_reload.json',{'epoch':parent_epoch,'head_tensor_sha256':initial_digest,
            'checkpoint_sha256':file_sha(protocol['initial_head']['checkpoint']['path']),
            'optimizer':'fresh AdamW','original_optimizer_state_available':False})
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
            for epoch in range(parent_epoch+1,parent_epoch+protocol['epochs']+1):
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
                    schedule=protocol['schedule']
                    lr=continuation_lr(global_step,schedule['total_steps'],peak_lr=protocol['learning_rate'],
                        start_lr=schedule['start_lr'],warmup_steps=schedule['warmup_steps'],
                        minimum_ratio=schedule['minimum_ratio'])
                    for group in optimizer.param_groups:group['lr']=lr
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
                            'epoch':epoch,'epochs':parent_epoch+protocol['epochs'],'images':len(coverage),'expected_images':len(sampler),
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
                rank_rng = [None]*world
                dist.all_gather_object(rank_rng, rng_state())
                if rank==0:
                    save_torch(output/'last_training.pt', {
                        'heads': heads.state_dict(), 'optimizer': optimizer.state_dict(),
                        'epoch': epoch, 'global_step': global_step, 'rank_rng': rank_rng,
                        'protocol_sha256': protocol['_sha256'], 'source_sha256': protocol['source_sha256'],
                        'base_checkpoint_sha256': base_receipt['checkpoint_sha256'],
                        'validation_history': history})
                dist.barrier()
        selection=select_validation(selection_history(protocol,history))
        selected_path=selection['selected']['checkpoint']['path']
        selected_protocol=parent_protocol if Path(selected_path).resolve()==Path(protocol['initial_head']['checkpoint']['path']).resolve() else protocol
        load_heads(selected_path,heads,selected_protocol)
        if rank==0:write_json(output/'selection.json',selection)
        require(tensor_digest(adapter.state_dict())==base_digest,'Frozen model weights changed')
        require(all(param.grad is None for param in adapter.parameters()),'Frozen model received gradients')
        receipt={'status':'completed','rank':rank,'pid':os.getpid(),'base_unchanged':True,
            'base_gradients_absent':True,'base_tensor_sha256_before_and_after':base_digest,
            'head_tensor_sha256':tensor_digest(heads.state_dict()),'global_steps':global_step,
            'initial_head_tensor_sha256':initial_digest,'parent_head_epoch':parent_epoch,
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'elapsed_seconds':time.monotonic()-started}
        write_json(output/f'completed_rank{rank}.json',receipt)
        dist.barrier()
        if rank==0:
            ranks=[read(output/f'completed_rank{r}.json') for r in range(world)]
            require(len({r['head_tensor_sha256'] for r in ranks})==1,'Rank point head tensors disagree')
            finish_validation_report(protocol,output,history,selection,ranks)
            write_json(output/'run.json',{'status':'completed','scope':protocol['scope'],'finished_at_utc':now(),
                'protocol_sha256':protocol['_sha256'],'elapsed_seconds':time.monotonic()-started})
    except BaseException as error:
        write_json(output/f'failure_rank{rank}.json',{'status':'failed','type':type(error).__name__,
            'error':str(error),'global_step':global_step,'at_utc':now()})
        raise
    finally:
        dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = checked(args.protocol)
    require(protocol.get('evaluation_policy') == 'validation_only', 'Test evaluation is disabled')
    train(protocol, args.output)


if __name__ == '__main__':
    main()
