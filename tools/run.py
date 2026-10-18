#!/usr/bin/env python3
"""Generate fixtures, audit, train, evaluate and benchmark reproducible runs."""
import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import platform
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from rgbd_seg.model import build_model,load_model


def write(path,obj):
    Path(path).write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def move(d,device):
    return {k:v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}


def hash_file(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def code_hash():
    root=Path(__file__).resolve().parents[1]
    return hashlib.sha256(b''.join(p.read_bytes() for folder in ['rgbd_seg','tools'] for p in sorted((root/folder).glob('*.py')))).hexdigest()


def sync(device):
    if str(device).startswith('cuda'): torch.cuda.synchronize(device)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    g=sub.add_parser('synthetic'); g.add_argument('--data',required=True)
    a=sub.add_parser('audit'); a.add_argument('--data',required=True); a.add_argument('--output')
    t=sub.add_parser('train'); t.add_argument('--data',required=True); t.add_argument('--output',required=True)
    t.add_argument('--config',required=True); t.add_argument('--steps',type=int,default=400)
    t.add_argument('--lr',type=float,default=0.002); t.add_argument('--batch-size',type=int,default=2)
    t.add_argument('--device',default='cpu'); t.add_argument('--threads',type=int,default=2)
    t.add_argument('--cache-data',action='store_true',help='Load the selected training subset once into CPU memory')
    t.add_argument('--init-rgb',help='Local Torchvision Mask R-CNN state_dict; only shape-compatible RGB/core tensors copied')
    t.add_argument('--foreground-mean-init',action='store_true',help='Initialize two-class predictors from COCO background and mean of 80 foreground heads')
    t.add_argument('--augment',action='store_true',help='Aligned flips and RGB-only mild color jitter')
    t.add_argument('--crop-probability',type=float,default=0.,help='Training-only object-centered crop probability')
    t.add_argument('--backbone-lr-factor',type=float,default=1.)
    t.add_argument('--warmup-steps',type=int,default=0)
    t.add_argument('--cosine-decay',action='store_true')
    t.add_argument('--cuda-memory-fraction',type=float,default=1.)
    e=sub.add_parser('evaluate'); e.add_argument('--data',required=True); e.add_argument('--checkpoint',required=True)
    e.add_argument('--split',choices=['val','test'],default='val'); e.add_argument('--device',default='cpu')
    e.add_argument('--threads',type=int,default=2); e.add_argument('--output',required=True)
    e.add_argument('--extra-output',help='Compute documented overlap/boundary metrics from the same predictions at score 0.5')
    args=p.parse_args()
    from rgbd_seg.data import generate_synthetic,audit_dataset,RGBDDataset,dataset_fingerprint
    if args.command=='synthetic':
        generate_synthetic(Path(args.data)); return
    if args.command=='audit':
        result=audit_dataset(Path(args.data))
        if args.output: write(args.output,result)
        print(json.dumps(result,ensure_ascii=False,indent=2)); return
    torch.set_num_threads(args.threads)
    if args.device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select --device cpu or a working allocated GPU')
    if args.command=='train' and args.device.startswith('cuda'):
        torch.cuda.set_per_process_memory_fraction(args.cuda_memory_fraction,args.device)
    root=Path(args.data); manifest=json.loads((root/'manifest.json').read_text())
    if args.command=='train':
        out=Path(args.output)
        if out.exists() and any(out.iterdir()): raise ValueError('Output directory must be new or empty to preserve runs')
        out.mkdir(parents=True,exist_ok=True)
        config=json.loads(Path(args.config).read_text())
        config['num_classes']=len(manifest['classes'])+1
        seed=config.get('seed',17); torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
        model=build_model(config).to(args.device)
        config=model.config
        init_info=None
        if args.init_rgb:
            from rgbd_seg.pretrained import initialize_rgb
            init_info=initialize_rgb(model,args.init_rgb,args.foreground_mean_init)
        ds=RGBDDataset(root,'train')
        if not len(ds): raise ValueError('Training split is empty')
        if args.cache_data:
            ds=[ds[i] for i in range(len(ds))]
        backbone=[];other=[]
        for name,parameter in model.named_parameters():
            if parameter.requires_grad:
                (backbone if name.startswith('core.backbone.rgb.') else other).append(parameter)
        opt=torch.optim.AdamW([{'params':backbone,'lr':args.lr*args.backbone_lr_factor},
                              {'params':other,'lr':args.lr}],lr=args.lr,weight_decay=0.0001)
        import math
        def schedule(step):
            warmup=min(1.,(step+1)/max(1,args.warmup_steps))
            decay=0.1+0.9*(1+math.cos(math.pi*step/max(1,args.steps)))/2 if args.cosine_decay else 1.
            return warmup*decay
        scheduler=torch.optim.lr_scheduler.LambdaLR(opt,schedule)
        run={'run_id':out.name,'provenance':manifest['provenance'],'command':sys.argv,
             'config':config,'manifest_sha256':hash_file(root/'manifest.json'),'dataset_sha256':dataset_fingerprint(root),'code_sha256':code_hash(),
             'python':platform.python_version(),'torch':str(torch.__version__),
             'seed':seed,'steps':args.steps,'batch_size':args.batch_size,'device':args.device,'cache_data':args.cache_data,
             'rgb_initialization':init_info or 'random; identical seed across variants',
             'augmentation':{'enabled':args.augment,'crop_probability':args.crop_probability},
             'optimizer':{'type':'AdamW','lr':args.lr,'backbone_lr_factor':args.backbone_lr_factor,
                          'warmup_steps':args.warmup_steps,'cosine_decay':args.cosine_decay},
             'selection':'fixed final step; validation not used for checkpoint selection'}
        write(out/'run.json',run)
        rng=random.Random(seed);aug_rng=random.Random(seed+1000);start=time.perf_counter();model.train()
        from rgbd_seg.augment import augment
        with (out/'loss.jsonl').open('w') as log:
            for step in range(args.steps):
                batch=[ds[rng.randrange(len(ds))] for _ in range(args.batch_size)]
                if args.augment:batch=[augment(s,t,aug_rng,args.crop_probability) for s,t in batch]
                samples=[move(s,args.device) for s,_ in batch]; targets=[move(t,args.device) for _,t in batch]
                opt.zero_grad(set_to_none=True); losses=model(samples,targets); total=sum(losses.values())
                if not torch.isfinite(total): raise FloatingPointError(f'Non-finite loss at step {step}')
                total.backward(); norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True); opt.step();scheduler.step()
                row={'step':step+1,'loss':float(total.detach()),'grad_norm':float(norm),
                     **{k:float(v.detach()) for k,v in losses.items()},'elapsed_s':time.perf_counter()-start,'lr':opt.param_groups[-1]['lr']}
                log.write(json.dumps(row)+'\n'); log.flush()
                if step==0 or (step+1)%25==0: print(json.dumps(row),flush=True)
        state={k:v.detach().cpu() for k,v in model.state_dict().items()}
        torch.save({'config':config,'state_dict':state,'provenance':manifest['provenance'],
                    'classes':manifest['classes'],'depth_scale':manifest['depth_scale'],
                    'manifest_sha256':run['manifest_sha256']},out/'model.pt')
        torch.save(state,out/'weights.pt')
        run.update({'duration_s':time.perf_counter()-start,'exit_status':'completed',
                    'checkpoint_sha256':hash_file(out/'model.pt'),'parameters':sum(p.numel() for p in model.parameters()),
                    'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
                    'weight_bytes':(out/'weights.pt').stat().st_size})
        if args.device.startswith('cuda'):
            run['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated(args.device)
            run['device_name']=torch.cuda.get_device_name(args.device)
        write(out/'run.json',run); print(json.dumps(run,ensure_ascii=False),flush=True)
    else:
        from rgbd_seg.evaluation import evaluate_predictions
        model,ckpt=load_model(args.checkpoint,args.device); ds=RGBDDataset(root,args.split)
        if not len(ds): raise ValueError('Evaluation split is empty')
        if ckpt['classes']!=manifest['classes']: raise ValueError('Checkpoint and dataset classes differ')
        # In-memory input -> transfer/transform -> prediction/postprocess; excludes file IO and plots.
        cached=[ds[i][0] for i in range(len(ds))]; predictions=[]; times=[]
        with torch.inference_mode():
            for _ in range(3): model([move(cached[0],args.device)])
            for sample in cached:
                sync(args.device); start=time.perf_counter()
                pred=model([move(sample,args.device)])[0]; sync(args.device)
                times.append(time.perf_counter()-start)
                predictions.append({k:v.cpu() for k,v in pred.items()})
        metrics=evaluate_predictions(ds,predictions)
        result={'provenance':manifest['provenance'],'split':args.split,'samples':len(ds),'metrics':metrics,
                'model':ckpt['config'],'checkpoint_sha256':hash_file(args.checkpoint),
                'manifest_sha256':hash_file(root/'manifest.json'),'dataset_sha256':dataset_fingerprint(root),'parameters':sum(p.numel() for p in model.parameters()),
                'checkpoint_bytes':Path(args.checkpoint).stat().st_size,
                'timing':{'device':args.device,'threads':args.threads,'batch':1,'warmup':3,'precision':'float32',
                    'scope':'in-memory tensors through preprocessing, device transfer, prediction and postprocessing; no IO/rendering',
                    'mean_ms':1000*np.mean(times),'median_ms':1000*np.median(times),'fps':len(ds)/sum(times)}}
        Path(args.output).parent.mkdir(parents=True,exist_ok=True); write(args.output,result)
        if args.extra_output:
            from rgbd_seg.instance_metrics import evaluate_extra_predictions
            extra=evaluate_extra_predictions(ds,predictions,0.5,0.5,0.003)
            extra['run']={'input_sha256':hash_file(args.checkpoint),'input_type':'checkpoint',
                          'dataset_sha256':dataset_fingerprint(root),'split':args.split,
                          'samples':len(ds),'provenance':manifest['provenance'],'device':args.device,
                          'model_config':ckpt['config']}
            Path(args.extra_output).parent.mkdir(parents=True,exist_ok=True)
            write(args.extra_output,extra)
        print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
