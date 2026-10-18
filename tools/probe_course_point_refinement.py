#!/usr/bin/env python3
"""Real-data forward/backward, reload, frozen-weight and timing probe."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F
from transformers.models.mask2former.modeling_mask2former import Mask2FormerHungarianMatcher
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import CourseDepthTrainingDataset, load_inference_sample
from rgbd_seg.course_optimized import load_for_inference, tensor_digest
from rgbd_seg.course_point_refinement import PairedPointHeads, training_batch, predict_paired
from rgbd_seg.course_distributed_eval import write_json
from tools.check_course_prediction_reload import compare_predictions


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    p=json.loads((ROOT/'reports/course_tiled_probability/protocol_full_v1.json').read_text())
    started=time.monotonic()
    adapter,receipt=load_for_inference(p['weights']['path'],p['training_protocol']['path'],device='cuda:0')
    original=tensor_digest(adapter.state_dict())
    heads=PairedPointHeads().cuda()
    cfg=adapter.model.config
    matcher=Mask2FormerHungarianMatcher(cost_class=cfg.class_weight,cost_mask=cfg.mask_weight,
                                       cost_dice=cfg.dice_weight,num_points=cfg.train_num_points)
    optimizer=torch.optim.AdamW(heads.parameters(),lr=.0003,weight_decay=.01)
    checks=[]
    with CourseArchive(p['archive'],'val') as archive:
        sample=load_inference_sample(archive,archive.image_ids[4])
        baseline=adapter.predict_sample(sample)
        zero=predict_paired(adapter,heads,sample)
        for name,items in zero.items():
            compare_predictions(baseline,[c.public() for c in items])
        checks.append('Untrained point heads exactly preserve native predictions')
    training_times=[]
    losses=[]
    with CourseDepthTrainingDataset(p['archive'],'train',augment=True,seed=17) as data:
        for step in range(4):
            began=time.monotonic()
            items=[data[step*2+i] for i in range(2)]
            values,counts=training_batch(adapter,[x[0] for x in items],[x[1] for x in items],matcher)
            fine,coarse,depth,truth=values
            predictions=heads(fine,coarse,depth)
            loss={name:F.binary_cross_entropy_with_logits(coarse+delta,truth) for name,delta in predictions.items()}
            optimizer.zero_grad(set_to_none=True)
            sum(loss.values()).backward()
            if not all(param.grad is not None and torch.isfinite(param.grad).all() for param in heads.parameters()):
                raise ValueError('Missing/non-finite head gradient')
            for head in (heads.plain,heads.depth):
                torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True)
            optimizer.step()
            torch.cuda.synchronize()
            training_times.append(time.monotonic()-began)
            losses.append({name:float(value.detach()) for name,value in loss.items()})
            print(json.dumps({'step':step+1,'instances':sum(counts),'seconds':training_times[-1],'losses':losses[-1]}),flush=True)
    if tensor_digest(adapter.state_dict())!=original or any(p.grad is not None for p in adapter.parameters()):
        raise ValueError('Frozen base model changed or received gradients')
    torch.save({'heads':heads.cpu().state_dict()},args.output/'smoke_heads.pt')
    restored=PairedPointHeads()
    restored.load_state_dict(torch.load(args.output/'smoke_heads.pt',weights_only=True)['heads'],strict=True)
    if tensor_digest(heads.state_dict())!=tensor_digest(restored.state_dict()):
        raise ValueError('Point head reload differs')
    heads.cuda().eval()
    with CourseArchive(p['archive'],'val') as archive:
        sample=load_inference_sample(archive,archive.image_ids[4])
        began=time.monotonic()
        outputs=predict_paired(adapter,heads,sample)
        torch.cuda.synchronize()
        prediction_seconds=time.monotonic()-began
        identities={name:[(c.query_id,c.score) for c in items] for name,items in outputs.items()}
        if any(items!=identities['native'] for items in identities.values()):
            raise ValueError('Point refinement changed identities/scores')
    result={'status':'passed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'base_reload':receipt,'base_model_tensor_sha256_before_and_after':original,
        'base_unchanged':True,'base_gradients_absent':True,'head_gradients_finite':True,
        'head_reload_exact':True,'zero_head_native_predictions_exact':True,
        'scores_and_instance_identities_preserved':True,'training_batch_size':2,
        'training_step_seconds':training_times,'losses':losses,
        'paired_inference_seconds':prediction_seconds,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'elapsed_seconds':time.monotonic()-started,'test_accessed':False,
        'checks':checks}
    write_json(args.output/'probe.json',result)
    print(json.dumps({'status':result['status'],'output':str(args.output),'seconds':result['elapsed_seconds']}),flush=True)


if __name__=='__main__':main()
