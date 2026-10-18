#!/usr/bin/env python3
"""Independent reload and prediction-only depth-cue diagnostic for trained heads."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch.nn import functional as F

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_inference_sample
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha, load_for_inference, tensor_digest
from rgbd_seg.course_point_refinement import (
    PairedPointHeads, _uncertain_grid, depth_cues, frozen_features, mask_boxes,
    point_sample, predict_paired, sample_shared,
)
from tools.check_course_prediction_reload import compare_predictions
from tools.run_course_point_refinement import checked, load_heads


def read(path):
    return json.loads(Path(path).read_text())


@torch.no_grad()
def run(protocol, experiment, output):
    if os.environ.get('CUDA_VISIBLE_DEVICES')!='5' or torch.cuda.device_count()!=1:
        raise ValueError('Use the original validation rank GPU 5 for strict prediction reload checks')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    result=read(experiment/'comparison.json')
    if result['status']!='completed' or protocol['scope']!='full_experiment':
        raise ValueError('Requires completed full experiment')
    checkpoint=result['selection']['selected']['checkpoint']
    if file_sha(checkpoint['path'])!=checkpoint['sha256']:
        raise ValueError('Selected point checkpoint bytes changed')
    heads=PairedPointHeads().cuda().eval()
    state=load_heads(checkpoint['path'],heads,protocol)
    if tensor_digest(heads.state_dict())!=checkpoint['head_tensor_sha256']:
        raise ValueError('Reloaded point tensors disagree with selection receipt')
    adapter,base=load_for_inference(protocol['weights']['path'],protocol['training_protocol']['path'],device='cuda:0')
    before=tensor_digest(adapter.state_dict())
    epoch=checkpoint['epoch']
    folder=experiment/f'val_epoch_{epoch:02d}'
    rank=protocol['gpus'].index(5)
    rank_ids=read(folder/f'rank{rank}.json')['ordered_ids']
    ids=[rank_ids[index] for index in (0,97,251,501)]
    references={}
    for name in ('native','point','point_depth'):
        references[name]={}
        with (folder/name/'predictions.jsonl').open() as source:
            for line in source:
                record=json.loads(line)
                if record['image_id'] in ids:
                    references[name][record['image_id']]=record['predictions']
        if set(references[name])!=set(ids):
            raise ValueError('Independent validation reload references incomplete')
    checks=[]
    with CourseArchive(protocol['archive'],'val') as archive:
        for image_id in ids:
            sample=load_inference_sample(archive,image_id)
            predictions=predict_paired(adapter,heads,sample)
            score_errors={name:compare_predictions(references[name][image_id],[c.public() for c in values])
                          for name,values in predictions.items()}
            outputs,fine,_,_,_=frozen_features(adapter,[sample])
            indices=torch.tensor([c.query_id for c in predictions['native'][:8]],device=adapter.device)
            if len(indices)==0:raise ValueError('Diagnostic image unexpectedly has no predictions')
            coarse=outputs.masks_queries_logits[0,indices].float()
            dense=F.interpolate(coarse[None],size=(1024,1024),mode='bilinear',align_corners=False)[0]
            boxes=mask_boxes(dense>0)
            _,coords,active=_uncertain_grid(dense,boxes,784)
            fine_points=sample_shared(fine,coords)
            coarse_points=point_sample(coarse[:,None],coords)
            cues=depth_cues(sample['depth_m'].cuda(),sample['depth_valid'].cuda())
            local=sample_shared(cues,coords)
            without_geometry=local.clone()
            without_geometry[:,1:5]=0
            ordinary=heads.plain(fine_points,coarse_points)
            learned=heads.depth(fine_points,coarse_points,local)
            altered=heads.depth(fine_points,coarse_points,without_geometry)
            difference=(learned-altered).abs()[:,0][active]
            checks.append({'image_id':image_id,'reloaded_masks_exact':True,
                'maximum_score_difference':score_errors,'queries_probed':len(indices),
                'active_points':int(active.sum()),
                'ordinary_residual_abs_max':float(ordinary.abs().max()),
                'depth_residual_abs_max':float(learned.abs().max()),
                'depth_difference_input_abs_max':float(local[:,1:5].abs().max()),
                'geometry_only_ablation_residual_abs_max':float(difference.max()),
                'geometry_only_ablation_residual_abs_mean':float(difference.mean())})
    after=tensor_digest(adapter.state_dict())
    if before!=after or any(p.grad is not None for p in adapter.parameters()):
        raise ValueError('Independent diagnostic changed the frozen model')
    learned_weights=heads.depth.layers[0].weight
    receipt={'status':'completed','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol_sha256':protocol['_sha256'],'selected_checkpoint':checkpoint,
        'head_reload_exact':True,'validation_predictions_reloaded_exact':True,
        'inference_input_keys':['image','depth_m','depth_valid','camera_K'],
        'ground_truth_used':False,'test_accessed':False,'physical_gpu':5,
        'base_tensor_sha256_before_and_after':before,'base_unchanged':True,
        'geometry_weight_l2':float(learned_weights[:,353:357].norm()),
        'validity_weight_l2':float(torch.cat((learned_weights[:,352:353],learned_weights[:,357:361]),dim=1).norm()),
        'ordinary_predictor_weight_l2':float(heads.plain.predictor.weight.norm()),
        'depth_predictor_weight_l2':float(heads.depth.predictor.weight.norm()),
        'geometry_effect_observed':any(c['geometry_only_ablation_residual_abs_max']>1e-8 for c in checks),
        'diagnostic_scope':'Four preselected validation images and first eight score-ranked native queries per image; local logit sensitivity, not a full AP ablation or causal effectiveness claim',
        'checks':checks,'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_sha(__file__)},
        'base_checkpoint_sha256':base['checkpoint_sha256']}
    write_json(output,receipt)
    print(json.dumps({'status':receipt['status'],'images':len(checks),'geometry_effect_observed':receipt['geometry_effect_observed'],'output':str(output)}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--experiment',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('Keep the diagnostic receipt immutable')
    run(checked(args.protocol),args.experiment,args.output)


if __name__=='__main__':main()
