#!/usr/bin/env python3
"""Freeze a separate continuation of the existing heads on the original e05 base."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_depth_data import CourseDepthTrainingDataset
from rgbd_seg.course_point_refinement import POLICY
from tools.run_course_point_refinement import checked


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['engineering_smoke','full_experiment'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engineering',type=Path)
    parser.add_argument('--engineering-audit',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    parent_path=ROOT/'reports/course_point_refinement/protocol_full_v1.json'
    p=json.loads(parent_path.read_text())
    parent_root=ROOT/'reports/course_point_refinement/formal_v1/experiment'
    head=parent_root/'heads_epoch_02.pt'
    validation=parent_root/'val_epoch_02/comparison.json'
    initial={'checkpoint':{'path':str(head),'sha256':file_sha(head),'epoch':2},
             'protocol':{'path':str(parent_path),'sha256':file_sha(parent_path)},
             'validation':{'path':str(validation),'sha256':file_sha(validation)}}
    p.pop('engineering',None)
    sources=dict(p['source_sha256'])
    for name in ('tools/run_course_point_continuation.py','tools/supervise_course_point_continuation.py',
                 'tools/build_course_point_continuation_protocol.py','rgbd_seg/course_longer_training.py'):
        sources[name]=file_sha(ROOT/name)
    for name,digest in sources.items():
        if file_sha(ROOT/name)!=digest:raise ValueError(f'Parent source changed: {name}')
    p.update(schema='course_point_continuation_v1',scope=args.scope,status='frozen',
        created_at_utc=datetime.now(timezone.utc).isoformat(),evaluation_policy='validation_only',
        point_policy=POLICY,source_sha256=sources,initial_head=initial,
        gpus=[4],batch_size=8,global_batch_size=8,learning_rate=1e-4,
        epochs=4 if args.scope=='full_experiment' else 1,
        training_images=14432 if args.scope=='full_experiment' else 16,
        schedule={'total_steps':4*(14432//8),'start_lr':3e-5,'warmup_steps':100,'minimum_ratio':.1},
        initialization='Restore both original e02 heads; fresh AdamW because the parent run saved no optimizer state',
        head_training='Frozen original e05 base; paired heads only; original global batch eight on one GPU',
        selection={'split':'val','metric':'mask_ap_small',
                   'rule':'Maximum full validation APs across the original e02 and all continued heads; tie AP then earlier epoch'},
        material_passport={'origin_skill':'academic-research-suite/experiment-agent','mode':'run',
            'verification_status':'ENGINEERING_PENDING' if args.scope=='engineering_smoke' else 'ENGINEERING_AUDITED'},
        limitations=['Not an exact optimizer-state resume',
                     'Changing four ranks to one changes point sampling RNG and forward batch shape',
                     'Whole-image head selection precedes the tiled comparison'],
        non_goals=['No base-model training in this branch','No test access or Kaggle selection',
                   'Do not modify ongoing base continuation or historical results'])
    if args.scope=='engineering_smoke':
        with CourseDepthTrainingDataset(p['archive'],'train',augment=True,seed=p['seed']) as data:
            p['training_indices']=sorted(range(len(data)),key=lambda i:(
                -len(data.annotations_by_image[data.image_ids[i]]),data.image_ids[i]))[:16]
        p['engineering_sample']='Sixteen most densely annotated training images, batch eight, two real updates'
    else:
        if args.engineering is None or args.engineering_audit is None:
            raise ValueError('Formal continuation requires an engineering receipt and artifact audit')
        receipt=json.loads(args.engineering.read_text())
        audit=json.loads(args.engineering_audit.read_text())
        if receipt['status']!='completed' or receipt['scope']!='engineering_smoke' or receipt['exit_code']!=0:
            raise ValueError('Engineering process did not complete')
        engineer=json.loads(Path(receipt['protocol']).read_text())
        for key in ('source_sha256','initial_head','weights','gpus','batch_size','schedule'):
            if engineer[key]!=p[key]:raise ValueError(f'Engineering input differs: {key}')
        if audit['status']!='passed' or audit['protocol_sha256']!=file_sha(receipt['protocol']):
            raise ValueError('Engineering outputs were not audited')
        p['engineering']={'path':str(args.engineering.resolve()),'sha256':file_sha(args.engineering)}
        p['engineering_audit']={'path':str(args.engineering_audit.resolve()),'sha256':file_sha(args.engineering_audit)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    checked(args.output)
    print(json.dumps({'status':'frozen','scope':args.scope,'path':str(args.output.resolve()),'sha256':file_sha(args.output)}))


if __name__=='__main__':main()
