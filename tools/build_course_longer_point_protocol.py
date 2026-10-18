#!/usr/bin/env python3
"""Bind longer point-head training to a base selected using full validation."""
import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_point_refinement import POLICY


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-selection',type=Path,required=True)
    parser.add_argument('--scope',choices=['engineering_smoke','full_experiment'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engineering',type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    decision=json.loads(args.base_selection.read_text())
    if decision['status'] not in ('selected_after_convergence_review','selected_by_user') or decision['selection_split']!='val' or decision['test_used']:
        raise ValueError('A reviewed or explicitly user-selected validation-only base is required')
    choice=decision['selected']
    if decision['status']=='selected_by_user':
        if not decision.get('user_instruction') or decision.get('user_requested_base_epoch')!=choice.get('epoch'):
            raise ValueError('Explicit base selection must identify the requested epoch')
        binding=decision['completed_base_audit']
        if file_sha(binding['path'])!=binding['sha256']:
            raise ValueError('User-selected base audit changed')
        audit=json.loads(Path(binding['path']).read_text())
        rows=[v for v in audit['epochs'] if v['epoch']==choice['epoch']]
        if (audit['status']!='passed' or audit['scope']!='completed_base_arm' or audit['test_used']
                or audit['arm']!=choice['arm'] or len(rows)!=1
                or rows[0]['checkpoint_sha256']!=choice['checkpoint_sha256']
                or rows[0]['validation_images']!=2584 or rows[0]['training_images']!=14432):
            raise ValueError('Requested base epoch lacks a complete audited training/validation record')
    weights,training_protocol=Path(choice['checkpoint']),Path(choice['protocol'])
    if file_sha(weights)!=choice['checkpoint_sha256']:
        raise ValueError('Selected checkpoint changed')
    base=json.loads(training_protocol.read_text())
    if base['status']!='frozen':
        raise ValueError('Selected base training protocol is not frozen')
    old=json.loads((ROOT/'reports/course_point_refinement/protocol_full_v1.json').read_text())
    p=copy.deepcopy(old)
    p.update(schema='course_longer_point_v1',scope=args.scope,status='frozen',
        created_at_utc=datetime.now(timezone.utc).isoformat(),point_policy=POLICY,
        evaluation_policy='validation_only',
        weights={'path':str(weights.resolve()),'sha256':file_sha(weights)},
        training_protocol={'path':str(training_protocol.resolve()),'sha256':file_sha(training_protocol)},
        base_selection={'path':str(args.base_selection.resolve()),'sha256':file_sha(args.base_selection)},
        gpus=[2,3,5,6],epochs=6 if args.scope=='full_experiment' else 1,
        training_images=14432 if args.scope=='full_experiment' else 16,
        initialization='Fresh paired heads with identical shared initialization; train against the newly selected frozen base',
        head_training='Paired heads only; complete validation each epoch; test is not opened',
        non_goals=['No test evaluation or submission during head selection',
                   'No modifications to historical results or frozen base weights'])
    sources={**old['source_sha256'],**base['source_sha256']}
    for name in ('tools/run_course_longer_point.py','tools/build_course_longer_point_protocol.py',
                 'tools/supervise_course_longer_point.py'):
        sources[name]=file_sha(ROOT/name)
    for name,digest in sources.items():
        if file_sha(ROOT/name)!=digest:
            raise ValueError(f'Frozen upstream source changed: {name}')
    p['source_sha256']=sources
    p['selection']['rule']='Maximum full validation APs, then AP, then earlier epoch; legacy gain gate is descriptive only'
    p['material_passport']={'origin_skill':'academic-research-suite/experiment-agent','mode':'run',
        'verification_status':'FOUR_GPU_ENGINEERING_PASSED' if args.scope=='full_experiment' else 'ENGINEERING_PENDING'}
    p.pop('engineering',None)
    if args.scope=='full_experiment':
        if args.engineering is None:
            raise ValueError('Full run requires an engineering receipt')
        receipt=json.loads(args.engineering.read_text())
        if receipt['status']!='completed' or receipt['scope']!='engineering_smoke':
            raise ValueError('Engineering is incomplete')
        engineer=json.loads(Path(receipt['protocol']).read_text())
        if engineer['source_sha256']!=sources or engineer['weights']!=p['weights']:
            raise ValueError('Engineering source/base identity differs')
        p['engineering']={'path':str(args.engineering.resolve()),'sha256':file_sha(args.engineering)}
    else:
        p['training_indices']=list(range(16))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'frozen','path':str(args.output.resolve()),'sha256':file_sha(args.output)}))


if __name__=='__main__':
    main()
