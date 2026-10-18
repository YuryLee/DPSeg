#!/usr/bin/env python3
"""Freeze the final whole-image/tiled comparison after point-head validation."""
import argparse
import copy
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_optimized import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--point-run',type=Path,required=True)
    parser.add_argument('--point-protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    report=json.loads((args.point_run/'comparison.json').read_text())
    pp=json.loads(args.point_protocol.read_text())
    if (report['status']!='completed' or report['scope']!='full_experiment' or report['test_used']
            or report['protocol_sha256']!=file_sha(args.point_protocol)
            or len(report['validation_history'])!=pp['epochs']):
        raise ValueError('Complete validation-only point training is required')
    choice=report['selection']['selected']
    if report['selection']['selection_split']!='val' or report['selection']['test_labels_used_for_selection']:
        raise ValueError('Head was not selected exclusively on validation')
    head=choice['checkpoint']
    if file_sha(head['path'])!=head['sha256']:
        raise ValueError('Selected head changed')
    reload=json.loads((args.point_run/'base_reload.json').read_text())
    if reload['checkpoint_sha256']!=pp['weights']['sha256']:
        raise ValueError('Point head base identity differs')
    p=copy.deepcopy(json.loads((ROOT/'reports/course_tiled_point_refinement/protocol.json').read_text()))
    for key in ('val_reference','test_reference','new_training'):
        p.pop(key,None)
    history=ROOT/'reports/course_tiled_point_refinement/formal_v1/comparison.json'
    p.update(schema='course_longer_tiled_v1',status='frozen',
        created_at_utc=datetime.now(timezone.utc).isoformat(),gpus=[2,3,5,6,7],
        archive=pp['archive'],
        head={'path':head['path'],'sha256':head['sha256'],'epoch':head['epoch']},
        head_kind='depth' if choice['variant'].startswith('point_depth') else 'plain',
        point_protocol={'path':str(args.point_protocol.resolve()),'sha256':file_sha(args.point_protocol)},
        historical_comparison={'path':str(history),'sha256':file_sha(history)},
        point_validation={'path':str((args.point_run/'comparison.json').resolve()),
                          'sha256':file_sha(args.point_run/'comparison.json')},
        base_epoch=reload['epoch'],worker_timeout_seconds=28800,
        selection='Maximum full validation APs across six whole-image, three tiled and historical candidates; tie AP, then retain historical',
        composition='Fixed tile640 and probability-depth settings; point head chosen by whole-image full validation',
        test_policy='Separate test phase requires unchanged completed full validation selection',
        limitations=['Best among evaluated candidates; single seed',
                     'Chosen point epoch/type precedes tiled validation; not an exhaustive tile/head-epoch search'])
    arm=re.sub('[^a-z0-9_]+','_',Path(pp['weights']['path']).parent.name.lower())
    arm=arm.replace('lr5e5','lr5em5').replace('lr2e5','lr2em5')
    p['base_tag']=f'base_e{reload["epoch"]:02d}_{arm}'
    sources={**p['source_sha256'],**pp['source_sha256']}
    for name in ('tools/run_course_longer_tiled.py','tools/build_course_longer_tiled_protocol.py'):
        sources[name]=file_sha(ROOT/name)
    for name,digest in sources.items():
        if file_sha(ROOT/name)!=digest:
            raise ValueError(f'Frozen source changed: {name}')
    p['source_sha256']=sources
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'frozen','path':str(args.output.resolve()),'sha256':file_sha(args.output)}))


if __name__=='__main__':main()
