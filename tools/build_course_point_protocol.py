#!/usr/bin/env python3
"""Freeze engineering/formal point-head protocols before observing outcomes."""
import argparse
from datetime import datetime,timezone
import importlib.metadata
import inspect
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from transformers.models.mask2former.modeling_mask2former import Mask2FormerHungarianMatcher
from rgbd_seg.course_point_refinement import POLICY
from rgbd_seg.course_optimized import file_sha


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scope',choices=['engineering_smoke','full_experiment'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--engineering',type=Path)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError('A frozen protocol must not be overwritten')
    base=json.loads((ROOT/'reports/course_tiled_probability/protocol_full_v1.json').read_text())
    sources=dict(json.loads((ROOT/'reports/course_resolution_ddp/protocol.json').read_text())['source_sha256'])
    names=['rgbd_seg/course_point_refinement.py','tools/run_course_point_refinement.py',
        'tools/supervise_course_point_refinement.py','tools/build_course_point_protocol.py',
        'tests/test_course_point_refinement.py','rgbd_seg/course_optimized.py','rgbd_seg/course_tiled_probability.py',
        'rgbd_seg/course_depth_refinement.py','tools/check_course_prediction_reload.py',
        str(Path(inspect.getfile(Mask2FormerHungarianMatcher)).resolve())]
    sources.update({name:file_sha(ROOT/name) for name in names})
    archive=Path(base['archive'])
    formal=args.scope=='full_experiment'
    p={'schema':'course_point_refinement_v1','status':'frozen','scope':args.scope,
        'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','mode':'run',
            'verification_status':'FOUR_GPU_ENGINEERING_PASSED' if formal else 'SINGLE_GPU_AND_CPU_PASSED_DDP_PENDING'},
        'point_policy':POLICY,'archive':str(archive),
        'archive_stat':{'bytes':archive.stat().st_size,'mtime_ns':archive.stat().st_mtime_ns},
        'weights':base['weights'],'training_protocol':base['training_protocol'],
        'source_sha256':sources,'gpus':[3,5,6,7],'seed':17,'batch_size':2,'global_batch_size':8,
        'learning_rate':.0003,'epochs':2 if formal else 1,'training_images':14432 if formal else 16,
        'selection':{'split':'val','metric':'mask_ap_small','minimum_gain_vs_native_probability_depth':.005,
                     'maximum_AP_drop':.005},
        'head_training':'paired point heads only; backbone/pixel/query/classification weights frozen',
        'versions':{name:importlib.metadata.version(name) for name in ('torch','transformers','pycocotools')},
        'point_reference':'https://arxiv.org/abs/1912.08193',
        'non_goals':['No Kaggle submission','No training on validation/test labels',
                     'No modification of historical frozen experiments']}
    if formal:
        if args.engineering is None:raise ValueError('Formal protocol requires engineering receipt')
        receipt=json.loads(args.engineering.read_text())
        if receipt['status']!='completed' or receipt['scope']!='engineering_smoke':
            raise ValueError('Four-GPU engineering did not complete')
        engineering_protocol=json.loads(Path(receipt['protocol']).read_text())
        if engineering_protocol['source_sha256']!=sources:raise ValueError('Source changed since engineering run')
        p['engineering']={'path':str(args.engineering.resolve()),'sha256':file_sha(args.engineering)}
    else:p['training_indices']=list(range(16))
    args.output.write_text(json.dumps(p,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'frozen','scope':p['scope'],'path':str(args.output),'sha256':file_sha(args.output)}))


if __name__=='__main__':main()
