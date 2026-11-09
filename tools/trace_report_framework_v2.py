#!/usr/bin/env python3
"""Record real intermediate outputs without changing the frozen tiled pipeline."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_backbone import load_sample
from rgbd_seg.course_depth_backbone_points import PairedPointHeads
from rgbd_seg import course_depth_backbone_system as system
from tools.run_course_depth_backbone_system import load_base
from tools.select_report_figure_cases import ARCHIVE,RUN,SOURCES,load_records,write,digest

def main():
    out=ROOT/'outputs/report_figures_swinl_v2/framework_trace';out.mkdir(exist_ok=True)
    iid=202577005
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    adapter=load_base(json.loads((RUN/'protocol_system.json').read_text()),RUN/'base15/final.pt',torch.device('cuda:0'))
    heads=PairedPointHeads(fine_channels=448)
    heads.load_state_dict(torch.load(RUN/'points6/heads_epoch_06.pt',map_location='cpu',weights_only=True)['heads'],strict=True)
    heads.eval().requires_grad_(False).to(adapter.device)
    captured={};original_merge=system.merge_candidates;original_refiner=system.refined_public
    def save_candidates(name,candidates):
        captured[name]=[dict(c.public(),view_id=c.view_id,query_id=c.query_id,bounds=list(c.bounds)) for c in candidates]
        # Fixed candidate index chosen from the completed run for illustration.
        # No GT, labels or selection criteria are passed to model prediction.
        c=candidates[72]
        full=np.zeros((1024,1024),np.float32);l,t,r,b=c.bounds
        full[t:b,l:r]=c.probability.detach().cpu().numpy()
        np.save(out/f'{name}_probability_72.npy',full)
    def trace_merge(candidates):
        selected,stats=original_merge(candidates);save_candidates('merged',selected)
        return selected,stats
    def trace_refiner(candidates,*args,**kwargs):
        save_candidates('point',candidates)
        return original_refiner(candidates,*args,**kwargs)
    system.merge_candidates=trace_merge;system.refined_public=trace_refiner
    try:
        with CourseArchive(ARCHIVE,'test') as data,torch.inference_mode():
            sample=load_sample(data,iid,'depth')
            assert set(sample)=={'image','depth_m','depth_valid'}
            point,final,stats=system.tiled_pair(adapter,heads.depth,sample)
    finally:
        system.merge_candidates=original_merge;system.refined_public=original_refiner
    old=load_records(SOURCES['M4'],{iid})[iid]
    assert [p['segmentation'] for p in final]==[p['segmentation'] for p in old]
    assert max(abs(p['score']-q['score']) for p,q in zip(final,old))<1e-6
    assert len(captured['merged'])==len(point)==len(final)
    for name,records in captured.items():write(out/f'{name}.json',records)
    write(out/'final.json',final)
    write(out/'run.json',dict(image_id=iid,identical_to_locked_M4=True,ground_truth_in_inference=False,
        training=False,precision='float32, TF32 disabled',merging=stats,
        base_sha256=digest(RUN/'base15/final.pt'),heads_sha256=digest(RUN/'points6/heads_epoch_06.pt'),
        script_sha256=digest(__file__),pipeline_sha256=digest(ROOT/'rgbd_seg/course_depth_backbone_system.py')))
    print('Exact full-system reproduction; real merged, point and final masks saved.',flush=True)

if __name__=='__main__':main()
