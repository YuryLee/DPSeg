#!/usr/bin/env python3
"""Frozen M2/M3 inference for selected report examples; no fitting or GT input."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_backbone import load_sample
from rgbd_seg.course_depth_backbone_points import PairedPointHeads, predict_paired
from rgbd_seg.course_tiled_probability import refined_public
from tools.run_course_depth_backbone_system import load_base
from tools.select_report_figure_cases import OUT, RUN, ARCHIVE, SOURCES, digest, load_records, write

def main():
    torch.set_num_threads(4)
    # Match the locked evaluation runner, including convolution arithmetic.
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    chosen = json.loads((OUT/'selected_cases.json').read_text())
    ids = sorted({r['image_id'] for r in chosen})
    output = OUT/'supplemental_predictions'
    output.mkdir(exist_ok=True)
    protocol = json.loads((RUN/'protocol_system.json').read_text())
    base_path, head_path = RUN/'base15/final.pt', RUN/'points6/heads_epoch_06.pt'
    metadata = dict(status='initializing', scope='selected qualitative examples only',
        image_ids=ids, ground_truth_in_inference=False, training=False,
        base=dict(path=str(base_path),sha256=digest(base_path)),
        heads=dict(path=str(head_path),sha256=digest(head_path)),
        input_keys=['image','depth_m','depth_valid'], precision='float32',
        gpu=torch.cuda.get_device_name(0), script_sha256=digest(__file__), tf32=False,
        stages={'M2':'whole image + trained depth point head',
                'M3':'M2 + existing probability-depth refinement; no tiling'})
    write(output/'run.json', metadata)
    adapter = load_base(protocol, base_path, torch.device('cuda:0'))
    heads = PairedPointHeads(fine_channels=448)
    saved = torch.load(head_path, map_location='cpu', weights_only=True)
    assert saved['base_checkpoint_sha256'] == metadata['base']['sha256']
    heads.load_state_dict(saved['heads'], strict=True)
    heads.eval().requires_grad_(False).to(adapter.device)
    del saved
    native = load_records(SOURCES['M1'], set(ids))
    checks = []
    with CourseArchive(ARCHIVE, split='test') as data, \
         (output/'M2.jsonl').open('w') as f2, (output/'M3.jsonl').open('w') as f3:
        with torch.inference_mode():
            for image_id in ids:
                started = time.monotonic()
                sample = load_sample(data, image_id, 'depth')
                assert set(sample) == {'image', 'depth_m', 'depth_valid'}
                paired = predict_paired(adapter, heads, sample, head_names=('point_depth',))
                point = [c.public() for c in paired['point_depth']]
                prob, stats = refined_public(paired['point_depth'], sample, 'probability', adapter.device)
                assert [(p['category_id'],p['score']) for p in point] == [(p['category_id'],p['score']) for p in prob]
                current_native = [c.public() for c in paired['native']]
                prev = native[image_id]
                checks.append(dict(image_id=image_id,
                    identical_native_rles=[p['segmentation'] for p in prev] == [p['segmentation'] for p in current_native],
                    old_count=len(prev), new_count=len(current_native),
                    max_score_difference=max((abs(p['score']-q['score']) for p,q in zip(prev,current_native)),default=0)))
                if not checks[-1]['identical_native_rles'] or checks[-1]['max_score_difference'] > 1e-6:
                    raise ValueError(f'Native output did not reproduce the locked evaluation: {checks[-1]}')
                for stream, values in ((f2,point),(f3,prob)):
                    stream.write(json.dumps(dict(image_id=image_id,predictions=values))+'\n')
                    stream.flush()
                print(f'{image_id}: M2/M3 done in {time.monotonic()-started:.1f}s', flush=True)
                del paired, sample, point, prob, current_native
                torch.cuda.empty_cache()
    metadata.update(status='completed', native_reproduction=checks,
        peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
        outputs={stage: dict(path=str(output/f'{stage}.jsonl'),sha256=digest(output/f'{stage}.jsonl'))
                 for stage in ('M2','M3')})
    write(output/'run.json', metadata)

if __name__ == '__main__':
    main()
