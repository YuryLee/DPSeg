#!/usr/bin/env python3
"""Record SAHI polygon-union and pixel-coordinate semantics on synthetic masks."""
import copy
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from pycocotools import mask as mu
import torch
from rgbd_seg.course_sahi_merge import (
    Candidate, box_nms, polygon_objects, official_public, encode_mask, DEPENDENCIES,
)
from sahi.postprocess.combine import NMSPostprocess


def candidates(masks):
    return [Candidate(score, i, i, (0, 0, 1024, 1024), torch.empty(0), encode_mask(mask))
            for i, (mask, score) in enumerate(zip(masks, (.9, .8)))]


def main():
    a = np.zeros((1024, 1024), bool)
    a[10:30, 10:30] = True
    a[15:25, 15:25] = False
    b = np.zeros_like(a)
    b[10:30, 19:39] = True
    b[15:25, 19:29] = False
    objects, converted, stats = polygon_objects(candidates((a, b)))
    before = copy.deepcopy([p.mask.segmentation for p in objects])
    merged = official_public(objects, 'GREEDYNMM')
    actual = mu.decode(merged[0]['segmentation']).astype(bool)
    expected = np.logical_or.reduce([mu.decode(c.segmentation).astype(bool) for c in converted])
    assert stats['changed_instances'] == 0 and len(merged) == 1
    assert before == [p.mask.segmentation for p in objects]
    assert (actual & ~expected).sum() == 100 and not (expected & ~actual).any()
    ring = dict(case='overlapping_rings', input_conversion=stats, input_unchanged=True,
                binary_union_area=int(expected.sum()), official_polygon_union_area=int(actual.sum()),
                added_vs_binary_union=int((actual & ~expected).sum()), removed_vs_binary_union=0,
                geometry={'a_outer_yx':[10,30,10,30], 'a_hole_yx':[15,25,15,25],
                          'b_outer_yx':[10,30,19,39], 'b_hole_yx':[15,25,19,29]})
    a[:] = False
    b[:] = False
    a[10:30, 10:31] = True
    b[10:30, 17:38] = True
    cs = candidates((a, b))
    objects, _, stats = polygon_objects(cs)
    rle_count = len(box_nms(cs, 'IOU'))
    polygon_count = len(NMSPostprocess(match_metric='IOU', match_threshold=.5,
                                     class_agnostic=False)(objects))
    assert stats['changed_instances'] == 0 and (rle_count, polygon_count) == (1, 2)
    boxes = dict(case='bounding_box_coordinate_convention', input_conversion=stats,
        rle_xywh=[mu.toBbox(c.segmentation).tolist() for c in cs],
        polygon_xyxy=[p.bbox.to_xyxy() for p in objects],
        rle_pixel_edge_iou=14/28, polygon_contour_iou=13/27,
        rle_nms_count=rle_count, polygon_nms_count=polygon_count)
    sources = [Path(__file__).resolve(), ROOT/'rgbd_seg/course_sahi_merge.py',
               DEPENDENCIES/'sahi/postprocess/utils.py', DEPENDENCIES/'sahi/utils/shapely.py',
               DEPENDENCIES/'sahi/annotation.py', DEPENDENCIES/'sahi/utils/cv.py']
    result = dict(status='passed', scope='CPU-only synthetic geometry audit; no dataset labels or model inference',
        source_sha256={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        cases=[ring, boxes], conclusions=[
            'Official polygon union can fill holes even when pre-merge polygon conversion is lossless.',
            'COCO pixel-edge bounds and SAHI contour bounds can differ at the overlap threshold.',
            'Both late official arms use the same polygon objects; compare them within that representation.',
            'Pre-merge conversion statistics do not measure changes introduced during polygon union.'])
    output = ROOT/'reports/course_sahi_merge_v1/supplementary_semantics.json'
    output.parent.mkdir(exist_ok=True, parents=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
