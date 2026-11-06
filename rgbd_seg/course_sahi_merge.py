"""Official SAHI postprocessors with frozen DPSeg views and mask refiners.

Early NMS replaces only candidate selection. Late comparisons refine every
candidate in its source view, then merge. An order control and a polygon
conversion control distinguish these integration effects from box merging.
This is not SAHI's full sliced-prediction/model-adapter API.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path
import sys

import numpy as np
from pycocotools import mask as mask_utils
import torch

ROOT = Path(__file__).resolve().parents[1]
DEPENDENCIES = ROOT / 'reports/course_sahi_merge_deps/site'
if DEPENDENCIES.is_dir():
    sys.path.insert(0, str(DEPENDENCIES))

import cv2
import sahi
import shapely
from sahi.postprocess.backends import set_postprocess_backend
from sahi.postprocess.combine import NMSPostprocess, GreedyNMMPostprocess
from sahi.prediction import ObjectPrediction
from sahi.utils.cv import get_coco_segmentation_from_bool_mask

from .course_depth_backbone_system import crop_sample
from .course_swinl_removal import refine_candidates
from .course_tile_size import extract_view
from .course_tiled_probability import (
    Candidate, encode_mask, merge_candidates, refined_public, view_bounds,
)

VARIANTS = ('sahi_nms_iou_early', 'mask_nms_late',
            'sahi_polygon_mask_nms_late', 'sahi_nms_ios_late', 'sahi_greedynmm_ios_late')
set_postprocess_backend('numpy')


def dependency_receipt():
    """Bind the exact official source and geometry binaries used by this run."""
    if sahi.__version__ != '0.14.0' or shapely.__version__ != '2.1.2':
        raise ValueError('This protocol requires SAHI 0.14.0 and Shapely 2.1.2')
    files = {}
    for module in (sahi, shapely, cv2):
        base = Path(module.__file__).resolve().parent
        for path in sorted(base.rglob('*')):
            if path.is_file() and path.suffix in ('.py', '.so'):
                files[f'{module.__name__}/{path.relative_to(base)}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    for path in sorted((DEPENDENCIES/'shapely.libs').glob('*')):
        if path.is_file():
            files[f'shapely.libs/{path.name}'] = hashlib.sha256(path.read_bytes()).hexdigest()
    wheel = DEPENDENCIES.parent/'wheels/sahi-0.14.0-py3-none-any.whl'
    checksum = hashlib.sha256(wheel.read_bytes()).hexdigest()
    if checksum != '7fb73078e82fb34fb170276ae6a017e9250c4b64842926c332a8b683aa7a8d81':
        raise ValueError('SAHI wheel differs from published PyPI SHA256')
    return dict(sahi=sahi.__version__, shapely=shapely.__version__, opencv=cv2.__version__,
                numpy=np.__version__, backend='numpy', official_wheel_sha256=checksum, files=files)


def box_nms(candidates, metric='IOU'):
    """Official NMS; identity mapping preserves exact RLE and source features."""
    ordered = sorted(candidates, key=lambda c: (-c.score, c.view_id, c.query_id))
    objects, source = [], {}
    for c in ordered:
        x, y, w, h = mask_utils.toBbox(c.segmentation).tolist()
        obj = ObjectPrediction(bbox=[x, y, x+w, y+h], score=c.score, category_id=1,
                               category_name='component', full_shape=[1024, 1024])
        objects.append(obj)
        source[id(obj)] = c
    selected = NMSPostprocess(match_metric=metric, match_threshold=.5, class_agnostic=False)(objects) if objects else []
    selected = sorted(selected, key=lambda p: -p.score.value)[:100]
    return [source[id(p)] for p in selected]


def polygon_objects(candidates):
    objects, converted = [], []
    stats = dict(input_instances=len(candidates), changed_instances=0,
                 dropped_instances=0, added_pixels=0, removed_pixels=0)
    for c in candidates:
        binary = mask_utils.decode(c.segmentation).astype(bool)
        segmentation = get_coco_segmentation_from_bool_mask(binary)
        # Official polygon representation cannot encode single pixels/lines.
        # Drop these explicitly in all polygon-based arms, never invent a box mask.
        if not segmentation:
            stats['dropped_instances'] += 1
            stats['removed_pixels'] += int(binary.sum())
            continue
        obj = ObjectPrediction(segmentation=segmentation, score=c.score, category_id=1,
                               category_name='component', full_shape=list(binary.shape))
        reconstructed = obj.mask.bool_mask
        if not reconstructed.any():
            raise ValueError('Unexpected empty SAHI polygon rasterization')
        stats['changed_instances'] += int(not np.array_equal(binary, reconstructed))
        stats['added_pixels'] += int((reconstructed & ~binary).sum())
        stats['removed_pixels'] += int((binary & ~reconstructed).sum())
        objects.append(obj)
        converted.append(replace(c, segmentation=encode_mask(reconstructed)))
    return objects, converted, stats


def official_public(objects, mode):
    cls = NMSPostprocess if mode == 'NMS' else GreedyNMMPostprocess
    # GreedyNMM replaces entries in its input list; isolate the container.
    result = cls(match_metric='IOS', match_threshold=.5, class_agnostic=False)(list(objects)) if objects else []
    result = sorted(result, key=lambda p: -p.score.value)[:100]
    values = []
    for p in result:
        if p.mask is None or not p.mask.bool_mask.any():
            raise ValueError('Official postprocessor returned an empty mask')
        values.append(dict(category_id=p.category.id, score=float(p.score.value),
                           segmentation=encode_mask(p.mask.bool_mask)))
    return values


def check_synthetic():
    """Check overlap semantics, union, source identity and representation loss."""
    def candidate(binary, score, view, query=0):
        return Candidate(score, query, view, (0, 0, 1024, 1024),
                         torch.empty(0), encode_mask(binary))
    a = np.zeros((1024, 1024), bool)
    a[10:30, 10:30] = True
    b = np.zeros_like(a)
    b[10:30, 21:41] = True
    # Box overlap .29 IoU, .45 IoS: shift to overlap above .5 IoS only.
    b[:] = False
    b[10:30, 19:39] = True
    cs = [candidate(a, .9, 0), candidate(b, .8, 1)]
    assert len(box_nms(cs, 'IOU')) == 2
    assert len(box_nms(cs, 'IOS')) == 1
    same_view = [candidate(a, .9, 0, 0), candidate(a, .8, 0, 1)]
    assert len(merge_candidates(same_view)[0]) == 2
    assert len(box_nms(same_view)) == 1
    upper, lower = np.zeros_like(a), np.zeros_like(a)
    upper[100:120, 100:120] = np.triu(np.ones((20, 20), bool), 1)
    lower[100:120, 100:120] = np.tril(np.ones((20, 20), bool), -1)
    disjoint = [candidate(upper, .9, 0), candidate(lower, .8, 1)]
    assert not (upper & lower).any()
    assert len(merge_candidates(disjoint)[0]) == 2 and len(box_nms(disjoint)) == 1
    objects, _, roundtrip = polygon_objects(cs)
    before = [obj.mask.segmentation for obj in objects]
    nms = official_public(objects, 'NMS')
    merged = official_public(objects, 'GREEDYNMM')
    assert len(nms) == len(merged) == 1 and merged[0]['score'] == .9
    assert np.array_equal(mask_utils.decode(merged[0]['segmentation']).astype(bool), a | b)
    assert before == [obj.mask.segmentation for obj in objects]
    pixel = np.zeros_like(a)
    pixel[4, 4] = True
    _, _, tiny = polygon_objects([candidate(pixel, .7, 0)])
    assert tiny['dropped_instances'] == 1
    ring = a.copy()
    ring[15:25, 15:25] = False
    _, _, holes = polygon_objects([candidate(ring, .8, 0)])
    assert box_nms(cs)[0] is cs[0]
    return dict(status='passed', bbox_iou_vs_ios=True, same_view_suppression_difference=True,
                disjoint_masks_overlapping_boxes=True,
                union_mask_and_max_score=True, source_identity_preserved=True,
                input_not_mutated=True, polygon_rectangle_roundtrip=roundtrip,
                single_pixel_conversion=tiny, ring_conversion=holes)


@torch.inference_mode()
def predict_mergers(adapter, head, sample):
    if set(sample) != {'image', 'depth_m', 'depth_valid'} or adapter.depth_encoding != 'depth':
        raise ValueError('Expected no-intrinsics RGB/depth/validity inputs only')
    full_size = tuple(sample['image'].shape[-2:])
    if full_size != (1024, 1024):
        raise ValueError('Expected 1024-square original images')
    cache, candidates = {}, []
    for view_id, bounds in enumerate(view_bounds(*full_size)):
        view = sample if view_id == 0 else crop_sample(sample, bounds)
        current, cache[view_id] = extract_view(adapter, view, bounds, full_size, view_id)
        candidates.extend(current)
    native = [c.public() for c in candidates if c.view_id == 0]
    selected, original_stats = merge_candidates(candidates)
    point = refine_candidates(adapter, head, selected, cache, full_size)
    legacy, _ = refined_public(point, sample, 'probability', adapter.device)
    del point
    early = box_nms(candidates)
    point = refine_candidates(adapter, head, early, cache, full_size)
    early_public, _ = refined_public(point, sample, 'probability', adapter.device)
    del point
    # Each candidate keeps its original view features until both refiners finish.
    # Do one view at a time to bound memory; no labels are used in any step.
    late, refinement_stats = [], []
    for view_id in cache:
        group = [c for c in candidates if c.view_id == view_id]
        point = refine_candidates(adapter, head, group, {view_id: cache[view_id]}, full_size)
        public, stats = refined_public(point, sample, 'probability', adapter.device)
        late.extend(replace(c, probability=torch.empty(0), segmentation=p['segmentation'])
                    for c, p in zip(group, public))
        refinement_stats.append(stats)
        del point, public
    late = sorted(late, key=lambda c: (-c.score, c.view_id, c.query_id))
    late_selected, late_stats = merge_candidates(late)
    objects, converted, conversion = polygon_objects(late)
    polygon_selected, polygon_stats = merge_candidates(converted)
    result = dict(sahi_nms_iou_early=early_public,
        mask_nms_late=[c.public() for c in late_selected],
        sahi_polygon_mask_nms_late=[c.public() for c in polygon_selected],
        sahi_nms_ios_late=official_public(objects, 'NMS'),
        sahi_greedynmm_ios_late=official_public(objects, 'GREEDYNMM'))
    stats = dict(model_views=5, candidates=len(candidates), original_merging=original_stats,
                 late_merging=late_stats, polygon_merging=polygon_stats,
                 polygon_conversion=conversion, late_refinement=refinement_stats,
                 output_counts={k: len(v) for k, v in result.items()})
    return result, stats, native, legacy
