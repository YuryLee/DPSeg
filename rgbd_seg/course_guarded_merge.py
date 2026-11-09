"""Fixed 640-pixel slice experiments protecting reliable whole-image masks.

Only candidate selection changes. Scores, source-view point refinement, depth
probability refinement, RLE masks and the final top-100 rule remain unchanged.
All geometry is computed from predictions in original-image coordinates.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils
import torch

from .course_depth_backbone_system import crop_sample
from .course_swinl_removal import refine_candidates
from .course_tile_size import extract_view
from .course_tiled_probability import (
    Candidate, encode_mask, merge_candidates, refined_public, view_bounds,
)

VARIANTS = ('global_guard', 'global_guard_edge', 'global_guard_consensus',
            'global_guard_edge_consensus', 'small_tiles_only')
POLICY = dict(tile_size_px=640, full_size=[1024, 1024], small_area_max=1024,
              reliable_score_min=.5, duplicate_iou_threshold=.5,
              internal_edge_distance_px=8, edge_priority_multiplier=.75,
              contained_fraction_min=.8, whole_to_tile_area_ratio_min=1.5,
              consensus_iou_min=.5, consensus_priority_floor=.75,
              consensus_priority_weight=.25, max_detections=100,
              final_score='unchanged', final_order='score descending, view_id, query_id',
              same_view_suppression=False, mask_union=False)
VARIANT_DESCRIPTIONS = {
    'global_guard': '优先保留分数≥0.5、预测面积>1024的整图候选，再进行原跨视图掩码去重',
    'global_guard_edge': '保护整图＋降低内部裁剪边缘候选优先级，剔除被可靠整图掩码包含的疑似截断候选',
    'global_guard_consensus': '保护整图＋按其他高置信度视图的掩码一致性调整选择优先级',
    'global_guard_edge_consensus': '保护整图＋裁剪边缘处理＋跨视图一致性',
    'small_tiles_only': '保护整图，且切片仅提供预测面积≤1024的小目标候选',
}


def identity(candidate):
    return candidate.view_id, candidate.query_id


def prediction_geometry(candidates):
    """Areas, pairwise overlaps and internal-edge distances; no GT is read."""
    n = len(candidates)
    if len({identity(c) for c in candidates}) != n:
        raise ValueError('Candidate view/query identities must be unique')
    rles = [c.segmentation for c in candidates]
    areas = np.array([int(mask_utils.area(rle)) for rle in rles], dtype=np.float64)
    if n and (areas <= 0).any():
        raise ValueError('Empty masks are not candidates')
    ious = mask_utils.iou(rles, rles, [0]*n) if n else np.zeros((0, 0))
    # Recover intersection from Jaccard overlap and areas without rasterizing.
    intersections = ious * (areas[:, None]+areas[None, :]) / (1+ious)
    distances = np.full(n, np.inf)
    height, width = POLICY['full_size']
    for i, candidate in enumerate(candidates):
        if candidate.view_id == 0:
            continue
        x, y, w, h = mask_utils.toBbox(candidate.segmentation).tolist()
        left, top, right, bottom = candidate.bounds
        edges = []
        if left > 0:
            edges.append(x-left)
        if top > 0:
            edges.append(y-top)
        if right < width:
            edges.append(right-(x+w))
        if bottom < height:
            edges.append(bottom-(y+h))
        if edges:
            distances[i] = min(edges)
            if distances[i] < 0:
                raise ValueError('Prediction extends outside its source view')
    views = np.array([c.view_id for c in candidates])
    scores = np.array([c.score for c in candidates])
    guard = (views == 0) & (areas > POLICY['small_area_max']) & (scores >= POLICY['reliable_score_min'])
    near = (views != 0) & (distances <= POLICY['internal_edge_distance_px'])
    support = np.zeros(n)
    for i in range(n):
        others = (views != views[i]) & (scores >= POLICY['reliable_score_min'])
        value = float(ious[i, others].max()) if others.any() else 0.
        support[i] = value if value >= POLICY['consensus_iou_min'] else 0.
    return dict(areas=areas, ious=ious, intersections=intersections,
                views=views, scores=scores, guard=guard, near=near, support=support)


def select_candidates(candidates, variant, geometry=None, legacy=None):
    """Greedy cross-view mask suppression, with optional selection priorities.

The priority is used only to choose a representative, never as its submitted
confidence. Every returned item is an original Candidate with its source view.
"""
    if variant not in VARIANTS:
        raise ValueError('Unknown guarded merging variant')
    g = prediction_geometry(candidates) if geometry is None else geometry
    n = len(candidates)
    eligible = np.ones(n, dtype=bool)
    priorities = g['scores'].copy()
    edge = variant in ('global_guard_edge', 'global_guard_edge_consensus')
    consensus = variant in ('global_guard_consensus', 'global_guard_edge_consensus')
    area_filtered = partial_filtered = 0
    if variant == 'small_tiles_only':
        eligible = (g['views'] == 0) | (g['areas'] <= POLICY['small_area_max'])
        area_filtered = int((~eligible).sum())
    if edge:
        priorities[g['near']] *= POLICY['edge_priority_multiplier']
        for i in np.flatnonzero(g['near']):
            containing = g['guard'] & (g['areas'] >= POLICY['whole_to_tile_area_ratio_min']*g['areas'][i])
            if np.any(g['intersections'][i, containing] / g['areas'][i] >= POLICY['contained_fraction_min']):
                eligible[i] = False
                partial_filtered += 1
    if consensus:
        priorities *= POLICY['consensus_priority_floor'] + POLICY['consensus_priority_weight']*g['support']
    order = sorted(np.flatnonzero(eligible).tolist(),
                   key=lambda i: (not bool(g['guard'][i]), -priorities[i],
                                  candidates[i].view_id, candidates[i].query_id))
    kept, duplicates = [], 0
    for i in order:
        if any(g['views'][j] != g['views'][i] and g['ious'][i, j] > POLICY['duplicate_iou_threshold'] for j in kept):
            duplicates += 1
        else:
            kept.append(i)
    # All arms apply the original public ranking and top-100 truncation.
    kept.sort(key=lambda i: (-candidates[i].score, candidates[i].view_id, candidates[i].query_id))
    selected = [candidates[i] for i in kept[:POLICY['max_detections']]]
    baseline = merge_candidates(candidates)[0] if legacy is None else legacy
    selected_ids, baseline_ids = {identity(c) for c in selected}, {identity(c) for c in baseline}
    stats = dict(candidates=n, protected_whole_candidates=int(g['guard'].sum()),
                 protected_whole_selected=sum(bool(g['guard'][i]) for i in kept[:POLICY['max_detections']]),
                 protected_whole_lost_to_top100=sum(bool(g['guard'][i]) for i in kept[POLICY['max_detections']:]),
                 near_internal_edge_candidates=int(g['near'].sum()),
                 cross_view_supported_candidates=int((g['support'] > 0).sum()),
                 area_filtered=area_filtered, partial_filtered=partial_filtered,
                 duplicates_removed=duplicates,
                 top100_removed=max(0, len(kept)-POLICY['max_detections']),
                 selected=len(selected), selected_whole=sum(c.view_id == 0 for c in selected),
                 selected_tiles=sum(c.view_id != 0 for c in selected),
                 selected_tile_small=sum(c.view_id != 0 and int(mask_utils.area(c.segmentation)) <= POLICY['small_area_max'] for c in selected),
                 added_vs_legacy=len(selected_ids-baseline_ids),
                 removed_vs_legacy=len(baseline_ids-selected_ids),
                 shared_vs_legacy=len(selected_ids & baseline_ids))
    return selected, stats


def check_synthetic():
    """Exercise complete-vs-partial masks, independent small objects and ranking."""
    def rectangle(x, y, w, h):
        a = np.zeros((1024, 1024), bool)
        a[y:y+h, x:x+w] = True
        return a

    def candidate(mask, score, view, query=0, bounds=(0, 0, 640, 640)):
        return Candidate(score, query, view, (0, 0, 1024, 1024) if view == 0 else bounds,
                         torch.empty(0), encode_mask(mask))

    whole = candidate(rectangle(100, 100, 50, 50), .7, 0)
    tile = candidate(rectangle(100, 100, 48, 50), .9, 1)
    small = candidate(rectangle(200, 100, 10, 10), .6, 1, 1)
    assert merge_candidates([whole, tile, small])[0][0] is tile
    for variant in VARIANTS:
        selected, _ = select_candidates([whole, tile, small], variant)
        assert selected == [whole, small]
        assert [c.score for c in selected] == [.7, .6]
    # Low-confidence whole predictions and small whole masks are not protected.
    low = candidate(rectangle(100, 100, 50, 50), .49, 0)
    assert select_candidates([low, tile], 'global_guard')[0] == [tile]
    assert select_candidates([candidate(rectangle(100, 100, 20, 20), .6, 0),
                              candidate(rectangle(100, 100, 20, 20), .9, 1)], 'global_guard')[0][0].view_id == 1
    # A cut-off fragment can have low IoU but high containment inside a whole mask.
    complete = candidate(rectangle(610, 300, 80, 30), .8, 0)
    fragment = candidate(rectangle(610, 300, 25, 30), .95, 1)
    assert len(select_candidates([complete, fragment], 'global_guard')[0]) == 2
    for variant in ('global_guard_edge', 'global_guard_edge_consensus'):
        selected, stats = select_candidates([complete, fragment], variant)
        assert selected == [complete] and stats['partial_filtered'] == 1
    # Image outer edges do not count as internal cut boundaries.
    outer = candidate(rectangle(0, 100, 10, 20), .8, 1)
    assert not prediction_geometry([outer])['near'][0]
    assert select_candidates([outer], 'global_guard_edge')[0] == [outer]
    # Consensus can choose a slightly lower-score mask supported by another view.
    a = candidate(rectangle(450, 100, 20, 20), .9, 1)
    b = candidate(rectangle(455, 100, 20, 20), .88, 2, bounds=(384, 0, 1024, 640))
    c = candidate(rectangle(455, 100, 20, 20), .7, 0)
    assert select_candidates([a, b, c], 'global_guard')[0] == [a]
    assert select_candidates([a, b, c], 'global_guard_consensus')[0] == [b]
    # Match the original strict >0.5 test and keep same-view candidates.
    same = candidate(rectangle(100, 100, 50, 50), .6, 0, 1)
    assert select_candidates([whole, same], 'global_guard')[0] == [whole, same]
    first = candidate(rectangle(100, 100, 20, 10), .9, 0)
    second = candidate(rectangle(100, 100, 10, 10), .8, 1)
    assert len(select_candidates([first, second], 'global_guard')[0]) == 2
    many = [candidate(rectangle(100, 100, 2, 2), .01+i/200, 0, i) for i in range(101)]
    selected, stats = select_candidates(many, 'global_guard')
    assert len(selected) == 100 and selected[0] is many[-1] and selected[-1] is many[1]
    assert stats['top100_removed'] == 1
    for variant in VARIANTS:
        assert select_candidates([], variant)[0] == []
    return dict(status='passed', whole_guard=True, low_score_and_small_whole_not_guarded=True,
                missing_small_supplement=True, partial_containment_filter=True,
                outer_boundary_exempt=True, consensus_changes_representative=True,
                original_scores_and_identity=True, strict_iou_and_same_view_rules=True,
                original_top100=True, empty_input=True)


@torch.inference_mode()
def predict_mergers(adapter, head, sample, *, verify_independent=False):
    if set(sample) != {'image', 'depth_m', 'depth_valid'} or adapter.depth_encoding != 'depth':
        raise ValueError('Expected no-intrinsics RGB/depth/validity inputs only')
    full_size = tuple(sample['image'].shape[-2:])
    if full_size != tuple(POLICY['full_size']):
        raise ValueError('Expected the original 1024-square image')
    bounds = view_bounds(*full_size)
    if len(bounds) != 5 or any((r-l, b-t) != (640, 640) for l, t, r, b in bounds[1:]):
        raise ValueError('This experiment requires the original four 640-pixel tiles')
    cache, candidates = {}, []
    for view_id, box in enumerate(bounds):
        view = sample if view_id == 0 else crop_sample(sample, box)
        current, cache[view_id] = extract_view(adapter, view, box, full_size, view_id)
        candidates.extend(current)
    native = [c.public() for c in candidates if c.view_id == 0]
    legacy_candidates, legacy_stats = merge_candidates(candidates)
    g = prediction_geometry(candidates)
    selections, merging = {}, {}
    for variant in VARIANTS:
        selections[variant], merging[variant] = select_candidates(candidates, variant, g, legacy_candidates)
    # Refine each selected source candidate once. Inference is per instance;
    # exact legacy comparison is required both by the gate and on every test image.
    union = {identity(c): c for c in legacy_candidates}
    for group in selections.values():
        union.update({identity(c): c for c in group})
    union_candidates = sorted(union.values(), key=lambda c: (-c.score, c.view_id, c.query_id))
    point = refine_candidates(adapter, head, union_candidates, cache, full_size)
    public, refinement = refined_public(point, sample, 'probability', adapter.device)
    refined = {identity(c): p for c, p in zip(union_candidates, public)}
    if len(refined) != len(union_candidates):
        raise ValueError('Refinement lost a source candidate')
    for c, p in zip(union_candidates, public):
        if (p['category_id'], p['score']) != (1, c.score):
            raise ValueError('Refinement changed public categories or confidence')
    values = {variant: [refined[identity(c)] for c in selections[variant]] for variant in VARIANTS}
    legacy = [refined[identity(c)] for c in legacy_candidates]
    del point
    if verify_independent:
        # GPU grouping/chunking must not confound the candidate-selection study.
        for variant, group in [('legacy_full', legacy_candidates), *selections.items()]:
            independent_point = refine_candidates(adapter, head, group, cache, full_size)
            independent_public, _ = refined_public(independent_point, sample, 'probability', adapter.device)
            expected = legacy if variant == 'legacy_full' else values[variant]
            if independent_public != expected:
                raise ValueError(f'Shared refinement differs from independent {variant} refinement')
            del independent_point, independent_public
    stats = dict(model_views=5, candidates=len(candidates), union_refined=len(union_candidates),
                 original_merging=legacy_stats, merging=merging, probability_refinement=refinement,
                 independent_refinement_verified=verify_independent,
                 output_counts={variant: len(values[variant]) for variant in VARIANTS})
    return values, stats, native, legacy
