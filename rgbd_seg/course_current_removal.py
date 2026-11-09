"""Independent removals from the adopted guarded640 DPSeg inference pipeline."""
from __future__ import annotations

import torch

from .course_depth_backbone_system import crop_sample
from .course_dpseg import DEFAULT_MERGE, OVERLAP_CONFIGS, _check_sample
from .course_guarded_merge import identity, select_candidates
from .course_swinl_removal import refine_candidates
from .course_tile_size import extract_view
from .course_tiled_probability import refined_public

VARIANTS = ('without_point', 'without_probability')
COMPONENTS = {
    'without_point': dict(rgbd=True, point=False, probability=True, tiles=True),
    'without_probability': dict(rgbd=True, point=True, probability=False, tiles=True),
}


def check_invariants(selected, point, full, values):
    """Only masks may differ across the two removals and their full reference."""
    expected = [(1, candidate.score) for candidate in selected]
    if [identity(candidate) for candidate in point] != [identity(candidate) for candidate in selected]:
        raise ValueError('Point refinement changed selected view/query identities')
    if tuple(values) != VARIANTS:
        raise ValueError('Unexpected removal variants')
    for predictions in (full, *values.values()):
        if [(p['category_id'], p['score']) for p in predictions] != expected:
            raise ValueError('Removal changed selected candidate scores, categories, count or ordering')


@torch.inference_mode()
def predict_current_removals(adapter, head, sample):
    _check_sample(adapter, sample)
    full_size = (1024, 1024)
    geometry = OVERLAP_CONFIGS['overlap40']
    cache, candidates, whole = {}, [], None
    for view_id, bounds in enumerate(geometry['bounds']):
        view = sample if view_id == 0 else crop_sample(sample, bounds)
        current, cache[view_id] = extract_view(adapter, view, bounds, full_size, view_id)
        if view_id == 0:
            whole = current
        candidates.extend(current)
    selected, merging = select_candidates(candidates, DEFAULT_MERGE)
    # This branch consumes coarse candidates directly; the learned head is absent.
    without_point, no_point_stats = refined_public(selected, sample, 'probability', adapter.device)
    point = refine_candidates(adapter, head, selected, cache, full_size)
    # This branch exports point-refined masks directly; no probability filter.
    without_probability = [candidate.public() for candidate in point]
    full, full_stats = refined_public(point, sample, 'probability', adapter.device)
    values = dict(without_point=without_point, without_probability=without_probability)
    check_invariants(selected, point, full, values)
    # Reproduce the existing S3 / without-tiles reference from the same whole view.
    whole_point = refine_candidates(adapter, head, whole, {0: cache[0]}, full_size)
    without_tiles, no_tile_stats = refined_public(whole_point, sample, 'probability', adapter.device)
    if [(p['category_id'], p['score']) for p in without_tiles] != [(1, c.score) for c in whole]:
        raise ValueError('Whole-view refinement changed selected scores or ordering')
    stats = dict(geometry=geometry, merge_strategy=DEFAULT_MERGE, merging=merging,
        selected_identity=[list(identity(c)) for c in selected],
        selected_scores=[c.score for c in selected], point_identity_unchanged=True,
        removal_scores_and_order_unchanged=True, whole_count=len(whole),
        model_views=5, actual_model_input_size_verified=True, model_input_sizes=[[1024, 1024]]*5,
        probability_changes=dict(full=full_stats, without_point=no_point_stats, without_tiles=no_tile_stats))
    return values, stats, full, without_tiles, [c.public() for c in whole]
