"""Current DPSeg inference: guarded merging with four fully covering tiles.

Changing overlap changes crop size at a fixed view count. Each crop is resized
to the same 1024-square model input; this is not a pure overlap-only study.
Historical inference modules are intentionally left unchanged.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .course_depth_backbone_system import crop_sample
from .course_guarded_merge import select_candidates
from .course_swinl_removal import refine_candidates
from .course_tile_size import extract_view
from .course_tiled_probability import refined_public

DEFAULT_MERGE = 'global_guard_edge_consensus'
DEFAULT_OVERLAP = .4
DEFAULT_MANIFEST_PATH = 'configs/dpseg_current.json'


def four_tile_geometry(overlap):
    """Round crop size, anchor at four corners, and report realized overlap."""
    if isinstance(overlap, bool) or not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('Overlap must be finite and in [0, 1)')
    length = 1024
    size = round(length / (2 - overlap))
    if not 512 <= size < length:
        raise ValueError('Requested overlap cannot produce four distinct integer crops')
    start = length - size
    bounds = [(0, 0, length, length)]
    bounds.extend((x, y, x + size, y + size) for y in (0, start) for x in (0, start))
    return dict(tile_size_px=size, target_overlap_ratio=overlap,
                actual_overlap_ratio=(2 * size - length) / size,
                actual_overlap_px=2 * size - length,
                axis_starts_px=[0, start], bounds=bounds, tiles=4,
                views_including_whole=5, model_input_size=[length, length],
                coverage_fraction=1., magnification=length / size)


OVERLAP_CONFIGS = {f'overlap{percent:02d}': four_tile_geometry(percent / 100)
                   for percent in (0, 20, 40, 60, 80)}
OVERLAP_VARIANTS = tuple(OVERLAP_CONFIGS)
NEW_OVERLAP_VARIANTS = tuple(key for key in OVERLAP_VARIANTS if key != 'overlap40')


def check_synthetic():
    from .course_tiled_probability import view_bounds
    expected_sizes = [512, 569, 640, 731, 853]
    for config, size in zip(OVERLAP_CONFIGS.values(), expected_sizes):
        bounds = config['bounds']
        assert config['tile_size_px'] == size
        assert len(bounds) == len(set(bounds)) == 5
        coverage = np.zeros((1024, 1024), dtype=np.uint8)
        for left, top, right, bottom in bounds[1:]:
            assert right - left == bottom - top == size
            assert 0 <= left < right <= 1024 and 0 <= top < bottom <= 1024
            coverage[top:bottom, left:right] += 1
        assert coverage.min() >= 1
        assert config['actual_overlap_px'] == 2 * size - 1024
    assert OVERLAP_CONFIGS['overlap40']['bounds'] == view_bounds(1024, 1024)
    assert OVERLAP_CONFIGS['overlap00']['actual_overlap_px'] == 0
    assert OVERLAP_CONFIGS['overlap00']['tiles'] == 4
    return dict(status='passed', full_coverage=True, exactly_four_distinct_tiles=True,
                zero_overlap_disjoint=True, reference40_bounds_exact=True,
                model_input_size=[1024, 1024], tile_sizes=expected_sizes)


def _check_sample(adapter, sample):
    if set(sample) != {'image', 'depth_m', 'depth_valid'} or adapter.depth_encoding != 'depth':
        raise ValueError('Expected no-intrinsics RGB/depth/validity inputs only')
    if tuple(sample['image'].shape[-2:]) != (1024, 1024):
        raise ValueError('Current four-tile inference expects a 1024-square image')


def _predict_arm(adapter, head, sample, geometry, native, whole_features):
    full_size = (1024, 1024)
    candidates, cache = list(native), {0: whole_features}
    for view_id, bounds in enumerate(geometry['bounds'][1:], 1):
        view = crop_sample(sample, bounds)
        current, cache[view_id] = extract_view(adapter, view, bounds, full_size, view_id)
        candidates.extend(current)
    selected, merging = select_candidates(candidates, DEFAULT_MERGE)
    point = refine_candidates(adapter, head, selected, cache, full_size)
    values, refinement = refined_public(point, sample, 'probability', adapter.device)
    if [(p['category_id'], p['score']) for p in values] != [(1, c.score) for c in selected]:
        raise ValueError('Mask refinement changed categories, scores or ordering')
    # extract_view asserts the actual padded input shape returned by the model
    # on every forward, including the shared whole-image forward.
    stats = dict(geometry=geometry, merging=merging, probability_changes=refinement,
                 model_views=geometry['views_including_whole'], merge_strategy=DEFAULT_MERGE,
                 actual_model_input_size_verified=True,
                 model_input_sizes=[[1024, 1024] for _ in geometry['bounds']])
    return values, stats


@torch.inference_mode()
def predict_dpseg(adapter, head, sample, overlap=DEFAULT_OVERLAP):
    """Run the current full method; return public predictions and diagnostics."""
    _check_sample(adapter, sample)
    geometry = four_tile_geometry(overlap)
    native, whole_features = extract_view(adapter, sample, (0, 0, 1024, 1024), (1024, 1024), 0)
    return _predict_arm(adapter, head, sample, geometry, native, whole_features)


@torch.inference_mode()
def predict_overlaps(adapter, head, sample, *, include_reference=False, verify_independent=False):
    """Evaluate new overlap arms, sharing only whole-image features/logits.

The 40% arm is optional because its full-test predictions already exist. The
engineering gate can compare it with the independently implemented historical
guarded-merge pipeline, including exact scores and compressed RLEs.
"""
    _check_sample(adapter, sample)
    native, whole_features = extract_view(adapter, sample, (0, 0, 1024, 1024), (1024, 1024), 0)
    values, statistics = {}, {}
    for variant in NEW_OVERLAP_VARIANTS:
        values[variant], statistics[variant] = _predict_arm(
            adapter, head, sample, OVERLAP_CONFIGS[variant], native, whole_features)
    reference40 = None
    if include_reference:
        reference40, statistics['overlap40'] = _predict_arm(
            adapter, head, sample, OVERLAP_CONFIGS['overlap40'], native, whole_features)
    if verify_independent:
        if not include_reference:
            raise ValueError('Independent reference verification requires include_reference=True')
        from .course_guarded_merge import predict_mergers
        independent, _, independent_native, _ = predict_mergers(adapter, head, sample)
        if independent[DEFAULT_MERGE] != reference40 or independent_native != [c.public() for c in native]:
            raise ValueError('Current default differs from the historical guarded-merge reference')
    return values, statistics, [c.public() for c in native], reference40
