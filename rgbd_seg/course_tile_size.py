"""Vary tile size while preserving the frozen DPSeg model and mask refiners.

The window grid follows SAHI's stride and end-alignment rule, with a nominal
overlap ratio. End alignment can increase the actual overlap. This module does
not use SAHI's detector adapter or prediction merging implementation.
"""
from __future__ import annotations

import math

import torch

from .course_depth_backbone_points import frozen_features
from .course_depth_backbone_system import crop_sample
from .course_swinl_removal import refine_candidates
from .course_tiled_probability import candidates_from_logits, merge_candidates, refined_public


def axis_starts(length, size, overlap):
    if type(length) is not int or type(size) is not int or not 0 < size <= length:
        raise ValueError('Tile size must be a positive integer within the image')
    if not math.isfinite(overlap) or not 0 <= overlap < 1:
        raise ValueError('Overlap must be finite and in [0, 1)')
    stride = size - int(size * overlap)
    starts, position = [], 0
    while True:
        position = min(position, length - size)
        starts.append(position)
        if position + size == length:
            return starts
        position += stride


def tile_bounds(height, width, size, overlap=.2):
    result = [(0, 0, width, height)]
    for top in axis_starts(height, size, overlap):
        for left in axis_starts(width, size, overlap):
            box = (left, top, left + size, top + size)
            if box not in result:
                result.append(box)
    return result


def geometry(size, overlap=.2):
    starts = axis_starts(1024, size, overlap)
    overlaps = [size - (right-left) for left, right in zip(starts, starts[1:])]
    bounds = tile_bounds(1024, 1024, size, overlap)
    return dict(tile_size_px=size, nominal_overlap_ratio=overlap,
        nominal_stride_px=size-int(size*overlap), axis_starts_px=starts,
        actual_adjacent_overlap_px=overlaps,
        actual_adjacent_overlap_ratios=[value/size for value in overlaps],
        tiles=len(bounds)-1, views_including_whole=len(bounds),
        model_input_size=[1024, 1024], magnification=1024/size, bounds=bounds)


def extract_view(adapter, view, bounds, full_size, view_id):
    outputs, fine, sizes, _, _ = frozen_features(adapter, [view])
    if sizes['padded_size'] != (1024, 1024):
        raise ValueError('Unexpected model coordinates')
    candidates = candidates_from_logits(
        outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
        padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
        original_size=sizes['original_sizes'][0], bounds=bounds,
        full_size=full_size, view_id=view_id)
    return candidates, (fine, outputs.masks_queries_logits[0], view)


@torch.inference_mode()
def predict_sizes(adapter, head, sample, sizes=(512, 768), overlap=.2):
    if set(sample) != {'image', 'depth_m', 'depth_valid'} or adapter.depth_encoding != 'depth':
        raise ValueError('Expected no-intrinsics RGB/depth/validity only')
    full_size = tuple(sample['image'].shape[-2:])
    if full_size != (1024, 1024) or len(set(sizes)) != len(sizes):
        raise ValueError('Expected 1024-square images and distinct sizes')
    native, whole_features = extract_view(adapter, sample, (0, 0, 1024, 1024), full_size, 0)
    outputs, statistics = {}, {}
    for size in sizes:
        # Each arm has its own candidates and refinement state. Only the
        # unchanged whole-image feature/logit tensors are reused across arms.
        candidates, cache = list(native), {0: whole_features}
        bounds = tile_bounds(*full_size, size, overlap)
        for view_id, box in enumerate(bounds[1:], 1):
            view = crop_sample(sample, box)
            current, cache[view_id] = extract_view(adapter, view, box, full_size, view_id)
            candidates.extend(current)
            del current
        selected, merging = merge_candidates(candidates)
        point = refine_candidates(adapter, head, selected, cache, full_size)
        values, refinement = refined_public(point, sample, 'probability', adapter.device)
        if [(p['category_id'], p['score']) for p in values] != [(1, c.score) for c in selected]:
            raise ValueError('Mask refinement changed selected scores or order')
        outputs[str(size)] = values
        statistics[str(size)] = dict(merging=merging, probability_changes=refinement,
                                     model_views=len(bounds))
        del cache, candidates, selected, point
    return outputs, statistics, [candidate.public() for candidate in native]
