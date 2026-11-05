"""Fixed-weight, leave-one-module-out DPSeg inference with shared view features.

All postprocessing rules come from the frozen full system. Four outputs are
kept separate: full, without the point head, without probability refinement,
and without tiles. No ground truth or camera intrinsics enter prediction.
"""
from dataclasses import replace

import numpy as np
import torch
from torch.nn import functional as F

from .course_depth_backbone_points import frozen_features, depth_cues, mask_boxes, refine_logits
from .course_depth_backbone_system import crop_sample
from .course_tiled_probability import (
    candidates_from_logits, encode_mask, merge_candidates, refined_public, view_bounds,
)

VARIANTS = ('full', 'without_point', 'without_probability', 'without_tiles')


@torch.inference_mode()
def refine_candidates(adapter, head, selected, cache, full_size):
    """Same source-view point prediction and empty-mask fallback as tiled_pair."""
    changed = {}
    for view_id, (fine, logits, view) in cache.items():
        group = [candidate for candidate in selected if candidate.view_id == view_id]
        if not group:
            continue
        coarse = logits[[candidate.query_id for candidate in group]].float()
        raw = torch.stack([candidate.probability > .5 for candidate in group])
        boxes = mask_boxes(raw)
        cues = depth_cues(view['depth_m'].to(adapter.device), view['depth_valid'].to(adapter.device))
        original_size = tuple(view['image'].shape[-2:])
        refined = refine_logits(head, coarse, fine, cues, boxes)
        if original_size != (1024, 1024):
            refined = F.interpolate(refined[None], size=original_size,
                                    mode='bilinear', align_corners=False)[0]
        binary, probability = refined > 0, refined.sigmoid()
        empty = ~binary.flatten(1).any(1)
        binary[empty] = raw[empty]
        for j, candidate in enumerate(group):
            if empty[j]:
                probability[j] = candidate.probability
        masks = binary.cpu().numpy()
        for j, candidate in enumerate(group):
            full = np.zeros(full_size, dtype=bool)
            left, top, right, bottom = candidate.bounds
            full[top:bottom, left:right] = masks[j]
            changed[(candidate.view_id, candidate.query_id)] = replace(
                candidate, probability=probability[j].clone(), segmentation=encode_mask(full))
    return [changed[(candidate.view_id, candidate.query_id)] for candidate in selected]


@torch.inference_mode()
def predict_removals(adapter, head, sample):
    if set(sample) != {'image', 'depth_m', 'depth_valid'} or adapter.depth_encoding != 'depth':
        raise ValueError('Expected no-intrinsics RGB/depth/validity inputs only')
    full_size = tuple(sample['image'].shape[-2:])
    if full_size != (1024, 1024):
        raise ValueError('Expected the original 1024-square course image')
    cache, candidates, whole = {}, [], None
    for view_id, bounds in enumerate(view_bounds(*full_size)):
        view = sample if view_id == 0 else crop_sample(sample, bounds)
        outputs, fine, sizes, _, _ = frozen_features(adapter, [view])
        if sizes['padded_size'] != (1024, 1024):
            raise ValueError('Unexpected model coordinates')
        current = candidates_from_logits(
            outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
            original_size=sizes['original_sizes'][0], bounds=bounds,
            full_size=full_size, view_id=view_id)
        if view_id == 0:
            whole = current
        candidates.extend(current)
        cache[view_id] = (fine, outputs.masks_queries_logits[0], view)
    selected, merging = merge_candidates(candidates)
    # Remove the point head while retaining the same merged candidates and
    # the original probability/depth filter.
    without_point, no_point_stats = refined_public(selected, sample, 'probability', adapter.device)
    point = refine_candidates(adapter, head, selected, cache, full_size)
    without_probability = [candidate.public() for candidate in point]
    full, full_stats = refined_public(point, sample, 'probability', adapter.device)
    del point
    # Whole-image candidates are selected before cross-view NMS. This matches
    # the original point+probability S3, including its ordering and top-100 cap.
    whole_point = refine_candidates(adapter, head, whole, {0: cache[0]}, full_size)
    without_tiles, no_tile_stats = refined_public(whole_point, sample, 'probability', adapter.device)
    identities = [(candidate.score, candidate.view_id, candidate.query_id) for candidate in selected]
    if [(p['category_id'], p['score']) for p in full] != [(1, value[0]) for value in identities]:
        raise ValueError('Full refinement changed selected candidate scores/order')
    for values in (without_point, without_probability):
        if [(p['category_id'], p['score']) for p in values] != [(p['category_id'], p['score']) for p in full]:
            raise ValueError('Removing mask refinement changed selected candidate scores/order')
    if [(p['category_id'], p['score']) for p in without_tiles] != [(1, candidate.score) for candidate in whole]:
        raise ValueError('Whole-image refinement changed candidates')
    return dict(full=full, without_point=without_point, without_probability=without_probability,
                without_tiles=without_tiles), dict(merging=merging,
        coarse_selected=identities, whole_count=len(whole),
        probability_changes={'full': full_stats, 'without_point': no_point_stats, 'without_tiles': no_tile_stats})
