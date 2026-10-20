"""Run one frozen S0-S4 configuration without computing the other ablations.

The tiled path follows course_progressive_ablation.tiled_pair, with probability
refinement optional so S3 does not pay for S4. The benchmark runner checks every
sampled output against the original full-validation predictions.
"""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from .course_point_refinement import (
    frozen_features, depth_cues, mask_boxes, refine_logits, predict_paired,
)
from .course_tiled_probability import (
    view_bounds, crop_sample, candidates_from_logits, merge_candidates,
    refined_public, encode_mask,
)


@torch.inference_mode()
def predict(stage, adapter, head, sample):
    if set(sample) != {'image', 'depth_m', 'depth_valid', 'camera_K'}:
        raise ValueError('Inference accepts only RGB, depth, validity and K')
    if stage == 'S0':
        return adapter.predict_sample(sample)
    if stage == 'S1':
        batch, sizes = adapter.prepare_batch([sample])
        outputs = adapter.model(**batch, return_dict=True, output_auxiliary_logits=False)
        h, w = sample['image'].shape[-2:]
        candidates = candidates_from_logits(
            outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
            original_size=(h, w), bounds=(0, 0, w, h), full_size=(h, w), view_id=0)
        return [c.public() for c in candidates]
    if stage == 'S2':
        paired = predict_paired(adapter, SimpleNamespace(depth=head), sample,
                                head_names=('point_depth',))
        return [c.public() for c in paired['point_depth']]
    if stage not in ('S3', 'S4'):
        raise ValueError(f'Unknown stage: {stage}')
    full_size = tuple(sample['image'].shape[-2:])
    cache, candidates = {}, []
    for view_id, bounds in enumerate(view_bounds(*full_size)):
        view = sample if view_id == 0 else crop_sample(sample, bounds)
        outputs, fine, sizes, _, _ = frozen_features(adapter, [view])
        if sizes['padded_size'] != (1024, 1024):
            raise ValueError('Unexpected model coordinates')
        candidates.extend(candidates_from_logits(
            outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
            original_size=sizes['original_sizes'][0], bounds=bounds,
            full_size=full_size, view_id=view_id))
        cache[view_id] = (fine, outputs.masks_queries_logits[0], view)
    selected, _ = merge_candidates(candidates)
    changed = {}
    for view_id, (fine, logits, view) in cache.items():
        group = [c for c in selected if c.view_id == view_id]
        if not group:
            continue
        coarse = logits[[c.query_id for c in group]].float()
        raw = torch.stack([c.probability > .5 for c in group])
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
    point_candidates = [changed[(c.view_id, c.query_id)] for c in selected]
    if stage == 'S3':
        return [c.public() for c in point_candidates]
    result, _ = refined_public(point_candidates, sample, 'probability', adapter.device)
    return result
