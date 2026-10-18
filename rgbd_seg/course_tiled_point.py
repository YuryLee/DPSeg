"""Frozen tiled representatives followed by the trained depth point head."""
from dataclasses import replace
import time

import numpy as np
import torch
from torch.nn import functional as F

from .course_point_refinement import frozen_features, depth_cues, mask_boxes, refine_logits
from .course_tiled_probability import (
    view_bounds, crop_sample, candidates_from_logits, merge_candidates,
    refined_public, encode_mask,
)
from .course_point_guard import guard_predictions

ARMS = ('tiled_probability_depth', 'tiled_point_depth_probability_depth',
        'tiled_point_depth_probability_depth_guard')


@torch.inference_mode()
def predict(adapter, head, sample, *, zero_check=False):
    """No annotations enter this function. Selection precedes either refiner."""
    began = time.monotonic()
    full_size = tuple(sample['image'].shape[-2:])
    cache, candidates = {}, []
    for view_id, bounds in enumerate(view_bounds(*full_size)):
        view = sample if view_id == 0 else crop_sample(sample, bounds)
        outputs, fine, sizes, _, _ = frozen_features(adapter, [view])
        if sizes['padded_size'] != (1024, 1024):
            raise ValueError('Unexpected model coordinates')
        values = candidates_from_logits(outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
            original_size=sizes['original_sizes'][0], bounds=bounds, full_size=full_size, view_id=view_id)
        candidates.extend(values)
        cache[view_id] = (fine, outputs.masks_queries_logits[0], view)
    selected, merging = merge_candidates(candidates)
    baseline, _ = refined_public(selected, sample, 'probability', adapter.device)
    changed = {}
    zero_views = []
    for view_id, (fine, logits, view) in cache.items():
        group = [c for c in selected if c.view_id == view_id]
        if not group:
            continue
        coarse = logits[[c.query_id for c in group]].float()
        raw = torch.stack([c.probability > .5 for c in group])
        boxes = mask_boxes(raw)
        cues = depth_cues(view['depth_m'].to(adapter.device), view['depth_valid'].to(adapter.device))
        original_size = tuple(view['image'].shape[-2:])
        if zero_check:
            # Same architecture, zero residual output: must restore original logits/masks.
            from .course_point_refinement import PointHead
            zero = PointHead(depth_channels=9).to(adapter.device).eval()
            restored = refine_logits(zero, coarse, fine, cues, boxes)
            if original_size != (1024, 1024):
                restored = F.interpolate(restored[None], size=original_size, mode='bilinear', align_corners=False)[0]
            if not torch.equal(restored > 0, raw):
                raise ValueError('Zero head changed crop/whole mask')
            for j, c in enumerate(group):
                torch.testing.assert_close(restored[j].sigmoid(), c.probability, rtol=0, atol=1e-7)
            zero_views.append(view_id)
            del zero, restored
        refined = refine_logits(head, coarse, fine, cues, boxes)
        if original_size != (1024, 1024):
            refined = F.interpolate(refined[None], size=original_size, mode='bilinear', align_corners=False)[0]
        binary, probability = refined > 0, refined.sigmoid()
        empty = ~binary.flatten(1).any(1)
        binary[empty] = raw[empty]
        for j, c in enumerate(group):
            if empty[j]:
                probability[j] = c.probability
        masks = binary.cpu().numpy()
        for j, c in enumerate(group):
            full = np.zeros(full_size, dtype=bool)
            left, top, right, bottom = c.bounds
            full[top:bottom, left:right] = masks[j]
            changed[(c.view_id, c.query_id)] = replace(c, probability=probability[j].clone(),
                segmentation=encode_mask(full))
    point_candidates = [changed[(c.view_id, c.query_id)] for c in selected]
    point, _ = refined_public(point_candidates, sample, 'probability', adapter.device)
    guarded, guard_stats = guard_predictions(baseline, point)
    outputs = dict(zip(ARMS, (baseline, point, guarded)))
    identities = [(v['category_id'], v['score']) for v in baseline]
    if any([(v['category_id'], v['score']) for v in values] != identities for values in outputs.values()):
        raise ValueError('Point head changed score/order/count')
    torch.cuda.synchronize(adapter.device)
    return outputs, {'seconds': time.monotonic()-began, 'merging': merging,
                     'guard': guard_stats, 'zero_check_views': zero_views,
                     'selected_identities': [[c.view_id,c.query_id] for c in selected]}
