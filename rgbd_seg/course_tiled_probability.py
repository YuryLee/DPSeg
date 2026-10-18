"""Fixed-view inference and probability-domain depth refinement.

The trained model and historical binary refiner are unchanged. Crop selection,
mask deduplication and refinement use predictions, measured depth and camera K
only. Cross-view NMS selects a representative; it does not average probability
maps produced at different scales. All public masks use original coordinates.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F

from .course_depth_refinement import POLICY as BINARY_POLICY, refine_masks

TILE_POLICY = {
    'tile_size_px': 640,
    'positions': 'four corners, row-major; retain whole-image view',
    'overlap_at_1024_px': 256,
    'internal_border_margin_px': 2,
    'duplicate_metric': 'mask IoU',
    'duplicate_iou_threshold': 0.5,
    'suppress': 'cross-view only; IoU strictly greater than threshold',
    'representative': 'higher original score; tie: view then query id',
    'max_detections': 100,
    'score_threshold': 0.01,
    'target_resize': 'existing 1024 shortest-edge model preprocessing',
    'geometry': 'aligned RGB/depth/valid crop; K_crop = translation @ K',
    'depth_normalization': 'existing model standardization applied per view',
}
PROBABILITY_POLICY = {
    **BINARY_POLICY,
    'input': 'float32 sigmoid of restored original-view mask logits',
    'operation': 'one joint bilateral weighted mean, then threshold once',
    'composition': 'replaces binary depth refinement; no serial smoothing',
    'scores': 'unchanged from raw mask quality score',
}
VARIANTS = ('native', 'native_binary_depth', 'native_probability_depth',
            'tiled', 'tiled_binary_depth', 'tiled_probability_depth')


@dataclass
class Candidate:
    score: float
    query_id: int
    view_id: int
    bounds: tuple[int, int, int, int]
    probability: torch.Tensor
    segmentation: dict

    def public(self):
        return {'category_id': 1, 'score': self.score, 'segmentation': self.segmentation}


def view_bounds(height, width):
    if min(height, width) < 1:
        raise ValueError('Image dimensions must be positive')
    ch, cw = min(height, TILE_POLICY['tile_size_px']), min(width, TILE_POLICY['tile_size_px'])
    result = [(0, 0, width, height)]
    for y in (0, height-ch):
        for x in (0, width-cw):
            box = (x, y, x+cw, y+ch)
            if box not in result:
                result.append(box)
    return result


def crop_sample(sample, box):
    if set(sample) != {'image', 'depth_m', 'depth_valid', 'camera_K'}:
        raise ValueError('Only RGB/depth/valid/K may enter inference')
    left, top, right, bottom = box
    h, w = sample['image'].shape[-2:]
    if not 0 <= left < right <= w or not 0 <= top < bottom <= h:
        raise ValueError('Crop outside image')
    k = sample['camera_K']
    affine = k.new_tensor([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
    return {name: sample[name][:, top:bottom, left:right].contiguous()
            for name in ('image', 'depth_m', 'depth_valid')} | {'camera_K': affine @ k}


def encode_mask(mask):
    value = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    return {'size': value['size'], 'counts': value['counts'].decode('ascii')}


@torch.inference_mode()
def candidates_from_logits(class_logits, mask_logits, *, padded_size, image_size,
                           original_size, bounds, full_size, view_id):
    """Preserve the frozen adapter's query, threshold, score and resize rules."""
    if class_logits.ndim != 2 or class_logits.shape[1] != 2:
        raise ValueError('Expected foreground/no-object logits')
    if mask_logits.ndim != 3 or len(mask_logits) != len(class_logits):
        raise ValueError('Mask/query mismatch')
    if not torch.isfinite(class_logits).all() or not torch.isfinite(mask_logits).all():
        raise ValueError('Non-finite model predictions')
    left, top, right, bottom = bounds
    if original_size != (bottom-top, right-left):
        raise ValueError('View shape differs from crop bounds')
    scores = class_logits.float().softmax(-1)[:, 0]
    scores, indices = scores.topk(min(100, len(scores)), sorted=False)
    result = []
    for start in range(0, len(indices), 8):
        stop = min(start+8, len(indices))
        logits = F.interpolate(mask_logits[indices[start:stop]].float()[None],
                               size=padded_size, mode='bilinear', align_corners=False)
        logits = logits[:, :, :image_size[0], :image_size[1]]
        logits = F.interpolate(logits, size=original_size, mode='bilinear', align_corners=False)[0]
        masks = logits > 0
        areas = masks.flatten(1).sum(1)
        probabilities = logits.sigmoid()
        quality = (probabilities*masks).flatten(1).sum(1)/(areas+1e-6)
        final_scores = scores[start:stop]*quality
        masks_cpu = masks.cpu().numpy()
        scores_cpu = final_scores.cpu().tolist()
        area_cpu = areas.cpu().tolist()
        for j, score in enumerate(scores_cpu):
            if not area_cpu[j] or score < TILE_POLICY['score_threshold']:
                continue
            binary = masks_cpu[j]
            if view_id:
                margin = TILE_POLICY['internal_border_margin_px']
                touches = ((left > 0 and binary[:, :margin].any())
                           or (top > 0 and binary[:margin, :].any())
                           or (right < full_size[1] and binary[:, -margin:].any())
                           or (bottom < full_size[0] and binary[-margin:, :].any()))
                if touches:
                    continue
            if bounds == (0, 0, full_size[1], full_size[0]):
                full_mask = binary
            else:
                full_mask = np.zeros(full_size, dtype=bool)
                full_mask[top:bottom, left:right] = binary
            result.append(Candidate(score, int(indices[start+j]), view_id, bounds,
                                    probabilities[j].clone(), encode_mask(full_mask)))
    result.sort(key=lambda c: (-c.score, c.query_id))
    return result


@torch.inference_mode()
def infer_view(adapter, sample, bounds, full_size, view_id):
    if adapter.training:
        raise ValueError('Frozen inference requires eval mode')
    batch, sizes = adapter.prepare_batch([sample])
    outputs = adapter.model(**batch, return_dict=True)
    return candidates_from_logits(outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
        padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
        original_size=sizes['original_sizes'][0], bounds=bounds, full_size=full_size, view_id=view_id)


def merge_candidates(candidates):
    ordered = sorted(candidates, key=lambda c: (-c.score, c.view_id, c.query_id))
    if not ordered:
        return [], {'candidates': 0, 'duplicates_removed': 0, 'top100_removed': 0, 'selected_views': []}
    rles = [{'size': c.segmentation['size'], 'counts': c.segmentation['counts'].encode('ascii')}
            for c in ordered]
    overlaps = mask_utils.iou(rles, rles, [0]*len(rles))
    kept, duplicates = [], 0
    for i, candidate in enumerate(ordered):
        if any(ordered[j].view_id != candidate.view_id
               and overlaps[i, j] > TILE_POLICY['duplicate_iou_threshold'] for j in kept):
            duplicates += 1
        else:
            kept.append(i)
    chosen = [ordered[i] for i in kept[:TILE_POLICY['max_detections']]]
    return chosen, {'candidates': len(candidates), 'duplicates_removed': duplicates,
                    'top100_removed': max(0, len(kept)-len(chosen)),
                    'selected_views': [c.view_id for c in chosen]}


@torch.inference_mode()
def refine_probability_masks(probabilities, original, depth, valid, *, radius=2,
                             spatial_sigma_px=1.5, depth_sigma_m=0.001, chunk_size=16):
    """Same geometry/support/fallback as the binary refiner, with soft values."""
    if (probabilities.ndim != 3 or probabilities.shape != original.shape
            or original.dtype != torch.bool or not probabilities.is_floating_point()
            or depth.ndim != 2 or depth.shape != valid.shape or valid.dtype != torch.bool
            or probabilities.shape[-2:] != depth.shape):
        raise ValueError('Expected aligned probabilities/original/depth/valid')
    if len({x.device for x in (probabilities, original, depth, valid)}) != 1:
        raise ValueError('Input devices differ')
    if (not torch.isfinite(probabilities).all() or (probabilities < 0).any()
            or (probabilities > 1).any() or not torch.isfinite(depth[valid]).all()
            or (depth[valid] <= 0).any()):
        raise ValueError('Invalid probability or metric depth')
    if type(radius) is not int or radius < 0 or type(chunk_size) is not int or chunk_size < 1:
        raise ValueError('Invalid radius/chunk size')
    if any(not math.isfinite(x) or x <= 0 for x in (spatial_sigma_px, depth_sigma_m)):
        raise ValueError('Invalid sigmas')
    if original.shape[0] and not original.flatten(1).any(1).all():
        raise ValueError('Original masks must be nonempty')
    h, w = depth.shape
    depth = depth.float()
    denominator = torch.zeros_like(depth)
    weights = []
    for dy in range(-radius, radius+1):
        for dx in range(-radius, radius+1):
            y0, y1 = max(0, -dy), min(h, h-dy)
            x0, x1 = max(0, -dx), min(w, w-dx)
            if y0 >= y1 or x0 >= x1:
                continue
            centre = (slice(y0, y1), slice(x0, x1))
            neighbor = (slice(y0+dy, y1+dy), slice(x0+dx, x1+dx))
            usable = valid[centre] & valid[neighbor]
            delta = torch.where(usable, depth[centre]-depth[neighbor], 0.)
            weight = (usable.float()*math.exp(-(dx*dx+dy*dy)/(2*spatial_sigma_px**2))
                      *torch.exp(-0.5*(delta/depth_sigma_m).square()))
            denominator[centre] += weight
            weights.append((centre, neighbor, weight))
    output = torch.empty_like(original)
    stats = {'instances': len(original), 'changed_instances': 0,
             'added_pixels': 0, 'removed_pixels': 0, 'empty_reverted': 0}
    for start in range(0, len(original), chunk_size):
        raw = original[start:start+chunk_size]
        soft = probabilities[start:start+chunk_size].float()
        numerator = torch.zeros_like(soft)
        for centre, neighbor, weight in weights:
            numerator[(slice(None), *centre)] += soft[(slice(None), *neighbor)]*weight
        midpoint = denominator[None]*0.5
        refined = torch.where((numerator == midpoint) | ~valid[None], raw, numerator > midpoint)
        empty = ~refined.flatten(1).any(1)
        refined[empty] = raw[empty]
        stats['empty_reverted'] += int(empty.sum())
        stats['changed_instances'] += int((raw != refined).flatten(1).any(1).sum())
        stats['added_pixels'] += int((refined & ~raw).sum())
        stats['removed_pixels'] += int((raw & ~refined).sum())
        output[start:start+chunk_size] = refined
    return output, stats


@torch.inference_mode()
def refined_public(candidates, sample, mode, device):
    if not candidates:
        return [], {'instances': 0, 'changed_instances': 0, 'added_pixels': 0,
                    'removed_pixels': 0, 'empty_reverted': 0}
    full_size = tuple(sample['image'].shape[-2:])
    rles = [{'size': c.segmentation['size'], 'counts': c.segmentation['counts'].encode('ascii')}
            for c in candidates]
    masks = torch.from_numpy(mask_utils.decode(rles).transpose(2, 0, 1).astype(bool)).to(device)
    depth = sample['depth_m'][0].to(device)
    valid = sample['depth_valid'][0].to(device)
    if mode == 'binary':
        refined, stats = refine_masks(masks, depth, valid, mode='depth')
    elif mode == 'probability':
        # Construct soft maps only for surviving candidates; crop backgrounds
        # outside each view are zero, with the fixed internal-edge filter above.
        probabilities = torch.zeros((len(candidates), *full_size), device=device, dtype=torch.float32)
        for i, candidate in enumerate(candidates):
            left, top, right, bottom = candidate.bounds
            probabilities[i, top:bottom, left:right] = candidate.probability.to(device)
        refined, stats = refine_probability_masks(probabilities, masks, depth, valid)
    else:
        raise ValueError('Unknown refinement domain')
    cpu = refined.cpu().numpy()
    return [{'category_id': 1, 'score': c.score, 'segmentation': encode_mask(binary)}
            for c, binary in zip(candidates, cpu)], stats


@torch.inference_mode()
def predict_variants(adapter, sample, *, native_check=None):
    """Five model views produce six paired outputs without using annotations."""
    import time
    device = adapter.device
    synchronize = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    synchronize()
    begin = time.perf_counter()
    full_size = tuple(sample['image'].shape[-2:])
    views = view_bounds(*full_size)
    native = infer_view(adapter, sample, views[0], full_size, 0)
    synchronize()
    native_seconds = time.perf_counter()-begin
    outputs = {'native': [c.public() for c in native]}
    if native_check is not None:
        native_check(outputs['native'])
    counts, ref_times = {}, {}
    for mode in ('binary', 'probability'):
        synchronize()
        start = time.perf_counter()
        key = f'native_{mode}_depth'
        outputs[key], counts[key] = refined_public(native, sample, mode, device)
        synchronize()
        ref_times[key] = time.perf_counter()-start
    synchronize()
    start = time.perf_counter()
    all_candidates = list(native)
    for i, bounds in enumerate(views[1:], 1):
        all_candidates.extend(infer_view(adapter, crop_sample(sample, bounds), bounds, full_size, i))
    selected, merging = merge_candidates(all_candidates)
    synchronize()
    tile_extra_seconds = time.perf_counter()-start
    outputs['tiled'] = [c.public() for c in selected]
    for mode in ('binary', 'probability'):
        synchronize()
        start = time.perf_counter()
        key = f'tiled_{mode}_depth'
        outputs[key], counts[key] = refined_public(selected, sample, mode, device)
        synchronize()
        ref_times[key] = time.perf_counter()-start
    return outputs, {'native_seconds': native_seconds, 'tile_extra_seconds': tile_extra_seconds,
                     'refinement_seconds': ref_times, 'merging': merging, 'refinement_stats': counts}
