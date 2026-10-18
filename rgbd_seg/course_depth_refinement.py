"""Inference-only joint bilateral smoothing of independent binary masks.

This is a fixed, depth-guided postprocessor, not a learned refiner or an
implementation of joint bilateral upsampling (the masks are already full size).
Scores, order and instance count are preserved. Invalid centre depths and
otherwise-empty refined instances fall back to their original predictions.
"""
from __future__ import annotations

import copy
import math
import numpy as np
from pycocotools import mask as mask_utils
import torch

POLICY = {'radius': 2, 'spatial_sigma_px': 1.5, 'depth_sigma_m': 0.001,
          'iterations': 1, 'threshold': 0.5, 'tie': 'original',
          'invalid_center': 'original', 'invalid_neighbor': 'exclude',
          'empty_result': 'original', 'scores': 'unchanged',
          'instance_order_and_count': 'unchanged', 'chunk_size': 16}


@torch.inference_mode()
def refine_masks(masks, depth, valid, *, mode='depth', radius=2,
                 spatial_sigma_px=1.5, depth_sigma_m=0.001, chunk_size=16):
    """Return bool[N,H,W] and counters; no labels or RGB enter this function."""
    if mode not in ('depth', 'spatial'):
        raise ValueError('Unknown refinement mode')
    if (type(radius) is not int or radius < 0 or type(chunk_size) is not int
            or chunk_size < 1):
        raise ValueError('Invalid integer radius/chunk_size')
    if any(not math.isfinite(x) or x <= 0 for x in (spatial_sigma_px, depth_sigma_m)):
        raise ValueError('Sigmas must be finite and positive')
    if (masks.ndim != 3 or depth.ndim != 2 or depth.shape != valid.shape
            or masks.shape[-2:] != depth.shape or min(depth.shape) < 1):
        raise ValueError('Expected aligned masks[N,H,W],depth[H,W],valid[H,W]')
    if masks.dtype != torch.bool or valid.dtype != torch.bool or not depth.is_floating_point():
        raise ValueError('Expected bool masks/valid and floating depth')
    if masks.device != depth.device or valid.device != depth.device:
        raise ValueError('Mismatched input devices')
    if not torch.isfinite(depth[valid]).all() or (depth[valid] <= 0).any():
        raise ValueError('Valid depths must be finite and positive metres')
    if masks.shape[0] and not masks.flatten(1).any(1).all():
        raise ValueError('Original instances must be nonempty')
    depth = depth.float()
    h, w = depth.shape
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
            weight = usable.float()*math.exp(-(dx*dx+dy*dy)/(2*spatial_sigma_px**2))
            if mode == 'depth':
                # Mask before the exponential so invalid NaN/Inf cannot leak.
                delta = torch.where(usable, depth[centre]-depth[neighbor], 0.)
                weight *= torch.exp(-0.5*(delta/depth_sigma_m).square())
            denominator[centre] += weight
            weights.append((centre, neighbor, weight))
    output = torch.empty_like(masks)
    counters = {'instances': masks.shape[0], 'changed_instances': 0,
                'added_pixels': 0, 'removed_pixels': 0, 'empty_reverted': 0}
    for start in range(0, masks.shape[0], chunk_size):
        original = masks[start:start+chunk_size]
        numerator = torch.zeros(original.shape, device=depth.device, dtype=torch.float32)
        for centre, neighbor, weight in weights:
            numerator[(slice(None), *centre)] += original[(slice(None), *neighbor)].float()*weight
        midpoint = denominator[None]*0.5
        refined = torch.where((numerator == midpoint) | ~valid[None], original, numerator > midpoint)
        empty = ~refined.flatten(1).any(1)
        counters['empty_reverted'] += int(empty.sum().item())
        refined[empty] = original[empty]
        counters['changed_instances'] += int((refined != original).flatten(1).any(1).sum().item())
        counters['added_pixels'] += int((refined & ~original).sum().item())
        counters['removed_pixels'] += int((original & ~refined).sum().item())
        output[start:start+chunk_size] = refined
    return output, counters


def refine_predictions(predictions, depth, valid, *, mode, device='cpu'):
    """Decode independent RLEs, refine once with the frozen policy, re-encode."""
    if not predictions:
        return [], {'instances': 0, 'changed_instances': 0, 'added_pixels': 0,
                    'removed_pixels': 0, 'empty_reverted': 0}
    rles = [{'size': p['segmentation']['size'],
             'counts': p['segmentation']['counts'].encode('ascii')} for p in predictions]
    masks = mask_utils.decode(rles).transpose(2, 0, 1).astype(bool)
    result, stats = refine_masks(torch.from_numpy(masks).to(device).contiguous(),
        torch.as_tensor(depth, dtype=torch.float32, device=device),
        torch.as_tensor(valid, dtype=torch.bool, device=device), mode=mode,
        radius=POLICY['radius'], spatial_sigma_px=POLICY['spatial_sigma_px'],
        depth_sigma_m=POLICY['depth_sigma_m'], chunk_size=POLICY['chunk_size'])
    output = copy.deepcopy(predictions)
    result = result.cpu().numpy()
    for item, mask in zip(output, result):
        encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
        item['segmentation'] = {'size': encoded['size'], 'counts': encoded['counts'].decode('ascii')}
    return output, stats
