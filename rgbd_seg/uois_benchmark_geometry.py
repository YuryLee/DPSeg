"""Geometry for the existing course DPSeg model's external benchmark diagnostics.

This mirrors the established generalization/data.py mapping without importing
that package or changing vendor paths. It is a fixed 1024 course-model input
path, not evidence of UOAIS-Sim training or matched-training reproduction.
Only model inputs and prediction RLEs are handled here; no GT is read.
"""
from __future__ import annotations

import numpy as np
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F


INPUT_SIZE = 1024
RGB_PADDING = (.485, .456, .406)
_SAMPLE_KEYS = {'image', 'depth_m', 'depth_valid', 'camera_K'}


def _content_box(shape):
    if (len(shape) != 2 or any(type(n) is not int or n <= 0 for n in shape)):
        raise ValueError('original_size must be positive integer [height, width]')
    height, width = shape
    scale = INPUT_SIZE / max(height, width)
    resized_h, resized_w = round(height * scale), round(width * scale)
    if min(resized_h, resized_w) < 1:
        raise ValueError('Aspect ratio leaves no pixels in the resized short dimension')
    top, left = (INPUT_SIZE - resized_h) // 2, (INPUT_SIZE - resized_w) // 2
    return [left, top, left + resized_w, top + resized_h]


def letterbox_sample(sample):
    """Return (four-key sample, geometry) at fixed 1024 x 1024 resolution.

    RGB uses bilinear align_corners=False; depth/valid use nearest-exact. RGB pad
    is ImageNet mean, depth pad is zero and invalid. The half-pixel affine uses
    the actual rounded width/height, so K remains consistent for odd rectangles.
    Inputs are not mutated. Geometry contains no GT or annotation-derived values.
    """
    if not isinstance(sample, dict) or set(sample) != _SAMPLE_KEYS:
        raise ValueError('Expected exactly image, depth_m, depth_valid and camera_K')
    image, depth, valid, k = (sample[key] for key in ('image', 'depth_m', 'depth_valid', 'camera_K'))
    if any(not isinstance(value, torch.Tensor) for value in (image, depth, valid, k)):
        raise ValueError('All model inputs must be torch tensors')
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('Expected RGB [3,H,W]')
    height, width = image.shape[-2:]
    if depth.shape != (1, height, width) or valid.shape != depth.shape or k.shape != (3, 3):
        raise ValueError('RGB, depth, validity and K dimensions disagree')
    if (image.dtype != torch.float32 or depth.dtype != torch.float32
            or k.dtype != torch.float32 or valid.dtype != torch.bool):
        raise ValueError('Expected float32 RGB/depth/K and boolean validity')
    if len({value.device for value in (image, depth, valid, k)}) != 1:
        raise ValueError('All model inputs must be on the same device')
    if (not torch.isfinite(image).all() or torch.any(image < 0) or torch.any(image > 1)
            or not torch.isfinite(depth).all() or torch.any(depth[valid] <= 0)
            or torch.any(depth[~valid] != 0)):
        raise ValueError('Expected RGB in [0,1], positive valid depth and zero invalid depth')
    if (not torch.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0
            or not torch.equal(k[2], k.new_tensor([0., 0., 1.]))
            or torch.linalg.det(k).abs() < 1e-8):
        raise ValueError('Expected finite invertible original pinhole K')

    left, top, right, bottom = _content_box([height, width])
    resized_h, resized_w = bottom - top, right - left
    rgb = image.new_tensor(RGB_PADDING)[:, None, None].expand(3, INPUT_SIZE, INPUT_SIZE).clone()
    rgb[:, top:bottom, left:right] = F.interpolate(
        image[None], (resized_h, resized_w), mode='bilinear', align_corners=False)[0]
    mapped_depth = F.interpolate(depth[None], (resized_h, resized_w), mode='nearest-exact')[0]
    mapped_valid = F.interpolate(valid[None].float(), (resized_h, resized_w), mode='nearest-exact')[0].bool()
    padding = (left, INPUT_SIZE-right, top, INPUT_SIZE-bottom)
    mapped_depth, mapped_valid = F.pad(mapped_depth, padding), F.pad(mapped_valid, padding)
    sx, sy = resized_w / width, resized_h / height
    affine = k.new_tensor([[sx, 0, left + (sx-1)/2], [0, sy, top + (sy-1)/2], [0, 0, 1]])
    mapped = {'image': rgb, 'depth_m': mapped_depth, 'depth_valid': mapped_valid,
              'camera_K': affine @ k}
    geometry = {'original_size': [height, width], 'content_box': [left, top, right, bottom],
                'input_size': INPUT_SIZE, 'affine': affine.cpu().tolist()}
    return mapped, geometry


def _decode_prediction(rle):
    rle = dict(rle)
    if rle.get('size') != [INPUT_SIZE, INPUT_SIZE]:
        raise ValueError('Prediction RLE must use the 1024 letterbox dimensions')
    counts = rle.get('counts')
    if isinstance(counts, list):
        if any(type(n) is not int or n < 0 for n in counts) or sum(counts) != INPUT_SIZE ** 2:
            raise ValueError('Invalid prediction RLE counts')
        rle = mask_utils.frPyObjects(rle, INPUT_SIZE, INPUT_SIZE)
    elif isinstance(counts, str):
        rle['counts'] = counts.encode('ascii')
    elif not isinstance(counts, bytes):
        raise ValueError('Expected compressed or uncompressed COCO RLE')
    mask = mask_utils.decode(rle)
    if mask.shape != (INPUT_SIZE, INPUT_SIZE) or mask.dtype != np.uint8 or mask.max(initial=0) > 1:
        raise ValueError('Expected one binary prediction mask')
    return mask


def restore_predictions(predictions, geometry):
    """Crop padding, nearest-exact restore RLEs, and retain nonempty masks in order.

    Score values are copied unchanged; there is no threshold, sorting, NMS or
    overlap resolution. Only masks with zero area AFTER restoration are removed.
    Other record fields are preserved, with existing COCO bbox/area updated to
    original coordinates. The input list, records and RLEs are not mutated.
    """
    shape = geometry['original_size']
    box = _content_box(shape)
    if geometry.get('input_size') != INPUT_SIZE or geometry.get('content_box') != box:
        raise ValueError('Geometry does not describe the fixed centered 1024 letterbox')
    height, width = shape
    left, top, right, bottom = box
    result = []
    for prediction in predictions:
        mask = _decode_prediction(prediction['segmentation'])
        crop = torch.from_numpy(mask[top:bottom, left:right].copy()).float()[None, None]
        restored = F.interpolate(crop, (height, width), mode='nearest-exact')[0, 0].bool().numpy()
        if not restored.any():
            continue
        rle = mask_utils.encode(np.asfortranarray(restored, dtype=np.uint8))
        rle['counts'] = rle['counts'].decode('ascii')
        record = {**prediction, 'segmentation': rle}
        if 'area' in record:
            record['area'] = int(restored.sum())
        if 'bbox' in record:
            record['bbox'] = mask_utils.toBbox(rle).tolist()
        result.append(record)
    return result
