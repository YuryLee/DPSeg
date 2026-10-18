"""Frozen-weight full-image, high-resolution and fixed Crop&Fuse inference.

Only RGB/depth/validity tensors enter this module. All returned predictions use
original image coordinates; no labels or target masks are used to select crops.
The internal-crop-edge filter and fixed area rule are project adaptations, not
a reproduction of the cited 2025 Crop&Fuse paper's original-workspace edge rule.
"""
from __future__ import annotations

import math

import torch
from torchvision.ops import box_iou


PROTOCOL = {
    'crop_fraction': 0.625,
    'crop_max_area': 96 ** 2,
    'internal_border_margin': 2,
    'merge_method': 'same-class greedy box NMS; suppress IoU > threshold',
    'merge_iou': 0.5,
    'max_detections': 100,
    'highres_min_size': 1152,
    'highres_max_size': 2048,
    'mask_threshold': 0.5,
    'crop_area_reference': 'binary prediction pixels in original crop coordinates',
    'output_coordinates': 'original full image',
    'tie_break': 'view order (full, TL, TR, BL, BR), then original candidate order',
}


def corner_crops(height: int, width: int) -> list[tuple[int, int, int, int]]:
    """Unique (left, top, right, bottom) crops, excluding a duplicate full view."""
    if height <= 0 or width <= 0:
        raise ValueError('Image dimensions must be positive')
    ch, cw = math.ceil(height * PROTOCOL['crop_fraction']), math.ceil(width * PROTOCOL['crop_fraction'])
    seen = {(0, 0, width, height)}
    result = []
    for top in (0, height - ch):
        for left in (0, width - cw):
            box = (left, top, left + cw, top + ch)
            if box not in seen:
                result.append(box)
                seen.add(box)
    return result


def _empty(height, width):
    return {'boxes': torch.empty((0, 4), dtype=torch.float32),
            'scores': torch.empty((0,), dtype=torch.float32),
            'labels': torch.empty((0,), dtype=torch.int64),
            'masks': torch.empty((0, 1, height, width), dtype=torch.bool)}


def _binary_cpu(prediction, height, width):
    """Validate before binarizing, so numerical failures cannot become background."""
    boxes, scores, labels, masks = (prediction[key] for key in ('boxes', 'scores', 'labels', 'masks'))
    n = len(scores)
    if boxes.shape != (n, 4) or scores.shape != (n,) or labels.shape != (n,) or masks.shape != (n, 1, height, width):
        raise ValueError('Prediction must use original view dimensions and aligned N×1×H×W masks')
    if not torch.isfinite(boxes).all() or not torch.isfinite(scores).all() or not torch.isfinite(masks).all():
        raise ValueError('Prediction contains non-finite values')
    if ((scores < 0) | (scores > 1)).any() or ((masks < 0) | (masks > 1)).any():
        raise ValueError('Scores and masks must be probabilities in [0, 1]')
    if labels.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8) or (labels <= 0).any():
        raise ValueError('Prediction class indices must be positive integers')
    return {'boxes': boxes.detach().to(device='cpu', dtype=torch.float32),
            'scores': scores.detach().to(device='cpu', dtype=torch.float32),
            'labels': labels.detach().to(device='cpu', dtype=torch.int64),
            'masks': (masks >= PROTOCOL['mask_threshold']).to(device='cpu')}


def _same_class_nms(boxes, scores, labels):
    """Deterministic CPU NMS; tied scores retain the earlier view/candidate."""
    order = torch.argsort(scores, descending=True, stable=True)
    kept = []
    while order.numel():
        current = int(order[0])
        kept.append(current)
        remaining = order[1:]
        if not remaining.numel():
            break
        overlaps = box_iou(boxes[current:current + 1], boxes[remaining])[0]
        suppress = (labels[remaining] == labels[current]) & (overlaps > PROTOCOL['merge_iou'])
        order = remaining[~suppress]
    return torch.tensor(kept, dtype=torch.int64)


@torch.inference_mode()
def run_inference(model, sample, method='full'):
    """Return (CPU TorchVision prediction, JSON-compatible diagnostic counts).

    `model` must already be in eval mode. Inputs may be on CPU or the model's
    device. The full/highres branches preserve the model's candidate order and
    boxes/scores/labels, removing only empty binary masks. Crop&Fuse additionally
    filters crop candidates and applies global same-class box NMS/top100.

    Highres temporarily changes the transform and is not safe to run concurrently
    with another call on the same model. The transform is restored on exceptions.
    """
    if method not in ('full', 'highres', 'crop_fuse'):
        raise ValueError('method must be full, highres or crop_fuse')
    if model.training:
        raise ValueError('Frozen-weight inference requires model.eval()')
    if set(sample) != {'image', 'depth', 'depth_valid'}:
        raise ValueError('Only image, depth and depth_valid inputs are accepted; no targets')
    image, depth, valid = (sample[key] for key in ('image', 'depth', 'depth_valid'))
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError('image must be 3×H×W')
    h, w = image.shape[-2:]
    if h <= 0 or w <= 0 or depth.shape != (1, h, w) or valid.shape != (1, h, w):
        raise ValueError('RGB, depth and validity must have aligned positive dimensions')
    parameter = next(model.parameters(), None)
    device = parameter.device if parameter is not None else image.device
    views = [(0, 0, w, h)]
    if method == 'crop_fuse':
        views.extend(corner_crops(h, w))
    counts = {'method': method, 'views': len(views), 'raw_candidates': 0,
              'empty_masks_removed': 0, 'crop_area_removed': 0, 'crop_border_removed': 0,
              'nms_removed': 0, 'topk_removed': 0, 'output_candidates': 0, 'output_source_views': [], 'per_view': []}
    transform = None
    if method == 'highres':
        core = getattr(model, 'core', model)
        transform = core.transform
        old_min, old_max = transform.min_size, transform.max_size
        transform.min_size, transform.max_size = (PROTOCOL['highres_min_size'],), PROTOCOL['highres_max_size']
    batches = []
    try:
        for view_index, (left, top, right, bottom) in enumerate(views):
            vh, vw = bottom - top, right - left
            inputs = {key: value[:, top:bottom, left:right].to(device) for key, value in sample.items()}
            raw = model([inputs])[0]
            pred = _binary_cpu(raw, vh, vw)
            del raw, inputs
            n = len(pred['scores'])
            nonempty = pred['masks'].flatten(1).any(dim=1)
            keep = nonempty.clone()
            area_removed = border_removed = 0
            if view_index > 0:
                area_ok = pred['masks'].flatten(1).sum(dim=1) <= PROTOCOL['crop_max_area']
                area_removed = int((keep & ~area_ok).sum())
                keep &= area_ok
                margin = PROTOCOL['internal_border_margin']
                boxes = pred['boxes']
                border = torch.zeros(n, dtype=torch.bool)
                if left > 0:
                    border |= boxes[:, 0] <= margin
                if top > 0:
                    border |= boxes[:, 1] <= margin
                if right < w:
                    border |= boxes[:, 2] >= vw - margin
                if bottom < h:
                    border |= boxes[:, 3] >= vh - margin
                border_removed = int((keep & border).sum())
                keep &= ~border
            pred = {key: value[keep] for key, value in pred.items()}
            pred['boxes'] += torch.tensor([left, top, left, top], dtype=torch.float32)
            batches.append((pred, (left, top, right, bottom)))
            view_counts = {'index': view_index, 'kind': 'full' if view_index == 0 else 'crop',
                           'bounds_xyxy': [left, top, right, bottom], 'raw_candidates': n,
                           'empty_masks_removed': int((~nonempty).sum()),
                           'crop_area_removed': area_removed, 'crop_border_removed': border_removed,
                           'after_filter': len(pred['scores'])}
            counts['per_view'].append(view_counts)
            for key in ('raw_candidates', 'empty_masks_removed', 'crop_area_removed', 'crop_border_removed'):
                counts[key] += view_counts[key]
    finally:
        if transform is not None:
            transform.min_size, transform.max_size = old_min, old_max
    if not any(len(pred['scores']) for pred, _ in batches):
        return _empty(h, w), counts
    boxes = torch.cat([pred['boxes'] for pred, _ in batches])
    scores = torch.cat([pred['scores'] for pred, _ in batches])
    labels = torch.cat([pred['labels'] for pred, _ in batches])
    if method == 'crop_fuse':
        keep = _same_class_nms(boxes, scores, labels)
        counts['nms_removed'] = len(scores) - len(keep)
        counts['topk_removed'] = max(0, len(keep) - PROTOCOL['max_detections'])
        keep = keep[:PROTOCOL['max_detections']]
    else:
        keep = torch.arange(len(scores))
    # Allocate full-resolution masks only for survivors; keep crop masks local
    # until NMS to avoid retaining 500 full-size float masks.
    masks = torch.zeros((len(keep), 1, h, w), dtype=torch.bool)
    references = [(view_index, local_index) for view_index, (pred, _) in enumerate(batches)
                  for local_index in range(len(pred['scores']))]
    source_views = []
    for output_index, candidate_index in enumerate(keep.tolist()):
        view_index, local_index = references[candidate_index]
        pred, (left, top, right, bottom) = batches[view_index]
        masks[output_index, :, top:bottom, left:right] = pred['masks'][local_index]
        source_views.append(view_index)
    counts['output_candidates'] = len(keep)
    counts['output_source_views'] = source_views
    return {'boxes': boxes[keep], 'scores': scores[keep], 'labels': labels[keep], 'masks': masks}, counts
