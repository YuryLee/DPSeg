"""Shared UOAIS geometry for the existing DPSeg base and point-head losses.

The caller chooses augmentation from train data only. There is no sampling,
amodal target, model initialization or test-set selection in this module.
"""
from __future__ import annotations
import torch
from torch.nn import functional as F
from .uois_benchmark_geometry import INPUT_SIZE, letterbox_sample


def prepare_training_example(sample, target, *, horizontal_flip=False,
                             crop_box=None, empty_crop_policy='error'):
    """Return the four-key sample, independent visible targets, and geometry.

``crop_box`` is (left, top, right, bottom) in original pixels. Only the explicit
``drop`` policy permits a crop to remove an instance. A nonempty mask lost during
resizing always raises. Empty images are retained. Flip uses F@K (negative fx),
as course_depth_data does; model_depth_channels accepts that K and reflects X.
    """
    if empty_crop_policy not in {'error', 'drop'} or type(horizontal_flip) is not bool:
        raise ValueError('Specify crop handling and a boolean horizontal flip')
    image = sample['image']
    height, width = image.shape[-2:]
    masks, labels = target['masks'], target['labels']
    if (masks.ndim != 3 or tuple(masks.shape[1:]) != (height, width)
            or masks.dtype not in (torch.uint8, torch.bool)
            or labels.shape != (len(masks),) or labels.dtype != torch.int64
            or not torch.all(labels == 1) or torch.any(masks > 1)):
        raise ValueError('Expected independent visible binary masks and class-one targets')
    if len(masks) and not masks.flatten(1).any(1).all():
        raise ValueError('Raw visible targets must already exclude empty instances')
    left, top, right, bottom = (0, 0, width, height) if crop_box is None else crop_box
    if (any(type(v) is not int for v in (left, top, right, bottom))
            or not 0 <= left < right <= width or not 0 <= top < bottom <= height):
        raise ValueError('Crop must be a nonempty integer box inside the original image')
    original_k = sample['camera_K']
    crop_affine = original_k.new_tensor([[1, 0, -left], [0, 1, -top], [0, 0, 1]])
    cropped = {key: value[..., top:bottom, left:right] for key, value in sample.items()
               if key != 'camera_K'}
    cropped['camera_K'] = crop_affine @ original_k
    masks = masks[:, top:bottom, left:right].bool()
    keep = masks.flatten(1).any(1)
    if (~keep).any() and empty_crop_policy != 'drop':
        raise ValueError('Crop removed a visible instance; explicit drop policy required')
    kept_indices = torch.where(keep)[0]
    dropped_indices = torch.where(~keep)[0]
    masks = masks[keep]
    mapped, geometry = letterbox_sample(cropped)
    x0, y0, x1, y1 = geometry['content_box']
    transformed = torch.zeros((len(masks), INPUT_SIZE, INPUT_SIZE),
                              dtype=torch.bool, device=masks.device)
    if len(masks):
        resized = F.interpolate(masks[:, None].float(), (y1-y0, x1-x0), mode='nearest-exact')[:, 0].bool()
        if not resized.flatten(1).any(1).all():
            raise ValueError('Resize removed a nonempty small visible instance')
        transformed[:, y0:y1, x0:x1] = resized
    affine = original_k.new_tensor(geometry['affine']) @ crop_affine
    if horizontal_flip:
        for key in ('image', 'depth_m', 'depth_valid'):
            mapped[key] = mapped[key].flip(-1)
        transformed = transformed.flip(-1)
        flip = original_k.new_tensor([[-1, 0, INPUT_SIZE-1], [0, 1, 0], [0, 0, 1]])
        mapped['camera_K'] = flip @ mapped['camera_K']
        affine = flip @ affine
    result = {}
    for key, value in target.items():
        if key in {'masks', 'boxes', 'area'}:
            continue
        if key != 'image_id' and isinstance(value, torch.Tensor) and value.ndim and len(value) == len(labels):
            result[key] = value[kept_indices].clone()
        else:
            result[key] = value.clone() if isinstance(value, torch.Tensor) else value
    boxes = []
    for mask in transformed:
        ys, xs = torch.where(mask)
        boxes.append(torch.stack((xs.min(), ys.min(), xs.max()+1, ys.max()+1)))
    result.update(masks=transformed.to(torch.uint8),
                  area=transformed.flatten(1).sum(1).float(),
                  boxes=(torch.stack(boxes).float() if boxes else
                         torch.zeros((0, 4), device=masks.device, dtype=torch.float32)))
    annotation_ids = target.get('annotation_ids')
    meta = {'original_size': [height, width], 'input_size': INPUT_SIZE,
            'crop_box': [left, top, right, bottom], 'horizontal_flip': horizontal_flip,
            'pixel_affine': affine.cpu().tolist(),
            'dropped_by_crop_indices': dropped_indices.tolist(),
            'dropped_by_crop_annotation_ids': [] if annotation_ids is None else annotation_ids[dropped_indices].tolist(),
            'empty_image_retained': len(transformed) == 0,
            'target_policy': 'independent visible masks; no amodal target or minimum-area filter'}
    return mapped, result, meta


def collate_training_examples(examples):
    if not examples or any(len(example) != 3 for example in examples):
        raise ValueError('Expected nonempty prepared (sample, target, metadata) examples')
    samples, targets, metadata = zip(*examples)
    return list(samples), list(targets), list(metadata)
