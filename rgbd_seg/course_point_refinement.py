"""Trainable PointRend-inspired residual point heads for a frozen DA-Fusion.

The RGB-D backbone and Mask2Former remain untouched. Training labels are used
only for matching and point supervision. Inference accepts RGB/depth/valid/K,
uses predicted mask support to select regions, and preserves original scores.
"""
from __future__ import annotations

from dataclasses import replace
import math
import time

import torch
from torch import nn
from torch.nn import functional as F

from .course_tiled_probability import candidates_from_logits, encode_mask


POLICY = {
    'family': 'PointRend-inspired residual point refinement; not an exact paper reproduction',
    'fine_features': 'stride4 fused Swin features (96) + pixel decoder features (256)',
    'hidden_layers': 3, 'hidden_channels': 256,
    'instance_conditioning': 'original per-query coarse mask logit, concatenated at every hidden layer',
    'output': 'residual logit; zero-initialized final layer preserves baseline',
    'depth_channels': 9, 'depth_difference_scale_m': .001,
    'depth_features': 'center validity + tanh of 4 signed neighbor differences + 4 pair-validity bits',
    'training_points_per_instance': 784, 'oversample_ratio': 3,
    'uncertain_fraction': .75, 'train_roi': 'GT bounding box plus 8 pixels; training supervision only',
    'inference_roi': 'original predicted foreground bounding box plus 8 pixels',
    'inference_points_per_stage_per_instance': 784,
    'inference_stages': '512 then 1024 for a 256 coarse map; carry interpolated residuals',
    'roi_padding_px': 8, 'loss': 'mean point BCE per instance',
    'scores': 'original baseline class-times-mask-quality score, unchanged',
    'empty_result': 'original mask', 'max_detections': 100,
    'base_precision': 'float32', 'head_precision': 'float32',
}


def point_sample(features, coords):
    """Bilinear point sampling in normalized image coordinates [0,1]."""
    if features.ndim != 4 or coords.ndim != 3 or coords.shape[-1] != 2:
        raise ValueError('Expected NCHW features and NP2 coordinates')
    return F.grid_sample(features, coords.mul(2).sub(1).unsqueeze(2),
                         mode='bilinear', padding_mode='zeros', align_corners=False).squeeze(-1)


def sample_shared(features, coords):
    """Sample one image's features for many instances without replicating a map."""
    if features.shape[0] != 1:
        raise ValueError('Shared features must contain one image')
    n, p, _ = coords.shape
    if not n:
        return features.new_empty((0, features.shape[1], p))
    sampled = point_sample(features, coords.reshape(1, n*p, 2))
    return sampled.reshape(features.shape[1], n, p).permute(1, 0, 2).contiguous()


def depth_cues(depth, valid):
    """Metric, bounded local cues; invalid values cannot enter differences."""
    if depth.ndim == 3:
        depth, valid = depth[0], valid[0]
    if depth.ndim != 2 or valid.shape != depth.shape or valid.dtype != torch.bool:
        raise ValueError('Expected aligned depth and boolean validity')
    if not torch.isfinite(depth[valid]).all() or (depth[valid] <= 0).any():
        raise ValueError('Valid depth must be finite positive metres')
    safe = torch.where(valid, depth.float(), 0.)
    h, w = safe.shape
    differences, flags = [], []
    for dy, dx in ((0,-1), (0,1), (-1,0), (1,0)):
        neighbor, usable = torch.zeros_like(safe), torch.zeros_like(valid)
        y0, y1, x0, x1 = max(0,-dy), min(h,h-dy), max(0,-dx), min(w,w-dx)
        if y0 < y1 and x0 < x1:
            neighbor[y0:y1,x0:x1] = safe[y0+dy:y1+dy,x0+dx:x1+dx]
            usable[y0:y1,x0:x1] = valid[y0:y1,x0:x1] & valid[y0+dy:y1+dy,x0+dx:x1+dx]
        differences.append(torch.where(usable, torch.tanh((neighbor-safe)/.001), 0.))
        flags.append(usable.float())
    return torch.stack([valid.float(), *differences, *flags])[None]


def mask_boxes(masks, padding_px=8):
    """Normalized support boxes; an empty mask receives the full image."""
    if masks.ndim != 3 or masks.dtype != torch.bool:
        raise ValueError('Expected boolean instance masks')
    n, h, w = masks.shape
    if not n:
        return torch.empty((0,4), device=masks.device)
    columns, rows = masks.any(1), masks.any(2)
    x = torch.arange(w, device=masks.device)[None]
    y = torch.arange(h, device=masks.device)[None]
    left = torch.where(columns, x, w).amin(1)
    right = torch.where(columns, x+1, 0).amax(1)
    top = torch.where(rows, y, h).amin(1)
    bottom = torch.where(rows, y+1, 0).amax(1)
    empty = right == 0
    left = torch.where(empty, 0, (left-padding_px).clamp(min=0))
    top = torch.where(empty, 0, (top-padding_px).clamp(min=0))
    right = torch.where(empty, w, (right+padding_px).clamp(max=w))
    bottom = torch.where(empty, h, (bottom+padding_px).clamp(max=h))
    return torch.stack((left/w, top/h, right/w, bottom/h), dim=-1).float()


@torch.no_grad()
def training_points(coarse, labels, *, points=784, oversample=3, importance=.75):
    """PointRend biased sampling within a supervised training region."""
    boxes = mask_boxes(labels > .5)
    n = len(coarse)
    random = torch.rand(n, points*oversample, 2, device=coarse.device)
    coords = boxes[:,None,:2] + random*(boxes[:,None,2:]-boxes[:,None,:2])
    uncertainty = -point_sample(coarse[:,None], coords)[:,0].abs()
    important = int(points*importance)
    selected = uncertainty.topk(important, dim=1).indices
    selected = coords.gather(1, selected[:,:,None].expand(-1,-1,2))
    random = torch.rand(n, points-important, 2, device=coarse.device)
    random = boxes[:,None,:2] + random*(boxes[:,None,2:]-boxes[:,None,:2])
    return torch.cat((selected, random), dim=1)


class PointHead(nn.Module):
    """Shared 3-layer MLP, implemented as 1x1 Conv1d over selected points."""
    def __init__(self, fine_channels=352, depth_channels=0, hidden=256):
        super().__init__()
        self.depth_channels = depth_channels
        channels = fine_channels + depth_channels + 1
        self.layers = nn.ModuleList()
        for _ in range(3):
            self.layers.append(nn.Conv1d(channels, hidden, 1))
            channels = hidden + 1
        self.predictor = nn.Conv1d(channels, 1, 1)
        nn.init.zeros_(self.predictor.weight)
        nn.init.zeros_(self.predictor.bias)

    def forward(self, fine, coarse, depth=None):
        if self.depth_channels:
            if depth is None or depth.shape[1] != self.depth_channels:
                raise ValueError('Depth-conditioned head requires local depth cues')
            fine = torch.cat((fine, depth), dim=1)
        x = torch.cat((fine, coarse), dim=1)
        for layer in self.layers:
            x = torch.cat((F.relu(layer(x)), coarse), dim=1)
        return self.predictor(x)


class PairedPointHeads(nn.Module):
    """Both arms share frozen features, matches, coordinates and target points."""
    def __init__(self):
        super().__init__()
        self.plain = PointHead()
        self.depth = PointHead(depth_channels=9)
        # Align shared initialization; added depth weights start at zero.
        with torch.no_grad():
            self.depth.layers[0].weight.zero_()
            self.depth.layers[0].weight[:,:352].copy_(self.plain.layers[0].weight[:,:352])
            self.depth.layers[0].weight[:,-1:].copy_(self.plain.layers[0].weight[:,-1:])
            self.depth.layers[0].bias.copy_(self.plain.layers[0].bias)
            for plain, depth in zip(self.plain.layers[1:], self.depth.layers[1:]):
                depth.load_state_dict(plain.state_dict())

    def forward(self, fine, coarse, depth):
        return {'point': self.plain(fine, coarse),
                'point_depth': self.depth(fine, coarse, depth)}


@torch.no_grad()
def frozen_features(adapter, samples, targets=None):
    """Extract the actual frozen model's finest fused features and mask logits."""
    if adapter.training or any(p.requires_grad for p in adapter.parameters()):
        raise ValueError('The base model must be frozen and in eval mode')
    batch, sizes = adapter.prepare_batch(samples, targets)
    labels = batch.pop('mask_labels', None)
    classes = batch.pop('class_labels', None)
    outputs = adapter.model(**batch, return_dict=True, output_hidden_states=True,
                            output_auxiliary_logits=False)
    fine = torch.cat((outputs.encoder_hidden_states[0],
                      outputs.pixel_decoder_last_hidden_state), dim=1).detach()
    if fine.shape[1] != 352 or fine.shape[-2:] != outputs.masks_queries_logits.shape[-2:]:
        raise ValueError('Unexpected feature layout')
    return outputs, fine, sizes, labels, classes


def training_batch(adapter, samples, targets, matcher):
    """Build a detached point minibatch; only returned point heads get gradients."""
    outputs, features, sizes, labels, classes = frozen_features(adapter, samples, targets)
    with torch.no_grad():
        matches = matcher(outputs.masks_queries_logits, outputs.class_queries_logits, labels, classes)
        fine_parts, coarse_parts, depth_parts, label_parts = [], [], [], []
        counts = []
        for b, (prediction_indices, target_indices) in enumerate(matches):
            prediction_indices, target_indices = prediction_indices.to(adapter.device), target_indices.to(adapter.device)
            count = len(prediction_indices)
            counts.append(count)
            if not count:
                continue
            coarse = outputs.masks_queries_logits[b, prediction_indices].float()
            truth = labels[b][target_indices].float()
            coords = training_points(coarse, truth)
            fine_parts.append(sample_shared(features[b:b+1], coords).float())
            coarse_parts.append(point_sample(coarse[:,None], coords).float())
            label_parts.append(point_sample(truth[:,None], coords).float())
            sample = samples[b]
            cues = depth_cues(sample['depth_m'].to(adapter.device), sample['depth_valid'].to(adapter.device))
            # Course whole-image training is exactly 1024; no padded coordinate ambiguity.
            if tuple(sample['image'].shape[-2:]) != sizes['padded_size']:
                raise ValueError('This experiment requires original 1024 whole-image training')
            depth_parts.append(sample_shared(cues, coords))
        if not fine_parts:
            return None, counts
        values = tuple(torch.cat(items).detach() for items in (fine_parts, coarse_parts, depth_parts, label_parts))
        if any(not torch.isfinite(value).all() for value in values):
            raise ValueError('Non-finite training features')
        return values, counts


@torch.no_grad()
def _uncertain_grid(logits, boxes, points):
    n, h, w = logits.shape
    y = (torch.arange(h, device=logits.device)+.5)/h
    x = (torch.arange(w, device=logits.device)+.5)/w
    usable = ((x[None,None,:] >= boxes[:,0,None,None]) &
              (x[None,None,:] < boxes[:,2,None,None]) &
              (y[None,:,None] >= boxes[:,1,None,None]) &
              (y[None,:,None] < boxes[:,3,None,None]))
    uncertainty = -logits.abs()
    uncertainty = uncertainty.masked_fill(~usable, -torch.inf).flatten(1)
    indices = uncertainty.topk(min(points, h*w), dim=1).indices
    active = uncertainty.gather(1, indices).isfinite()
    coords = torch.stack(((indices % w+.5)/w, (indices//w+.5)/h), dim=-1)
    return indices, coords, active


@torch.no_grad()
def refine_logits(head, coarse, fine_features, cues, boxes, *, output_size=(1024,1024),
                  points=784, chunk_size=8):
    """Two-stage sparse subdivision with an exact baseline at zero residual."""
    if output_size != (1024,1024) or tuple(coarse.shape[-2:]) != (256,256):
        raise ValueError('Frozen experiment supports 256 -> 512 -> 1024 only')
    output = coarse.new_empty((len(coarse), *output_size), dtype=torch.float32)
    for start in range(0, len(coarse), chunk_size):
        stop = min(start+chunk_size, len(coarse))
        original = coarse[start:stop,None].float()
        residual = torch.zeros_like(original)
        for size in ((512,512), output_size):
            residual = F.interpolate(residual, size=size, mode='bilinear', align_corners=False)
            base = F.interpolate(original, size=size, mode='bilinear', align_corners=False)
            indices, coords, active = _uncertain_grid((base+residual)[:,0], boxes[start:stop], points)
            fine = sample_shared(fine_features, coords)
            coarse_points = point_sample(original, coords)
            depth = sample_shared(cues, coords) if head.depth_channels else None
            delta = head(fine, coarse_points, depth)[:,0]
            if not torch.isfinite(delta).all():
                raise ValueError('Non-finite point prediction')
            old = residual[:,0].flatten(1).gather(1, indices)
            residual[:,0].flatten(1).scatter_(1, indices, torch.where(active, delta, old))
        output[start:stop] = (base+residual)[:,0]
    return output


@torch.no_grad()
def predict_paired(adapter, heads, sample, *, bounds=None, full_size=None, view_id=0,
                   head_names=('point','point_depth'), return_timing=False):
    """Whole-image or aligned square crop predictions, without GT arguments."""
    original_size = tuple(sample['image'].shape[-2:])
    if original_size not in ((1024,1024),(640,640)):
        raise ValueError('Frozen experiment supports 1024 images and 640 square views')
    if any(name not in ('point','point_depth') for name in head_names):
        raise ValueError('Unknown point head')
    full_size = original_size if full_size is None else full_size
    bounds = (0,0,original_size[1],original_size[0]) if bounds is None else bounds
    synchronize = (lambda: torch.cuda.synchronize(adapter.device)) if adapter.device.type == 'cuda' else lambda: None
    synchronize()
    began = time.perf_counter()
    outputs, fine, sizes, _, _ = frozen_features(adapter, [sample])
    if sizes['padded_size'] != (1024,1024):
        raise ValueError('Point subdivision expects model input 1024')
    candidates = candidates_from_logits(outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
        padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
        original_size=sizes['original_sizes'][0], bounds=bounds, full_size=full_size, view_id=view_id)
    synchronize()
    timing = {'native_seconds':time.perf_counter()-began}
    results = {'native': candidates}
    if not candidates:
        results.update({name:[] for name in head_names})
        timing.update({f'{name}_extra_seconds':0. for name in head_names})
        return (results,timing) if return_timing else results
    began = time.perf_counter()
    indices = torch.tensor([c.query_id for c in candidates], device=adapter.device)
    coarse = outputs.masks_queries_logits[0,indices].float()
    raw = torch.stack([c.probability > .5 for c in candidates])
    boxes = mask_boxes(raw)
    synchronize()
    timing['roi_preparation_seconds'] = time.perf_counter()-began
    began = time.perf_counter()
    cues = (depth_cues(sample['depth_m'].to(adapter.device), sample['depth_valid'].to(adapter.device))
            if 'point_depth' in head_names else None)
    synchronize()
    timing['depth_cue_seconds'] = time.perf_counter()-began
    for name in head_names:
        synchronize()
        began = time.perf_counter()
        head = heads.plain if name == 'point' else heads.depth
        logits = refine_logits(head, coarse, fine, cues, boxes)
        if original_size != (1024,1024):
            logits = F.interpolate(logits[None],size=original_size,mode='bilinear',align_corners=False)[0]
        binary = logits > 0
        empty = ~binary.flatten(1).any(1)
        binary[empty] = raw[empty]
        # An empty fallback also restores the original soft map for later filtering.
        probabilities = logits.sigmoid()
        for j, original in enumerate(candidates):
            if empty[j]:
                probabilities[j] = original.probability
        values = binary.cpu().numpy()
        results[name] = []
        for j,c in enumerate(candidates):
            if original_size == full_size and bounds == (0,0,full_size[1],full_size[0]):
                binary = values[j]
            else:
                import numpy as np
                binary = np.zeros(full_size,dtype=bool)
                left,top,right,bottom = bounds
                binary[top:bottom,left:right] = values[j]
            results[name].append(replace(c,probability=probabilities[j].clone(),segmentation=encode_mask(binary)))
        synchronize()
        timing[f'{name}_extra_seconds'] = time.perf_counter()-began
    return (results,timing) if return_timing else results
