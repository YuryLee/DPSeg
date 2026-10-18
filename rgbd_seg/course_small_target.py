"""Training-only small-instance point quotas for the existing DA-Fusion model.

Keep native Hungarian matching, losses, weights, point budget and inference.
For matched targets with ORIGINAL annotation area <= 1024, replace the last
256 random points with 128 foreground centres and 128 morphological-band
centres. This deliberately changes the sampled loss distribution, rather than
claiming an unbiased estimate of the original loss. No target enters inference.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import json
from pathlib import Path

import torch
from torch.nn import functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss, dice_loss, sample_point, sigmoid_cross_entropy_loss,
)

from .course_dafusion import CourseTrainableDAFusion, build_trainable as build_base
from .course_mask2former import DEFAULT_CHECKPOINT, _sha256

POLICY = {
    'name': 'small_target_quota_128fg_128band_v1',
    'small_original_area_max': 1024,
    'foreground_points': 128,
    'boundary_band_points': 128,
    'boundary_kernel': 3,
    'replacement': 'Last 256 points of native random tail; uncertain-point quota unchanged',
    'coordinates': 'Uniform with replacement over resized GT pixel centres; align_corners=False',
    'empty_after_resize': 'Native sampling fallback; instance and class retained',
    'empty_boundary_band': 'Foreground pool fallback',
    'scope': 'Matched-mask BCE and Dice, including all native auxiliary layers',
    'matching': 'Unchanged native shared random-point Hungarian matcher',
    'inference': 'Unchanged DA-Fusion model and postprocessing; no GT context',
    'loss_interpretation': 'Intentional GT-aware training sampling distribution, not importance-corrected',
}


class SmallTargetQuotaLoss(Mask2FormerLoss):
    def __init__(self, config, weight_dict):
        super().__init__(config, weight_dict)
        random_tail = self.num_points - int(self.importance_sample_ratio * self.num_points)
        if random_tail < POLICY['foreground_points'] + POLICY['boundary_band_points']:
            raise ValueError('Small-target quota must fit in the native random-point tail')
        self._target_context = None
        self.last_sampling_stats = None

    @contextmanager
    def bind_targets(self, masks, original_areas):
        """Cache pixel pools once per batch, shared by final and auxiliary losses."""
        if self._target_context is not None:
            raise RuntimeError('Nested target context is unsupported')
        if len(masks) != len(original_areas) or not masks:
            raise ValueError('Mask and original-area batches must align')
        h = max(int(m.shape[-2]) for m in masks)
        w = max(int(m.shape[-1]) for m in masks)
        pools, stats = {}, {'small_targets': 0, 'small_empty_after_resize': 0,
            'small_boundary_fallback': 0, 'matched_small_pairs_across_layers': 0,
            'foreground_points_forced': 0, 'boundary_points_forced': 0,
            'loss_mask_calls': 0, 'point_budget_per_pair': self.num_points}
        with torch.no_grad():
            for batch, (target_masks, areas) in enumerate(zip(masks, original_areas)):
                if areas.ndim != 1 or len(areas) != len(target_masks):
                    raise ValueError('Original areas must align with independent target masks')
                if not torch.isfinite(areas).all() or (areas <= 0).any():
                    raise ValueError('Original target areas must be finite and positive')
                for target in (areas <= POLICY['small_original_area_max']).nonzero().flatten().tolist():
                    stats['small_targets'] += 1
                    binary = target_masks[target]
                    fg = binary.nonzero(as_tuple=False)
                    if not len(fg):
                        stats['small_empty_after_resize'] += 1
                        continue
                    dilated = F.max_pool2d(binary[None, None], 3, 1, 1)[0, 0]
                    eroded = -F.max_pool2d(-binary[None, None], 3, 1, 1)[0, 0]
                    band = (dilated > eroded).nonzero(as_tuple=False)
                    if not len(band):
                        stats['small_boundary_fallback'] += 1
                        band = fg
                    def centres(yx):
                        xy = yx[:, [1, 0]].to(dtype=torch.float32)
                        return (xy + .5) / xy.new_tensor([w, h])
                    pools[(batch, target)] = (centres(fg), centres(band))
        self._target_context = (pools, stats)
        try:
            yield stats
        finally:
            self.last_sampling_stats = dict(stats)
            self._target_context = None

    def quota_coordinates(self, native_coordinates, indices):
        if self._target_context is None:
            raise RuntimeError('Loss requires a scoped training-target context')
        pools, stats = self._target_context
        if native_coordinates.shape[1:] != (self.num_points, 2):
            raise ValueError('Native coordinate budget differs')
        cursor = 0
        for batch, (_, targets) in enumerate(indices):
            for target in targets.tolist():
                pool = pools.get((batch, int(target)))
                if pool is not None:
                    foreground, band = pool
                    nf, nb = POLICY['foreground_points'], POLICY['boundary_band_points']
                    fg = foreground[torch.randint(len(foreground), (nf,), device=foreground.device)]
                    bd = band[torch.randint(len(band), (nb,), device=band.device)]
                    native_coordinates[cursor, -(nf+nb):-nb] = fg
                    native_coordinates[cursor, -nb:] = bd
                    stats['matched_small_pairs_across_layers'] += 1
                    stats['foreground_points_forced'] += nf
                    stats['boundary_points_forced'] += nb
                cursor += 1
        if cursor != len(native_coordinates):
            raise ValueError('Matching indices and sampled masks differ')
        return native_coordinates

    def loss_masks(self, masks_queries_logits, mask_labels, indices, num_masks):
        if self._target_context is None:
            raise RuntimeError('Training targets were not bound')
        self._target_context[1]['loss_mask_calls'] += 1
        if not self._target_context[0]:
            # Exact native behavior and RNG consumption when no usable small GT.
            return super().loss_masks(masks_queries_logits, mask_labels, indices, num_masks)
        src_idx = self._get_predictions_permutation_indices(indices)
        tgt_idx = self._get_targets_permutation_indices(indices)
        pred_masks = masks_queries_logits[src_idx][:, None]
        target_masks, _ = self._pad_images_to_max_in_batch(mask_labels)
        target_masks = target_masks[tgt_idx][:, None]
        if not len(pred_masks):
            zero = pred_masks.sum() * 0
            return {'loss_mask': zero, 'loss_dice': zero}
        with torch.no_grad():
            coordinates = self.sample_points_using_uncertainty(pred_masks,
                self.calculate_uncertainty, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            coordinates = self.quota_coordinates(coordinates, indices)
            labels = sample_point(target_masks, coordinates, align_corners=False).squeeze(1)
        logits = sample_point(pred_masks, coordinates, align_corners=False).squeeze(1)
        return {'loss_mask': sigmoid_cross_entropy_loss(logits, labels, num_masks),
                'loss_dice': dice_loss(logits, labels, num_masks)}


class CourseSmallTargetDAFusion(CourseTrainableDAFusion):
    """Same model/inference interface; targets scoped to training loss only."""
    sampling_audit_path = None

    def forward(self, samples, targets):
        if targets is None:
            raise ValueError('Training requires targets; inference uses predict_sample')
        batch, self.last_batch_metadata = self.prepare_batch(samples, targets)
        areas = [target['area'] for target in targets]
        self.model._captured_losses = None
        with self.model.criterion.bind_targets(batch['mask_labels'], areas):
            outputs = self.model(**batch, return_dict=True)
            losses = self.model._captured_losses
            self.model._captured_losses = None
        if not losses or outputs.loss is None:
            raise RuntimeError('Mask2Former losses were not captured')
        self.last_loss_total = outputs.loss.detach()
        stats = self.model.criterion.last_sampling_stats
        self.last_batch_metadata['small_target_sampling'] = stats
        if self.sampling_audit_path is not None:
            record = {'image_ids': [int(t['image_id'].item()) for t in targets], **stats}
            with Path(self.sampling_audit_path).open('a') as stream:
                stream.write(json.dumps(record, allow_nan=False)+'\n')
        return losses


def build_trainable(checkpoint_dir=DEFAULT_CHECKPOINT, device='cpu', *, seed=17):
    base, base_metadata = build_base(checkpoint_dir, device=device, seed=seed)
    criterion = SmallTargetQuotaLoss(base.model.config, base.model.criterion.weight_dict).to(device)
    criterion.load_state_dict(base.model.criterion.state_dict(), strict=True)
    base.model.criterion = criterion
    metadata = copy.deepcopy(base_metadata)
    metadata['model_id'] += '+small-target-quota-v1'
    metadata['small_target_policy'] = dict(POLICY)
    metadata['small_target_source_sha256'] = _sha256(__file__)
    metadata['loss']['small_target_sampling'] = dict(POLICY)
    metadata['initialization']['base_model_id'] = base_metadata['model_id']
    adapter = CourseSmallTargetDAFusion(base.model,
        mean=base.image_mean.flatten().tolist(), std=base.image_std.flatten().tolist(),
        short_edge=base.short_edge, max_size=base.max_size, metadata=metadata).to(device).train()
    if sum(p.numel() for p in adapter.model.parameters()) != metadata['parameters']:
        raise RuntimeError('Sampling optimization must not add model parameters')
    return adapter, metadata
