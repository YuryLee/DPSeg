"""Mask BCE -> sigmoid focal only; keep matching, point sampling and Dice."""
from __future__ import annotations

import copy

import torch
from torch.nn import functional as F
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss, sample_point, dice_loss, sigmoid_cross_entropy_loss,
)

from .course_resolution import local_num_masks


FOCAL_POLICY = {
    'name': 'mask_focal_gamma2_no_alpha_v1', 'gamma': 2., 'alpha': None,
    'scope': 'Matched mask loss, including all auxiliary layers',
    'outer_weights': 'Unchanged classification/BCE-replacement/Dice = 2/5/5',
    'matching': 'Native BCE/Dice/class Hungarian cost unchanged',
    'sampling': 'Native uncertainty and random points unchanged; no GT quota',
    'soft_labels': 'Keep bilinearly sampled targets; p_t = sigmoid(x)*y+(1-sigmoid(x))*(1-y)',
    'normalization': 'Mean over sampled points, sum matched masks, divide by native local target count',
    'interpretation': 'Intentional objective and gradient-scale change; no empirical loss rescaling',
    'inference': 'Identical architecture and postprocessing; no labels enter inference',
}


def focal_mask_loss(logits, labels, num_masks, *, gamma=2.):
    if gamma < 0:
        raise ValueError('Focusing parameter must be nonnegative')
    if gamma == 0:
        return sigmoid_cross_entropy_loss(logits, labels, num_masks)
    cross_entropy = F.binary_cross_entropy_with_logits(logits, labels, reduction='none')
    probabilities = logits.sigmoid()
    p_t = probabilities*labels + (1-probabilities)*(1-labels)
    return (cross_entropy*(1-p_t).pow(gamma)).mean(1).sum()/num_masks


class MaskFocalLoss(Mask2FormerLoss):
    get_num_masks = local_num_masks

    def __init__(self, config, weight_dict, *, gamma=2.):
        super().__init__(config, weight_dict)
        self.gamma = gamma
        self.mask_loss_calls = 0

    def loss_masks(self, masks_queries_logits, mask_labels, indices, num_masks):
        self.mask_loss_calls += 1
        source = self._get_predictions_permutation_indices(indices)
        target = self._get_targets_permutation_indices(indices)
        predictions = masks_queries_logits[source][:, None]
        labels, _ = self._pad_images_to_max_in_batch(mask_labels)
        labels = labels[target][:, None]
        if not len(predictions):
            zero = predictions.sum()*0
            return {'loss_mask': zero, 'loss_dice': zero}
        with torch.no_grad():
            coordinates = self.sample_points_using_uncertainty(predictions,
                self.calculate_uncertainty, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            point_labels = sample_point(labels, coordinates, align_corners=False).squeeze(1)
        point_logits = sample_point(predictions, coordinates, align_corners=False).squeeze(1)
        return {'loss_mask': focal_mask_loss(point_logits, point_labels, num_masks, gamma=self.gamma),
                'loss_dice': dice_loss(point_logits, point_labels, num_masks)}


def apply_focal(adapter, metadata):
    """Replace only the criterion; tensor names, values and parameter count stay."""
    criterion = MaskFocalLoss(adapter.model.config, adapter.model.criterion.weight_dict,
                              gamma=FOCAL_POLICY['gamma']).to(adapter.device)
    criterion.load_state_dict(adapter.model.criterion.state_dict(), strict=True)
    adapter.model.criterion = criterion
    metadata = copy.deepcopy(metadata)
    metadata['model_id'] += '+mask-focal-v1'
    metadata['experiment_family'] = 'small_target_independent_focal_v1'
    metadata['loss']['focal_policy'] = copy.deepcopy(FOCAL_POLICY)
    adapter.metadata = metadata
    return adapter, metadata
