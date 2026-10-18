"""Isolated trainable paper adaptations; legacy model loading is unchanged."""
from __future__ import annotations

import copy
import threading

import torch
from torchvision.models.detection import roi_heads as tv_roi_heads

from rgbd_seg.model import RGBDMaskRCNN, build_model
from rgbd_seg.pretrained import initialize_rgb

VARIANTS = ('baseline', 'bce2', 'bace', 'freqfusion')
_LOSS_LOCK = threading.RLock()


def variant_config(variant):
    if variant not in VARIANTS:
        raise ValueError(f'Unknown paper variant: {variant}')
    if variant == 'freqfusion':
        from rgbd_seg.freqfusion_fpn import FORWARD_CONFIG
        return copy.deepcopy(FORWARD_CONFIG)
    if variant == 'bace':
        return {'roi_scale': 4, 'mask_bce_weight': 1.0, 'aux_weight': 1.0,
                'refined_logit': 'block_max(target) + logit - block_max(logit)',
                'inference': 'unchanged original mask logits; no GT at inference'}
    if variant == 'bce2':
        return {'mask_bce_weight': 2.0, 'purpose': 'mask-loss magnitude control'}
    return {'mask_bce_weight': 1.0}


class PaperModel(RGBDMaskRCNN):
    """Retain core.* parameter names and unchanged inference for loss variants.

    BACE temporarily wraps the process-local Torchvision mask loss only while
    this training forward runs. V5 training uses one model in one thread/process;
    it must not share a process with uncoordinated external Torchvision training.
    The original function is restored even after errors. No Torchvision files
    are edited. A separate demo process is unaffected.
    """
    def __init__(self, core, config, base_config, variant, rgb_parameter_ids):
        super().__init__(core, config)
        self.base_config = dict(base_config)
        self.variant = variant
        self._original_rgb_parameter_ids = set(rgb_parameter_ids)

    def forward(self, samples, targets=None):
        if not self.training:
            return super().forward(samples, targets)
        if self.variant != 'bace':
            losses = super().forward(samples, targets)
            if self.variant == 'bce2':
                losses['loss_mask_aux'] = losses['loss_mask']
            return losses
        from rgbd_seg.boundary_loss import bace_components
        captured = []
        with _LOSS_LOCK:
            original = tv_roi_heads.maskrcnn_loss

            def augmented(mask_logits, proposals, gt_masks, gt_labels, matched_idxs):
                base = original(mask_logits, proposals, gt_masks, gt_labels, matched_idxs)
                labels = torch.cat([labels[index] for labels, index in zip(gt_labels, matched_idxs)])
                projected = torch.cat([
                    tv_roi_heads.project_masks_on_boxes(masks, boxes, index, mask_logits.shape[-1])
                    for masks, boxes, index in zip(gt_masks, proposals, matched_idxs)
                ])
                selected = mask_logits[torch.arange(len(labels), device=labels.device), labels]
                components = bace_components(selected, projected, scale=4)
                captured.append(components['aux'])
                return base

            tv_roi_heads.maskrcnn_loss = augmented
            try:
                losses = super().forward(samples, targets)
            finally:
                tv_roi_heads.maskrcnn_loss = original
        if len(captured) != 1:
            raise RuntimeError('Expected exactly one ROI mask-loss invocation')
        losses['loss_mask_aux'] = captured[0]
        return losses


def build_v5_model(base_config, variant, init_rgb=None):
    """Initialize shared weights first; extensions do not alter subsequent RNG streams."""
    base_config = dict(base_config)
    if any(key.startswith('paper_') for key in base_config):
        raise ValueError('base_config must not contain paper-variant fields')
    if base_config.get('mode') != 'E2' or base_config.get('arch') != 'r50_v2':
        raise ValueError('V5 is scoped to E2 with the existing r50_v2 architecture')
    settings = variant_config(variant)
    base = build_model(base_config)
    info = initialize_rgb(base, init_rgb, foreground_mean=False) if init_rgb is not None else None
    original_rgb_ids = {id(p) for p in base.core.backbone.rgb.parameters()}
    if variant == 'freqfusion':
        from rgbd_seg.freqfusion_fpn import FreqFusionFPN
        # Do not use torch.manual_seed here: it also resets CUDA generators.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(base.config['seed'] + 2000)
            base.core.backbone.rgb.fpn = FreqFusionFPN(base.core.backbone.rgb.fpn)
    config = {**base.config, 'paper_variant': variant, 'paper_variant_config': settings}
    model = PaperModel(base.core, config, base.config, variant, original_rgb_ids)
    return model, info


def optimizer_groups(model, lr=0.0002, backbone_factor=0.1):
    """Reuse original RGB parameter identities; newly added fusion uses head LR."""
    if lr <= 0 or backbone_factor <= 0:
        raise ValueError('Learning rates must be positive')
    buckets = {'pretrained_rgb': [], 'heads_depth_new_fusion': []}
    names = {key: [] for key in buckets}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        group = 'pretrained_rgb' if id(parameter) in model._original_rgb_parameter_ids else 'heads_depth_new_fusion'
        buckets[group].append(parameter)
        names[group].append(name)
    groups, metadata = [], []
    for key, parameters in buckets.items():
        rate = lr * backbone_factor if key == 'pretrained_rgb' else lr
        groups.append({'params': parameters, 'lr': rate})
        metadata.append({'name': key, 'lr': rate, 'parameter_names': names[key],
                         'numel': sum(p.numel() for p in parameters)})
    ids = [id(p) for group in groups for p in group['params']]
    if len(ids) != len(set(ids)):
        raise RuntimeError('Optimizer contains duplicate parameter objects')
    return groups, metadata


def load_v5_model(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint.get('format') != 'rgbd_v5_v1':
        raise ValueError('Expected an isolated rgbd_v5_v1 checkpoint')
    model, _ = build_v5_model(checkpoint['base_config'], checkpoint['variant'])
    if model.config != checkpoint['config']:
        raise ValueError('Checkpoint variant configuration differs from current implementation')
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    return model.to(device).eval(), checkpoint
