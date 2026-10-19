"""RGB-only control for the fixed 1024px, 5+10 epoch progressive ablation."""
from __future__ import annotations

import copy
import math
from pathlib import Path
import types

import torch

from .course_finetune_mask2former import (
    CourseTrainableMask2Former, build_trainable as build_rgb, component_predictions,
)
from .course_resolution import local_num_masks
from .course_optimized import file_sha


class RGBControl(CourseTrainableMask2Former):
    """Use only image tensors, while retaining the shared dataset interface."""

    def prepare_batch(self, samples, targets=None):
        if not isinstance(samples, (tuple, list)) or not samples:
            raise ValueError('Expected a nonempty list of inference samples')
        required = {'image', 'depth_m', 'depth_valid', 'camera_K'}
        if any(not isinstance(s, dict) or set(s) != required for s in samples):
            raise ValueError('Inference inputs must not contain annotation fields')
        return super().prepare_batch([s['image'] for s in samples], targets)

    @torch.inference_mode()
    def predict_sample(self, sample, *, score_threshold=.01, max_detections=100):
        if self.training:
            raise ValueError('Prediction requires eval mode')
        batch, sizes = self.prepare_batch([sample])
        outputs = self.model(**batch, return_dict=True)
        values = component_predictions(
            outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
            original_size=sizes['original_sizes'][0], score_threshold=score_threshold,
            max_detections=max_detections)
        return [{k: v[k] for k in ('category_id', 'score', 'segmentation')} for v in values]


def build_trainable(checkpoint, *, device='cpu', seed=17, resolution=1024):
    if resolution != 1024:
        raise ValueError('The progressive experiment fixes 1024px input')
    original, metadata = build_rgb(checkpoint, device='cpu', seed=seed)
    metadata = copy.deepcopy(metadata)
    metadata.update(model_id='Mask2Former-Swin-Tiny-RGB-progressive-control-1024',
                    experiment_family='progressive_ablation_v1', depth_used=False,
                    control_adapter_sha256=file_sha(Path(__file__)))
    metadata['preprocess'].update(shortest_edge=1024,
        hf_padding_limitation='Course square input is 1024 square, with no padding.')
    metadata['loss']['normalization'] = 'Native local microbatch mask count; image-weighted DDP accumulation'
    adapter = RGBControl(original.model, mean=metadata['preprocess']['mean'],
        std=metadata['preprocess']['std'], short_edge=1024, max_size=1333,
        metadata=metadata).to(device).train()
    adapter.model.criterion.get_num_masks = types.MethodType(local_num_masks, adapter.model.criterion)
    return adapter, metadata


def check_parent_checkpoint(checkpoint, parent_protocol, parent_sha, metadata, expected_epoch):
    """Continue the completed final epoch, whether or not it was the best epoch.

    The main ablation fixes epoch15 in both RGB and RGB-D arms. Requiring the
    stage-one final epoch to be its validation winner would change that budget.
    """
    if (checkpoint['protocol_sha256'] != parent_sha
            or checkpoint['source_sha256'] != parent_protocol['source_sha256']
            or checkpoint['model_metadata'] != metadata):
        raise ValueError('Parent provenance or model construction differs')
    state = checkpoint['state']
    steps = math.ceil(14432 / parent_protocol['training']['effective_batch_size']) * expected_epoch
    if (state['epoch'] != expected_epoch or state['images_in_epoch'] != 0
            or state['stage'] != 'training' or state['global_step'] != steps
            or len(state['val_results']) != expected_epoch):
        raise ValueError('Parent must contain all completed training and validation epochs')
    if [v['epoch'] for v in state['val_results']] != list(range(1, expected_epoch + 1)):
        raise ValueError('Parent validation epoch coverage differs')
    return state['val_results'][-1]
