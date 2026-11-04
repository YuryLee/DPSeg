"""Course depth-encoding and backbone experiments, separate from released models.

The no-intrinsics arm exposes only RGB, metric depth and validity to the model.
The Swin-L arm keeps the historical XYZ encoding and all four DS/DC stages.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import types

import numpy as np
from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from .course_dafusion import (CourseTrainableDAFusion, DualSwinFusionBackbone,
                             _load_upstream, upstream_source_sha256)
from .course_depth_data import (camera_intrinsics, load_depth_record,
                                model_depth_channels, transform_depth)
from .course_finetune_mask2former import (BACKBONE_PREFIX, CourseTrainableMask2Former,
                                         LossReportingMask2Former)
from .course_mask2former import _sha256
from .course_resolution import build_trainable as build_tiny, local_num_masks
from .course_training_data import CourseTrainingDataset

ENCODINGS = ('depth', 'xyz')


def depth_channels(sample, size, device=None):
    """Median/p95 absolute deviation of Z, repeated three times, plus validity.

    No pixel rays, focal lengths, principal point, inferred calibration or XYZ
    are constructed. Keeping four channels preserves the adapter parameter count.
    Normalization is computed per view over measured depths only, after resize.
    """
    if set(sample) != {'image', 'depth_m', 'depth_valid'}:
        raise ValueError('The depth arm accepts RGB/depth/validity only, without camera parameters')
    depth = sample['depth_m'].to(device=device, dtype=torch.float32)
    valid = sample['depth_valid'].to(device=depth.device)
    if (depth.shape != valid.shape or depth.ndim != 3 or depth.shape[0] != 1
            or valid.dtype != torch.bool or sample['image'].shape[-2:] != depth.shape[-2:]
            or not torch.isfinite(depth).all() or (depth[~valid] != 0).any()
            or (depth[valid] <= 0).any()):
        raise ValueError('Expected aligned finite metric depth and boolean validity')
    if len(size) != 2 or any(type(v) is not int or v < 1 for v in size):
        raise ValueError('Expected a positive integer output size')
    depth = F.interpolate(depth[None], size=size, mode='bilinear', align_corners=False)[0]
    valid = F.interpolate(valid[None].float(), size=size, mode='nearest')[0].bool()
    normalized = torch.zeros_like(depth)
    values = depth[valid]
    if values.numel():
        center = torch.quantile(values, .5)
        scale = torch.quantile((values-center).abs(), .95) + 1e-6
        normalized[valid] = (values-center)/scale
    return torch.cat((normalized.repeat(3, 1, 1), valid.float()), dim=0)


class CourseVariantDataset(CourseTrainingDataset):
    def __init__(self, archive, split='train', augment=True, seed=17, encoding='depth'):
        if encoding not in ENCODINGS:
            raise ValueError('Unknown depth encoding')
        self.encoding = encoding
        super().__init__(archive, split=split, augment=augment, seed=seed)

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        record = self.images[index]
        depth, valid = load_depth_record(self._handle(), record)
        if self.encoding == 'xyz':
            depth, valid, k = transform_depth(depth, valid, camera_intrinsics(record),
                                              horizontal_flip=self._flip_for(record['id']))
            return {'image': image, 'depth_m': depth, 'depth_valid': valid, 'camera_K': k}, target
        if self._flip_for(record['id']):
            depth, valid = depth.flip(-1), valid.flip(-1)
        return {'image': image, 'depth_m': depth, 'depth_valid': valid}, target

    def summary(self):
        return {**super().summary(), 'depth_encoding': self.encoding,
                'camera_intrinsics_read_by_input_loader': self.encoding == 'xyz',
                'model_sample_keys': ['image', 'depth_m', 'depth_valid']
                  + (['camera_K'] if self.encoding == 'xyz' else [])}


def load_sample(archive, image_id, encoding):
    if encoding not in ENCODINGS:
        raise ValueError('Unknown depth encoding')
    index = archive.image_ids.index(image_id)
    record = archive.images[index]
    image = torch.from_numpy(np.array(archive.load_rgb(index), dtype=np.uint8, copy=True))
    depth, valid = load_depth_record(archive._image_archive(), record)
    sample = {'image': image.permute(2, 0, 1).float().div_(255),
              'depth_m': depth, 'depth_valid': valid}
    if encoding == 'xyz':
        sample['camera_K'] = camera_intrinsics(record)
    return sample


class DepthBackboneAdapter(CourseTrainableDAFusion):
    def __init__(self, *args, encoding, **kwargs):
        if encoding not in ENCODINGS:
            raise ValueError('Unknown depth encoding')
        super().__init__(*args, **kwargs)
        self.depth_encoding = encoding

    def prepare_batch(self, samples, targets=None):
        required = {'image', 'depth_m', 'depth_valid'}
        if self.depth_encoding == 'xyz':
            required.add('camera_K')
        if not samples or any(set(s) != required for s in samples):
            raise ValueError(f'Inference sample keys must be exactly {sorted(required)}')
        batch, sizes = CourseTrainableMask2Former.prepare_batch(self, [s['image'] for s in samples], targets)
        pixels = batch['pixel_values']
        depths = pixels.new_zeros((len(samples), 4, *sizes['padded_size']))
        encoder = depth_channels if self.depth_encoding == 'depth' else model_depth_channels
        with torch.autocast(device_type=self.device.type, enabled=False):
            for i, (sample, size) in enumerate(zip(samples, sizes['resized_sizes'])):
                depths[i, :, :size[0], :size[1]] = encoder(sample, size, device=self.device)
        batch['pixel_values'] = torch.cat((pixels, depths), dim=1)
        return batch, sizes


class LargeDualSwin(DualSwinFusionBackbone):
    def __init__(self, rgb_encoder):
        nn.Module.__init__(self)
        FusionStage, DepthAdapter = _load_upstream()
        self.swin_rgb = rgb_encoder
        self.swin_d = copy.deepcopy(rgb_encoder)
        self.channels = list(rgb_encoder.channels)
        if self.channels != [192, 384, 768, 1536]:
            raise ValueError('Expected the four-stage Swin-Large backbone')
        self.depth_adapter = DepthAdapter(in_chans=4, out_chans=3, hidden=128, zero_init=True)
        self.fusions = nn.ModuleList([
            FusionStage(c, num_heads=8, n_groups=4, stride=s, range_factor=2.,
                        use_rpb=True, geom_prior=False)
            for c, s in zip(self.channels, (8, 4, 2, 1))])
        self.fusion_checkpointing = True
        for stage, fusion in enumerate(self.fusions):
            for module in fusion.modules():
                if module.__class__.__name__ == 'ContinuousPositionBias':
                    module.max_pairs = 0 if stage < 2 else 8_000_000


def build_trainable(checkpoint, *, encoding, backbone='tiny', device='cpu', seed=17,
                    resolution=1024, backbone_checkpointing=True):
    if resolution != 1024 or encoding not in ENCODINGS or backbone not in ('tiny', 'large'):
        raise ValueError('Expected declared depth encoding, Swin backbone and 1024 input')
    checkpoint = Path(checkpoint).resolve()
    if backbone == 'tiny':
        original, metadata = build_tiny(checkpoint, device='cpu', seed=seed, resolution=resolution)
        model = original.model
        metadata = copy.deepcopy(metadata)
    else:
        receipt = json.loads((checkpoint/'source_receipt.json').read_text())
        if (receipt.get('repository') != 'facebook/mask2former-swin-large-coco-instance'
                or _sha256(checkpoint/'model.safetensors') != receipt['weights_sha256']):
            raise ValueError('Swin-L COCO initialization identity differs')
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            model, loading = LossReportingMask2Former.from_pretrained(checkpoint,
                local_files_only=True, use_safetensors=True, torch_dtype=torch.float32,
                output_loading_info=True)
            if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
                raise ValueError(f'COCO Swin-L loading differs: {loading}')
            if model.config.num_labels != 80:
                raise ValueError('Expected 80-class COCO initialization')
            torch.default_generator.manual_seed(seed)
            classifier = nn.Linear(model.config.hidden_dim, 2)
            model._init_weights(classifier)
            model.class_predictor = classifier
            model.config.num_labels = 1
            model.config.id2label, model.config.label2id = {0: 'component'}, {'component': 0}
            model.criterion = Mask2FormerLoss(model.config, model.weight_dict)
            allowed = {'class_predictor.weight', 'class_predictor.bias', 'criterion.empty_weight'}
            unchanged = 0
            with safe_open(checkpoint/'model.safetensors', framework='pt', device='cpu') as source:
                state = model.state_dict()
                if set(state) != set(source.keys()):
                    raise ValueError('Unexpected pretrained state key mismatch')
                for name in source.keys():
                    if name not in allowed:
                        if not torch.equal(source.get_tensor(name), state[name]):
                            raise ValueError('Unexpected initialization change: '+name)
                        unchanged += 1
            encoder = LargeDualSwin(model.model.pixel_level_module.encoder)
            model.model.pixel_level_module.encoder = encoder
        processor = json.loads((checkpoint/'preprocessor_config.json').read_text())
        metadata = {
            'pretrained_sha256': receipt['weights_sha256'], 'pretrained_receipt': receipt,
            'upstream_source_sha256': upstream_source_sha256(),
            'initialization': {'seed': seed, 'unchanged_pretrained_tensors': unchanged,
                'both_backbones': 'Same COCO Swin-L tensors, independent trainable storage',
                'classifier': 'Fresh seeded two-output foreground/no-object classifier',
                'fusion': 'Fresh seeded upstream DS/DC at all four scales',
                'queries_and_decoder': 'Retain Swin-L COCO checkpoint query and decoder tensors'},
            'backbone_parameter_prefixes': [BACKBONE_PREFIX+'swin_rgb.', BACKBONE_PREFIX+'swin_d.'],
            'preprocess': {'mean': processor['image_mean'], 'std': processor['image_std'],
                           'shortest_edge': 1024, 'longest_edge': 1333},
            'fusion': {'channels': encoder.channels, 'heads': 8, 'offset_groups': 4,
                       'strides': [8, 4, 2, 1], 'rpb_stage_max_pairs': [0, 0, 8000000, 8000000],
                       'geometry_prior': False},
            'loss': {'weights': model.weight_dict, 'no_object_weight': model.config.no_object_weight,
                     'train_num_points': model.config.train_num_points,
                     'oversample_ratio': model.config.oversample_ratio,
                     'importance_sample_ratio': model.config.importance_sample_ratio,
                     'decoder_layers': model.config.decoder_layers,
                     'matching_and_loss_dtype': 'float32 native BCE/Dice/classification plus auxiliary losses'},
            'postprocess': {'score_threshold': .01, 'max_detections': 100, 'category_id': 1}}
    encoder = model.model.pixel_level_module.encoder
    if backbone_checkpointing:
        for branch in (encoder.swin_rgb, encoder.swin_d):
            branch.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.criterion.get_num_masks = types.MethodType(local_num_masks, model.criterion)
    metadata.update(model_id=f'DA-Fusion-course-{backbone}-{encoding}-1024-v1',
        experiment_family='course_depth_backbone_v1', depth_encoding=encoding,
        camera_intrinsics_used=encoding == 'xyz', backbone=backbone,
        backbone_checkpointing=bool(backbone_checkpointing),
        parameters=sum(p.numel() for p in model.parameters()))
    metadata['preprocess'].update(depth_encoding=encoding,
        depth_description=('Metric Z -> bilinear resize -> valid median/p95 absolute deviation -> repeated Z,Z,Z + validity'
                           if encoding == 'depth' else 'Historical calibrated XYZ + validity'),
        depth_augmentation='Same deterministic horizontal flip as RGB/targets')
    adapter = DepthBackboneAdapter(model, mean=metadata['preprocess']['mean'],
        std=metadata['preprocess']['std'], short_edge=1024, max_size=1333,
        metadata=metadata, encoding=encoding).to(device).train()
    return adapter, metadata
