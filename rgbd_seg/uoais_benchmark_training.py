"""Shared UOAIS augmentation and adapters for native DA-Fusion and DPSeg.

This is a declared local training protocol, not a claim of exact paper reproduction.
No course-trained weights or real benchmark labels are used here.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .uoais_dpseg_training import prepare_training_example

ROOT = Path(__file__).resolve().parents[1]


def augment_visible(sample, target, *, scale=1., offset=(0, 0)):
    """Shared resize/crop to the original canvas; padding has no valid depth/GT.

    Offsets are (top, left) in the scaled image. Only instances that become empty
    under the declared training transform are dropped. Evaluation never uses it.
    """
    h, w = sample['image'].shape[-2:]
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    top, left = offset
    if (not .5 <= scale <= 2.5 or min(top, left) < 0
            or top > max(0, nh-h) or left > max(0, nw-w)):
        raise ValueError('Invalid shared training resize/crop')
    ch, cw = min(h, nh-top), min(w, nw-left)
    result = {}
    for key in ('image', 'depth_m', 'depth_valid'):
        tensor = sample[key]
        kwargs = {'mode': 'bilinear', 'align_corners': False} if key == 'image' else {'mode': 'nearest-exact'}
        resized = F.interpolate(tensor[None].float(), (nh, nw), **kwargs)[0]
        if key == 'image':
            # CPU bilinear arithmetic can put saturated white one float32 ULP
            # above one (observed on official train image 194 with one worker
            # thread). Restore the documented RGB range before model validation.
            resized.clamp_(0., 1.)
        canvas = tensor.new_zeros((tensor.shape[0], h, w))
        if key == 'image':
            canvas[:] = canvas.new_tensor([.485, .456, .406])[:, None, None]
        canvas[:, :ch, :cw] = resized[:, top:top+ch, left:left+cw].to(tensor.dtype)
        result[key] = canvas
    affine = sample['camera_K'].new_tensor([
        [nw/w, 0, (nw/w-1)/2-left], [0, nh/h, (nh/h-1)/2-top], [0, 0, 1]])
    result['camera_K'] = affine @ sample['camera_K']
    original = target['masks']
    masks = torch.zeros((len(original), h, w), dtype=torch.bool)
    if len(original):
        resized = F.interpolate(original[:, None].float(), (nh, nw), mode='nearest-exact')[:, 0].bool()
        masks[:, :ch, :cw] = resized[:, top:top+ch, left:left+cw]
    keep = masks.flatten(1).any(1)
    mapped = {key: (value[keep].clone() if key != 'image_id' and isinstance(value, torch.Tensor)
                    and value.ndim and len(value) == len(original) else copy.deepcopy(value))
              for key, value in target.items() if key not in ('masks', 'area', 'boxes')}
    masks = masks[keep]
    boxes = []
    for mask in masks:
        yy, xx = torch.where(mask)
        boxes.append(torch.stack((xx.min(), yy.min(), xx.max()+1, yy.max()+1)))
    mapped.update(masks=masks.to(torch.uint8), area=masks.flatten(1).sum(1).float(),
                  boxes=torch.stack(boxes).float() if boxes else torch.zeros(0, 4))
    return result, mapped, {'scale': scale, 'scaled_size': [nh, nw], 'offset': [top, left],
                            'dropped_instances': int((~keep).sum())}


def flip_native(sample, target):
    width = sample['image'].shape[-1]
    result = {key: value.flip(-1) for key, value in sample.items() if key != 'camera_K'}
    result['camera_K'] = sample['camera_K'].new_tensor([[-1, 0, width-1], [0, 1, 0], [0, 0, 1]]) @ sample['camera_K']
    target = {**target, 'masks': target['masks'].flip(-1), 'boxes': target['boxes'].clone()}
    target['boxes'][:, [0, 2]] = width - target['boxes'][:, [2, 0]]
    return result, target


class TrainingExamples(torch.utils.data.Dataset):
    def __init__(self, dataset, family, seed):
        self.dataset, self.family, self.seed = dataset, family, seed

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, key):
        index, epoch = key
        sample, target = self.dataset[index]
        rng = np.random.default_rng(np.random.SeedSequence([self.seed, epoch, index]))
        scale = float(rng.uniform(.5, 2.5))
        h, w = sample['image'].shape[-2:]
        offset = (int(rng.integers(max(0, round(h*scale)-h)+1)),
                  int(rng.integers(max(0, round(w*scale)-w)+1)))
        sample, target, meta = augment_visible(sample, target, scale=scale, offset=offset)
        flip = bool(rng.random() < .5)
        if self.family == 'dpseg':
            sample, target, _ = prepare_training_example(sample, target, horizontal_flip=flip)
        elif flip:
            sample, target = flip_native(sample, target)
        meta.update(index=index, epoch=epoch, horizontal_flip=flip)
        return sample, target, meta


def global_num_masks(self, class_labels, device):
    count = torch.tensor(sum(map(len, class_labels)), dtype=torch.float32, device=device)
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(count)
        count /= torch.distributed.get_world_size()
    return count.clamp(min=1)


class NativeDAFusion(nn.Module):
    def __init__(self, native):
        super().__init__()
        self.native = native

    @property
    def device(self):
        return next(self.parameters()).device

    def inputs(self, samples, targets=None):
        from detectron2.structures import Instances
        from .course_depth_data import model_depth_channels
        batch = []
        for i, sample in enumerate(samples):
            h, w = sample['image'].shape[-2:]
            record = {'image': sample['image'].to(self.device) * 255,
                      'depth_image': model_depth_channels(sample, (h, w), device=self.device),
                      'height': h, 'width': w}
            if targets is not None:
                instances = Instances((h, w))
                instances.gt_masks = targets[i]['masks'].to(self.device).bool()
                instances.gt_classes = torch.zeros(len(instances.gt_masks), device=self.device, dtype=torch.long)
                record['instances'] = instances
            batch.append(record)
        return batch

    def forward(self, samples, targets):
        return self.native(self.inputs(samples, targets))

    @torch.inference_mode()
    def predict_sample(self, sample, *, score_threshold=.01, max_detections=100):
        from pycocotools import mask as mask_utils
        if self.training:
            raise ValueError('Inference requires eval mode')
        instances = self.native(self.inputs([sample]))[0]['instances'].to('cpu')
        result = []
        for mask, score in zip(instances.pred_masks[:max_detections], instances.scores[:max_detections]):
            if float(score) <= score_threshold or not mask.any():
                continue
            rle = mask_utils.encode(np.asfortranarray(mask.numpy(), dtype=np.uint8))
            rle['counts'] = rle['counts'].decode('ascii')
            result.append({'category_id': 1, 'score': float(score), 'segmentation': rle})
        return result


def build_model(family, checkpoint, seed):
    torch.manual_seed(seed)
    if family == 'dpseg':
        from .course_resolution import build_trainable
        model, metadata = build_trainable(checkpoint, device='cpu', seed=seed, resolution=1024)
        model.model.criterion.get_num_masks = types.MethodType(global_num_masks, model.model.criterion)
        metadata['loss']['normalization'] = 'Global instance count averaged across DDP ranks'
        return model, {**metadata, 'experiment_family': 'uoais_sim_comparison_v1',
                       'model_id': 'DPSeg-base-UOAIS-Swin-T',
                       'historical_metadata_note': 'Inherited course factory descriptions are historical; run protocol overrides data/training budget.'}
    from PIL import Image
    if not hasattr(Image, 'LINEAR'):
        Image.LINEAR = Image.Resampling.BILINEAR
    vendor = ROOT / 'third_party/DA-Fusion'
    sys.path.insert(0, str(vendor / 'dafusion'))
    import dafusion
    from dafusion.config import add_dafusion_config
    from detectron2.config import get_cfg
    from detectron2.modeling import build_model as native_build
    cfg = get_cfg()
    add_dafusion_config(cfg)
    cfg.merge_from_file(str(vendor / 'dafusion/configs/dafusion_rgbd_uoais_stage1.yaml'))
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.WEIGHTS = ''
    cfg.MODEL.MASK_FORMER.NO_OBJECT_WEIGHT = .1
    cfg.freeze()
    native = native_build(cfg)
    initial = torch.load(checkpoint, map_location='cpu', weights_only=True)['model']
    if getattr(initial, '_metadata', {}).get('sem_seg_head', {}).get('version') != 2:
        raise ValueError('Native initialization requires verified version-2 metadata')
    missing, unexpected = native.load_state_dict(initial, strict=False)
    allowed = ('backbone.fusions.', 'backbone.depth_adapter.', 'sem_seg_head.predictor.class_embed.',
               'sem_seg_head.predictor.query_feat.', 'sem_seg_head.predictor.query_embed.')
    if unexpected or any(k != 'criterion.empty_weight' and not k.startswith(allowed) for k in missing):
        raise ValueError(f'Unexplained initialization: {missing}, {unexpected}')
    model = NativeDAFusion(native)
    metadata = {'model_id': 'DA-Fusion-public-reimplementation-UOAIS-Swin-L',
                'experiment_family': 'uoais_sim_comparison_v1', 'resolved_config': cfg.dump(),
                'source_commit': json.loads((vendor / 'source_receipt.json').read_text())['commit'],
                'backbone_parameter_prefixes': ['native.backbone.swin_rgb.', 'native.backbone.swin_d.'],
                'fresh_tensor_names': missing, 'loaded_tensors': len(initial),
                'parameters': sum(p.numel() for p in model.parameters()),
                'training_override': 'Shared runner augmentation/optimizer; no public mapper noise or hole patterns; no-object weight .1'}
    return model, metadata
