"""Single-component Mask R-CNN initialized from the verified COCO checkpoint.

Only class-dependent output layers are replaced. Course targets keep independent
instance masks; RGB normalization/resizing and losses use torchvision unchanged.
"""
from pathlib import Path
import hashlib

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch import nn
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT = ROOT / 'pretrained/maskrcnn_resnet50_fpn_v2_coco_mirror_state_dict.pt'
CHECKPOINT_SHA256 = 'a015f8e1988c0d3171679bb32ff9ef2204f544cebd7d29d18b59be5001957064'
BACKBONE_PREFIX = 'model.backbone.'


class CourseTrainableMaskRCNN(nn.Module):
    def __init__(self, checkpoint=DEFAULT_CHECKPOINT, *, seed=17):
        super().__init__()
        checkpoint = Path(checkpoint).resolve()
        with checkpoint.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        if checksum != CHECKPOINT_SHA256:
            raise ValueError('Unexpected COCO initialization checksum')
        # Initializing all layers locally first prevents implicit downloads.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(seed)
            self.model = maskrcnn_resnet50_fpn_v2(
                weights=None, weights_backbone=None, num_classes=91,
                min_size=800, max_size=1333, box_score_thresh=.01,
                box_detections_per_img=100,
            )
            pretrained = torch.load(checkpoint, map_location='cpu', weights_only=True)
            self.model.load_state_dict(pretrained, strict=True)
            self.model.roi_heads.box_predictor = FastRCNNPredictor(
                self.model.roi_heads.box_predictor.cls_score.in_features, 2)
            logits = self.model.roi_heads.mask_predictor.mask_fcn_logits
            self.model.roi_heads.mask_predictor.mask_fcn_logits = nn.Conv2d(logits.in_channels, 2, 1)
            nn.init.kaiming_normal_(self.model.roi_heads.mask_predictor.mask_fcn_logits.weight,
                                   mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(self.model.roi_heads.mask_predictor.mask_fcn_logits.bias)
        reinitialized = [key for key in self.model.state_dict()
                         if key.startswith('roi_heads.box_predictor.')
                         or key.startswith('roi_heads.mask_predictor.mask_fcn_logits.')]
        inherited = []
        for key, tensor in self.model.state_dict().items():
            if key not in reinitialized:
                if not torch.equal(tensor, pretrained[key]):
                    raise RuntimeError(f'Unexpected initialization change: {key}')
                inherited.append(key)
        self.metadata = {
            'model': 'maskrcnn', 'architecture': 'Mask R-CNN ResNet50-FPN v2',
            'initial_checkpoint': str(checkpoint), 'initial_checkpoint_sha256': checksum,
            'initialization_seed': seed, 'foreground_classes': {'1': 'component'},
            'background_index': 0, 'reinitialized_state_keys': reinitialized,
            'exactly_inherited_state_keys': inherited,
            'preprocess': {'shortest_edge': 800, 'longest_edge': 1333,
                           'rgb_range': [0, 1], 'interpolation': 'bilinear, align_corners=False, antialias=False',
                           'mean': self.model.transform.image_mean, 'std': self.model.transform.image_std},
            'postprocess': {'score_threshold': .01, 'mask_threshold': .5,
                            'max_detections': 100, 'nms': 'native single-component box NMS',
                            'independent_overlapping_masks': True},
            'normalization_training': 'BatchNorm running statistics frozen; affine parameters trainable',
            'backbone_parameter_prefix': BACKBONE_PREFIX,
            'loss': 'unmodified torchvision detection, box, RPN and instance-mask losses',
            'parameters': sum(p.numel() for p in self.parameters()),
            'depth_used': False,
        }
        self.train()

    def train(self, mode=True):
        super().train(mode)
        # Preserve pretrained BN statistics with the per-GPU batch used here.
        if mode:
            for layer in self.model.modules():
                if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                    layer.eval()
        return self

    def forward(self, images, targets):
        if not self.training:
            raise RuntimeError('Training loss requires train()')
        device = next(self.parameters()).device
        images = [image.to(device, non_blocking=True) for image in images]
        targets = [{key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
                    for key, value in target.items()} for target in targets]
        if any(not bool((target['labels'] == 1).all()) for target in targets):
            raise ValueError('Course foreground labels must be component=1')
        return self.model(images, targets)

    @torch.inference_mode()
    def predict(self, image):
        if self.training:
            raise RuntimeError('Call eval() before predict()')
        if not isinstance(image, Image.Image) or image.mode != 'RGB':
            raise ValueError('predict requires a PIL RGB image')
        device = next(self.parameters()).device
        tensor = pil_to_tensor(image).to(device, dtype=torch.float32).div_(255)
        output = self.model([tensor])[0]
        masks = (output['masks'][:, 0] >= .5).to(torch.uint8).cpu().numpy()
        results = []
        for mask, score, label in zip(masks, output['scores'].tolist(), output['labels'].tolist()):
            if label != 1:
                raise RuntimeError('Unexpected component prediction label')
            if score < .01 or not mask.any():
                continue
            rle = mask_utils.encode(np.asfortranarray(mask))
            rle['counts'] = rle['counts'].decode('ascii')
            results.append({'category_id': 1, 'score': float(score), 'segmentation': rle})
        return results


def build_trainable(checkpoint=DEFAULT_CHECKPOINT, device='cpu', seed=17):
    adapter = CourseTrainableMaskRCNN(checkpoint, seed=seed).to(device)
    return adapter, adapter.metadata
