"""RGB Mask2Former control using the same Swin-L COCO initialization as DPSeg."""
from __future__ import annotations

import json
from pathlib import Path
import types

from safetensors import safe_open
import torch
from torch import nn
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from .course_finetune_mask2former import (
    BACKBONE_PREFIX, CourseTrainableMask2Former, LossReportingMask2Former,
)
from .course_mask2former import _sha256
from .course_resolution import local_num_masks


def build_trainable(checkpoint, *, device='cpu', seed=17, resolution=1024,
                    backbone_checkpointing=True):
    if resolution != 1024:
        raise ValueError('The matched Large comparison uses 1024px input')
    checkpoint = Path(checkpoint).resolve()
    receipt = json.loads((checkpoint / 'source_receipt.json').read_text())
    if (receipt.get('repository') != 'facebook/mask2former-swin-large-coco-instance'
            or _sha256(checkpoint / 'model.safetensors') != receipt['weights_sha256']):
        raise ValueError('Swin-L COCO initialization identity differs')
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model, loading = LossReportingMask2Former.from_pretrained(
            checkpoint, local_files_only=True, use_safetensors=True,
            torch_dtype=torch.float32, output_loading_info=True)
        if any(loading.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError(f'Pretrained loading mismatch: {loading}')
        encoder = model.model.pixel_level_module.encoder
        if (model.config.num_labels != 80 or list(encoder.channels) != [192, 384, 768, 1536]
                or [len(stage.blocks) for stage in encoder.encoder.layers] != [2, 2, 18, 2]):
            raise ValueError('Expected the original COCO instance Swin-L model')
        # Match the fresh component classifier used by the Large RGB-D arm.
        torch.default_generator.manual_seed(seed)
        classifier = nn.Linear(model.config.hidden_dim, 2)
        model._init_weights(classifier)
        model.class_predictor = classifier
        model.config.num_labels = 1
        model.config.id2label, model.config.label2id = {0: 'component'}, {'component': 0}
        model.criterion = Mask2FormerLoss(model.config, model.weight_dict)
    allowed = {'class_predictor.weight', 'class_predictor.bias', 'criterion.empty_weight'}
    unchanged, changed = 0, []
    with safe_open(checkpoint / 'model.safetensors', framework='pt', device='cpu') as source:
        state = model.state_dict()
        if set(state) != set(source.keys()):
            raise ValueError('Pretrained state keys differ')
        for name in source.keys():
            if torch.equal(source.get_tensor(name), state[name]):
                unchanged += 1
            elif name in allowed:
                changed.append(name)
            else:
                raise ValueError('Unexpected initialization change: ' + name)
    if set(changed) != allowed:
        raise ValueError('Expected only classifier and no-object buffer replacement')
    if backbone_checkpointing:
        encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    model.criterion.get_num_masks = types.MethodType(local_num_masks, model.criterion)
    processor = json.loads((checkpoint / 'preprocessor_config.json').read_text())
    metadata = {
        'model_id': 'Mask2Former-Swin-Large-RGB-course-1024-v1',
        'experiment_family': 'course_swinl_rgb_v1', 'backbone': 'large',
        'input_modality': 'RGB', 'depth_used': False, 'camera_intrinsics_used': False,
        'backbone_checkpointing': bool(backbone_checkpointing),
        'pretrained_sha256': receipt['weights_sha256'], 'pretrained_receipt': receipt,
        'pretrained_config_sha256': _sha256(checkpoint / 'config.json'),
        'pretrained_preprocessor_sha256': _sha256(checkpoint / 'preprocessor_config.json'),
        'parameters': sum(p.numel() for p in model.parameters()),
        'backbone_parameter_prefix': BACKBONE_PREFIX,
        'initialization': {'seed': seed, 'unchanged_pretrained_tensors': unchanged,
            'changed_tensors': sorted(changed), 'all_other_tensors_exactly_equal': True,
            'classifier': 'Fresh seeded two-output component/no-object classifier',
            'backbone_and_decoder': 'Same public COCO Swin-L checkpoint as the RGB-D arm'},
        'preprocess': {'mean': processor['image_mean'], 'std': processor['image_std'],
            'shortest_edge': 1024, 'longest_edge': 1333, 'size_divisibility': 32,
            'rgb_resize': 'torch bilinear align_corners=False antialias=False',
            'mask_resize': 'nearest per independent binary instance',
            'augmentation': 'Same deterministic seed/epoch/image horizontal flip as RGB-D'},
        'loss': {'weights': model.weight_dict, 'no_object_weight': model.config.no_object_weight,
            'train_num_points': model.config.train_num_points,
            'oversample_ratio': model.config.oversample_ratio,
            'importance_sample_ratio': model.config.importance_sample_ratio,
            'decoder_layers': model.config.decoder_layers,
            'matching_and_loss_dtype': 'float32 native BCE/Dice/classification plus auxiliary losses',
            'normalization': 'Local microbatch mask count; image-weighted DDP accumulation'},
        'postprocess': {'score_threshold': .01, 'max_detections': 100, 'category_id': 1,
            'mask_threshold': .5, 'score': 'Foreground probability times mean positive-mask probability'},
    }
    adapter = CourseTrainableMask2Former(model, mean=processor['image_mean'],
        std=processor['image_std'], short_edge=1024, max_size=1333,
        metadata=metadata).to(device).train()
    return adapter, metadata
