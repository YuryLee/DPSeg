"""Frozen RGB Swin-L base for DPSeg without RGB-D feature fusion.

Only RGB reaches Mask2Former. Metric depth/validity remain in the caller's
sample for the unchanged nine-channel point cues and probability refinement.
This ablation therefore still uses depth; it is not an RGB-only full system.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path

import torch

from .course_finetune_mask2former import CourseTrainableMask2Former, component_predictions

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PROTOCOL = ROOT / 'reports/course_swinl_rgb_v1/protocol.json'
DEFAULT_WEIGHTS = ROOT / 'reports/course_swinl_rgb_v1/base15/final.pt'
DEFAULT_TRAINING_RECEIPT = ROOT / 'reports/course_swinl_rgb_v1/base15/run.json'
DEFAULT_PROTOCOL_SHA256 = 'bfa367bcbc8b4c57d4445a4607de02f8243409261aedbb7c59476073cd48ac42'
DEFAULT_WEIGHTS_SHA256 = '6d350cac2ef76ee22d372c09cdcb1b01210ccc55c7c2739000b86e2cec302c15'
DEFAULT_TRAINING_RECEIPT_SHA256 = '0c92b0c1599eda340c7d414e4b9ce5b78c0b4dd9d96fb8925d6bc681fbe1e974'
FINE_CHANNELS = 448


def _sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def _checked_path(path, expected, name):
    path = Path(path).expanduser().resolve()
    if not isinstance(expected, str) or _sha(path) != expected:
        raise ValueError(f'{name} SHA256 differs from the declared RGB reference')
    return path


class RGBRefinementAdapter(CourseTrainableMask2Former):
    """Bridge sample dictionaries to the original RGB preprocessing interface.

    ``depth_encoding`` describes the shared sample representation only. It does
    not add a depth encoder, depth input tensor or feature fusion to the model.
    RGB tensor lists remain accepted so the original ``predict(PIL)`` API also
    works. Sample dictionaries reject annotation fields and camera intrinsics.
    """

    depth_encoding = 'depth'
    fine_channels = FINE_CHANNELS

    def __init__(self, model, *, mean, std, short_edge=1024, max_size=1333, metadata=None):
        metadata = copy.deepcopy(metadata or {})
        metadata.update(
            ablation='without_rgbd_feature_fusion',
            input_modality='RGB encoder with depth retained for mask refinements',
            encoder_input_modality='RGB', rgbd_feature_fusion=False,
            depth_used=True, encoder_depth_used=False, camera_intrinsics_used=False,
            sample_depth_encoding='depth', point_depth_channels=9,
            point_fine_channels=FINE_CHANNELS,
            point_fine_features='RGB Swin-L stride4 features (192) + pixel decoder features (256)',
            depth_used_for_point_refinement=True, depth_used_for_probability_refinement=True,
            base_frozen=True)
        super().__init__(model, mean=mean, std=std, short_edge=short_edge,
                         max_size=max_size, metadata=metadata)
        self.requires_grad_(False).eval()

    def prepare_batch(self, samples, targets=None):
        if not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError('Expected a nonempty list of RGB tensors or no-intrinsics samples')
        if all(torch.is_tensor(sample) for sample in samples):
            images = samples
        else:
            required = {'image', 'depth_m', 'depth_valid'}
            images = []
            for sample in samples:
                if not isinstance(sample, dict) or set(sample) != required:
                    raise ValueError('Expected only image, depth_m and depth_valid; no annotations or camera K')
                image, depth, valid = (sample[key] for key in ('image', 'depth_m', 'depth_valid'))
                if (not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] != 3
                        or not torch.is_tensor(depth) or not depth.is_floating_point()
                        or depth.ndim not in (2, 3) or (depth.ndim == 3 and depth.shape[0] != 1)
                        or not torch.is_tensor(valid) or valid.dtype != torch.bool
                        or valid.shape != depth.shape or depth.shape[-2:] != image.shape[-2:]):
                    raise ValueError('Expected aligned RGB, metric depth and boolean validity tensors')
                images.append(image)
        # pixel_mask is RGB padding validity, never the depth-validity map.
        # In particular, depth contents are not read, normalized or forwarded.
        return super().prepare_batch(images, targets)

    @torch.inference_mode()
    def predict_sample(self, sample, *, score_threshold=.01, max_detections=100):
        if self.training or any(parameter.requires_grad for parameter in self.parameters()):
            raise ValueError('RGB refinement base must be frozen and in eval mode')
        batch, sizes = self.prepare_batch([sample])
        outputs = self.model(**batch, return_dict=True)
        values = component_predictions(
            outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
            original_size=sizes['original_sizes'][0], score_threshold=score_threshold,
            max_detections=max_detections)
        return [{key: item[key] for key in ('category_id', 'score', 'segmentation')} for item in values]


def load_rgb_base(protocol_path=DEFAULT_PROTOCOL, weights_path=DEFAULT_WEIGHTS, *, device='cpu',
                  expected_protocol_sha256=DEFAULT_PROTOCOL_SHA256,
                  expected_weights_sha256=DEFAULT_WEIGHTS_SHA256,
                  training_receipt_path=DEFAULT_TRAINING_RECEIPT,
                  expected_training_receipt_sha256=DEFAULT_TRAINING_RECEIPT_SHA256):
    """Return ``(frozen_adapter, receipt)`` for the completed fixed epoch15 RGB base.

    No point head is loaded or selected here. Callers train their own 448+9 input
    point head, and record the chosen epoch from full-validation evaluation.
    Weight/protocol paths may be relocated without changing their pinned bytes.
    """
    from .course_swinl_rgb import build_trainable
    from tools.train_course_swinl_rgb import sources

    protocol_path = _checked_path(protocol_path, expected_protocol_sha256, 'RGB training protocol')
    weights_path = _checked_path(weights_path, expected_weights_sha256, 'RGB epoch15 weights')
    training_receipt_path = _checked_path(training_receipt_path, expected_training_receipt_sha256,
                                         'RGB completed-training receipt')
    protocol = json.loads(protocol_path.read_text())
    training = json.loads(training_receipt_path.read_text())
    if (protocol['schema'] != 'course_swinl_rgb_training_v1' or protocol['status'] != 'frozen'
            or protocol['model'] != {'backbone': 'large', 'input_modality': 'RGB',
                                      'backbone_checkpointing': True}
            or protocol['training']['epochs'] != 15
            or protocol['source_sha256'] != sources()):
        raise ValueError('RGB model, full training budget or frozen sources differ')
    expected_steps = math.ceil(14432 / protocol['training']['effective_batch_size']) * 15
    if (training['status'] != 'completed' or training['scope'] != 'full_train_val'
            or training['epoch'] != 15 or training['test_used'] is not False
            or training['optimizer_steps'] != expected_steps
            or training['training_image_exposures'] != 14432 * 15
            or training['protocol_sha256'] != expected_protocol_sha256
            or training['source_sha256'] != protocol['source_sha256']
            or training['final_checkpoint_sha256'] != expected_weights_sha256):
        raise ValueError('RGB base is not the completed full-data epoch15 reference')
    saved = torch.load(weights_path, map_location='cpu', weights_only=True, mmap=True)
    if (saved['epoch'] != 15 or saved['source_sha256'] != protocol['source_sha256']
            or saved['protocol_sha256'] != expected_protocol_sha256
            or saved['model_metadata'] != training['model_metadata']):
        raise ValueError('RGB checkpoint provenance differs from the completed training run')
    base, metadata = build_trainable(protocol['checkpoint'], device='cpu',
        seed=protocol['training']['seed'], backbone_checkpointing=protocol['model']['backbone_checkpointing'])
    if metadata != saved['model_metadata']:
        raise ValueError('RGB checkpoint construction differs from the original architecture')
    base.load_state_dict(saved['model'], strict=True)
    del saved
    encoder = base.model.model.pixel_level_module.encoder
    if (list(encoder.channels) != [192, 384, 768, 1536]
            or base.model.config.mask_feature_size != 256
            or any('swin_d' in name or 'fusions' in name for name in base.state_dict())):
        raise ValueError('Expected one RGB Swin-L encoder and a 448-channel point feature map')
    adapter = RGBRefinementAdapter(base.model, mean=metadata['preprocess']['mean'],
        std=metadata['preprocess']['std'], short_edge=1024, max_size=1333, metadata=metadata)
    adapter.requires_grad_(False).eval().to(device)
    if adapter.training or any(parameter.requires_grad for parameter in adapter.parameters()):
        raise ValueError('RGB base freeze failed')
    receipt = dict(status='loaded_frozen', base_epoch=15, base_input_modality='RGB',
        rgbd_feature_fusion=False, encoder_depth_used=False, camera_intrinsics_used=False,
        sample_depth_encoding='depth', depth_retained_for=['point_refinement', 'probability_refinement'],
        point_fine_channels=FINE_CHANNELS, point_depth_channels=9,
        model_parameters_frozen=True, model_eval=True,
        protocol=dict(path=str(protocol_path), sha256=expected_protocol_sha256),
        weights=dict(path=str(weights_path), sha256=expected_weights_sha256),
        training_receipt=dict(path=str(training_receipt_path), sha256=expected_training_receipt_sha256),
        base_source_sha256=protocol['source_sha256'],
        adapter_source_sha256=_sha(Path(__file__)),
        completed_base_optimizer_steps=training['optimizer_steps'],
        completed_base_training_image_exposures=training['training_image_exposures'],
        base_model_metadata=metadata, adapter_metadata=copy.deepcopy(adapter.metadata))
    return adapter, receipt
