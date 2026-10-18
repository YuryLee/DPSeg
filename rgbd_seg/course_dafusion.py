"""DA-Fusion public DS/DC reimplementation adapted to the course benchmark.

Two independently trainable COCO-initialized Swin-Tiny backbones feed four
upstream FusionStage modules and the unchanged pretrained Mask2Former decoder.
This is a course adaptation, not a reproduction of the paper's benchmark scores.
The pinned upstream source stays unmodified. Its position-bias size cutoff and
sampling-coordinate conventions are deliberately retained and disclosed.
"""
from __future__ import annotations

import copy
import importlib
import importlib.util
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import BackboneOutput

from .course_finetune_mask2former import (
    BACKBONE_PREFIX, CourseTrainableMask2Former, component_predictions,
    build_trainable as build_rgb_model,
)
from .course_mask2former import DEFAULT_CHECKPOINT, _sha256


ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'third_party' / 'DA-Fusion'
UPSTREAM_COMMIT = 'c5bf9e0395c11ed1c2fbb2b88e7d0589a90df570'
_PACKAGE = '_course_dafusion_pinned_fusion'
_SOURCE_FILES = (
    'dafusion/dafusion/modeling/fusion/__init__.py',
    'dafusion/dafusion/modeling/fusion/deformable_self_attention.py',
    'dafusion/dafusion/modeling/fusion/deformable_cross_attention.py',
    'dafusion/dafusion/modeling/fusion/fusion_stage.py',
    'dafusion/dafusion/modeling/fusion/geometry_prior.py',
    'dafusion/dafusion/modeling/fusion/offset_network.py',
    'dafusion/dafusion/modeling/fusion/relative_position_bias.py',
    'dafusion/dafusion/modeling/fusion/sampling.py',
    'dafusion/dafusion/modeling/backbone/depth_adapter.py',
    'dafusion/dafusion/modeling/backbone/dual_fusion_backbone.py',
    'dafusion/dafusion/data/depth_xyz.py',
    'dafusion/dafusion/data/hha.py',
    'dafusion/dafusion/data/dataset_mappers/dafusion_mapper.py',
    'dafusion/dafusion/config.py',
    'dafusion/configs/Base-DAFusion-SwinL.yaml',
    'dafusion/configs/dafusion_rgbd_uoais_stage1.yaml',
    'dafusion/README.md',
)


def upstream_source_sha256():
    """Validate source receipt and return project-relative provenance hashes."""
    receipt_path = UPSTREAM / 'source_receipt.json'
    receipt = json.loads(receipt_path.read_text())
    if receipt.get('commit') != UPSTREAM_COMMIT or receipt.get('all_git_blobs_verified') is not True:
        raise ValueError('Unverified or unexpected DA-Fusion upstream commit')
    result = {str(receipt_path.relative_to(ROOT)): _sha256(receipt_path)}
    for name in _SOURCE_FILES:
        path = UPSTREAM / name
        digest = _sha256(path)
        if digest != receipt['files'][name]['sha256']:
            raise ValueError(f'Pinned upstream file changed: {name}')
        result[str(path.relative_to(ROOT))] = digest
    return result


def contiguous_multihead_attend(q, k, v, num_heads, bias=None):
    """Same equations as upstream; contiguous SDPA inputs permit CUDA kernels.

    The biased path intentionally uses the upstream explicit softmax formula.
    No attention term, sampling point, dropout, or bias cutoff is changed.
    Floating-point results of fused SDPA may differ within numerical tolerance.
    """
    b, c, nq = q.shape
    nk, dh = k.shape[2], c // num_heads
    q = q.reshape(b, num_heads, dh, nq).transpose(-1, -2).contiguous()
    k = k.reshape(b, num_heads, dh, nk).transpose(-1, -2).contiguous()
    v = v.reshape(b, num_heads, dh, nk).transpose(-1, -2).contiguous()
    if bias is not None:
        attention = torch.matmul(q, k.transpose(-1, -2)) / (dh ** .5)
        attention = attention + (bias if bias.dim() == 4 else bias.unsqueeze(0))
        out = torch.matmul(attention.softmax(dim=-1), v)
    else:
        out = F.scaled_dot_product_attention(q, k, v)
    return out.transpose(-1, -2).reshape(b, c, nq)


def _load_upstream():
    """Import only tensor modules, avoiding Detectron2's registration package."""
    upstream_source_sha256()
    if _PACKAGE not in sys.modules:
        path = UPSTREAM / 'dafusion/dafusion/modeling/fusion/__init__.py'
        spec = importlib.util.spec_from_file_location(_PACKAGE, path,
                                                     submodule_search_locations=[str(path.parent)])
        package = importlib.util.module_from_spec(spec)
        sys.modules[_PACKAGE] = package
        spec.loader.exec_module(package)
        ds = importlib.import_module(_PACKAGE + '.deformable_self_attention')
        dc = importlib.import_module(_PACKAGE + '.deformable_cross_attention')
        # Scoped to this private, pinned package; the source tree is not edited.
        ds.upstream_multihead_attend = ds.multihead_attend
        ds.multihead_attend = contiguous_multihead_attend
        dc.multihead_attend = contiguous_multihead_attend
        adapter_name = _PACKAGE + '.depth_adapter'
        spec = importlib.util.spec_from_file_location(adapter_name,
            UPSTREAM / 'dafusion/dafusion/modeling/backbone/depth_adapter.py')
        module = importlib.util.module_from_spec(spec)
        sys.modules[adapter_name] = module
        spec.loader.exec_module(module)
    fusion = importlib.import_module(_PACKAGE + '.fusion_stage')
    depth = importlib.import_module(_PACKAGE + '.depth_adapter')
    return fusion.FusionStage, depth.DepthAdapter


class DualSwinFusionBackbone(nn.Module):
    """Complete both Swin branches, then fuse matching scales as upstream does."""

    def __init__(self, rgb_encoder, *, fusion_checkpointing=True):
        super().__init__()
        FusionStage, DepthAdapter = _load_upstream()
        self.swin_rgb = rgb_encoder
        self.swin_d = copy.deepcopy(rgb_encoder)
        self.channels = list(rgb_encoder.channels)
        if self.channels != [96, 192, 384, 768]:
            raise ValueError(f'Expected the pinned four-stage Swin-Tiny: {self.channels}')
        self.depth_adapter = DepthAdapter(in_chans=4, out_chans=3, hidden=128, zero_init=True)
        self.fusions = nn.ModuleList([
            FusionStage(dim, num_heads=8, n_groups=4, stride=stride,
                        range_factor=2., use_rpb=True, geom_prior=False)
            for dim, stride in zip(self.channels, [8, 4, 2, 1])
        ])
        self.fusion_checkpointing = bool(fusion_checkpointing)

    def forward(self, pixel_values, **kwargs):
        if pixel_values.ndim != 4 or pixel_values.shape[1] != 7:
            raise ValueError('Expected normalized RGB + standardized XYZ + validity (7 channels)')
        rgb = self.swin_rgb(pixel_values[:, :3]).feature_maps
        depth = self.swin_d(self.depth_adapter(pixel_values[:, 3:])).feature_maps
        fused = []
        for fusion, fr, fd in zip(self.fusions, rgb, depth):
            if fr.shape != fd.shape:
                raise RuntimeError('RGB and depth backbone features are not aligned')
            if self.fusion_checkpointing and self.training and torch.is_grad_enabled():
                fused.append(checkpoint(fusion, fr, fd, use_reentrant=False))
            else:
                fused.append(fusion(fr, fd))
        return BackboneOutput(feature_maps=tuple(fused))


class CourseTrainableDAFusion(CourseTrainableMask2Former):
    """Native Mask2Former losses and independent masks, with required RGB-D."""

    def prepare_batch(self, samples, targets=None):
        from .course_depth_data import model_depth_channels
        if not isinstance(samples, (list, tuple)) or not samples:
            raise ValueError('Expected a nonempty list of RGB-D sample dictionaries')
        required = {'image', 'depth_m', 'depth_valid', 'camera_K'}
        if any(not isinstance(sample, dict) or set(sample) != required for sample in samples):
            raise ValueError('RGB-D samples must contain only image/depth_m/depth_valid/camera_K')
        batch, sizes = super().prepare_batch([s['image'] for s in samples], targets)
        pixels = batch['pixel_values']
        depths = pixels.new_zeros((len(samples), 4, *sizes['padded_size']))
        with torch.autocast(device_type=self.device.type, enabled=False):
            for index, (sample, size) in enumerate(zip(samples, sizes['resized_sizes'])):
                depth = model_depth_channels(sample, size, device=self.device)
                depths[index, :, :size[0], :size[1]] = depth
        batch['pixel_values'] = torch.cat((pixels, depths), dim=1)
        return batch, sizes

    @torch.inference_mode()
    def predict_sample(self, sample, *, score_threshold=.01, max_detections=100):
        if self.training:
            raise ValueError('Call adapter.eval() before predict_sample')
        batch, sizes = self.prepare_batch([sample])
        outputs = self.model(**batch, return_dict=True)
        predictions = component_predictions(outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes['padded_size'], image_size=sizes['resized_sizes'][0],
            original_size=sizes['original_sizes'][0], score_threshold=score_threshold,
            max_detections=max_detections)
        return [{key: item[key] for key in ('category_id', 'score', 'segmentation')} for item in predictions]

    def predict(self, image, **kwargs):
        raise ValueError('DA-Fusion requires RGB, measured depth, validity and K; use predict_sample')


def build_trainable(checkpoint_dir=DEFAULT_CHECKPOINT, device='cpu', *, seed=17,
                    fusion_checkpointing=True):
    """Initialize from the same public COCO checkpoint as the RGB baseline."""
    base, base_metadata = build_rgb_model(checkpoint_dir, device='cpu', seed=seed)
    encoder = base.model.model.pixel_level_module.encoder
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        fused_encoder = DualSwinFusionBackbone(encoder, fusion_checkpointing=fusion_checkpointing)
    if any(not torch.equal(value, fused_encoder.swin_d.state_dict()[name])
           for name, value in encoder.state_dict().items()):
        raise RuntimeError('Depth initialization must exactly copy RGB backbone tensors')
    base.model.model.pixel_level_module.encoder = fused_encoder
    metadata = {
        'model_id': 'DA-Fusion-public-DS-DC-course-Swin-Tiny',
        'method_scope': 'Course adaptation of pinned public DS/DC reimplementation; not paper-score reproduction',
        'upstream_repository': 'https://github.com/teamROBI/DA-Fusion',
        'upstream_commit': UPSTREAM_COMMIT, 'upstream_source_sha256': upstream_source_sha256(),
        'adapter_sha256': _sha256(__file__),
        'pretrained_sha256': base_metadata['pretrained_sha256'],
        'initialization': {
            'seed': seed, 'RGB_and_depth_backbones_equal_at_start': True,
            'both_backbones': 'Same public COCO Swin-Tiny; independent trainable parameter storage',
            'decoder': 'Unchanged public COCO Mask2Former pixel/query/mask decoder',
            'classifier': 'Same seeded fresh foreground/no-object head as RGB baseline',
            'fusion': 'New upstream DS/DC modules, native PyTorch random initialization',
            'depth_adapter': 'Upstream 4-to-3 adapter, hidden=128, with zero-initialized final projection and XYZ residual',
            'base_loading_verification': base_metadata['initialization'],
        },
        'backbone_parameter_prefixes': [BACKBONE_PREFIX + 'swin_rgb.', BACKBONE_PREFIX + 'swin_d.'],
        'parameters': sum(p.numel() for p in base.model.parameters()),
        'fusion': {'scales': [4, 8, 16, 32], 'channels': fused_encoder.channels,
                   'heads': 8, 'offset_groups': 4, 'strides': [8, 4, 2, 1], 'range_factor': 2.,
                   'relative_position_bias': True, 'rpb_max_pairs': 4000000,
                   'rpb_active_at_800': [False, False, True, True], 'geometry_prior': False,
                   'fusion_checkpointing': bool(fusion_checkpointing),
                   'attention_layout': 'Contiguous Q/K/V for SDPA; same explicit upstream biased attention',
                   'sampling': 'Unchanged upstream reference-grid/grid_sample conventions',
                   'flow': 'Independent complete RGB/depth Swin, then DS_rgb + DS_depth + bidirectional DC at all four scales'},
        'preprocess': {**base_metadata['preprocess'], 'depth_used': True,
                       'depth_encoding': 'camera XYZ metres -> aligned bilinear resize, nearest valid, reflect X on hflip -> per-channel median and isotropic p95 radius -> XYZ+valid',
                       'depth_normalization': 'Upstream standardize_xyz, valid points only; no GT/extrinsics/support mask',
                       'depth_augmentation': 'Same geometric flip as RGB baseline; no extra holes/noise/copy-paste'},
        'loss': base_metadata['loss'], 'postprocess': base_metadata['postprocess'],
        'baseline_initialization_metadata': base_metadata,
        'differences_from_paper': [
            'Uses current public DS/DC code, whose README calls it a reimplementation without completed paper benchmark validation',
            'Swin-Tiny replaces Swin-L; Hugging Face Mask2Former replaces Detectron2 implementation',
            'Current public stage1 XYZ+valid and DepthAdapter instead of paper HHA',
            'Public DC samples each modality own KV then cross-attends; paper equations describe cross-modal offsets',
            'Public continuous RPB cutoff and coordinate conventions retained',
            'Course fixed data/resize/hflip/loss/pretraining and five-epoch effective-batch-eight protocol',
            'No-object weight 0.1 retained from RGB baseline, versus 0.05 in public XYZ stage1 config',
        ],
    }
    adapter = CourseTrainableDAFusion(base.model, mean=base.image_mean.flatten().tolist(),
        std=base.image_std.flatten().tolist(), metadata=metadata).to(device).train()
    return adapter, metadata
