"""Load the audited course model with explicit, relocatable artifact paths.

Historical protocol bytes and model sources stay unchanged. Only the locations
of weights, public initialization files and the installed Transformers module
are resolved here; their recorded content hashes remain mandatory.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path, PurePosixPath
import time

import torch

from .course_optimized import FAMILIES, build_trainable, file_sha, tensor_digest
from .course_point_refinement import POLICY, PairedPointHeads, predict_paired
from .course_tiled_probability import refined_public

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = (
    'native', 'native_probability_depth', 'point',
    'point_probability_depth', 'point_depth', 'point_depth_probability_depth',
)
DEFAULT_MANIFEST = ROOT / 'configs/course_e15_point.json'
TRANSFORMERS_SOURCE = 'transformers/models/mask2former/modeling_mask2former.py'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def check_file(path, expected, description):
    path = Path(path).resolve()
    if not path.is_file() or file_sha(path) != expected:
        raise ValueError(f'{description} is missing or its SHA-256 differs: {path}')
    return path


def source_path(name, root=ROOT):
    """Resolve source identity without falling back to the old project folder."""
    logical = PurePosixPath(name)
    if logical.is_absolute():
        if not name.endswith('/' + TRANSFORMERS_SOURCE):
            raise ValueError(f'Unsupported external source in the protocol: {name}')
        spec = importlib.util.find_spec('transformers.models.mask2former.modeling_mask2former')
        if spec is None or spec.origin is None:
            raise ValueError('The pinned Transformers implementation is not installed')
        return Path(spec.origin).resolve()
    root = Path(root).resolve()
    path = (root / Path(*logical.parts)).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f'Source path leaves the project: {name}')
    return path


def verify_sources(mapping, root=ROOT):
    resolved = {}
    for name, expected in mapping.items():
        path = check_file(source_path(name, root), expected, 'Training source')
        resolved[name] = str(path)
    return resolved


def load_base(weights_path, protocol_path, *, coco_checkpoint, device='cpu', expected=None):
    """Reconstruct the original model, with an explicit public COCO directory."""
    weights_path, protocol_path = Path(weights_path).resolve(), Path(protocol_path).resolve()
    protocol = read_json(protocol_path)
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=True)
    protocol_sha = file_sha(protocol_path)
    if protocol.get('status') != 'frozen' or protocol_sha != checkpoint['protocol_sha256']:
        raise ValueError('Base checkpoint does not match its frozen protocol')
    if checkpoint['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Base checkpoint source identity differs from its protocol')
    resolved = verify_sources(checkpoint['source_sha256'])
    actual_file_sha = file_sha(weights_path)
    if expected is not None:
        if actual_file_sha != expected['sha256'] or checkpoint['epoch'] != expected['epoch']:
            raise ValueError('Base weights differ from the selected model manifest')
        if protocol_sha != expected['protocol_sha256']:
            raise ValueError('Base protocol differs from the selected model manifest')
    saved = checkpoint['model_metadata']
    method = protocol.get('intervention', {}).get('method', 'native')
    resolution = saved['preprocess']['shortest_edge']
    if (method not in FAMILIES or saved['experiment_family'] != FAMILIES[method]
            or resolution not in (800, 1024)
            or protocol.get('selected_resolution', resolution) != resolution):
        raise ValueError('Base method or resolution differs from the checkpoint')
    if not Path(coco_checkpoint).is_dir():
        raise ValueError('Provide the local public COCO initialization directory')
    adapter, metadata = build_trainable(
        Path(coco_checkpoint).resolve(), method=method, resolution=resolution,
        seed=protocol['training']['seed'], device=device,
    )
    if metadata != saved:
        differing = sorted(k for k in set(metadata) | set(saved) if metadata.get(k) != saved.get(k))
        raise ValueError(f'Reconstructed model metadata differs: {differing}')
    adapter.load_state_dict(checkpoint['model'], strict=True)
    digest = tensor_digest(adapter.state_dict())
    if digest != checkpoint['selection']['model_sha256']:
        raise ValueError('Reloaded base tensor digest differs from the selected checkpoint')
    if expected is not None and digest != expected['model_tensor_sha256']:
        raise ValueError('Base tensor digest differs from the selected model manifest')
    adapter.eval().requires_grad_(False)
    return adapter, {
        'checkpoint': str(weights_path), 'checkpoint_sha256': actual_file_sha,
        'protocol': str(protocol_path), 'protocol_sha256': protocol_sha,
        'epoch': checkpoint['epoch'], 'model_sha256': digest,
        'method': method, 'resolution': resolution, 'model_metadata': metadata,
        'verified_source_locations': resolved,
    }


def load_model(*, base_weights, head_weights, coco_checkpoint, manifest=DEFAULT_MANIFEST,
               device='cpu'):
    """Load the exact selected base/head pair; no downloads or path rewriting."""
    manifest_path = Path(manifest).resolve()
    bundle = read_json(manifest_path)
    if bundle.get('schema') != 'rgbd-course-model-v1':
        raise ValueError('Unsupported course model manifest')
    base_spec, head_spec = bundle['base'], bundle['head']
    base_protocol = check_file(manifest_path.parent / base_spec['protocol'],
                               base_spec['protocol_sha256'], 'Base protocol')
    point_protocol = check_file(manifest_path.parent / head_spec['protocol'],
                                head_spec['protocol_sha256'], 'Point protocol')
    base_weights = check_file(base_weights, base_spec['sha256'], 'Selected base weights')
    head_weights = check_file(head_weights, head_spec['sha256'], 'Selected point weights')
    point = read_json(point_protocol)
    if (point.get('status') != 'frozen' or point['point_policy'] != POLICY
            or point['weights']['sha256'] != base_spec['sha256']
            or point['training_protocol']['sha256'] != base_spec['protocol_sha256']):
        raise ValueError('Point protocol is bound to a different base or point policy')
    point_sources = verify_sources(point['source_sha256'])
    adapter, base_receipt = load_base(
        base_weights, base_protocol, coco_checkpoint=coco_checkpoint,
        device=device, expected=base_spec,
    )
    value = torch.load(head_weights, map_location='cpu', weights_only=True)
    if (value['protocol_sha256'] != head_spec['protocol_sha256']
            or value['source_sha256'] != point['source_sha256']
            or value['point_policy'] != POLICY
            or value['base_checkpoint_sha256'] != base_receipt['checkpoint_sha256']
            or value['base_model_tensor_sha256'] != base_receipt['model_sha256']
            or value['epoch'] != head_spec['epoch']):
        raise ValueError('Point weights are bound to a different protocol, base or epoch')
    heads = PairedPointHeads().to(device)
    heads.load_state_dict(value['heads'], strict=True)
    digest = tensor_digest(heads.state_dict())
    if digest != head_spec['head_tensor_sha256']:
        raise ValueError('Reloaded point tensor digest differs from the model manifest')
    heads.eval().requires_grad_(False)
    receipt = {
        'model_name': bundle['name'], 'manifest': str(manifest_path),
        'manifest_sha256': file_sha(manifest_path), 'base': base_receipt,
        'head': {'checkpoint': str(head_weights), 'checkpoint_sha256': file_sha(head_weights),
                 'epoch': value['epoch'], 'head_tensor_sha256': digest,
                 'protocol_sha256': head_spec['protocol_sha256']},
        'verified_point_source_locations': point_sources,
        'loader_sha256': file_sha(__file__), 'downloads_performed': False,
    }
    return adapter, heads, receipt


@torch.inference_mode()
def predict_variants(adapter, heads, sample):
    """Reuse the six frozen validation paths, preserving scores/count/order."""
    if set(sample) != {'image', 'depth_m', 'depth_valid', 'camera_K'}:
        raise ValueError('Prediction accepts RGB, metric depth, validity and K only')
    paired, timing = predict_paired(adapter, heads, sample, return_timing=True)
    outputs = {name: [candidate.public() for candidate in values]
               for name, values in paired.items()}
    refinement_times = {}
    synchronize = (lambda: torch.cuda.synchronize(adapter.device)) if adapter.device.type == 'cuda' else lambda: None
    for name, candidates in paired.items():
        synchronize()
        began = time.perf_counter()
        target = 'native_probability_depth' if name == 'native' else name + '_probability_depth'
        outputs[target], _ = refined_public(candidates, sample, 'probability', adapter.device)
        synchronize()
        refinement_times[target] = time.perf_counter() - began
    identities = [(v['category_id'], v['score']) for v in outputs['native']]
    if any([(v['category_id'], v['score']) for v in outputs[name]] != identities for name in VARIANTS):
        raise ValueError('A refiner changed instance identities, scores or order')
    timing['probability_refinement_seconds'] = refinement_times
    return {name: outputs[name] for name in VARIANTS}, timing
