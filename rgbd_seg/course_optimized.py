"""Shared construction and strict reload of resolution/crop/Focal course models."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import torch

from .course_crop import CROP_POLICY
from .course_focal import apply_focal
from .course_resolution import build_trainable as build_resolution

FAMILIES = {'native':'resolution_ddp_v1', 'crop':'small_target_independent_crop_v1',
            'focal':'small_target_independent_focal_v1'}


def build_trainable(checkpoint, *, method, resolution, device='cpu', seed=17):
    if method not in FAMILIES:
        raise ValueError('Unknown independent intervention')
    adapter,metadata = build_resolution(checkpoint,device=device,seed=seed,resolution=resolution)
    if method == 'focal':
        return apply_focal(adapter,metadata)
    if method == 'crop':
        metadata = copy.deepcopy(metadata)
        metadata['model_id'] += '+crop512-v1'
        metadata['experiment_family'] = FAMILIES[method]
        metadata['preprocess']['training_crop_policy'] = copy.deepcopy(CROP_POLICY)
        adapter.metadata = metadata
    return adapter,metadata


def file_sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_digest(state):
    digest = hashlib.sha256()
    for name,tensor in state.items():
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str((tuple(value.shape),value.dtype)).encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def load_for_inference(weights_path, protocol_path, *, device='cpu'):
    """Reload a selected local checkpoint; never fall back to 800 preprocessing.

    This verifies model identity and construction only. A caller exporting test
    results must separately enforce the final experiment-selection decision.
    """
    weights_path,protocol_path = Path(weights_path),Path(protocol_path)
    protocol = json.loads(protocol_path.read_text())
    checkpoint = torch.load(weights_path,map_location='cpu',weights_only=True)
    if protocol['status'] != 'frozen' or file_sha(protocol_path) != checkpoint['protocol_sha256']:
        raise ValueError('Selected checkpoint does not match the frozen protocol')
    if checkpoint['source_sha256'] != protocol['source_sha256']:
        raise ValueError('Checkpoint source identity differs from protocol')
    root = Path(__file__).resolve().parents[1]
    for name,expected in checkpoint['source_sha256'].items():
        path = Path(name) if Path(name).is_absolute() else root/name
        if file_sha(path) != expected:
            raise ValueError(f'Training source changed: {name}')
    method = protocol.get('intervention',{}).get('method','native')
    saved = checkpoint['model_metadata']
    resolution = saved['preprocess']['shortest_edge']
    if (method not in FAMILIES or saved['experiment_family'] != FAMILIES[method]
            or resolution not in (800,1024)
            or protocol.get('selected_resolution',resolution) != resolution):
        raise ValueError('Method/resolution differs from selected checkpoint')
    adapter,metadata = build_trainable(protocol['checkpoint'],method=method,resolution=resolution,
                                       seed=protocol['training']['seed'],device=device)
    if metadata != saved:
        differing = sorted(k for k in set(metadata)|set(saved) if metadata.get(k) != saved.get(k))
        raise ValueError(f'Reconstructed model metadata differs: {differing}')
    adapter.load_state_dict(checkpoint['model'],strict=True)
    actual = tensor_digest(adapter.state_dict())
    if actual != checkpoint['selection']['model_sha256']:
        raise ValueError('Loaded tensor values differ from the selected validation model')
    adapter.eval()
    adapter.requires_grad_(False)
    receipt = {'checkpoint':str(weights_path.resolve()),'checkpoint_sha256':file_sha(weights_path),
        'protocol':str(protocol_path.resolve()),'protocol_sha256':checkpoint['protocol_sha256'],
        'method':method,'resolution':resolution,'epoch':checkpoint['epoch'],'model_sha256':actual,
        'model_metadata':metadata,'whole_image_inference':True,'test_inference_performed':False,
        'loader_source_sha256':file_sha(Path(__file__))}
    return adapter,receipt
