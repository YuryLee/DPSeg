"""UOAIS point-head integration; established point/refinement sources stay unchanged."""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.nn import functional as F


def point_loss_sums(predictions, coarse, truth, *, real_instances):
    """Per-instance BCE sums, including a zero-loss dummy for an empty rank.

    The caller divides by the global matched-instance count and compensates for
    DDP's rank averaging. Empty images still count toward training coverage.
    """
    if real_instances < 0 or (real_instances and real_instances != len(coarse)):
        raise ValueError('Matched-instance count differs from point tensors')
    if truth.shape != coarse.shape or coarse.ndim != 3 or coarse.shape[1] != 1:
        raise ValueError('Expected aligned N x 1 x P coarse and target tensors')
    sums = {}
    for name, delta in predictions.items():
        if delta.shape != coarse.shape:
            raise ValueError('Point residual and coarse-logit dimensions differ')
        values = F.binary_cross_entropy_with_logits(coarse+delta, truth, reduction='none')
        sums[name] = values.mean((1, 2)).sum() if real_instances else values.sum()*0.
    return sums


def empty_point_batch(device):
    return (torch.zeros(1, 352, 784, device=device), torch.zeros(1, 1, 784, device=device),
            torch.zeros(1, 9, 784, device=device), torch.zeros(1, 1, 784, device=device))


def load_uoais_base(directory, *, engineering=False, device='cpu'):
    """Load only this experiment's base, rejecting course or unfinished selections."""
    from tools.train_uoais_benchmark import source_hashes, sha
    from .uoais_benchmark_training import build_model
    directory = Path(directory).resolve()
    run = json.loads((directory/'run.json').read_text())
    protocol_path = directory/'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    if run['family'] != 'dpseg' or sha(protocol_path) != run['protocol_sha256']:
        raise ValueError('Point training requires an identified UOAIS DPSeg base protocol')
    if engineering:
        if run['scope'] != 'engineering_smoke' or run['status'] != 'smoke_passed':
            raise ValueError('Engineering mode requires a completed engineering base')
        checkpoint_path = directory/'last.pt'
    else:
        if (run['scope'] != 'full_train_val' or run['status'] != 'training_completed'
                or run['state']['epoch'] != protocol['training']['epochs']
                or len(run['state']['validation']) != protocol['training']['epochs']):
            raise ValueError('Point training waits for the completed full UOAIS base run')
        checkpoint_path = directory/'best.pt'
    source = source_hashes()
    if source != run['source_sha256']:
        raise ValueError('Base model/training sources changed')
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    if (state['scope'] != run['scope'] or state['family'] != 'dpseg'
            or state['source_sha256'] != source or state['protocol_sha256'] != run['protocol_sha256']):
        raise ValueError('Base checkpoint provenance differs')
    if not engineering:
        if state['selection']['scope'] != 'full_validation' or state['selection']['images'] != 5000:
            raise ValueError('Base was not selected on the complete UOAIS validation split')
    model, metadata = build_model('dpseg', protocol['models']['dpseg']['pretrained'], protocol['training']['seed'])
    model.load_state_dict(state['model'], strict=True)
    model.to(device).eval().requires_grad_(False)
    receipt = {'base_run': str(directory), 'checkpoint': str(checkpoint_path),
               'checkpoint_sha256': sha(checkpoint_path), 'protocol_sha256': run['protocol_sha256'],
               'scope': run['scope'], 'metadata': metadata,
               'selection': state.get('selection'), 'test_used': False}
    return model, protocol, receipt
