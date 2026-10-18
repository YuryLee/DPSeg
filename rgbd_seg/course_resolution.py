"""Resolution experiment helpers; historical DA-Fusion sources stay frozen."""
from __future__ import annotations

import types

import torch

from .course_dafusion import build_trainable as build_original


def local_num_masks(self, class_labels, device):
    """Explicit native microbatch normalization, independent of Accelerate state.

    DDP averages gradients of equal-image microbatches. This experiment compares
    two resolutions using this identical rule; it does not claim equivalence to
    the historical batch-four, single-GPU loss normalization.
    """
    return torch.tensor(sum(map(len, class_labels)), dtype=torch.float32,
                        device=device).clamp(min=1)


def rank_epoch_order(order, *, rank, world_size, batch_size):
    """Disjoint equal-sized microbatches, retaining the global image order.

    No padding, duplicated samples, or dropped tail is permitted. The course
    split is divisible by both the distributed microbatch and effective batch.
    """
    if world_size < 1 or batch_size < 1 or not 0 <= rank < world_size:
        raise ValueError('Invalid rank/world/batch')
    stride = world_size * batch_size
    if len(order) % stride:
        raise ValueError('Uneven distributed tail requires an explicit new protocol')
    return [value for start in range(0, len(order), stride)
            for value in order[start + rank*batch_size:start + (rank+1)*batch_size]]


def heterogeneous_rank_order(order, *, rank, batch_sizes):
    """Disjoint rank chunks with explicitly specified per-GPU batch sizes."""
    if (not batch_sizes or any(type(b) is not int or b < 1 for b in batch_sizes)
            or not 0 <= rank < len(batch_sizes)):
        raise ValueError('Invalid batch layout')
    stride = sum(batch_sizes)
    if len(order) % stride:
        raise ValueError('Uneven tail requires an explicit new protocol')
    offset = sum(batch_sizes[:rank])
    return [value for start in range(0, len(order), stride)
            for value in order[start+offset:start+offset+batch_sizes[rank]]]


def rank_batches(order, *, rank, batch_sizes):
    """Image-exact batches, including proportional nonempty tail batches.

    Remainder images go to ranks with largest fractional quota (rank breaks
    ties). No padded/duplicate samples enter training. Tiny tails that cannot
    provide one example per rank are rejected before training.
    """
    if (not batch_sizes or any(type(b) is not int or b < 1 for b in batch_sizes)
            or not 0 <= rank < len(batch_sizes)):
        raise ValueError('Invalid batch layout')
    total = sum(batch_sizes)
    result = []
    for start in range(0, len(order), total):
        chunk = order[start:start+total]
        counts = [len(chunk)*b//total for b in batch_sizes]
        priorities = sorted(range(len(counts)), key=lambda r: (-(len(chunk)*batch_sizes[r] % total), r))
        for r in priorities[:len(chunk)-sum(counts)]:
            counts[r] += 1
        if min(counts) == 0:
            raise ValueError('Tail must provide a nonempty batch on every GPU')
        offset = sum(counts[:rank])
        result.append(chunk[offset:offset+counts[rank]])
    return result


def build_trainable(checkpoint, *, device='cpu', seed=17, resolution=1024):
    if resolution not in (800, 1024):
        raise ValueError('This ablation only permits 800 and 1024 square input')
    adapter, metadata = build_original(checkpoint, device=device, seed=seed)
    adapter.short_edge = resolution
    adapter.max_size = 1333
    encoder = adapter.model.model.pixel_level_module.encoder
    bias_modules = []
    for stage, fusion in enumerate(encoder.fusions):
        found = []
        for name, module in fusion.named_modules():
            if module.__class__.__name__ == 'ContinuousPositionBias':
                # Raising the threshold globally would also enable stage 2 at
                # 800! Preserve the historical [off, off, on, on] explicitly.
                module.max_pairs = 0 if stage < 2 else 8_000_000
                expected_active = stage >= 2
                def verify_bias(module, inputs, output, expected=expected_active):
                    if (output is not None) != expected:
                        raise RuntimeError('Actual RPB activation differs from the resolution contract')
                module.register_forward_hook(verify_bias)
                found.append({'name': name, 'max_pairs': module.max_pairs})
        if len(found) != 4:
            raise RuntimeError(f'Unexpected position-bias modules at stage {stage}: {found}')
        bias_modules.append(found)
    adapter.model.criterion.get_num_masks = types.MethodType(
        local_num_masks, adapter.model.criterion)
    metadata.update(model_id=f'DA-Fusion-course-Swin-Tiny-DDP-resolution-{resolution}',
                    experiment_family='resolution_ddp_v1')
    metadata['preprocess']['shortest_edge'] = resolution
    metadata['preprocess']['hf_padding_limitation'] = (
        f'Course square input becomes {resolution} square, divisible by 32; no padding.')
    metadata['fusion'].pop('rpb_max_pairs')
    metadata['fusion']['rpb_stage_max_pairs'] = [0, 0, 8_000_000, 8_000_000]
    metadata['fusion']['rpb_active_at_input'] = [False, False, True, True]
    metadata['fusion']['rpb_modules'] = bias_modules
    metadata['loss']['normalization'] = 'Native local microbatch mask count; DDP mean across ranks'
    metadata['differences_from_paper'].append(
        'Resolution ablation explicitly fixes RPB active stages; DDP local-microbatch mean losses')
    adapter.metadata = metadata
    return adapter, metadata
