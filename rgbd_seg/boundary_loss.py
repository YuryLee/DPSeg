"""Training-only BACE arithmetic for class-selected ROI mask logits.

Independent implementation of the loss equation, checked against Pan and Pei's
BARD-ERA author code (``mask_boundary_aware_loss``, default scale=4):
https://github.com/PANpinchi/BARD-ERA/tree/942b5869d0101a39ea53483773e8d90f7252244d
Paper: https://arxiv.org/html/2504.19643v1#S3.SS3

No third-party implementation is copied here. The author package metadata names
Apache 2.0, but no top-level license text was located; its loss file carries an
OpenMMLab notice.
See reports/v5/boundary_method_notes.md for provenance and adaptation limits.

Max pooling is nonlinear. The upsampled block maximum is not a linear projector.
The refined quantity below is a training logit, not an inference probability.
Importing this module does not replace torchvision functions or modify a model.
"""
from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F


def bace_components(logits: Tensor, targets: Tensor, scale: int = 4) -> dict[str, Tensor]:
    """Return graph-connected ``{'base': BCE, 'aux': BACE}`` scalar losses.

    Inputs are aligned N×H×W tensors after positive ROI/class selection and
    target projection. Targets may be soft values in [0, 1], allowing at most
    eight machine epsilons of ROIAlign roundoff; values are neither clamped nor
    thresholded. BACE uses ``q = block_max(targets) + logits - block_max(logits)``
    in BCEWithLogits, where block_max pools with kernel=stride=scale and repeats
    each maximum with nearest-neighbor interpolation.

    Scale=4 matches the author's default. Adding base and aux with unit weights
    matches the published default loss. Active BACE requires H/W divisible by scale;
    there is no implicit padding or adaptive change of the author's operator.
    Half/bfloat inputs are computed in float32. Empty ROIs return a graph-linked
    zero. Both returned losses retain gradients; detach only when logging.
    """
    return _components(logits, targets, scale=scale, auxiliary_enabled=True)


def _components(logits: Tensor, targets: Tensor, *, scale: int, auxiliary_enabled: bool) -> dict[str, Tensor]:
    if not isinstance(logits, Tensor) or not isinstance(targets, Tensor):
        raise TypeError('logits and targets must be tensors; targets are training supervision')
    if logits.ndim != 3 or logits.shape != targets.shape:
        raise ValueError('Expected matching N×H×W logits and targets')
    if not logits.is_floating_point() or targets.is_complex():
        raise TypeError('Logits must be real floating point; targets must be real')
    if logits.device != targets.device:
        raise ValueError('Logits and targets must be on the same device')
    if isinstance(scale, bool) or not isinstance(scale, int) or scale < 1:
        raise ValueError('scale must be a positive integer')
    if logits.shape[-2] == 0 or logits.shape[-1] == 0:
        raise ValueError('ROI height and width must be positive')
    if not torch.isfinite(logits).all() or not torch.isfinite(targets).all():
        raise ValueError('Logits and targets must be finite')
    # CUDA ROIAlign can produce 1 + one float32 epsilon from binary GT. Keep
    # these soft values exactly as Torchvision's BCE does. Half/bfloat targets
    # use float32 tolerance after promotion, not their much coarser epsilon.
    range_dtype = torch.float32 if targets.dtype in (torch.float16, torch.bfloat16) else targets.dtype
    tolerance = 8 * torch.finfo(range_dtype).eps if targets.is_floating_point() else 0.0
    if ((targets < -tolerance) | (targets > 1 + tolerance)).any():
        raise ValueError(f'Targets must lie in [0, 1] within floating point tolerance {tolerance:g}')

    work = logits.float() if logits.dtype in (torch.float16, torch.bfloat16) else logits
    target = targets.detach().to(dtype=work.dtype)
    zero = work.reshape(-1)[:1].sum() * 0
    base = zero
    auxiliary = zero
    if work.shape[0]:
        base = F.binary_cross_entropy_with_logits(work, target)
        if auxiliary_enabled:
            if any(size % scale != 0 for size in work.shape[-2:]):
                raise ValueError('Active BACE requires ROI height and width divisible by scale')

            def block_max(x: Tensor) -> Tensor:
                coarse = F.max_pool2d(x[:, None], kernel_size=scale, stride=scale)
                return F.interpolate(coarse, scale_factor=scale, mode='nearest')[:, 0]

            refined_logits = block_max(target) + (work - block_max(work))
            if not torch.isfinite(refined_logits).all():
                raise ValueError('Refined logits overflowed; use a wider floating point dtype')
            auxiliary = F.binary_cross_entropy_with_logits(refined_logits, target)
    return {'base': base, 'aux': auxiliary}


def bace_loss(
    logits: Tensor,
    targets: Tensor,
    *,
    scale: int = 4,
    alpha: float = 1.0,
    beta: float = 1.0,
    return_components: bool = False,
) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
    """Weighted convenience API; alpha/beta are project names, both default 1.

    Beta=0 skips pooling and exactly recovers the original BCE when alpha=1.
    Optional logging components are detached. Use ``bace_components`` when the
    caller needs both component tensors with their gradient connections intact.
    """
    if any(isinstance(w, bool) or not isinstance(w, (int, float)) or not math.isfinite(w) or w < 0
           for w in (alpha, beta)):
        raise ValueError('alpha and beta must be finite nonnegative numbers')
    components = _components(logits, targets, scale=scale, auxiliary_enabled=beta != 0)
    base, auxiliary = components['base'], components['aux']
    total = alpha * base + beta * auxiliary
    if return_components:
        return total, {'bce': base.detach(), 'bace': auxiliary.detach(), 'total': total.detach()}
    return total
