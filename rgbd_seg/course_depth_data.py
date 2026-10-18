"""Course RGB-D inputs from ZIP, without support masks or scene-label features.

Depth is camera-axis Z in metres, not ray length or an estimated depth image.
The optional model helper follows the official DA-Fusion XYZ encoding. It does
not implement HHA or infer gravity.
Targets retain original areas; model adapters own target resizing.
"""
from __future__ import annotations

import io
import math

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from .course_training_data import CourseTrainingDataset, collate_fn


def camera_intrinsics(record):
    """Original pinhole K, integer pixel centres; calibration only, no GT."""
    camera = record.get('camera', {})
    if camera.get('pixel_center_convention') != 'integer':
        raise ValueError('Expected explicit integer pixel-centre convention')
    if camera.get('resolution') != [record['width'], record['height']]:
        raise ValueError('Camera resolution disagrees with RGB/depth metadata')
    k = torch.as_tensor(camera.get('K'), dtype=torch.float32)
    if (k.shape != (3, 3) or not torch.isfinite(k).all()
            or k[0, 0] <= 0 or k[1, 1] <= 0
            or not torch.equal(k[2], torch.tensor([0., 0., 1.]))
            or k[0, 1] != 0 or k[1, 0] != 0):
        raise ValueError('Expected finite, positive-focal-length pinhole K')
    return k.clone()


def load_depth_record(archive_handle, record):
    """Decode one aligned uint16 PNG to depth[1,H,W], valid[1,H,W]."""
    if record.get('depth_semantics') != 'camera_axis_z':
        raise ValueError('Depth must explicitly use camera_axis_z semantics')
    scale = record.get('depth_scale_m')
    invalid = record.get('depth_invalid_code')
    if type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0:
        raise ValueError('Depth scale must be finite and positive')
    if type(invalid) is not int or not 0 <= invalid <= 65535:
        raise ValueError('Expected explicit uint16 invalid code')
    with Image.open(io.BytesIO(archive_handle.read(record['depth_file_name']))) as image:
        raw = np.array(image)
    if raw.dtype != np.uint16 or raw.shape != (record['height'], record['width']):
        raise ValueError('Expected aligned original-resolution uint16 depth PNG')
    valid = (raw != invalid) & (raw > 0)
    with np.errstate(over='ignore', invalid='ignore'):
        depth = raw.astype(np.float32) * scale
    if not np.isfinite(depth[valid]).all():
        raise ValueError('Metric depth overflows float32')
    depth[~valid] = 0
    return torch.from_numpy(depth).unsqueeze(0), torch.from_numpy(valid).unsqueeze(0)


def _check_depth(depth, valid):
    if depth.ndim != 3 or depth.shape[0] != 1 or valid.shape != depth.shape:
        raise ValueError('Depth and valid must have matching [1,H,W] shapes')
    if not depth.is_floating_point() or valid.dtype != torch.bool:
        raise ValueError('Depth must be floating point and valid boolean')
    if (depth.device != valid.device or not torch.isfinite(depth).all()
            or torch.any(depth[valid] <= 0) or torch.any(depth[~valid] != 0)):
        raise ValueError('Invalid depth values or device mismatch')


def transform_depth(depth, valid, camera_k, *, horizontal_flip=False, size=None):
    """Apply aligned flip then nearest resize, and return the pixel-to-ray K.

    K' = A @ K. Flipping yields a negative fx: XYZ remains in the ORIGINAL
    camera coordinate system. Resizing follows the align_corners=False
    half-pixel affine. Depth/valid use nearest sampling, never interpolate zeros.
    This transforms only depth/calibration; callers transform RGB/targets with
    the same geometry. No clipping, filling, or scene-dependent normalization.
    """
    _check_depth(depth, valid)
    k = torch.as_tensor(camera_k, dtype=depth.dtype, device=depth.device).clone()
    if k.shape != (3, 3) or not torch.isfinite(k).all() or torch.linalg.det(k) == 0:
        raise ValueError('Camera K must be a finite invertible 3x3 matrix')
    h, w = depth.shape[-2:]
    if horizontal_flip:
        depth, valid = depth.flip(-1), valid.flip(-1)
        a = k.new_tensor([[-1, 0, w-1], [0, 1, 0], [0, 0, 1]])
        k = a @ k
    if size is not None:
        if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
            raise ValueError('size must be positive integer (height, width)')
        nh, nw = size
        depth = F.interpolate(depth[None], size=(nh, nw), mode='nearest')[0]
        valid = F.interpolate(valid[None].float(), size=(nh, nw), mode='nearest')[0].bool()
        sx, sy = nw/w, nh/h
        a = k.new_tensor([[sx, 0, (sx-1)/2], [0, sy, (sy-1)/2], [0, 0, 1]])
        k = a @ k
    return depth, valid, k


def depth_to_xyz(depth, valid, camera_k):
    """Project metric Z to camera XYZ[3,H,W]; invalid points are exactly zero.

    Returns physical metres, with no normalization or artificial RGB encoding.
    Accepts the reflected K from transform_depth; no extrinsics/GT are used.
    """
    _check_depth(depth, valid)
    k = torch.as_tensor(camera_k, dtype=depth.dtype, device=depth.device)
    if k.shape != (3, 3) or not torch.isfinite(k).all() or torch.linalg.det(k) == 0:
        raise ValueError('Camera K must be a finite invertible 3x3 matrix')
    h, w = depth.shape[-2:]
    y, x = torch.meshgrid(torch.arange(h, device=depth.device, dtype=depth.dtype),
                          torch.arange(w, device=depth.device, dtype=depth.dtype), indexing='ij')
    rays = torch.linalg.solve(k, torch.stack((x, y, torch.ones_like(x))).flatten(1))
    if torch.any(rays[2].abs() < 1e-12):
        raise ValueError('Camera rays have zero Z')
    xyz = (rays / rays[2:3] * depth.flatten(1)).reshape(3, h, w)
    return xyz.masked_fill(~valid.expand_as(xyz), 0)


def standardize_xyz(xyz, valid):
    """Official DA-Fusion median/p95 normalization, in Torch float32.

    XYZ is [3,H,W]; valid is boolean [1,H,W] or [H,W]. Component-wise
    medians use the mean of the two middle values for even populations, matching
    numpy.median. Radius percentile uses linear interpolation, matching the
    author's numpy.percentile(..., 95). Invalid points remain exactly zero.
    This is an input encoding, not a dataset-fitted normalization.
    """
    if xyz.ndim != 3 or xyz.shape[0] != 3 or not xyz.is_floating_point():
        raise ValueError('XYZ must be floating point [3,H,W]')
    if valid is None:
        return torch.zeros_like(xyz, dtype=torch.float32)
    if valid.shape == (1, *xyz.shape[1:]):
        valid = valid[0]
    if valid.shape != xyz.shape[1:] or valid.dtype != torch.bool or valid.device != xyz.device:
        raise ValueError('Validity must be aligned boolean data on the XYZ device')
    with torch.autocast(device_type=xyz.device.type, enabled=False):
        xyz = xyz.float()
        out = torch.zeros_like(xyz)
        points = xyz[:, valid].T
        if points.numel() == 0:
            return out
        if not torch.isfinite(points).all():
            raise ValueError('Valid XYZ points must be finite')
        center = torch.quantile(points, .5, dim=0, interpolation='linear')
        radius = torch.linalg.vector_norm(points - center, dim=1)
        scale = torch.quantile(radius, .95, interpolation='linear') + 1e-6
        out[:, valid] = ((points - center) / scale).T
        return out


def model_depth_channels(sample, size, device=None):
    """Return official XYZ+valid float32 [4,H,W] for the depth adapter.

    Input must be the dataset's ORIGINAL resolution sample, already flipped if
    requested. Backproject its raw metric Z; bilinearly resize XYZ (no antialias),
    nearest-resize validity, reflect X when K has negative fx, THEN standardize.
    Negative fx represents the dataset's horizontal flip, not arbitrary camera
    calibration. The official mapper also reflects X after its spatial flip.
    Do not call resize_rgbd_sample first: resizing metric Z and backprojecting
    does not reproduce resizing XYZ. No RGB, annotation or target statistic is
    used to construct these channels.
    """
    if len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError('size must be positive integer (height, width)')
    depth = sample['depth_m'].to(device=device, dtype=torch.float32)
    valid = sample['depth_valid'].to(device=depth.device)
    k = sample['camera_K'].to(device=depth.device, dtype=torch.float32)
    if sample['image'].shape[-2:] != depth.shape[-2:]:
        raise ValueError('RGB and depth are not aligned')
    with torch.autocast(device_type=depth.device.type, enabled=False):
        xyz = depth_to_xyz(depth, valid, k)
        xyz = F.interpolate(xyz[None], size=size, mode='bilinear',
                            align_corners=False, antialias=False)[0]
        valid = F.interpolate(valid[None].float(), size=size, mode='nearest')[0].bool()
        if k[0, 0] < 0:
            xyz = xyz * xyz.new_tensor([-1., 1., 1.])[:, None, None]
        xyz = standardize_xyz(xyz, valid)
        return torch.cat((xyz, valid.float()), dim=0)


def resize_rgbd_sample(sample, size):
    """Adapter helper: bilinear RGB, nearest depth/valid, matching calibration.

    No target transformation here; independent instance masks must separately
    use nearest resize, and original target area must stay unchanged.
    """
    if sample['image'].shape[-2:] != sample['depth_m'].shape[-2:]:
        raise ValueError('RGB and depth are not aligned')
    depth, valid, k = transform_depth(sample['depth_m'], sample['depth_valid'], sample['camera_K'], size=size)
    image = F.interpolate(sample['image'][None], size=size, mode='bilinear', align_corners=False, antialias=False)[0]
    return {**sample, 'image': image, 'depth_m': depth, 'depth_valid': valid, 'camera_K': k}


def load_inference_sample(archive, image_id):
    """Read RGB/depth/K by official ID from an open CourseArchive, without GT.

    No annotations, support_rle, CAD identity, target centre, or extrinsics are
    returned or used. The existing CourseArchive owns the read-only ZIP handle.
    """
    if type(image_id) is not int:
        raise ValueError('Expected integer official image_id')
    try:
        index = archive.image_ids.index(image_id)
    except ValueError as exc:
        raise ValueError(f'Unknown official image_id: {image_id}') from exc
    record = archive.images[index]
    image = torch.from_numpy(np.array(archive.load_rgb(index), dtype=np.uint8, copy=True)).permute(2, 0, 1).float().div_(255)
    depth, valid = load_depth_record(archive._image_archive(), record)
    return {'image': image, 'depth_m': depth, 'depth_valid': valid, 'camera_K': camera_intrinsics(record)}


class CourseDepthTrainingDataset(CourseTrainingDataset):
    """Existing train/val RGB+targets plus depth; original 1024 size retained.

    __getitem__ -> (sample, target), sample keys are image, depth_m, depth_valid,
    camera_K only. An explicit seed is required for augmented training, so the
    inherited RGB/mask flip can be replayed without consuming a second RNG draw.
    """
    def __init__(self, archive, split='train', augment=True, seed=17):
        if split == 'train' and augment and seed is None:
            raise ValueError('Aligned augmented depth requires an explicit seed')
        super().__init__(archive, split=split, augment=augment, seed=seed)
        for record in self.images:
            camera_intrinsics(record)
            if record.get('depth_semantics') != 'camera_axis_z':
                raise ValueError('Expected camera_axis_z depth')

    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        record = self.images[index]
        depth, valid = load_depth_record(self._handle(), record)
        depth, valid, k = transform_depth(depth, valid, camera_intrinsics(record),
                                         horizontal_flip=self._flip_for(record['id']))
        return {'image': image, 'depth_m': depth, 'depth_valid': valid, 'camera_K': k}, target

    def summary(self):
        result = super().summary()
        result.update(rgb_only=False, depth_input={
            'representation': 'raw metric camera-axis Z; encoding/fusion selected by adapter',
            'scale_m_values': sorted({r['depth_scale_m'] for r in self.images}),
            'invalid_codes': sorted({r['depth_invalid_code'] for r in self.images}),
            'invalid_output': 'depth=0; separate boolean validity mask',
            'geometry': 'same original RGB/target flip; K transformed by pixel affine',
            'resize': 'adapter-owned; nearest depth and valid, bilinear RGB',
            'calibration': 'camera.K only; no extrinsics, support masks, gravity or scene-label inputs',
            'XYZ_optional': 'physical camera coordinates; flipped K has negative fx',
            'DAFusion_model_XYZ': 'original XYZ; bilinear resize; nearest valid; flipped X sign; valid median/p95+1e-6; concat valid',
            'HHA': 'not implemented; no explicit gravity/ground-plane metadata supplied'})
        return result
