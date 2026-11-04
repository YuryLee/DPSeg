"""Index original OSD/OCID evaluation data without altering its annotations.

This module does not download data or infer camera calibration. Calibration and
depth units are explicit inputs, including whether they are measured or assumed.
The manifest is a receipt for actual files, not evidence of paper reproduction.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_array(path):
    with Image.open(path) as image:
        return np.array(image, copy=True)


def validate_calibration(calibration):
    required = {'camera_K', 'depth_scale_m', 'source', 'status'}
    if not required <= set(calibration):
        raise ValueError(f'Calibration requires {sorted(required)}')
    matrix = np.asarray(calibration['camera_K'], dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or matrix[0, 0] <= 0 or matrix[1, 1] <= 0
            or not np.allclose(matrix[2], [0, 0, 1])
            or abs(np.linalg.det(matrix)) < 1e-8):
        raise ValueError('Invalid camera intrinsics')
    scale = calibration['depth_scale_m']
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Depth scale must be finite positive metres per raw unit')
    if calibration['status'] not in {'provided', 'recovered', 'assumed'}:
        raise ValueError('Calibration status must be provided, recovered or assumed')
    if not isinstance(calibration['source'], str) or not calibration['source'].strip():
        raise ValueError('Calibration provenance is required')
    return calibration


def _triples(rgb_dir, depth_dir, label_dir):
    folders = [Path(rgb_dir), Path(depth_dir), Path(label_dir)]
    if any(not p.is_dir() for p in folders):
        raise FileNotFoundError(f'Missing RGB/depth/original annotation folder: {folders}')
    files = [{p.name: p for p in folder.glob('*.png')} for folder in folders]
    if not files[0] or any(set(v) != set(files[0]) for v in files[1:]):
        raise ValueError(f'RGB, depth and annotation filenames must match exactly: {folders}')
    for name in sorted(files[0]):
        yield tuple(v[name] for v in files)


def build_manifest(dataset, root, calibration, *, expected_images=None):
    """Decode and hash every image; retain all original instances and empty scenes.

    OCID background conventions are selected from the scene path: floor 0/1,
    table 0/1/2. No label-derived prediction filter or test-dependent correction
    is applied. OSD always reads ``annotation``, never ``annotation_fixed``.
    """
    root = Path(root).resolve()
    validate_calibration(calibration)
    groups = []
    if dataset == 'osd':
        groups.append((root / 'image_color', root / 'disparity', root / 'annotation', [0], 'osd'))
    elif dataset == 'ocid':
        for subset in ('ARID20', 'YCB10', 'ARID10'):
            for rgb_dir in sorted((root / subset).rglob('rgb')):
                if not rgb_dir.is_dir():
                    continue
                relative = rgb_dir.parent.relative_to(root)
                scene_types = set(relative.parts) & {'floor', 'table'}
                if len(scene_types) != 1:
                    raise ValueError(f'Cannot determine OCID background labels: {relative}')
                scene_type = next(iter(scene_types))
                background = [0, 1] if scene_type == 'floor' else [0, 1, 2]
                groups.append((rgb_dir, rgb_dir.parent / 'depth', rgb_dir.parent / 'label',
                               background, str(relative)))
    else:
        raise ValueError('Only original OSD and OCID layouts are implemented')
    records = []
    for rgb_dir, depth_dir, label_dir, background, scene in groups:
        for rgb_path, depth_path, label_path in _triples(rgb_dir, depth_dir, label_dir):
            rgb, depth, labels = (read_array(p) for p in (rgb_path, depth_path, label_path))
            if (rgb.ndim != 3 or rgb.shape[2] != 3 or depth.ndim != 2
                    or labels.ndim != 2 or rgb.shape[:2] != depth.shape or labels.shape != depth.shape):
                raise ValueError(f'RGB/depth/label dimensions differ: {rgb_path}')
            if depth.dtype.kind != 'u' or depth.dtype.itemsize != 2:
                raise ValueError(f'Expected raw uint16 metric depth, not a display image: {depth_path}')
            if labels.dtype.kind not in 'ui' or np.any(labels < 0):
                raise ValueError(f'Expected nonnegative instance labels: {label_path}')
            height, width = depth.shape
            if 'image_size' in calibration and calibration['image_size'] != [height, width]:
                raise ValueError('Calibration resolution differs from original image dimensions')
            ids, counts = np.unique(labels, return_counts=True)
            instances = [{'label_id': int(i), 'area': int(n)} for i, n in zip(ids, counts)
                         if int(i) not in background]
            records.append({
                'id': f'{dataset}/{rgb_path.relative_to(root).as_posix()}',
                'scene_id': scene, 'split': 'test', 'height': height, 'width': width,
                'rgb': str(rgb_path), 'depth': str(depth_path), 'instances': str(label_path),
                'background_labels': background, 'objects': instances,
                'sha256': {name: file_sha256(path) for name, path in
                           [('rgb', rgb_path), ('depth', depth_path), ('instances', label_path)]},
            })
    if not records:
        raise ValueError('No complete evaluation samples found')
    if len({r['id'] for r in records}) != len(records):
        raise ValueError('Duplicate image identifiers')
    if expected_images is not None and len(records) != expected_images:
        raise ValueError(f'Expected {expected_images} images, found {len(records)}; resolve version difference first')
    return {
        'schema': 'uois-benchmark-data-v1', 'dataset': dataset, 'root': str(root),
        'status': 'indexed_and_decoded', 'calibration': calibration,
        'images': len(records), 'objects': sum(len(r['objects']) for r in records),
        'expected_images': expected_images,
        'annotation_policy': 'original annotation files; documented background only; no object removal, correction or resize',
        'scope': 'external test only; no training or model selection on these images',
        'samples': records,
    }


class BenchmarkDataset:
    def __init__(self, manifest_path, *, verify_hashes=True):
        self.path = Path(manifest_path)
        self.manifest = json.loads(self.path.read_text())
        if self.manifest.get('schema') != 'uois-benchmark-data-v1':
            raise ValueError('Unsupported benchmark manifest')
        self.calibration = validate_calibration(self.manifest['calibration'])
        self.records = self.manifest['samples']
        if not self.records or len(self.records) != self.manifest['images']:
            raise ValueError('Incomplete manifest')
        if len({r['id'] for r in self.records}) != len(self.records):
            raise ValueError('Duplicate sample IDs')
        if any(r['split'] != 'test' for r in self.records):
            raise ValueError('This loader accepts external test data only')
        if verify_hashes:
            for record in self.records:
                for name in ('rgb', 'depth', 'instances'):
                    if file_sha256(record[name]) != record['sha256'][name]:
                        raise ValueError(f'Dataset file changed after indexing: {record[name]}')

    def __len__(self):
        return len(self.records)

    def inference_sample(self, index):
        """Return only the four model inputs; do not open annotations here."""
        import torch
        record = self.records[index]
        rgb, raw = read_array(record['rgb']), read_array(record['depth'])
        depth = raw.astype(np.float32) * self.calibration['depth_scale_m']
        invalid = self.calibration.get('invalid_depth_codes', [0])
        valid = np.isfinite(depth) & (depth > 0) & ~np.isin(raw, invalid)
        depth[~valid] = 0
        return {'image': torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                'depth_m': torch.from_numpy(depth)[None],
                'depth_valid': torch.from_numpy(valid)[None],
                'camera_K': torch.tensor(self.calibration['camera_K'], dtype=torch.float32)}

    def ground_truth(self, index):
        record = self.records[index]
        labels = read_array(record['instances']).astype(np.int32)
        labels[np.isin(labels, record['background_labels'])] = 0
        return labels
