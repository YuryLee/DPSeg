"""Original-resolution UOAIS-Sim train/val inputs and independent visible masks.

Layout follows the pinned DA-Fusion register_uois.py/uois_json.py. No test split,
amodal fallback, augmentation, resize, label-map flattening or camera estimation
is implemented here. The runner owns any shared RGB/depth/target geometry.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np
from pycocotools import mask as mask_utils
import torch
from torch.utils.data import Dataset

from .uois_benchmark_data import read_array, validate_calibration


def _integer(value, name):
    if type(value) is not int or not 0 <= value <= np.iinfo(np.int64).max:
        raise ValueError(f'{name} must be a nonnegative int64-compatible integer')
    return value


def _input_path(split_root, name):
    relative = PurePosixPath(name) if isinstance(name, str) else None
    if (relative is None or not name or relative.is_absolute() or '..' in relative.parts
            or '\\' in name):
        raise ValueError('Expected a safe relative path inside the declared split')
    path = (split_root / name).resolve()
    if not path.is_relative_to(split_root) or not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f'Missing, empty or wrong-split input: {path}')
    return path


def load_uoais_splits(root):
    """Validate both official JSON indices and file references without decoding images.

    Annotation IDs must be unique; IDs may restart in the other split. The
    author's JSON repeats each image once per instance with a fresh timestamp.
    Collapse those rows only when every field except date_captured agrees.
    Resolved paths and file inodes must be distinct across train/val (including
    symlink/hardlink aliases). This does not prove different scenes or detect
    duplicated image content in independent copied files. Image/mask pixels are
    checked on dataset access, rather than claiming an unperformed full decode.
    """
    root = Path(root).resolve()
    result, receipts, file_owners = {}, {}, {}
    for split in ('train', 'val'):
        split_root = (root / split).resolve()
        if split_root != root / split or not split_root.is_dir():
            raise ValueError(f'Expected a distinct original {split} directory')
        annotation_path = root / 'annotations' / f'coco_anns_uoais_sim_{split}.json'
        if not annotation_path.resolve().is_relative_to(root):
            raise ValueError('Annotation file must remain inside the dataset root')
        payload = annotation_path.read_bytes()
        document = json.loads(payload)
        if document.get('split', split) != split:
            raise ValueError('Annotation metadata split mismatch')
        categories = [_integer(c['id'], 'category ID') for c in document['categories']]
        if len(set(categories)) != len(categories) or not categories:
            raise ValueError('Categories must have unique IDs')
        records, image_metadata = {}, {}
        repeated_images = 0
        for image in document['images']:
            image_id = _integer(image['id'], 'image ID')
            metadata = {key: value for key, value in image.items() if key != 'date_captured'}
            if image_id in records:
                if metadata != image_metadata[image_id]:
                    raise ValueError(f'Conflicting repeated image metadata within {split}: {image_id}')
                repeated_images += 1
                continue
            if image.get('split', split) != split:
                raise ValueError('Image metadata split mismatch')
            h, w = _integer(image['height'], 'height'), _integer(image['width'], 'width')
            if min(h, w) == 0:
                raise ValueError('Image dimensions must be positive')
            paths = {}
            for field in ('file_name', 'depth_file_name'):
                path = _input_path(split_root, image[field])
                identity = (path.stat().st_dev, path.stat().st_ino)
                if identity in file_owners:
                    raise ValueError(f'Duplicate or shared input file: {path}; {file_owners[identity]}')
                file_owners[identity] = f'{split}/{image_id}/{field}'
                paths[field] = str(path)
            records[image_id] = {'id': image_id, 'sample_id': f'{split}:{image_id}',
                                 'split': split, 'height': h, 'width': w,
                                 **paths, 'annotations': []}
            image_metadata[image_id] = metadata
        if not records:
            raise ValueError(f'No {split} images in the original JSON')
        annotation_ids = set()
        for annotation in document['annotations']:
            aid = _integer(annotation['id'], 'annotation ID')
            if aid in annotation_ids:
                raise ValueError('Duplicate annotation ID within a split')
            annotation_ids.add(aid)
            image_id = _integer(annotation['image_id'], 'annotation image ID')
            if image_id not in records:
                raise ValueError('Annotation references an unknown image ID')
            if _integer(annotation['category_id'], 'annotation category ID') not in categories:
                raise ValueError('Annotation references an unknown category ID')
            crowd = annotation.get('iscrowd', 0)
            if type(crowd) is not int or crowd not in (0, 1):
                raise ValueError('iscrowd must be 0 or 1')
            if annotation.get('ignore', 0) != 0:
                raise ValueError('Ignored annotations need an explicit protocol; refusing to reinterpret them')
            if not crowd and ('visible_mask' not in annotation or annotation['visible_mask'] is None):
                raise ValueError(f'Non-crowd annotation {aid} lacks visible_mask; amodal fallback is forbidden')
            # Do not retain amodal segmentation/area/bbox as training targets.
            records[image_id]['annotations'].append({
                'id': aid, 'iscrowd': crowd, 'visible_mask': annotation.get('visible_mask')})
        result[split] = [records[key] for key in sorted(records)]
        receipts[split] = {'path': str(annotation_path), 'sha256': hashlib.sha256(payload).hexdigest(),
                           'images': len(records), 'annotations': len(annotation_ids),
                           'source_image_rows': len(document['images']),
                           'repeated_image_rows_collapsed': repeated_images,
                           'repeat_ignored_metadata_fields': ['date_captured']}
        # Free the full amodal JSON before reading the next split. Only visible
        # instance targets are retained in result.
        del document, payload, image_metadata
    return result, receipts


def _visible_mask(segmentation, shape):
    """Decode a visible COCO RLE or polygon list; explicit [] denotes empty."""
    if isinstance(segmentation, dict):
        rle = dict(segmentation)
        if rle.get('size') != list(shape):
            raise ValueError('visible_mask RLE dimensions disagree with the original image')
        counts = rle.get('counts')
        if isinstance(counts, list):
            if (any(type(n) is not int or n < 0 for n in counts)
                    or sum(counts) != shape[0] * shape[1]):
                raise ValueError('Invalid visible_mask RLE counts')
            rle = mask_utils.frPyObjects(rle, *shape)
        elif isinstance(counts, str):
            rle['counts'] = counts.encode('ascii')
        elif not isinstance(counts, bytes):
            raise ValueError('Invalid visible_mask RLE encoding')
    elif isinstance(segmentation, list):
        if not segmentation:
            return np.zeros(shape, dtype=np.uint8)
        for polygon in segmentation:
            if (not isinstance(polygon, list) or len(polygon) < 6 or len(polygon) % 2
                    or any(type(v) not in (int, float) or not np.isfinite(v) for v in polygon)):
                raise ValueError('Invalid visible_mask polygon')
        rle = mask_utils.merge(mask_utils.frPyObjects(segmentation, *shape))
    else:
        raise ValueError('visible_mask must be a COCO RLE or polygon list')
    mask = mask_utils.decode(rle)
    if mask.shape != shape or mask.dtype != np.uint8 or mask.max(initial=0) > 1:
        raise ValueError('Decoded visible_mask is not a binary original-size mask')
    return mask


class UOAISTrainingDataset(Dataset):
    """Return (four-key RGB-D sample, visible target) for train or val only.

    Non-crowd annotations require visible_mask. Explicit empty visible masks and
    crowds are omitted from instance targets, matching the author's visible-only
    mapper; images with no remaining instances stay in the dataset. Overlapping
    visible masks remain independent. Areas/boxes are measured from visible masks,
    never taken from COCO's potentially amodal area/bbox fields.
    """
    def __init__(self, root, split, calibration):
        if split not in ('train', 'val'):
            raise ValueError('Training adapter accepts only train/val, never test')
        self.root, self.split = Path(root).resolve(), split
        self.calibration = copy.deepcopy(validate_calibration(calibration))
        k = torch.tensor(self.calibration['camera_K'], dtype=torch.float32)
        if not torch.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError('Camera intrinsics must remain finite and positive in float32')
        if self.calibration.get('depth_semantics') != 'camera_axis_z':
            raise ValueError('Explicit camera_axis_z depth semantics are required')
        size = self.calibration.get('image_size')
        if (not isinstance(size, list) or len(size) != 2
                or any(type(n) is not int or n <= 0 for n in size)):
            raise ValueError('Calibration requires its original image_size [height, width]')
        invalid = self.calibration.get('invalid_depth_codes')
        if (not isinstance(invalid, list)
                or any(type(n) is not int or not 0 <= n <= 65535 for n in invalid)):
            raise ValueError('Explicit uint16 invalid_depth_codes are required (possibly [])')
        splits, self.annotation_receipts = load_uoais_splits(self.root)
        self._splits = splits
        self.records = splits[split]
        if any([r['height'], r['width']] != size for r in self.records):
            raise ValueError('Calibration resolution differs from original image dimensions')

    def for_split(self, split):
        """Share the already validated, read-only indices without reparsing 1.5 GB."""
        if split not in ('train', 'val'):
            raise ValueError('Training adapter accepts only train/val, never test')
        result = copy.copy(self)
        result.split, result.records = split, self._splits[split]
        if any([r['height'], r['width']] != self.calibration['image_size'] for r in result.records):
            raise ValueError('Calibration resolution differs from original image dimensions')
        return result

    def __len__(self):
        return len(self.records)

    def raw_sample(self, index):
        """Read only RGB/depth; annotation and category metadata are not model inputs."""
        record = self.records[index]
        shape = (record['height'], record['width'])
        rgb, raw = read_array(record['file_name']), read_array(record['depth_file_name'])
        if rgb.shape != (*shape, 3) or rgb.dtype != np.uint8:
            raise ValueError('Expected original-resolution uint8 RGB')
        if raw.shape != shape or raw.dtype != np.uint16:
            raise ValueError('Expected aligned original-resolution uint16 depth')
        valid = (raw > 0) & ~np.isin(raw, self.calibration['invalid_depth_codes'])
        with np.errstate(over='ignore', invalid='ignore'):
            depth = raw.astype(np.float32) * self.calibration['depth_scale_m']
        if not np.isfinite(depth[valid]).all() or np.any(depth[valid] <= 0):
            raise ValueError('Metric depth must remain finite and positive in float32')
        depth[~valid] = 0
        return {'image': torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                'depth_m': torch.from_numpy(depth)[None],
                'depth_valid': torch.from_numpy(valid)[None],
                'camera_K': torch.tensor(self.calibration['camera_K'], dtype=torch.float32)}

    def raw_target(self, index):
        record = self.records[index]
        shape = (record['height'], record['width'])
        masks, boxes, areas, ids = [], [], [], []
        for annotation in record['annotations']:
            if annotation['iscrowd']:
                continue
            mask = _visible_mask(annotation['visible_mask'], shape)
            if not mask.any():
                continue
            ys, xs = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
            masks.append(mask)
            boxes.append([int(xs[0]), int(ys[0]), int(xs[-1])+1, int(ys[-1])+1])
            areas.append(int(mask.sum()))
            ids.append(annotation['id'])
        count = len(masks)
        return {'masks': torch.from_numpy(np.stack(masks) if masks else np.zeros((0, *shape), np.uint8)),
                'labels': torch.ones(count, dtype=torch.int64),
                'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
                'area': torch.tensor(areas, dtype=torch.float32),
                'iscrowd': torch.zeros(count, dtype=torch.int64),
                'annotation_ids': torch.tensor(ids, dtype=torch.int64),
                'image_id': torch.tensor([record['id']], dtype=torch.int64)}

    def __getitem__(self, index):
        return self.raw_sample(index), self.raw_target(index)

    def summary(self):
        return {'dataset': 'UOAIS-Sim', 'root': str(self.root), 'split': self.split,
                'images': len(self), 'annotation_receipts': copy.deepcopy(self.annotation_receipts),
                'calibration': copy.deepcopy(self.calibration),
                'crowd_annotations_excluded': sum(a['iscrowd'] for r in self.records for a in r['annotations']),
                'target_policy': 'independent visible_mask only; no amodal fallback; omit empty/crowd instances; retain empty images',
                'geometry': 'original resolution; runner owns all augmentation and resize',
                'validation': 'both JSONs, IDs and file references checked at construction; image/mask pixels checked on access',
                'split_identity': 'COCO IDs are local to each split; sample_id includes the split; no shared resolved files/inodes'}
