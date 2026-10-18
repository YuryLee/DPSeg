"""Shared RGB-only course train/val dataset, reading directly from the ZIP.

Keeps independent (potentially overlapping) masks at original resolution. Model
adapters own resizing. Explicit seed+epoch produces architecture-independent
horizontal flips; use set_epoch before constructing a nonpersistent DataLoader.
Only compressed annotation dictionaries are cached, never decoded pixel arrays.
"""
from __future__ import annotations

from collections import defaultdict
import copy
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch.utils.data import Dataset


class CourseTrainingDataset(Dataset):
    def __init__(self, archive, split='train', augment=True, seed=None):
        if split not in ('train', 'val'):
            raise ValueError('Training dataset accepts only train/val; test is a separate inference API')
        if seed is not None and (type(seed) is not int or seed < 0):
            raise ValueError('seed must be a nonnegative integer or None')
        self.zip_path = Path(archive).resolve()
        self.split = split
        self.augment = bool(augment) and split == 'train'
        self.seed = seed
        self.epoch = 0
        self._archive = None
        self._archive_pid = None
        self.annotations_by_image = defaultdict(list)
        self.annotation_members = []
        self.categories = None
        records, annotation_ids = {}, set()
        zero_area = 0
        self.annotation_sha256 = hashlib.sha256()
        with ZipFile(self.zip_path) as source:
            names = source.namelist()
            if len(names) != len(set(names)):
                raise ValueError('Duplicate ZIP member names')
            self.members = frozenset(names)
            self.annotation_members = sorted(name for name in names if name.startswith(f'annotations/{split}/') and name.endswith('.json.gz'))
            for member in self.annotation_members:
                encoded = source.read(member)
                self.annotation_sha256.update(member.encode()+b'\0'+encoded)
                shard = json.loads(gzip.decompress(encoded))
                categories = shard.get('categories')
                if self.categories is not None and self.categories != categories:
                    raise ValueError('Category definitions differ across shards')
                if categories != [{'id': 1, 'name': 'component', 'supercategory': 'component'}]:
                    raise ValueError('Expected original single component category with id=1')
                self.categories = categories
                shard_images = {}
                for image in shard['images']:
                    image_id = image['id']
                    if type(image_id) is not int or image_id in records:
                        raise ValueError('Image IDs must be unique integers')
                    if image.get('split', split) != split:
                        raise ValueError('Image metadata split mismatch')
                    for key, folder in [('file_name', 'rgb'), ('depth_file_name', 'depth')]:
                        name = image.get(key)
                        path = PurePosixPath(name) if isinstance(name, str) else None
                        if path is None or path.is_absolute() or '..' in path.parts or path.parts[:2] != (folder, split) or name not in self.members:
                            raise ValueError('Missing, unsafe, or wrong-split RGB/depth member')
                    if type(image['height']) is not int or type(image['width']) is not int or min(image['height'], image['width']) <= 0:
                        raise ValueError('Invalid image dimensions')
                    records[image_id] = image
                    shard_images[image_id] = image
                for annotation in shard['annotations']:
                    aid = annotation['id']
                    if type(aid) is not int or aid in annotation_ids:
                        raise ValueError('Annotation IDs must be unique integers')
                    annotation_ids.add(aid)
                    image_id = annotation['image_id']
                    if image_id not in shard_images:
                        raise ValueError('Annotation references an image outside its shard')
                    if type(annotation['category_id']) is not int or annotation['category_id'] != 1:
                        raise ValueError('Unexpected annotation category')
                    if annotation['iscrowd'] not in (0, 1):
                        raise ValueError('Invalid original iscrowd')
                    area = annotation['area']
                    if type(area) not in (int, float) or not math.isfinite(area) or area < 0:
                        raise ValueError('Invalid original area')
                    rle = annotation['segmentation']
                    image = shard_images[image_id]
                    if not isinstance(rle, dict) or rle.get('size') != [image['height'], image['width']] or not isinstance(rle.get('counts'), str):
                        raise ValueError('Expected original-size compressed RLE')
                    measured = int(mask_utils.area(rle))
                    if measured != area:
                        raise ValueError(f'Original GT area disagrees with RLE: annotation {aid}')
                    zero_area += measured == 0
                    self.annotations_by_image[image_id].append(annotation)
        if not records:
            raise ValueError(f'No annotated {split} images in ZIP')
        self.images = [records[key] for key in sorted(records)]
        self.image_ids = [image['id'] for image in self.images]
        if len({r['file_name'] for r in self.images}) != len(self.images):
            raise ValueError('Multiple image IDs reference one RGB member')
        self.annotation_sha256 = self.annotation_sha256.hexdigest()
        self.annotation_count = len(annotation_ids)
        self.zero_area_annotation_count = zero_area
        self.empty_image_count = sum(not any(a['area'] > 0 for a in self.annotations_by_image[r['id']]) for r in self.images)

    def __len__(self):
        return len(self.images)

    def set_epoch(self, epoch):
        if type(epoch) is not int or epoch < 0:
            raise ValueError('epoch must be a nonnegative integer')
        self.epoch = epoch

    def _flip_for(self, image_id):
        if not self.augment:
            return False
        if self.seed is None:
            return bool(torch.rand(()) < .5)
        key = f'course_hflip_v1:{self.seed}:{self.epoch}:{image_id}'.encode()
        return int.from_bytes(hashlib.sha256(key).digest()[:8], 'big') < (1 << 63)

    def _handle(self):
        if self._archive is None or self._archive_pid != os.getpid():
            self.close()
            self._archive = ZipFile(self.zip_path, 'r')
            self._archive_pid = os.getpid()
        return self._archive

    def close(self):
        if self._archive is not None:
            self._archive.close()
        self._archive = None
        self._archive_pid = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_archive'] = None
        state['_archive_pid'] = None
        return state

    def load_rgb(self, index):
        record = self.images[index]
        with Image.open(io.BytesIO(self._handle().read(record['file_name']))) as source:
            if source.size != (record['width'], record['height']):
                raise ValueError('RGB dimensions disagree with metadata')
            return source.convert('RGB')

    def __getitem__(self, index):
        record = self.images[index]
        h, w = record['height'], record['width']
        flip = self._flip_for(record['id'])
        rgb = np.asarray(self.load_rgb(index), dtype=np.uint8)
        if flip:
            rgb = rgb[:, ::-1]
        image = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float().div_(255)
        masks, boxes, areas, crowds = [], [], [], []
        for annotation in self.annotations_by_image[record['id']]:
            # Only genuinely empty masks are dropped from training targets.
            # Original COCO GT remains unchanged in coco_ground_truth().
            if annotation['area'] == 0:
                continue
            binary = mask_utils.decode(annotation['segmentation'])
            if binary.shape != (h, w) or binary.dtype != np.uint8 or binary.max(initial=0) > 1:
                raise ValueError('Decoded mask shape/type is invalid')
            if int(binary.sum()) != annotation['area']:
                raise ValueError('Decoded mask does not preserve original area')
            if flip:
                binary = binary[:, ::-1]
            # Pixel-inclusive support -> exclusive xyxy, so one-pixel masks
            # have width=height=1 and are retained, never called degenerate.
            ys = np.flatnonzero(binary.any(axis=1))
            xs = np.flatnonzero(binary.any(axis=0))
            if not len(xs) or not len(ys):
                raise ValueError('Positive-area annotation decoded to empty mask')
            boxes.append([xs[0], ys[0], xs[-1]+1, ys[-1]+1])
            masks.append(np.ascontiguousarray(binary))
            areas.append(annotation['area'])
            crowds.append(annotation['iscrowd'])
        count = len(masks)
        target = {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(count, 4),
                  'masks': torch.from_numpy(np.stack(masks) if masks else np.zeros((0, h, w), dtype=np.uint8)),
                  'labels': torch.ones(count, dtype=torch.int64),
                  'image_id': torch.tensor([record['id']], dtype=torch.int64),
                  'area': torch.tensor(areas, dtype=torch.float32),
                  'iscrowd': torch.tensor(crowds, dtype=torch.int64)}
        return image, target

    def summary(self):
        """JSON-serializable data/augmentation contract for run protocols."""
        areas = [ann['area'] for group in self.annotations_by_image.values() for ann in group]
        return {'archive': str(self.zip_path), 'split': self.split, 'images': len(self),
                'annotation_shards': len(self.annotation_members), 'annotations': self.annotation_count,
                'scenes': len({image.get('scene_id') for image in self.images}),
                'categories': copy.deepcopy(self.categories),
                'annotation_member_name_and_bytes_sha256': self.annotation_sha256,
                'small_gt_area_le_1024': sum(0 <= area <= 1024 for area in areas),
                'zero_area_annotations_excluded_from_targets': self.zero_area_annotation_count,
                'images_with_no_positive_area_gt': self.empty_image_count,
                'positive_area_instances_kept': self.annotation_count-self.zero_area_annotation_count,
                'sizes': sorted({(image['height'], image['width']) for image in self.images}),
                'rgb_only': True, 'original_resolution_retained': True,
                'target_area': 'original GT area; horizontal reflection preserves it; model adapter owns resize',
                'box_rule': 'exclusive xyxy from each decoded mask, including one-pixel instances',
                'independent_overlapping_masks_preserved': True,
                'augmentation': {'horizontal_flip_probability': .5 if self.augment else 0,
                                 'seed': self.seed, 'epoch': self.epoch,
                                 'rule': 'SHA256(course_hflip_v1:seed:epoch:official_image_id), first uint64 < 2**63' if self.seed is not None else 'torch global RNG',
                                 'persistent_workers': 'set_epoch before iterator; use persistent_workers=False'},
                'cache': 'compressed annotations only; no decoded pixel cache; lazy per-PID ZIP handle'}

    def coco_ground_truth(self):
        """Independent unaugmented original annotations for train/val evaluation."""
        return {'info': {'source': str(self.zip_path), 'split': self.split},
                'images': copy.deepcopy(self.images), 'categories': copy.deepcopy(self.categories),
                'annotations': copy.deepcopy([ann for image in self.images for ann in self.annotations_by_image[image['id']]])}


def collate_fn(batch):
    images, targets = zip(*batch)
    return list(images), list(targets)
