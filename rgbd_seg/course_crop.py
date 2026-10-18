"""Training-only small-instance crops with aligned RGB-D camera geometry."""
from __future__ import annotations

import copy
import hashlib
import random

import torch

from .course_depth_data import CourseDepthTrainingDataset


CROP_POLICY = {
    'name': 'small_instance_crop512_mix_v1',
    'size': 512,
    'probability_on_eligible_images': .5,
    'small_original_area_max': 1024,
    'anchor': 'Uniform small instance, then uniform foreground pixel; crop contains that pixel',
    'origin': 'Uniform valid integer top/left among windows containing the anchor',
    'fallback': 'Whole image if no small instance or crop coin is false',
    'targets': 'All positive-area cropped instances retained independently, including partial masks',
    'geometry': 'After native hflip, crop all modalities; K_crop = translation(-left,-top) @ K',
    'evaluation': 'Whole image only; no GT or crops at inference',
    'rng': 'Image/epoch/seed keyed local Python RNG, no consumption of global RNG',
}


def choose_crop(target, *, seed, epoch, size=512, probability=.5):
    """Return (top,left,height,width,anchor_instance) or None, deterministically."""
    if not 0 <= probability <= 1 or type(size) is not int or size < 1:
        raise ValueError('Invalid crop policy')
    masks = target['masks']
    height, width = masks.shape[-2:]
    if size > min(height, width):
        raise ValueError('Crop must fit within the image')
    image_id = int(target['image_id'].item())
    key = f'course_crop512_v1:{seed}:{epoch}:{image_id}'.encode()
    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(), 'big'))
    if rng.random() >= probability:
        return None
    areas = target.get('original_area', target['area'])
    eligible = ((areas > 0) & (areas <= CROP_POLICY['small_original_area_max'])).nonzero().flatten().tolist()
    if not eligible:
        return None
    anchor = eligible[rng.randrange(len(eligible))]
    foreground = masks[anchor].nonzero(as_tuple=False)
    if not len(foreground):
        raise ValueError('A positive-area crop anchor cannot be empty')
    y, x = foreground[rng.randrange(len(foreground))].tolist()
    top = rng.randrange(max(0, y-size+1), min(y, height-size)+1)
    left = rng.randrange(max(0, x-size+1), min(x, width-size)+1)
    return top, left, size, size, anchor


def crop_sample_target(sample, target, window):
    """Pure transform, no mutation of original tensors or decoded labels."""
    top, left, height, width, anchor = window
    full_h, full_w = sample['image'].shape[-2:]
    if (min(top, left) < 0 or min(height, width) < 1
            or top+height > full_h or left+width > full_w):
        raise ValueError('Crop extends beyond the image')
    if any(tuple(sample[k].shape[-2:]) != (full_h, full_w)
           for k in ('depth_m', 'depth_valid')) or tuple(target['masks'].shape[-2:]) != (full_h, full_w):
        raise ValueError('RGB/depth/targets must be spatially aligned')
    region = (..., slice(top, top+height), slice(left, left+width))
    transformed = {k: sample[k][region].clone() for k in ('image','depth_m','depth_valid')}
    k = sample['camera_K']
    affine = k.new_tensor([[1,0,-left],[0,1,-top],[0,0,1]])
    transformed['camera_K'] = affine @ k
    masks = target['masks'][region].clone()
    keep = masks.flatten(1).bool().any(1)
    if anchor is not None and not bool(keep[anchor]):
        raise ValueError('Selected anchor was lost by the crop')
    result = {}
    for key, value in target.items():
        if key in ('boxes','masks','labels','area','iscrowd','original_area'):
            result[key] = value[keep].clone()
        else:
            result[key] = copy.deepcopy(value)
    result['masks'] = masks[keep]
    result['original_area'] = target.get('original_area', target['area'])[keep].clone()
    result['area'] = result['masks'].flatten(1).sum(1).to(torch.float32)
    boxes = []
    for binary in result['masks']:
        y, x = binary.nonzero(as_tuple=True)
        boxes.append(torch.stack((x.min(), y.min(), x.max()+1, y.max()+1)))
    result['boxes'] = (torch.stack(boxes).to(torch.float32) if boxes else
                       torch.empty((0,4), dtype=torch.float32, device=masks.device))
    result['crop_info'] = {'applied': True, 'top': top, 'left': left,
        'height': height, 'width': width, 'anchor_original_index': anchor,
        'original_instances': len(masks), 'retained_instances': int(keep.sum()),
        'truncated_instances': int((result['area'] < result['original_area']).sum())}
    return transformed, result


class CourseCropTrainingDataset(CourseDepthTrainingDataset):
    """Fixed crop policy only for augmented training; validation stays native."""
    def __getitem__(self, index):
        sample, target = super().__getitem__(index)
        if self.split != 'train' or not self.augment:
            return sample, target
        window = choose_crop(target, seed=self.seed, epoch=self.epoch,
            size=CROP_POLICY['size'], probability=CROP_POLICY['probability_on_eligible_images'])
        if window is None:
            target['crop_info'] = {'applied': False}
            return sample, target
        return crop_sample_target(sample, target, window)

    def summary(self):
        summary = super().summary()
        summary.update(crop_policy=copy.deepcopy(CROP_POLICY),
            original_resolution_retained=False,
            target_area='Crop-visible area; original_area retained separately; evaluation original GT untouched')
        return summary
