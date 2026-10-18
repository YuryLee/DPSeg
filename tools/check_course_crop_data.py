#!/usr/bin/env python3
"""Bounded CPU check on actual course train images, including spawned loading."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from torch.utils.data import DataLoader
from rgbd_seg.course_crop import CourseCropTrainingDataset, crop_sample_target
from rgbd_seg.course_depth_data import CourseDepthTrainingDataset, collate_fn, depth_to_xyz
from rgbd_seg.course_resolution import build_trainable
from tools.run_course_dafusion import sha256, now, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    dataset = CourseCropTrainingDataset(args.archive, split='train', augment=True, seed=17)
    selected, reference, counts = [], {}, {True:0, False:0}
    try:
        for index in range(min(128,len(dataset))):
            sample,target = dataset[index]
            cropped = target['crop_info']['applied']
            if counts[cropped] >= 2:
                continue
            original, labels = CourseDepthTrainingDataset.__getitem__(dataset,index)
            info = target['crop_info']
            if cropped:
                top,left,h,w = [info[key] for key in ('top','left','height','width')]
                expected, expected_target = crop_sample_target(original,labels,
                    (top,left,h,w,info['anchor_original_index']))
                xyz_before = depth_to_xyz(original['depth_m'],original['depth_valid'],original['camera_K'])
                xyz_after = depth_to_xyz(sample['depth_m'],sample['depth_valid'],sample['camera_K'])
                torch.testing.assert_close(xyz_after,xyz_before[:,top:top+h,left:left+w])
            else:
                expected,expected_target = original,labels
            for key in sample:
                torch.testing.assert_close(sample[key],expected[key],rtol=0,atol=0)
            for key in ('masks','boxes','labels','area','iscrowd','image_id'):
                torch.testing.assert_close(target[key],expected_target[key],rtol=0,atol=0)
            if (target['masks'].flatten(1).sum(1) <= 0).any():
                raise AssertionError('Empty retained mask')
            selected.append(index)
            reference[int(target['image_id'].item())] = sample,target
            counts[cropped] += 1
            if min(counts.values()) == 2:
                break
        if min(counts.values()) != 2:
            raise AssertionError('Bounded check did not cover crop and whole-image cases')
        dataset.close()
        observed = []
        loader = DataLoader(dataset,batch_sampler=[selected],num_workers=1,
            multiprocessing_context='spawn',collate_fn=collate_fn,
            timeout=60,
            generator=torch.Generator().manual_seed(17))
        for samples,targets in loader:
            for sample,target in zip(samples,targets):
                image_id = int(target['image_id'].item())
                expected,labels = reference[image_id]
                for key in sample:
                    torch.testing.assert_close(sample[key],expected[key],rtol=0,atol=0)
                for key,value in target.items():
                    if isinstance(value,torch.Tensor):
                        torch.testing.assert_close(value,labels[key],rtol=0,atol=0)
                    elif value != labels[key]:
                        raise AssertionError(key)
                observed.append({'image_id':image_id,'input_hw':list(sample['image'].shape[-2:]),
                    'instances':len(target['labels']),'crop_info':target['crop_info']})
            adapter,_ = build_trainable(ROOT/'pretrained/mask2former_swin_tiny_coco_instance',
                                        device='cpu',seed=17,resolution=1024)
            prepared,metadata = adapter.prepare_batch(samples,targets)
            if tuple(prepared['pixel_values'].shape) != (4,7,1024,1024):
                raise AssertionError('Mixed crop/full samples failed common model input sizing')
            if metadata['resized_empty_instances'] != [0]*4:
                raise AssertionError('Upsampling must not lose a nonempty instance')
            for masks,target in zip(prepared['mask_labels'],targets):
                if len(masks) != len(target['labels']):
                    raise AssertionError('Model preprocessing lost an instance')
        record = {'status':'passed','finished_at_utc':now(),'scope':'CPU data and preprocessing only; no training or AP claim',
            'train_images':len(dataset),'selected_indices':selected,'samples':observed,
            'spawn_loading_exact_match':True,'camera_xyz_checked':True,
            'mixed_batch_shape':[4,7,1024,1024],
            'source_sha256':{name:sha256(ROOT/name) for name in (
                'tools/check_course_crop_data.py','rgbd_seg/course_crop.py','rgbd_seg/course_depth_data.py',
                'rgbd_seg/course_resolution.py')}}
        write_json(args.output,record)
        print(json.dumps(record),flush=True)
    finally:
        dataset.close()


if __name__ == '__main__':
    main()
