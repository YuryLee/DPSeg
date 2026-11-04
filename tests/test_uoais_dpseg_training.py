import copy
import unittest
import torch
from rgbd_seg.uoais_dpseg_training import prepare_training_example, collate_training_examples
from rgbd_seg.course_depth_data import model_depth_channels


class SharedTrainingGeometryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.sample = {'image': torch.zeros(3, 6, 8), 'depth_m': torch.ones(1, 6, 8),
                       'depth_valid': torch.ones(1, 6, 8, dtype=torch.bool),
                       'camera_K': torch.tensor([[100., 0, 3.5], [0, 100., 2.5], [0, 0, 1.]])}
        masks = torch.zeros(2, 6, 8, dtype=torch.uint8)
        masks[0, 1:4, 1:4] = 1
        masks[1, 2:5, 2:5] = 1
        self.target = {'masks': masks, 'labels': torch.ones(2, dtype=torch.long),
                       'annotation_ids': torch.tensor([11, 22]), 'image_id': torch.tensor([9]),
                       'area': torch.tensor([9., 9.]), 'boxes': torch.zeros(2, 4),
                       'iscrowd': torch.zeros(2, dtype=torch.long)}

    def test_shared_flip_has_valid_downstream_depth_and_independent_masks(self):
        before = copy.deepcopy(self.sample)
        sample, target, meta = prepare_training_example(self.sample, self.target, horizontal_flip=True)
        affine = torch.tensor(meta['pixel_affine'])
        torch.testing.assert_close(sample['camera_K'], affine @ self.sample['camera_K'])
        self.assertLess(sample['camera_K'][0, 0], 0)
        channels = model_depth_channels(sample, (64, 64), device='cpu')
        self.assertEqual(tuple(channels.shape), (4, 64, 64))
        self.assertTrue(torch.isfinite(channels).all())
        self.assertTrue((target['masks'].sum(0) > 1).any())
        self.assertEqual(target['annotation_ids'].tolist(), [11, 22])
        for key in before:
            torch.testing.assert_close(before[key], self.sample[key])

    def test_crop_removal_requires_explicit_policy_and_logs_identity(self):
        crop = (4, 4, 8, 6)
        with self.assertRaisesRegex(ValueError, 'explicit drop'):
            prepare_training_example(self.sample, self.target, crop_box=crop)
        _, target, meta = prepare_training_example(self.sample, self.target, crop_box=crop,
                                                   empty_crop_policy='drop')
        self.assertEqual(target['annotation_ids'].tolist(), [22])
        self.assertEqual(meta['dropped_by_crop_annotation_ids'], [11])
        self.assertGreater(target['area'][0], 0)

    def test_empty_crop_retains_image_and_zero_instance_tensors(self):
        sample, target, meta = prepare_training_example(self.sample, self.target,
            crop_box=(6, 0, 8, 2), empty_crop_policy='drop')
        self.assertEqual(target['masks'].shape, (0, 1024, 1024))
        self.assertEqual(target['labels'].shape, (0,))
        self.assertEqual(target['boxes'].shape, (0, 4))
        self.assertTrue(meta['empty_image_retained'])
        self.assertEqual(sample['image'].shape, (3, 1024, 1024))

    def test_tiny_mask_preserved_and_batch_contract(self):
        self.target['masks'][0] = 0
        self.target['masks'][0, 0, 0] = 1
        example = prepare_training_example(self.sample, self.target)
        self.assertGreater(example[1]['area'][0], 0)
        samples, targets, metadata = collate_training_examples([example])
        self.assertEqual(set(samples[0]), {'image', 'depth_m', 'depth_valid', 'camera_K'})
        self.assertEqual(targets[0]['labels'].tolist(), [1, 1])
        self.assertEqual(len(metadata), 1)


if __name__ == '__main__':
    unittest.main()
