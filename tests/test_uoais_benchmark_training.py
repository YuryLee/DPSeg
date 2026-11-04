import unittest
import torch

from rgbd_seg.uoais_benchmark_training import augment_visible, flip_native


class SharedAugmentationTests(unittest.TestCase):
    def example(self):
        sample = {'image': torch.zeros(3, 4, 8), 'depth_m': torch.ones(1, 4, 8),
                  'depth_valid': torch.ones(1, 4, 8, dtype=torch.bool),
                  'camera_K': torch.tensor([[4., 0., 3.5], [0., 4., 1.5], [0., 0., 1.]])}
        masks = torch.zeros(2, 4, 8, dtype=torch.uint8)
        masks[0, 0:2, 0:2] = 1
        masks[1, 2:4, 6:8] = 1
        sample['image'][0] = masks[0]
        target = {'masks': masks, 'labels': torch.ones(2, dtype=torch.long),
                  'annotation_ids': torch.tensor([5, 9]), 'image_id': torch.tensor([31])}
        return sample, target

    def test_padding_is_background_and_invalid_depth(self):
        sample, target = self.example()
        mapped, truth, meta = augment_visible(sample, target, scale=.5)
        self.assertFalse(mapped['depth_valid'][:, 2:].any())
        self.assertFalse(mapped['depth_valid'][:, :, 4:].any())
        self.assertTrue((mapped['depth_m'][~mapped['depth_valid']] == 0).all())
        self.assertFalse(truth['masks'][:, 2:].any())
        self.assertEqual(truth['area'].tolist(), [1, 1])
        torch.testing.assert_close(mapped['camera_K'], torch.tensor([[2., 0., 1.5], [0., 2., .5], [0., 0., 1.]]))

    def test_crop_tracks_dropped_instance_without_dropping_image(self):
        sample, target = self.example()
        mapped, truth, meta = augment_visible(sample, target, scale=2, offset=(4, 8))
        self.assertEqual(truth['annotation_ids'].tolist(), [9])
        self.assertEqual(meta['dropped_instances'], 1)
        self.assertEqual(truth['image_id'].tolist(), [31])
        self.assertEqual(truth['boxes'].tolist(), [[4, 0, 8, 4]])
        torch.testing.assert_close(mapped['camera_K'], torch.tensor([[8., 0., -0.5], [0., 8., -.5], [0., 0., 1.]]))

    def test_flip_geometry_and_target_round_trip(self):
        sample, target = self.example()
        sample, target, _ = augment_visible(sample, target)
        flipped, truth = flip_native(sample, target)
        twice, twice_truth = flip_native(flipped, truth)
        for key in sample:
            torch.testing.assert_close(twice[key], sample[key])
        for key in target:
            torch.testing.assert_close(twice_truth[key], target[key])
        self.assertEqual(float(flipped['camera_K'][0, 0]), -4.)

    def test_saturated_rgb_stays_in_range_after_fractional_resize(self):
        sample = {'image': torch.ones(3, 480, 640), 'depth_m': torch.ones(1, 480, 640),
                  'depth_valid': torch.ones(1, 480, 640, dtype=torch.bool),
                  'camera_K': torch.tensor([[591., 0., 320.], [0., 590., 240.], [0., 0., 1.]])}
        target = {'masks': torch.zeros(0, 480, 640, dtype=torch.uint8),
                  'labels': torch.ones(0, dtype=torch.long)}
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            mapped, truth, _ = augment_visible(sample, target, scale=.9648532983463887)
        finally:
            torch.set_num_threads(previous_threads)
        self.assertGreaterEqual(float(mapped['image'].min()), 0.)
        self.assertLessEqual(float(mapped['image'].max()), 1.)
        self.assertEqual(len(truth['masks']), 0)


if __name__ == '__main__':
    unittest.main()
