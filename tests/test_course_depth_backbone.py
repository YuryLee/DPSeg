"""Check that the new depth arm genuinely has no calibration dependency."""
import unittest
from unittest.mock import patch

import torch

from rgbd_seg.course_depth_backbone import depth_channels, CourseVariantDataset, load_sample
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_backbone_system import crop_sample
from rgbd_seg.course_depth_backbone_points import PairedPointHeads

ARCHIVE = '/data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip'


class DepthEncodingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)

    def sample(self):
        depth = torch.arange(1, 17, dtype=torch.float32).reshape(1, 4, 4)/100
        depth[:, 0, 0] = 0
        return {'image': torch.ones(3, 4, 4), 'depth_m': depth, 'depth_valid': depth > 0}

    def test_depth_input_excludes_camera_and_annotation_fields(self):
        sample = self.sample()
        for extra in ('camera_K', 'annotations', 'support_rle'):
            with self.assertRaises(ValueError):
                depth_channels({**sample, extra: torch.eye(3)}, (4, 4))

    def test_depth_channels_retain_order_and_separate_validity(self):
        sample = self.sample()
        actual = depth_channels(sample, (4, 4))
        self.assertEqual(actual.shape, (4, 4, 4))
        self.assertTrue(torch.equal(actual[0], actual[1]))
        self.assertTrue(torch.equal(actual[1], actual[2]))
        self.assertTrue(torch.equal(actual[3].bool(), sample['depth_valid'][0]))
        self.assertTrue(torch.equal(actual[:3, 0, 0], torch.zeros(3)))
        values = actual[0][sample['depth_valid'][0]]
        self.assertTrue(torch.all(values[1:] > values[:-1]))

    def test_encoding_is_finite_for_missing_or_constant_depth(self):
        sample = self.sample()
        for value in (0., .7):
            depth = torch.full_like(sample['depth_m'], value)
            actual = depth_channels({**sample, 'depth_m': depth, 'depth_valid': depth > 0}, (8, 8))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertTrue(torch.equal(actual[:3], torch.zeros(3, 8, 8)))

    def test_depth_flip_and_scale_do_not_need_calibration(self):
        sample = self.sample()
        a = depth_channels(sample, (4, 4))
        flipped = {k: v.flip(-1) for k, v in sample.items()}
        self.assertTrue(torch.equal(depth_channels(flipped, (4, 4)), a.flip(-1)))
        b = depth_channels({**sample, 'depth_m': sample['depth_m']*10}, (4, 4))
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)

    def test_no_intrinsics_crop_keeps_modalities_aligned(self):
        sample = self.sample()
        cropped = crop_sample(sample, (1, 1, 4, 3))
        self.assertEqual(set(cropped), set(sample))
        for key in sample:
            self.assertTrue(torch.equal(cropped[key], sample[key][:, 1:3, 1:4]))

    def test_point_heads_support_both_backbone_widths_and_start_at_zero(self):
        for width in (352, 448):
            torch.manual_seed(17)
            heads = PairedPointHeads(fine_channels=width)
            torch.testing.assert_close(heads.depth.layers[0].weight[:, :width],
                                       heads.plain.layers[0].weight[:, :width], rtol=0, atol=0)
            self.assertTrue(torch.equal(heads.depth.layers[0].weight[:, width:width+9],
                                        torch.zeros_like(heads.depth.layers[0].weight[:, width:width+9])))
            coarse = torch.randn(2, 1, 11)
            values = heads(torch.randn(2, width, 11), coarse, torch.randn(2, 9, 11))
            self.assertEqual(set(values), {'point', 'point_depth'})
            for delta in values.values():
                self.assertTrue(torch.equal(delta, torch.zeros_like(coarse)))

    def test_real_loaders_do_not_call_camera_intrinsics(self):
        with patch('rgbd_seg.course_depth_backbone.camera_intrinsics', side_effect=AssertionError('K read')):
            dataset = CourseVariantDataset(ARCHIVE, split='train', augment=True, seed=17, encoding='depth')
            try:
                sample, target = dataset[0]
                self.assertEqual(set(sample), {'image', 'depth_m', 'depth_valid'})
                self.assertEqual(target['masks'].shape[-2:], (1024, 1024))
                self.assertTrue(torch.isfinite(depth_channels(sample, (1024, 1024))).all())
            finally:
                dataset.close()
            with CourseArchive(ARCHIVE, split='val') as dataset:
                sample = load_sample(dataset, dataset.image_ids[0], 'depth')
                self.assertEqual(set(sample), {'image', 'depth_m', 'depth_valid'})


if __name__ == '__main__':
    unittest.main()
