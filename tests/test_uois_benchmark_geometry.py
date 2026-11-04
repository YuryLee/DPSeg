import copy
import unittest

import numpy as np
from pycocotools import mask as mask_utils
import torch

from rgbd_seg.uois_benchmark_geometry import (
    INPUT_SIZE, RGB_PADDING, letterbox_sample, restore_predictions,
)
from rgbd_seg.uois_benchmark_metrics import predictions_to_label_map


def sample(height, width):
    return {'image': torch.full((3, height, width), .3),
            'depth_m': torch.ones((1, height, width)),
            'depth_valid': torch.ones((1, height, width), dtype=torch.bool),
            'camera_K': torch.tensor([[401., 0., (width-1)/2],
                                      [0., 403., (height-1)/2], [0., 0., 1.]])}


def record(mask, score, identifier):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return {'score': score, 'category_id': 1, 'segmentation': rle, 'id': identifier}


class UoisBenchmarkGeometryTests(unittest.TestCase):
    def test_landscape_portrait_and_rounded_intrinsics_preserve_rays(self):
        for height, width, expected_box in [
            (480, 640, [0, 128, 1024, 896]),
            (640, 480, [128, 0, 896, 1024]),
            (7, 11, [0, 186, 1024, 838]),
            (3, 11, [0, 372, 1024, 651]),  # Odd padding: bottom has one extra pixel.
        ]:
            with self.subTest(shape=(height, width)):
                original = sample(height, width)
                saved = {key: value.clone() for key, value in original.items()}
                mapped, geometry = letterbox_sample(original)
                self.assertEqual(set(mapped), set(original))
                self.assertEqual(geometry['content_box'], expected_box)
                self.assertEqual(geometry['original_size'], [height, width])
                self.assertEqual(tuple(mapped['image'].shape), (3, 1024, 1024))
                affine = torch.tensor(geometry['affine'], dtype=torch.float64)
                # Geometric pinhole rays remain identical after the pixel affine.
                points = torch.tensor([[0., width-1., (width-1)/2],
                                       [0., height-1., (height-1)/2], [1., 1., 1.]], dtype=torch.float64)
                before = torch.linalg.solve(original['camera_K'].double(), points)
                after = torch.linalg.solve(mapped['camera_K'].double(), affine @ points)
                torch.testing.assert_close(after, before, rtol=1e-6, atol=1e-7)
                left, top, right, bottom = expected_box
                sx, sy = (right-left)/width, (bottom-top)/height
                self.assertAlmostEqual(mapped['camera_K'][0, 2].item(), left + (sx*width-1)/2, places=4)
                self.assertAlmostEqual(mapped['camera_K'][1, 2].item(), top + (sy*height-1)/2, places=4)
                if (height, width) == (7, 11):
                    self.assertNotEqual(sx, sy)  # Integer rounding needs separate affine scales.
                for key in original:
                    torch.testing.assert_close(original[key], saved[key], rtol=0, atol=0)

    def test_rgb_bilinear_depth_discrete_and_padding_invalid(self):
        original = sample(2, 4)
        original['image'][0] = torch.arange(4).repeat(2, 1) / 3
        original['image'][1] = torch.tensor([[0.]*4, [1.]*4])
        original['depth_m'] = torch.arange(8, dtype=torch.float32).reshape(1, 2, 4)
        original['depth_valid'] = original['depth_m'] > 0
        mapped, geometry = letterbox_sample(original)
        self.assertEqual(geometry['content_box'], [0, 256, 1024, 768])
        self.assertEqual(set(mapped['depth_m'].unique().tolist()), set(range(8)))
        self.assertEqual(mapped['depth_valid'].sum().item(), 7*256*256)
        self.assertEqual(mapped['depth_m'][0, 256, 255].item(), 0)
        self.assertEqual(mapped['depth_m'][0, 256, 256].item(), 1)
        # Original red is a linear ramp; bilinear interpolation follows pixel centers.
        expected_red = (((384+.5)/256)-.5)/3
        self.assertAlmostEqual(mapped['image'][0, 400, 384].item(), expected_red, places=6)
        torch.testing.assert_close(mapped['image'][:, 0, 0], torch.tensor(RGB_PADDING))
        self.assertFalse(mapped['depth_valid'][:, :256].any())
        self.assertFalse(mapped['depth_valid'][:, 768:].any())
        self.assertFalse(mapped['depth_m'][:, :256].any())
        self.assertFalse(mapped['depth_m'][:, 768:].any())

    def test_native_square_is_identity_without_mutation(self):
        original = sample(1024, 1024)
        original['image'][0, 17, 29] = .9
        original['depth_valid'][0, 17, 29] = False
        original['depth_m'][0, 17, 29] = 0
        mapped, geometry = letterbox_sample(original)
        self.assertEqual(geometry['content_box'], [0, 0, 1024, 1024])
        self.assertEqual(geometry['affine'], [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])
        for key in original:
            torch.testing.assert_close(mapped[key], original[key], rtol=0, atol=0)

    def test_restore_crops_padding_before_downsampling_and_only_drops_empty(self):
        _, geometry = letterbox_sample(sample(4, 8))
        padding_only = np.zeros((1024, 1024), np.uint8); padding_only[:256] = 1
        cross = padding_only.copy(); cross[256:512, 0:256] = 1
        full = np.ones_like(cross)
        items = [record(padding_only, .99, 'pad'), record(cross, .001, 'low'), record(full, .7, 'full')]
        items[1].update(area=300000, bbox=[0, 0, 1024, 512])
        saved = copy.deepcopy(items)
        restored = restore_predictions(items, geometry)
        self.assertEqual([r['id'] for r in restored], ['low', 'full'])
        self.assertEqual([r['score'] for r in restored], [.001, .7])
        expected = np.zeros((4, 8), np.uint8); expected[:2, :2] = 1
        np.testing.assert_array_equal(mask_utils.decode(restored[0]['segmentation']), expected)
        np.testing.assert_array_equal(mask_utils.decode(restored[1]['segmentation']), np.ones((4, 8), np.uint8))
        self.assertEqual(restored[0]['bbox'], [0., 0., 2., 2.])
        self.assertEqual(restored[0]['area'], 4)
        self.assertEqual(items, saved)

    def test_restore_preserves_overlap_list_order_for_author_overwrite_rule(self):
        _, geometry = letterbox_sample(sample(4, 8))
        high = np.zeros((1024, 1024), np.uint8); high[256:768, :768] = 1
        low = np.zeros_like(high); low[256:768, 256:] = 1
        records = [record(high, .95, 'high'), record(low, .6, 'low')]
        restored = restore_predictions(records, geometry)
        self.assertEqual([r['id'] for r in restored], ['high', 'low'])
        self.assertEqual(sum(mask_utils.decode(r['segmentation'])[1, 3] for r in restored), 2)
        labels = predictions_to_label_map(restored, (4, 8))
        self.assertEqual(labels[1, 3], 2)  # Later low-score mask owns overlap.
        reverse = restore_predictions(list(reversed(records)), geometry)
        self.assertEqual([r['id'] for r in reverse], ['low', 'high'])
        self.assertEqual(restore_predictions([], geometry), [])

    def test_restore_bytes_and_uncompressed_masks_and_empty_after_sampling(self):
        _, geometry = letterbox_sample(sample(4, 8))
        empty = np.zeros((1024, 1024), np.uint8)
        sparse = empty.copy(); sparse[256, 0] = 1  # Not hit by nearest-exact downsample.
        one = record(sparse, .9, 'vanished')
        one['segmentation']['counts'] = one['segmentation']['counts'].encode('ascii')
        full = {'score': 0., 'segmentation': {'size': [1024, 1024], 'counts': [0, 1024**2]}}
        restored = restore_predictions([one, full], geometry)
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored[0]['score'], 0.)
        self.assertEqual(mask_utils.area(restored[0]['segmentation']), 32)

    def test_bad_input_or_geometry_fails_instead_of_silent_resize(self):
        original = sample(4, 8)
        bad = dict(original, target=torch.ones(1))
        with self.assertRaisesRegex(ValueError, 'exactly'):
            letterbox_sample(bad)
        bad = {**original, 'depth_m': torch.zeros((1, 4, 8))}
        with self.assertRaisesRegex(ValueError, 'positive valid depth'):
            letterbox_sample(bad)
        bad = {**original, 'depth_valid': torch.zeros((1, 4, 8), dtype=torch.bool)}
        with self.assertRaises(ValueError):
            letterbox_sample(bad)
        _, geometry = letterbox_sample(original)
        prediction = record(np.zeros((64, 64), np.uint8), .9, 'wrong-size')
        with self.assertRaisesRegex(ValueError, '1024 letterbox'):
            restore_predictions([prediction], geometry)
        corrupted = {**geometry, 'content_box': [0, 0, 1024, 1024]}
        with self.assertRaisesRegex(ValueError, 'centered'):
            restore_predictions([], corrupted)


if __name__ == '__main__':
    unittest.main()
