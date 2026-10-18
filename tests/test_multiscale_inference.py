"""Geometry/protocol tests with fake models: no weights, downloads or GPU needed."""
from types import SimpleNamespace
import unittest

import torch

from rgbd_seg.multiscale_inference import PROTOCOL, corner_crops, run_inference


def sample(height=100, width=100):
    coordinate = torch.arange(height * width).reshape(1, height, width).float()
    return {'image': coordinate.expand(3, -1, -1) / max(height * width, 1),
            'depth': coordinate.clone(), 'depth_valid': (coordinate % 3 != 0).float()}


def prediction(height, width, boxes=(), scores=None, labels=None, masks=None):
    n = len(boxes)
    if masks is None:
        masks = torch.zeros((n, 1, height, width))
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            masks[i, 0, int(y1):int(y2), int(x1):int(x2)] = .9
    return {'boxes': torch.tensor(boxes, dtype=torch.float32).reshape(n, 4),
            'scores': torch.tensor(scores if scores is not None else [.9] * n),
            'labels': torch.tensor(labels if labels is not None else [1] * n, dtype=torch.int64),
            'masks': masks}


class FakeModel(torch.nn.Module):
    def __init__(self, callback):
        super().__init__()
        self.callback = callback
        self.calls = []
        self.core = SimpleNamespace(transform=SimpleNamespace(min_size=(480, 640, 720), max_size=1280))
        self.eval()

    def forward(self, inputs):
        current = inputs[0]
        self.calls.append(current)
        return [self.callback(len(self.calls) - 1, current, self.core.transform)]


class MultiscaleInferenceTest(unittest.TestCase):
    def test_corner_geometry_and_deduplication(self):
        self.assertEqual(corner_crops(720, 1280), [(0, 0, 800, 450), (480, 0, 1280, 450),
                                                (0, 270, 800, 720), (480, 270, 1280, 720)])
        self.assertEqual(corner_crops(1, 1), [])
        self.assertEqual(corner_crops(2, 4), [(0, 0, 3, 2), (1, 0, 4, 2)])

    def test_crops_align_every_channel_and_paste_bottom_right(self):
        original = sample(100, 120)
        expected_views = [(0, 0, 120, 100), *corner_crops(100, 120)]
        def callback(index, current, transform):
            left, top, right, bottom = expected_views[index]
            for key in original:
                self.assertTrue(torch.equal(current[key], original[key][:, top:bottom, left:right]))
            h, w = current['image'].shape[-2:]
            return prediction(h, w, [[10, 10, 20, 20]]) if index == 4 else prediction(h, w)
        result, counts = run_inference(FakeModel(callback), original, 'crop_fuse')
        left, top, _, _ = expected_views[4]
        self.assertEqual(result['boxes'].tolist(), [[left + 10, top + 10, left + 20, top + 20]])
        self.assertEqual(result['masks'].shape, (1, 1, 100, 120))
        self.assertEqual(int(result['masks'].sum()), 100)
        self.assertTrue(result['masks'][0, 0, top + 10:top + 20, left + 10:left + 20].all())
        self.assertEqual(counts['output_source_views'], [4])
        self.assertEqual(counts['views'], 5)

    def test_full_preserves_candidates_and_only_removes_empty_masks(self):
        original = prediction(100, 100, [[0, 0, 99, 99], [1, 1, 98, 98], [20, 20, 25, 25]], [.7, .9, .8])
        original['masks'][2].zero_()
        original['masks'][0, 0, 0, 0] = .5
        model = FakeModel(lambda *args: original)
        result, counts = run_inference(model, sample(), 'full')
        self.assertEqual(len(model.calls), 1)
        self.assertTrue(torch.equal(result['boxes'], original['boxes'][:2]))
        self.assertTrue(torch.equal(result['scores'], original['scores'][:2]))
        self.assertTrue(torch.equal(result['labels'], original['labels'][:2]))
        self.assertTrue(torch.equal(result['masks'], original['masks'][:2] >= .5))
        self.assertEqual(result['masks'].dtype, torch.bool)
        self.assertEqual(counts['empty_masks_removed'], 1)
        self.assertEqual(counts['nms_removed'], 0)
        self.assertEqual(int(result['masks'][0].sum()), 99 ** 2)

    def test_internal_edge_margin_inclusive_outer_image_edge_allowed(self):
        # At left=0/top=0, only crop right/bottom edges are internal.
        # At right=W/top=0, only crop left/bottom edges are internal.
        def callback(index, current, transform):
            h, w = current['image'].shape[-2:]
            if index == 1:
                return prediction(h, w, [[0, 0, 4, 4], [10, 10, w - 2, 20], [10, 10, 20, h - 2]])
            if index == 2:
                return prediction(h, w, [[2, 10, 8, 20], [w - 4, 0, w, 4]])
            if index == 3:
                return prediction(h, w, [[10, 2, 20, 8], [0, h - 4, 4, h]])
            if index == 4:
                return prediction(h, w, [[2, 10, 8, 20], [10, 2, 20, 8], [w - 4, h - 4, w, h]])
            return prediction(h, w)
        result, counts = run_inference(FakeModel(callback), sample(), 'crop_fuse')
        self.assertEqual(counts['crop_border_removed'], 6)
        self.assertEqual(result['boxes'].tolist(), [[0, 0, 4, 4], [96, 0, 100, 4], [0, 96, 4, 100], [96, 96, 100, 100]])

    def test_crop_area_uses_binary_prediction_pixels_in_original_crop(self):
        def callback(index, current, transform):
            h, w = current['image'].shape[-2:]
            if index == 0:
                return prediction(h, w, [[0, 0, 100, 100]], [.8], [3])
            if index == 1:
                masks = torch.zeros((2, 1, h, w))
                masks[:, :, 1:97, 1:97] = .5
                masks[1, 0, 97, 97] = .5
                return prediction(h, w, [[1, 1, 98, 98], [1, 1, 98, 98]], [.9, .95], [1, 2], masks)
            return prediction(h, w)
        result, counts = run_inference(FakeModel(callback), sample(200, 200), 'crop_fuse')
        self.assertEqual(counts['crop_area_removed'], 1)
        self.assertEqual(result['labels'].tolist(), [1, 3])
        self.assertEqual(result['masks'].flatten(1).sum(1).tolist(), [9216, 10000])

    def test_cross_view_box_nms_preserves_other_classes_and_stable_ties(self):
        def callback(index, current, transform):
            h, w = current['image'].shape[-2:]
            if index == 0:
                return prediction(h, w, [[10, 10, 20, 20]], [.9], [1])
            if index == 1:
                return prediction(h, w, [[10, 10, 20, 20], [10, 10, 20, 20]], [.9, .9], [1, 2])
            return prediction(h, w)
        result, counts = run_inference(FakeModel(callback), sample(), 'crop_fuse')
        self.assertEqual(result['labels'].tolist(), [1, 2])
        self.assertEqual(counts['output_source_views'], [0, 1])
        self.assertEqual(counts['nms_removed'], 1)

    def test_empty_result_shapes_and_duplicate_full_crop_skipped(self):
        model = FakeModel(lambda index, current, t: prediction(*current['image'].shape[-2:]))
        result, counts = run_inference(model, sample(1, 1), 'crop_fuse')
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(result['masks'].shape, (0, 1, 1, 1))
        self.assertEqual(result['boxes'].shape, (0, 4))
        self.assertEqual(result['labels'].dtype, torch.int64)
        self.assertEqual(counts['output_source_views'], [])

    def test_highres_restores_transform_on_success(self):
        def callback(index, current, transform):
            self.assertEqual(transform.min_size, (1152,))
            self.assertEqual(transform.max_size, 2048)
            self.assertEqual(current['image'].shape[-2:], (100, 100))
            return prediction(100, 100, [[5, 5, 10, 10]])
        model = FakeModel(callback)
        before_min = model.core.transform.min_size
        result, counts = run_inference(model, sample(), 'highres')
        self.assertIs(model.core.transform.min_size, before_min)
        self.assertEqual(model.core.transform.max_size, 1280)
        self.assertEqual(result['masks'].shape, (1, 1, 100, 100))
        self.assertEqual(counts['views'], 1)

    def test_highres_restores_transform_on_forward_and_validation_errors(self):
        for invalid_result in (False, True):
            def callback(index, current, transform):
                if invalid_result:
                    return prediction(99, 100)
                raise RuntimeError('intentional model error')
            model = FakeModel(callback)
            before = model.core.transform.min_size
            with self.assertRaises((RuntimeError, ValueError)):
                run_inference(model, sample(), 'highres')
            self.assertIs(model.core.transform.min_size, before)
            self.assertEqual(model.core.transform.max_size, 1280)

    def test_global_top100_after_nms_and_score_order(self):
        # Distinct classes avoid suppression so the final global cap is isolated.
        boxes = [[10, 10, 20, 20]] * 101
        scores = [.1 + .8 * i / 100 for i in range(101)]
        def callback(index, current, transform):
            h, w = current['image'].shape[-2:]
            return prediction(h, w, boxes, scores, list(range(1, 102))) if index == 0 else prediction(h, w)
        result, counts = run_inference(FakeModel(callback), sample(), 'crop_fuse')
        self.assertEqual(len(result['scores']), 100)
        self.assertTrue((result['scores'][1:] <= result['scores'][:-1]).all())
        self.assertEqual(counts['topk_removed'], 1)
        self.assertEqual(result['labels'][0].item(), 101)
        self.assertEqual(result['labels'][-1].item(), 2)

    def test_no_target_input_and_nonfinite_mask_is_rejected(self):
        model = FakeModel(lambda *args: prediction(100, 100))
        with self.assertRaisesRegex(ValueError, 'no targets'):
            run_inference(model, {**sample(), 'masks': torch.zeros(1, 100, 100)})
        def callback(*args):
            result = prediction(100, 100, [[1, 1, 10, 10]])
            result['masks'][0, 0, 0, 0] = float('nan')
            return result
        with self.assertRaisesRegex(ValueError, 'non-finite'):
            run_inference(FakeModel(callback), sample())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
