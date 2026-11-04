"""Protocol regression cases against the hash-pinned author's evaluator."""
import sys
import unittest
import warnings

import numpy as np
from pycocotools import mask as mask_utils

from rgbd_seg.uois_benchmark_metrics import (
    evaluate_label_maps, load_reference, macro_average,
    predictions_to_label_map, reference_provenance,
)


def rle_record(mask, score=.9, text=True):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    if text:
        rle['counts'] = rle['counts'].decode('ascii')
    return {'score': score, 'segmentation': rle}


class UoisBenchmarkMetricTests(unittest.TestCase):
    def assert_native(self, prediction, truth):
        prediction_before, truth_before = prediction.copy(), truth.copy()
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            expected, assignments = load_reference().multilabel_metrics(
                prediction, truth, None, None, return_assign=True)
        result = evaluate_label_maps(prediction, truth)
        self.assertEqual(set(result['metrics']), set(expected))
        for key in expected:
            # Exact numeric equality, including NaN, not just rounded table values.
            np.testing.assert_equal(result['metrics'][key], expected[key], err_msg=key)
        self.assertEqual(result['assignments'], assignments)
        np.testing.assert_array_equal(prediction, prediction_before)
        np.testing.assert_array_equal(truth, truth_before)
        return result

    def test_empty_cases_and_native_full_image_boundary(self):
        empty = np.zeros((17, 31), dtype=np.int32)
        obj = empty.copy(); obj[3:8, 6:12] = 1
        for pred, gt in [(empty, empty), (empty, obj), (obj, empty)]:
            with self.subTest(pred_empty=not pred.any(), gt_empty=not gt.any()):
                self.assert_native(pred, gt)
        full = np.ones_like(empty)
        result = self.assert_native(full, full)
        self.assertTrue(np.isnan(result['metrics']['Boundary Precision']))
        self.assertTrue(np.isnan(result['metrics']['Boundary Recall']))
        self.assertEqual(result['metrics']['Boundary F-measure'], 0)

    def test_split_merge_and_rectangular_assignment_matrix(self):
        truth = np.zeros((23, 37), dtype=np.int32)
        truth[3:7, 3:7], truth[3:7, 17:21] = 1, 20
        merged = (truth > 0).astype(np.int32)
        self.assert_native(merged, truth)
        self.assert_native(truth, merged)
        more_pred = truth.copy(); more_pred[13:16, 28:32] = 7
        self.assert_native(more_pred, truth)
        self.assert_native(truth, more_pred)

    def test_matching_tie_preserves_native_munkres_assignment(self):
        truth = np.zeros((29, 43), dtype=np.int32)
        truth[4:12, 6:10], truth[4:12, 10:14] = 9, 23
        prediction = np.zeros_like(truth)
        prediction[4:8, 6:14], prediction[8:12, 6:14] = 6, 80
        # All four pair F-scores are exactly .5: solver tie behavior is observable.
        result = self.assert_native(prediction, truth)
        self.assertEqual(result['assignments'], [[9, 6], [23, 80]])
        self.assertEqual(result['metrics']['Objects F-measure'], .5)

    def test_hungarian_assignment_not_greedy(self):
        truth = np.zeros((13, 29), dtype=np.int32)
        truth[2, 1:11], truth[5, 1:11] = 1, 2
        prediction = np.zeros_like(truth)
        prediction[2, 1:8], prediction[5, 1:7], prediction[2, 8:11] = 1, 1, 2
        result = self.assert_native(prediction, truth)
        self.assertEqual(result['assignments'], [[1, 2], [2, 1]])

    def test_boundary_tolerance_native_rectangular_dimensions(self):
        for shape, shift in [((39, 57), 1), ((39, 57), 5), ((360, 480), 2)]:
            truth = np.zeros(shape, dtype=np.int32); truth[10:22, 11:27] = 1
            prediction = np.zeros_like(truth); prediction[10:22, 11+shift:27+shift] = 7
            result = self.assert_native(prediction, truth)
            self.assertLess(result['metrics']['Objects F-measure'], 1)
            if shift in [1, 2]:
                self.assertEqual(result['metrics']['Boundary F-measure'], 1)
            else:
                self.assertLess(result['metrics']['Boundary F-measure'], 1)

    def test_strict_f075_endpoint_and_inclusive_diagnostic(self):
        truth = np.zeros((11, 19), dtype=np.int32); truth[3, 2:6] = 1
        prediction = np.zeros_like(truth); prediction[3, 3:7] = 11
        # Both areas are 4, intersection 3, exactly F=.75 (IoU=.6).
        result = self.assert_native(prediction, truth)
        self.assertEqual(result['metrics']['obj_detected_075_percentage'], 0)
        self.assertEqual(result['diagnostics'], {
            'obj_detected_ge075': 1, 'obj_detected_ge075_percentage': 1,
            'obj_matched_exactly075': 1})
        result = self.assert_native(truth, truth)
        self.assertEqual(result['metrics']['obj_detected_075_percentage'], 1)
        self.assertEqual(result['diagnostics']['obj_matched_exactly075'], 0)

    def test_sparse_label_ids_and_one_pixel_gt_are_unchanged(self):
        truth = np.zeros((19, 31), dtype=np.uint16)
        truth[2, 3], truth[7:12, 9:17], truth[15:18, 25:29] = 1, 999, 60000
        prediction = np.zeros_like(truth)
        prediction[truth == 1], prediction[truth == 999], prediction[truth == 60000] = 90, 3, 7
        result = self.assert_native(prediction, truth)
        self.assertEqual(result['metrics']['obj_gt'], 3)
        self.assertEqual(result['metrics']['obj_detected_075_percentage'], 1)

    def test_fixed_seed_geometric_cases_against_native(self):
        rng = np.random.default_rng(260630)
        for case in range(24):
            truth = np.zeros((31, 47), dtype=np.int32)
            prediction = np.zeros_like(truth)
            for labels in [truth, prediction]:
                for label in range(1, int(rng.integers(2, 7))):
                    y, x = int(rng.integers(0, 28)), int(rng.integers(0, 42))
                    h, w = int(rng.integers(1, 11)), int(rng.integers(1, 12))
                    labels[y:y+h, x:x+w] = label * 13
            with self.subTest(case=case):
                self.assert_native(prediction, truth)

    def test_rle_overlap_follows_input_order_even_when_scores_differ(self):
        first = np.zeros((13, 21), dtype=np.uint8); first[2:8, 3:10] = 1
        last = np.zeros_like(first); last[5:11, 7:15] = 1
        records = [rle_record(first, .95), rle_record(last, .6, text=False)]
        result = predictions_to_label_map(records, first.shape)
        expected = np.zeros_like(first, dtype=np.int32)
        expected[first > 0] = 1; expected[last > 0] = 2
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result[6, 8], 2)  # Lower confidence last mask owns overlap.
        self.assert_native(result, expected)
        reverse = predictions_to_label_map(list(reversed(records)), first.shape)
        reversed_expected = np.zeros_like(expected)
        reversed_expected[last > 0] = 1; reversed_expected[first > 0] = 2
        np.testing.assert_array_equal(reverse, reversed_expected)
        self.assert_native(reverse, expected)

    def test_score_endpoint_and_uncompressed_rle(self):
        mask = np.ones((3, 5), dtype=np.uint8)
        for score in [.49, .5]:
            self.assertFalse(predictions_to_label_map([rle_record(mask, score)], mask.shape).any())
        prediction = predictions_to_label_map(
            [{'score': np.nextafter(.5, 1), 'segmentation': {'size': [3, 5], 'counts': [0, 15]}}],
            mask.shape)
        np.testing.assert_array_equal(prediction, mask)
        np.testing.assert_array_equal(predictions_to_label_map([], mask.shape), np.zeros_like(mask))

    def test_macro_average_is_per_image_and_retains_undefined_values(self):
        gt1 = np.zeros((17, 29), dtype=np.int32); gt1[1:3, 1:3] = 1
        gt2 = np.zeros_like(gt1)
        for i in range(3):
            gt2[6:9, 2+7*i:5+7*i] = i + 1
        rows = [self.assert_native(gt1, gt1), self.assert_native(np.zeros_like(gt2), gt2)]
        result = macro_average(rows)
        self.assertEqual(result['images'], 2)
        self.assertEqual(result['metrics']['obj_detected_075_percentage'], .5)  # pooled: .25
        self.assertEqual(result['metrics']['obj_gt'], 2)
        full = self.assert_native(np.ones_like(gt1), np.ones_like(gt1))
        result = macro_average([rows[0], full])
        self.assertTrue(np.isnan(result['metrics']['Boundary Precision']))
        self.assertEqual(result['undefined_images']['Boundary Precision'], 1)
        with self.assertRaises(ValueError):
            macro_average([])

    def test_input_validation_and_fixed_reference(self):
        good = np.zeros((5, 7), dtype=np.int32)
        for invalid in [np.zeros((5, 7), dtype=float), -np.ones((5, 7), dtype=int),
                        np.zeros((5, 7, 1), dtype=int), np.zeros((0, 7), dtype=int)]:
            with self.assertRaises(ValueError):
                evaluate_label_maps(invalid, good)
        with self.assertRaises(ValueError):
            evaluate_label_maps(np.zeros((7, 5), dtype=int), good)
        with self.assertRaises(ValueError):
            predictions_to_label_map([rle_record(good, float('nan'))], good.shape)
        with self.assertRaises(ValueError):
            predictions_to_label_map([rle_record(good)], (7, 5))
        with self.assertRaises(ValueError):
            predictions_to_label_map([{'score': 1, 'segmentation': {'size': [5, 7], 'counts': [36]}}], good.shape)
        self.assertEqual(len(reference_provenance()['files']), 3)
        self.assertNotIn('detectron2', sys.modules)


if __name__ == '__main__':
    unittest.main()
