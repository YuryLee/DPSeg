import unittest
import numpy as np
import torch

from rgbd_seg.instance_metrics import instance_metrics, predictions_to_label_map, evaluate_extra_predictions


class InstanceMetricTests(unittest.TestCase):
    def test_perfect_permuted_ids_and_boundary(self):
        gt = np.zeros((20, 30), dtype=np.int32)
        gt[2:6, 3:8], gt[12:17, 20:25] = 88, 7
        pred = np.zeros_like(gt)
        pred[gt == 88], pred[gt == 7] = 1, 99
        result = instance_metrics(pred, gt)
        self.assertTrue(all(value == 100 for value in result['metrics'].values()))
        self.assertEqual(result['counts']['matched_f_gt075'], 2)
        self.assertEqual({m['gt_label_id'] for m in result['matches_f1']}, {7, 88})
        self.assertEqual(instance_metrics(np.zeros((720,1280),dtype=np.uint8), np.zeros((720,1280),dtype=np.uint8))['boundary_tolerance_pixels'], 5)

    def test_merge_split_and_false_positive_are_penalized(self):
        gt = np.zeros((10, 15), dtype=np.int32)
        gt[2:4, 2:4], gt[2:4, 8:10] = 1, 2
        merged = (gt > 0).astype(np.int32)
        for pred, truth in ((merged, gt), (gt, merged)):
            result = instance_metrics(pred, truth)
            self.assertAlmostEqual(result['metrics']['overlap_f1'], 50)
            self.assertEqual(result['counts']['matched_f_gt075'], 0)
            self.assertEqual(len(result['matches_f1']), 1)
        one_gt = gt.copy(); one_gt[gt == 2] = 0
        result = instance_metrics(gt, one_gt)
        self.assertAlmostEqual(result['metrics']['overlap_precision'], 50)
        self.assertAlmostEqual(result['metrics']['overlap_recall'], 100)
        self.assertAlmostEqual(result['metrics']['overlap_f1'], 200 / 3)

    def test_hungarian_assignment_beats_greedy_pair_selection(self):
        gt = np.zeros((10, 15), dtype=np.int32)
        gt[2, 1:11], gt[5, 1:11] = 1, 2
        pred = np.zeros_like(gt)
        pred[2, 1:8], pred[5, 1:7] = 1, 1
        pred[2, 8:11] = 2
        result = instance_metrics(pred, gt)
        # Highest single pair is GT1/pred1; globally optimal matching instead
        # assigns GT1/pred2 and GT2/pred1, recovering 3+6 matched pixels.
        pairs = {(m['gt_label_id'], m['pred_label_id']) for m in result['matches_f1']}
        self.assertEqual(pairs, {(1, 2), (2, 1)})
        self.assertEqual(result['counts']['matched_intersection_pixels'], 9)

    def test_f075_is_dice_strict_and_iou_diagnostic_separate(self):
        gt = np.zeros((8, 10), dtype=np.int32)
        gt[2, 1:7] = 1  # area 6
        pred = np.zeros_like(gt); pred[2, 1:5] = 1  # area 4, IoU=2/3, F=.8
        result = instance_metrics(pred, gt)
        self.assertEqual(result['metrics']['object_recall_f_gt075'], 100)
        self.assertEqual(result['metrics']['object_recall_iou_ge075'], 0)
        gt[:] = 0; gt[2, 1:6] = 1
        pred[:] = 0; pred[2, 1:4] = 1  # F=6/8 exactly .75
        self.assertEqual(instance_metrics(pred, gt)['metrics']['object_recall_f_gt075'], 0)
        gt[2, 5] = 0  # IoU=3/4 exactly .75
        self.assertEqual(instance_metrics(pred, gt)['metrics']['object_recall_iou_ge075'], 100)

    def test_empty_cases_and_full_image_boundaries(self):
        empty = np.zeros((8, 9), dtype=np.int32)
        obj = empty.copy(); obj[3:5, 3:5] = 1
        both = instance_metrics(empty, empty)['metrics']
        self.assertTrue(all(value == 100 for value in both.values()))
        miss = instance_metrics(empty, obj)['metrics']
        self.assertEqual((miss['overlap_precision'], miss['overlap_recall'], miss['overlap_f1']), (100, 0, 0))
        fp = instance_metrics(obj, empty)['metrics']
        self.assertEqual((fp['overlap_precision'], fp['overlap_recall'], fp['overlap_f1']), (0, 100, 0))
        self.assertEqual(fp['object_recall_f_gt075'], 0)
        full = np.ones_like(empty)
        self.assertEqual(instance_metrics(full, full)['metrics']['boundary_f1'], 100)

    def test_boundary_tolerance_separate_from_region_overlap(self):
        gt = np.zeros((40, 40), dtype=np.int32); gt[10:20, 10:20] = 1
        pred = np.zeros_like(gt); pred[10:20, 11:21] = 1
        result = instance_metrics(pred, gt, boundary_tolerance=1)
        self.assertEqual(result['metrics']['boundary_f1'], 100)
        self.assertLess(result['metrics']['overlap_f1'], 100)
        farther = np.zeros_like(gt); farther[10:20, 15:25] = 1
        self.assertLess(instance_metrics(farther, gt, boundary_tolerance=1)['metrics']['boundary_f1'], 100)

    def test_prediction_adapter_threshold_and_score_priority(self):
        masks = np.zeros((3, 1, 8, 8), dtype=np.float32)
        masks[0, 0, 2:5, 2:5] = 1
        masks[1] = masks[0]
        masks[2, 0, 6, 6] = 1
        prediction = {'masks': masks, 'scores': np.array([.6, .9, .2])}
        result, counts = predictions_to_label_map(prediction, (8, 8), .5)
        self.assertEqual(result[3, 3], 2)
        self.assertEqual(result[6, 6], 0)
        self.assertEqual(counts['score_retained_predictions'], 2)
        self.assertEqual(counts['visible_predictions'], 1)
        self.assertEqual(counts['empty_or_fully_occluded_predictions'], 1)

    def test_macro_and_micro_are_distinct(self):
        dataset, predictions = [], []
        for i, area in enumerate((10,100)):
            mask = torch.zeros((1, 16, 16), dtype=torch.uint8)
            mask.view(-1)[:area] = 1
            dataset.append(({'image': torch.zeros(3,16,16)}, {'masks': mask, 'image_id': torch.tensor([i])}))
            predictions.append({'masks': mask[:,None].float() if i == 0 else torch.zeros(0,1,16,16), 'scores': torch.ones(1) if i == 0 else torch.zeros(0)})
        result = evaluate_extra_predictions(dataset, predictions)
        self.assertEqual(result['metrics_macro']['overlap_f1'], 50)
        self.assertAlmostEqual(result['metrics_micro']['overlap_f1'], 100/6)
        self.assertEqual(result['totals']['gt_objects'], 2)
        self.assertTrue(all(x is None for x in evaluate_extra_predictions([],[])['metrics_macro'].values()))

    def test_invalid_inputs_fail(self):
        gt = np.zeros((8,8),dtype=np.int32)
        for invalid in (np.zeros((8,8),dtype=float), np.full((8,8),-1), np.zeros((7,8),dtype=int)):
            with self.assertRaises(ValueError): instance_metrics(invalid,gt)
        with self.assertRaises(ValueError): instance_metrics(gt,gt,boundary_tolerance=float('nan'))
        with self.assertRaises(ValueError): predictions_to_label_map({'masks':np.zeros((1,8,8)), 'scores':np.array([np.nan])}, (8,8))


if __name__ == '__main__':
    unittest.main()
