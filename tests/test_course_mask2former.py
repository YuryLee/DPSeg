"""CPU regressions for independent instance masks and COCO label fidelity."""
import unittest

import numpy as np
from pycocotools import mask as mask_utils
import torch
from torch.nn import functional as F
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights

from rgbd_seg.course_mask2former import COCO_CATEGORY_IDS, instance_predictions


class CourseMask2FormerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_coco_sparse_ids_match_torchvision(self):
        categories = MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta["categories"]
        self.assertEqual(list(COCO_CATEGORY_IDS), [i for i, label in enumerate(categories)
            if label not in ("N/A", "__background__")])
        self.assertEqual(len(COCO_CATEGORY_IDS), 80)

    def test_overlapping_masks_remain_independent_and_original_size(self):
        cls = torch.full((2, 81), -20.)
        cls[0, 0] = cls[1, 11] = 20.
        masks = torch.full((2, 2, 2), 2.)
        predictions = instance_predictions(cls, masks, input_size=(4, 4), original_size=(8, 8))
        self.assertEqual(len(predictions), 2)
        self.assertEqual({p["category_id"] for p in predictions}, {1, 13})
        for item in predictions:
            decoded = mask_utils.decode(item["segmentation"])
            self.assertEqual(decoded.shape, (8, 8))
            self.assertEqual(decoded.sum(), 64)
            self.assertAlmostEqual(item["score"], torch.sigmoid(torch.tensor(2.)).item(), places=6)
            self.assertEqual(item["bbox"], [0., 0., 8., 8.])

    def test_no_object_stays_in_softmax_denominator(self):
        cls = torch.full((1, 81), -20.)
        cls[0, 0] = 5.
        cls[0, 80] = 20.
        predictions = instance_predictions(cls, torch.ones(1, 2, 2),
            input_size=(4, 4), original_size=(4, 4))
        self.assertEqual(predictions, [])

    def test_restore_logits_before_threshold_and_score(self):
        cls = torch.full((1, 81), -20.)
        cls[0, 2] = 20.
        logits = torch.tensor([[[-3., 2.], [1., -2.]]])
        reference = F.interpolate(F.interpolate(logits[None], size=(3, 4), mode="bilinear",
            align_corners=False), size=(7, 9), mode="bilinear", align_corners=False)[0, 0]
        binary = reference > 0
        expected_score = (reference.sigmoid() * binary).sum() / (binary.sum() + 1e-6)
        prediction = instance_predictions(cls, logits, input_size=(3, 4), original_size=(7, 9),
            max_detections=1)[0]
        np.testing.assert_array_equal(mask_utils.decode(prediction["segmentation"]), binary.numpy())
        self.assertAlmostEqual(prediction["score"], expected_score.item(), places=7)

    def test_author_query_class_topk_can_repeat_query(self):
        cls = torch.full((1, 81), -20.)
        cls[0, 0], cls[0, 1] = 10., 9.
        predictions = instance_predictions(cls, torch.ones(1, 2, 2),
            input_size=(2, 2), original_size=(2, 2), max_detections=2)
        self.assertEqual(len(predictions), 2)
        self.assertEqual([p["category_id"] for p in predictions], [1, 2])
        self.assertTrue(all(p["query_id"] == 0 for p in predictions))

    def test_zero_logits_and_empty_queries_produce_no_masks(self):
        for classes, masks in [(torch.zeros(2, 81), torch.zeros(2, 2, 2)),
                               (torch.zeros(0, 81), torch.zeros(0, 2, 2))]:
            self.assertEqual(instance_predictions(classes, masks, input_size=(2, 2),
                original_size=(2, 2), score_threshold=0), [])

    def test_chunking_preserves_masks_scores_and_candidate_limit(self):
        generator = torch.Generator().manual_seed(17)
        classes = torch.randn(12, 81, generator=generator)
        masks = torch.randn(12, 3, 3, generator=generator)
        kwargs = dict(input_size=(8, 8), original_size=(16, 16), score_threshold=0, max_detections=7)
        first = instance_predictions(classes, masks, chunk_size=1, **kwargs)
        second = instance_predictions(classes, masks, chunk_size=8, **kwargs)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 7)

    def test_invalid_logits_and_configuration_fail(self):
        valid_classes, valid_masks = torch.zeros(1, 81), torch.ones(1, 2, 2)
        for changed in (dict(max_detections=101), dict(score_threshold=float("nan")),
                        dict(chunk_size=0), dict(original_size=(0, 1))):
            kwargs = dict(input_size=(2, 2), original_size=(2, 2))
            kwargs.update(changed)
            with self.assertRaises(ValueError):
                instance_predictions(valid_classes, valid_masks, **kwargs)
        for classes, masks in [(torch.zeros(1, 134), valid_masks),
                               (valid_classes, torch.full((1, 2, 2), float("nan")))]:
            with self.assertRaises(ValueError):
                instance_predictions(classes, masks, input_size=(2, 2), original_size=(2, 2))


if __name__ == "__main__":
    unittest.main()
