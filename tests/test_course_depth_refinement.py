import copy
import unittest
import numpy as np
import torch
from pycocotools import mask as mask_utils
from rgbd_seg.course_depth_refinement import refine_masks, refine_predictions


class RefinementTests(unittest.TestCase):
    def test_depth_step_blocks_cross_edge_votes(self):
        masks = torch.tensor([[[0, 1, 0, 1, 1]]], dtype=torch.bool)
        depth = torch.tensor([[1., 1., 2., 2., 2.]])
        valid = torch.ones_like(depth, dtype=torch.bool)
        depth_result, _ = refine_masks(masks, depth, valid, radius=1, spatial_sigma_px=10.)
        spatial_result, _ = refine_masks(masks, depth, valid, mode='spatial', radius=1, spatial_sigma_px=10.)
        self.assertTrue(torch.equal(depth_result, masks))
        self.assertEqual(spatial_result.tolist(), [[[False, False, True, True, True]]])

    def test_constant_depth_equals_spatial_control(self):
        masks = torch.zeros((2, 9, 11), dtype=torch.bool)
        masks[0, 2:6, 3:7] = True
        masks[1, 1:8, 4] = True
        depth = torch.ones((9, 11))
        valid = torch.ones_like(depth, dtype=torch.bool)
        a, sa = refine_masks(masks, depth, valid, chunk_size=1)
        b, sb = refine_masks(masks, depth, valid, mode='spatial', chunk_size=16)
        self.assertTrue(torch.equal(a, b))
        self.assertEqual(sa, sb)

    def test_invalid_centres_and_empty_fallback(self):
        masks = torch.zeros((1, 5, 5), dtype=torch.bool)
        masks[0, 2, 2] = True
        depth = torch.ones((5, 5))
        result, stats = refine_masks(masks, depth, torch.ones_like(depth, dtype=torch.bool), mode='spatial')
        self.assertTrue(torch.equal(result, masks))
        self.assertEqual(stats['empty_reverted'], 1)
        result, stats = refine_masks(masks, depth*float('nan'), torch.zeros_like(depth, dtype=torch.bool))
        self.assertTrue(torch.equal(result, masks))
        self.assertEqual(stats['changed_instances'], 0)

    def test_border_and_identity_radius(self):
        masks = torch.ones((2, 3, 4), dtype=torch.bool)
        depth = torch.ones((3, 4))
        result, _ = refine_masks(masks, depth, torch.ones_like(depth, dtype=torch.bool))
        self.assertTrue(torch.equal(result, masks))
        masks[:, 1, 1] = False
        result, _ = refine_masks(masks, depth, torch.ones_like(depth, dtype=torch.bool), radius=0)
        self.assertTrue(torch.equal(result, masks))

    def test_rle_roundtrip_preserves_scores_order_count_and_inputs(self):
        masks = [np.eye(5, dtype=np.uint8), np.ones((5, 5), dtype=np.uint8)]
        predictions = []
        for i, mask in enumerate(masks):
            encoded = mask_utils.encode(np.asfortranarray(mask))
            predictions.append({'category_id': 1, 'score': .9-i*.1,
                'segmentation': {'size': encoded['size'], 'counts': encoded['counts'].decode('ascii')}})
        before = copy.deepcopy(predictions)
        result, _ = refine_predictions(predictions, np.ones((5, 5)), np.ones((5, 5), dtype=bool), mode='depth')
        self.assertEqual(predictions, before)
        self.assertEqual([p['score'] for p in result], [.9, .8])
        self.assertEqual([p['category_id'] for p in result], [1, 1])
        for p in result:
            mask = mask_utils.decode({'size': p['segmentation']['size'], 'counts': p['segmentation']['counts'].encode('ascii')})
            self.assertEqual(mask.shape, (5, 5))
            self.assertGreater(mask.sum(), 0)


if __name__ == '__main__':
    unittest.main()
