import unittest

import torch
from torch.nn import functional as F

from rgbd_seg.course_point_refinement import (
    PairedPointHeads, PointHead, depth_cues, mask_boxes, point_sample,
    refine_logits, sample_shared, training_points,
)


class PointRefinementTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_shared_point_sampling_preserves_image_and_instance_coordinates(self):
        feature = torch.arange(16.).reshape(1,1,4,4)
        coords = torch.tensor([[[.125,.125],[.875,.875]], [[.375,.125],[.125,.625]]])
        actual = sample_shared(feature, coords)
        self.assertTrue(torch.equal(actual, point_sample(feature.repeat(2,1,1,1), coords)))
        self.assertEqual(actual[:,0].tolist(), [[0.,15.],[1.,8.]])

    def test_depth_cues_exclude_missing_values_and_image_wraparound(self):
        z = torch.ones(5,6)*.5
        z[:,3:] += .01
        valid = torch.ones_like(z,dtype=torch.bool)
        valid[1,1] = False
        z[1,1] = float('nan')
        a = depth_cues(z,valid)
        z[1,1] = 9999.
        self.assertTrue(torch.equal(a,depth_cues(z,valid)))
        self.assertTrue(torch.isfinite(a).all())
        self.assertEqual(a.shape,(1,9,5,6))
        self.assertEqual(float(a[0,2,2,2]),1.)
        self.assertEqual(float(a[0,1,2,0]),0.)
        self.assertEqual(float(a[0,5,2,0]),0.)

    def test_one_pixel_gt_is_retained_and_training_points_stay_in_roi(self):
        truth = torch.zeros(1,32,32)
        truth[0,13,17] = 1
        boxes = mask_boxes(truth.bool())
        coords = training_points(torch.zeros(1,8,8),truth,points=64)
        self.assertTrue((coords >= boxes[:,None,:2]).all())
        self.assertTrue((coords < boxes[:,None,2:]).all())
        self.assertTrue((boxes[:,2:] > boxes[:,:2]).all())

    def test_zero_residual_is_exact_original_upsampling(self):
        torch.manual_seed(17)
        coarse = torch.randn(2,256,256)
        fine = torch.randn(1,4,256,256)
        head = PointHead(fine_channels=4,hidden=8)
        boxes = torch.tensor([[.1,.2,.2,.3],[.5,.5,.7,.7]])
        output = refine_logits(head,coarse,fine,None,boxes,points=16,chunk_size=1)
        reference = F.interpolate(coarse[None],size=(1024,1024),mode='bilinear',align_corners=False)[0]
        self.assertTrue(torch.equal(output,reference))

    def test_shared_initialization_and_gradient_path_through_new_heads(self):
        torch.manual_seed(17)
        heads = PairedPointHeads()
        fine = torch.randn(3,352,16)
        coarse = torch.randn(3,1,16)
        cues = torch.randn(3,9,16)
        target = torch.randint(0,2,coarse.shape).float()
        optimizer = torch.optim.AdamW(heads.parameters(),lr=.001)
        for _ in range(2):
            result = heads(fine,coarse,cues)
            loss = sum(F.binary_cross_entropy_with_logits(coarse+delta,target) for delta in result.values())
            optimizer.zero_grad()
            loss.backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in heads.parameters()))
            optimizer.step()
        self.assertGreater(float(heads.plain.layers[0].weight.grad.abs().sum()),0.)
        self.assertGreater(float(heads.depth.layers[0].weight.grad[:,352:361].abs().sum()),0.)

    def test_nonzero_head_actually_changes_selected_pixels(self):
        head = PointHead(fine_channels=4,hidden=8)
        head.predictor.bias.data.fill_(2.)
        coarse = torch.full((1,256,256),-1.)
        output = refine_logits(head,coarse,torch.zeros(1,4,256,256),None,
            torch.tensor([[.4,.4,.6,.6]]),points=32)
        self.assertGreater(int((output>0).sum()),0)
        self.assertEqual(float(output[0,0,0]),-1.)


if __name__ == '__main__':
    unittest.main()
