import unittest
import torch
from torch.nn import functional as F
from rgbd_seg.uoais_point_training import point_loss_sums, empty_point_batch
from rgbd_seg.course_point_refinement import PairedPointHeads


class UOAISPointLossTests(unittest.TestCase):
    def test_unequal_rank_counts_recover_global_instance_mean(self):
        coarse = torch.tensor([[[0., 1.]], [[-2., 3.]], [[4., -5.]]])
        truth = torch.tensor([[[0., 1.]], [[1., 0.]], [[0., 0.]]])
        delta = torch.randn_like(coarse, requires_grad=True)
        sums = []
        for sl in (slice(0, 1), slice(1, 3)):
            sums.append(point_loss_sums({'depth': delta[sl]}, coarse[sl], truth[sl],
                                        real_instances=len(coarse[sl]))['depth'])
        ddp_average = sum(value*2/3 for value in sums)/2
        expected = F.binary_cross_entropy_with_logits(coarse+delta, truth)
        torch.testing.assert_close(ddp_average, expected)
        actual_grad, = torch.autograd.grad(ddp_average, delta, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected, delta)
        torch.testing.assert_close(actual_grad, expected_grad)

    def test_empty_rank_has_zero_loss_and_defined_zero_gradients(self):
        heads = PairedPointHeads()
        fine, coarse, depth, truth = empty_point_batch('cpu')
        losses = point_loss_sums(heads(fine, coarse, depth), coarse, truth, real_instances=0)
        total = sum(losses.values())
        self.assertEqual(float(total.detach()), 0.)
        total.backward()
        for parameter in heads.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertFalse(parameter.grad.any())

    def test_zero_initialized_heads_preserve_coarse_logits(self):
        heads = PairedPointHeads()
        fine, coarse, depth, truth = empty_point_batch('cpu')
        fine.normal_(); coarse.normal_(); depth.normal_()
        for delta in heads(fine, coarse, depth).values():
            torch.testing.assert_close(coarse+delta, coarse, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
