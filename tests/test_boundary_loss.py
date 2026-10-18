import importlib
import unittest

import torch
from torch.nn import functional as F
from torchvision.models.detection import roi_heads

from rgbd_seg.boundary_loss import bace_components, bace_loss


class BoundaryLossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(912)

    def test_28_by_28_matches_explicit_block_construction(self):
        z = torch.randn(2, 28, 28, dtype=torch.float64, requires_grad=True)
        t = torch.rand_like(z)
        # A separate reshape/repeat formulation checks non-overlapping blocks.
        def repeated_max(x):
            blocks = x.reshape(2, 7, 4, 7, 4)
            coarse = blocks.amax(dim=(2, 4))
            return coarse.repeat_interleave(4, -2).repeat_interleave(4, -1)
        q = repeated_max(t) + (z - repeated_max(z))
        expected = F.binary_cross_entropy_with_logits(z, t) + F.binary_cross_entropy_with_logits(q, t)
        actual, parts = bace_loss(z, t, return_components=True)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertEqual(set(parts), {'bce', 'bace', 'total'})
        self.assertTrue(all(v.ndim == 0 and not v.requires_grad for v in parts.values()))

    def test_soft_targets_are_preserved(self):
        z = torch.randn(1, 8, 12, requires_grad=True)
        t = torch.linspace(0, 1, z.numel()).reshape_as(z)
        soft = bace_loss(z, t)
        hard = bace_loss(z, (t >= .5).float())
        self.assertGreater(abs(float((soft - hard).detach())), 1e-4)
        soft.backward()
        self.assertTrue(torch.isfinite(z.grad).all())

    def test_component_api_keeps_auxiliary_gradient(self):
        z = torch.randn(1, 28, 28, requires_grad=True)
        t = torch.rand_like(z)
        parts = bace_components(z, t)
        self.assertEqual(set(parts), {'base', 'aux'})
        self.assertTrue(all(v.requires_grad for v in parts.values()))
        torch.testing.assert_close(parts['base'], F.binary_cross_entropy_with_logits(z, t), rtol=0, atol=0)
        parts['aux'].backward()
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(z.grad.abs().sum().item(), 0)

    def test_observed_roi_align_one_ulp_overshoot_is_preserved(self):
        # Actual seed17 step13 CUDA ROIAlign value recorded by the diagnostic.
        z = torch.linspace(-2, 2, 16, dtype=torch.float32).reshape(1, 4, 4).requires_grad_()
        t = torch.linspace(0, 1, 16, dtype=torch.float32).reshape_as(z)
        t[0, 0, 0] = 1.0000001192092896
        original_target = t.clone()
        parts = bace_components(z, t)
        original_base = F.binary_cross_entropy_with_logits(z, t)
        torch.testing.assert_close(parts['base'], original_base, rtol=0, atol=0)
        ga = torch.autograd.grad(parts['base'], z, retain_graph=True)[0]
        ge = torch.autograd.grad(original_base, z, retain_graph=True)[0]
        torch.testing.assert_close(ga, ge, rtol=0, atol=0)
        # Clamping would change this pixel's gradient, so equality is substantive.
        gc = torch.autograd.grad(F.binary_cross_entropy_with_logits(z, t.clamp(0, 1)), z)[0]
        self.assertNotEqual(ga[0, 0, 0].item(), gc[0, 0, 0].item())
        self.assertTrue(torch.equal(t, original_target))
        self.assertTrue(torch.isfinite(parts['aux']))

    def test_roundoff_tolerance_is_bounded_by_target_precision(self):
        for dtype in (torch.float32, torch.float64):
            eps = torch.finfo(dtype).eps
            z = torch.zeros(1, 4, 4, dtype=dtype, requires_grad=True)
            for endpoint in (-8 * eps, 1 + 8 * eps):
                t = torch.full_like(z, endpoint)
                self.assertTrue(torch.isfinite(bace_loss(z, t)))
            for endpoint in (-9 * eps, 1 + 9 * eps, -.001, 1.001):
                with self.assertRaisesRegex(ValueError, 'tolerance'):
                    bace_loss(z, torch.full_like(z, endpoint))
        # Low precision does not expand accepted semantic range by ~0.01.
        for dtype in (torch.float16, torch.bfloat16):
            z = torch.zeros(1, 4, 4, dtype=dtype)
            with self.assertRaisesRegex(ValueError, 'tolerance'):
                bace_loss(z, torch.full_like(z, 1 + torch.finfo(dtype).eps))

    def test_zero_auxiliary_exactly_recovers_original_bce_and_gradient(self):
        z = torch.randn(2, 28, 28, requires_grad=True)
        t = torch.rand_like(z)
        actual = bace_loss(z, t, beta=0)
        expected = F.binary_cross_entropy_with_logits(z, t)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        ga = torch.autograd.grad(actual, z, retain_graph=True)[0]
        ge = torch.autograd.grad(expected, z)[0]
        torch.testing.assert_close(ga, ge, rtol=0, atol=0)

    def test_weights_and_both_zero(self):
        z = torch.randn(1, 4, 4, requires_grad=True)
        t = torch.rand_like(z)
        _, parts = bace_loss(z, t, return_components=True)
        value = bace_loss(z, t, alpha=.25, beta=.75)
        torch.testing.assert_close(value, .25 * parts['bce'] + .75 * parts['bace'])
        zero = bace_loss(z, t, alpha=0, beta=0)
        zero.backward()
        self.assertEqual(float(zero), 0)
        self.assertTrue((z.grad == 0).all())

    def test_empty_rois_are_connected_zero(self):
        z = torch.empty(0, 28, 28, requires_grad=True)
        result, parts = bace_loss(z, torch.empty_like(z), return_components=True)
        result.backward()
        self.assertEqual(result.item(), 0)
        self.assertIsNotNone(z.grad)
        self.assertEqual(z.grad.shape, z.shape)
        self.assertTrue(all(v.item() == 0 for v in parts.values()))

    def test_large_logits_have_finite_loss_and_gradient(self):
        z = torch.linspace(-10000, 10000, 28 * 28, dtype=torch.float32).reshape(1, 28, 28)
        z.requires_grad_()
        loss = bace_loss(z, torch.rand_like(z))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(z.grad).all())
        self.assertGreater(float(z.grad.abs().sum()), 0)

    def test_low_precision_is_promoted_before_difference(self):
        for dtype in (torch.float16, torch.bfloat16):
            z = torch.linspace(-60000, 60000, 16).reshape(1, 4, 4).to(dtype).requires_grad_()
            value = bace_loss(z, torch.ones_like(z))
            value.backward()
            self.assertEqual(value.dtype, torch.float32)
            self.assertTrue(torch.isfinite(value))
            self.assertTrue(torch.isfinite(z.grad).all())

    def test_small_and_nondivisible_rois_fail_without_silent_operator_change(self):
        for shape in ((1, 2, 2), (1, 7, 8), (1, 8, 9)):
            with self.assertRaisesRegex(ValueError, 'divisible'):
                bace_loss(torch.zeros(shape), torch.zeros(shape))
            # The disabled-auxiliary control has no pooling requirement.
            self.assertTrue(torch.isfinite(bace_loss(torch.zeros(shape), torch.zeros(shape), beta=0)))
        self.assertTrue(torch.isfinite(bace_loss(torch.zeros(1, 4, 4), torch.zeros(1, 4, 4))))

    def test_scale_one_has_constant_auxiliary_and_zero_auxiliary_gradient(self):
        z = torch.randn(1, 3, 5, dtype=torch.float64, requires_grad=True)
        t = torch.rand_like(z)
        loss = bace_loss(z, t, scale=1, alpha=0)
        torch.testing.assert_close(loss, F.binary_cross_entropy_with_logits(t, t))
        loss.backward()
        torch.testing.assert_close(z.grad, torch.zeros_like(z), rtol=0, atol=0)

    def test_auxiliary_block_shift_invariance_and_gradient(self):
        z = torch.randn(1, 8, 8, dtype=torch.float64, requires_grad=True)
        t = torch.rand_like(z)
        shift = torch.tensor([[[2., -3.], [9., -.2]]]).repeat_interleave(4, -2).repeat_interleave(4, -1)
        first = bace_loss(z, t, alpha=0)
        second = bace_loss(z + shift, t, alpha=0)
        torch.testing.assert_close(first, second)
        grad = torch.autograd.grad(first, z)[0].reshape(1, 2, 4, 2, 4)
        torch.testing.assert_close(grad.sum(dim=(2, 4)), torch.zeros(1, 2, 2, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_double_precision_gradcheck_away_from_maximum_ties(self):
        z = torch.randn(1, 4, 4, dtype=torch.float64, requires_grad=True)
        t = torch.rand_like(z)
        self.assertTrue(torch.autograd.gradcheck(lambda x: bace_loss(x, t), (z,)))

    def test_no_import_side_effect_or_inference_target_fallback(self):
        original = roi_heads.maskrcnn_loss
        importlib.reload(importlib.import_module('rgbd_seg.boundary_loss'))
        self.assertIs(roi_heads.maskrcnn_loss, original)
        with self.assertRaises(TypeError):
            bace_loss(torch.zeros(1, 4, 4), None)

    def test_invalid_inputs_are_rejected(self):
        z, t = torch.zeros(1, 4, 4), torch.zeros(1, 4, 4)
        for kwargs in ({'scale': 0}, {'scale': True}, {'scale': 2.5}, {'alpha': -1}, {'beta': float('nan')}):
            with self.assertRaises(ValueError):
                bace_loss(z, t, **kwargs)
        for bad in (torch.full_like(t, -0.1), torch.full_like(t, 1.1), torch.full_like(t, float('nan'))):
            with self.assertRaises(ValueError):
                bace_loss(z, bad)
        with self.assertRaises(ValueError):
            bace_loss(z, torch.zeros(2, 4, 4))
        with self.assertRaises(ValueError):
            bace_loss(torch.full_like(z, float('inf')), t)


if __name__ == '__main__':
    unittest.main()
