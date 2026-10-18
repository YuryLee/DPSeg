"""Behavioral checks for sampling quotas, loss gradients and inference isolation."""
import unittest
from unittest.mock import patch

import torch
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss, sample_point

from rgbd_seg.course_small_target import POLICY, SmallTargetQuotaLoss, build_trainable
from rgbd_seg.course_dafusion import build_trainable as build_base


class SamplingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        self.config = Mask2FormerConfig(num_labels=1, train_num_points=12544,
            oversample_ratio=3., importance_sample_ratio=.75)
        self.weights = {'loss_mask': 5., 'loss_dice': 5., 'loss_cross_entropy': 2.}
        self.loss = SmallTargetQuotaLoss(self.config, self.weights)

    def test_original_area_cutoff_matching_permutation_and_native_tail_preservation(self):
        masks = torch.zeros(2, 32, 32)
        masks[0, 8:12, 7:10] = 1
        masks[1, 20:25, 20:25] = 1
        original = torch.rand(2, 12544, 2)
        indices = [(torch.tensor([0, 1]), torch.tensor([1, 0]))]
        with self.loss.bind_targets([masks], [torch.tensor([1024., 1025.])]) as stats:
            actual = self.loss.quota_coordinates(original.clone(), indices)
            self.assertTrue(torch.equal(actual[0], original[0]))
            self.assertTrue(torch.equal(actual[1, :-256], original[1, :-256]))
            labels = sample_point(masks[0:1, None], actual[1:2, -256:-128], align_corners=False)
            torch.testing.assert_close(labels, torch.ones_like(labels), atol=1e-6, rtol=0)
            pool = self.loss._target_context[0][(0, 0)][1]
            self.assertTrue(((actual[1, -128:, None] == pool[None]).all(-1).any(-1)).all())
            self.assertEqual(stats['small_targets'], 1)
            self.assertEqual(stats['foreground_points_forced'], 128)
        self.assertIsNone(self.loss._target_context)

    def test_large_instances_have_native_loss_gradients_and_rng(self):
        native = Mask2FormerLoss(self.config, self.weights)
        masks = torch.zeros(1, 32, 32); masks[0, 5:20, 6:25] = 1
        indices = [(torch.tensor([0]), torch.tensor([0]))]
        first = torch.randn(1, 2, 8, 8, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        torch.manual_seed(41)
        expected = native.loss_masks(first, [masks], indices, 1)
        expected_grad = torch.autograd.grad(sum(expected.values()), first)[0]
        expected_rng = torch.get_rng_state()
        torch.manual_seed(41)
        with self.loss.bind_targets([masks], [torch.tensor([2048.])]):
            actual = self.loss.loss_masks(second, [masks], indices, 1)
        actual_grad = torch.autograd.grad(sum(actual.values()), second)[0]
        for key in expected:
            self.assertTrue(torch.equal(actual[key], expected[key]))
        self.assertTrue(torch.equal(actual_grad, expected_grad))
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_one_pixel_edges_and_resize_empty_fallback(self):
        masks = torch.zeros(2, 16, 16); masks[0, 0, 0] = 1
        indices = [(torch.tensor([0, 1]), torch.tensor([0, 1]))]
        original = torch.rand(2, 12544, 2)
        with self.loss.bind_targets([masks], [torch.tensor([1., 1.])]) as stats:
            got = self.loss.quota_coordinates(original.clone(), indices)
            self.assertEqual(stats['small_empty_after_resize'], 1)
            self.assertTrue(torch.equal(got[1], original[1]))
            self.assertTrue(((got >= 0) & (got <= 1)).all())
            labels = sample_point(masks[0:1, None], got[0:1, -256:-128], align_corners=False)
            torch.testing.assert_close(labels, torch.ones_like(labels))

    def test_empty_band_falls_back_and_context_is_cleared_after_failure(self):
        masks = torch.ones(1, 16, 16)
        with self.assertRaisesRegex(RuntimeError, 'sentinel'):
            with self.loss.bind_targets([masks], [torch.tensor([256.])]) as stats:
                self.assertEqual(stats['small_boundary_fallback'], 1)
                raise RuntimeError('sentinel')
        self.assertIsNone(self.loss._target_context)
        with self.assertRaisesRegex(RuntimeError, 'scoped'):
            self.loss.quota_coordinates(torch.rand(1, 12544, 2), [])

    def test_mixed_small_large_auxiliary_losses_are_finite_and_differentiable(self):
        masks = torch.zeros(2, 32, 32)
        masks[0, 2:4, 2:4] = 1; masks[1, 10:25, 10:25] = 1
        pred = torch.randn(1, 3, 8, 8, requires_grad=True)
        cls = torch.randn(1, 3, 2, requires_grad=True)
        aux = torch.randn(1, 3, 8, 8, requires_grad=True)
        with self.loss.bind_targets([masks], [torch.tensor([4., 2048.])]) as stats:
            losses = self.loss(pred, cls, [masks], [torch.zeros(2, dtype=torch.long)],
                [{'masks_queries_logits': aux, 'class_queries_logits': cls}])
            self.assertEqual(stats['loss_mask_calls'], 2)
            self.assertEqual(stats['matched_small_pairs_across_layers'], 2)
        sum(losses.values()).backward()
        for value in (pred, cls, aux):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_invalid_quota_and_original_area_alignment_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'fit'):
            SmallTargetQuotaLoss(Mask2FormerConfig(train_num_points=100), self.weights)
        for areas in [torch.tensor([0.]), torch.tensor([float('nan')]), torch.tensor([1., 2.])]:
            with self.subTest(areas=areas), self.assertRaises(ValueError):
                with self.loss.bind_targets([torch.ones(1, 16, 16)], [areas]): pass


class ModelIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.base, _ = build_base(seed=17)
        cls.variant, cls.metadata = build_trainable(seed=17)

    @classmethod
    def tearDownClass(cls):
        del cls.base, cls.variant

    def test_identical_initial_parameters_and_prediction_tensors(self):
        base = self.base.state_dict(); variant = self.variant.state_dict()
        self.assertEqual(set(base), set(variant))
        for name, value in base.items(): self.assertTrue(torch.equal(value, variant[name]), name)
        self.assertEqual(self.metadata['parameters'], 87517981)
        self.base.eval(); self.variant.eval()
        pixels = torch.randn(1, 7, 64, 64); pixels[:, 6] = 1
        with torch.no_grad():
            expected = self.base.model(pixel_values=pixels, return_dict=True)
            with patch.object(self.variant.model.criterion, 'forward', side_effect=AssertionError('Inference accessed targets')):
                actual = self.variant.model(pixel_values=pixels, return_dict=True)
        self.assertTrue(torch.equal(expected.class_queries_logits, actual.class_queries_logits))
        self.assertTrue(torch.equal(expected.masks_queries_logits, actual.masks_queries_logits))
        self.assertIsNone(self.variant.model.criterion._target_context)

    def test_real_decoder_training_gradients_context_and_target_metadata(self):
        adapter = self.variant.train()
        adapter.short_edge = 64; adapter.max_size = 64
        image = torch.rand(3, 64, 64)
        sample = {'image': image, 'depth_m': torch.ones(1,64,64),
            'depth_valid': torch.ones(1,64,64,dtype=torch.bool),
            'camera_K': torch.tensor([[60.,0.,32.],[0.,60.,32.],[0.,0.,1.]])}
        masks = torch.zeros(2,64,64,dtype=torch.uint8)
        masks[0,4:6,4:6]=1; masks[1,15:55,15:55]=1
        target={'masks':masks,'labels':torch.ones(2,dtype=torch.long),
            'area':torch.tensor([4.,1600.]),'image_id':torch.tensor([123])}
        with torch.autocast('cpu',dtype=torch.bfloat16):
            losses=adapter([sample],[target])
        self.assertTrue(all(torch.isfinite(v) for v in losses.values()))
        sum(losses.values()).backward()
        stats=adapter.last_batch_metadata['small_target_sampling']
        self.assertEqual(stats['small_targets'],1)
        self.assertGreater(stats['matched_small_pairs_across_layers'],1)
        self.assertEqual(stats['foreground_points_forced'],128*stats['matched_small_pairs_across_layers'])
        self.assertIsNone(adapter.model.criterion._target_context)
        self.assertTrue(torch.isfinite(adapter.model.class_predictor.weight.grad).all())
        adapter.zero_grad(set_to_none=True)


if __name__ == '__main__': unittest.main()
