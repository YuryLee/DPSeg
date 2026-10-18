"""CPU fidelity checks for the pinned DA-Fusion course adaptation.

No training/evaluation data or GPU is used. Small spatial inputs test mechanics,
not the accuracy or peak memory of the formal 800-pixel experiment.
"""
import importlib
import unittest
from unittest.mock import patch

from safetensors import safe_open
import torch

from rgbd_seg import course_dafusion as model_module
from rgbd_seg.course_mask2former import DEFAULT_CHECKPOINT


class FusionMechanicsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.default_generator.manual_seed(17)
        self.FusionStage, _ = model_module._load_upstream()
        self.ds = importlib.import_module(model_module._PACKAGE + '.deformable_self_attention')

    def test_attention_matches_upstream_output_and_input_bias_gradients(self):
        for bias_dims in (0, 3, 4):
            with self.subTest(bias_dims=bias_dims):
                tensors = [torch.randn(2, 32, n, dtype=torch.float64) for n in (13, 7, 7)]
                if bias_dims:
                    tensors.append(torch.randn(*((2, 4, 13, 7) if bias_dims == 4 else (4, 13, 7)),
                                               dtype=torch.float64))
                original = [x.clone().requires_grad_() for x in tensors]
                adapted = [x.clone().requires_grad_() for x in tensors]
                expected = self.ds.upstream_multihead_attend(*original[:3], 4,
                    original[3] if bias_dims else None)
                actual = model_module.contiguous_multihead_attend(*adapted[:3], 4,
                    adapted[3] if bias_dims else None)
                weight = torch.randn_like(expected)
                expected_grads = torch.autograd.grad((expected * weight).sum(), original)
                actual_grads = torch.autograd.grad((actual * weight).sum(), adapted)
                torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-11)
                for got, want in zip(actual_grads, expected_grads):
                    torch.testing.assert_close(got, want, rtol=1e-9, atol=1e-10)

    def test_all_four_offset_paths_receive_finite_nonzero_gradients(self):
        fusion = self.FusionStage(32, num_heads=4, n_groups=4, stride=2, use_rpb=True)
        rgb, depth = [torch.randn(2, 32, 12, 12, requires_grad=True) for _ in range(2)]
        output = fusion(rgb, depth)
        (output * torch.randn_like(output)).mean().backward()
        for unit in (fusion.ds_rgb.unit, fusion.ds_d.unit,
                     fusion.dc.unit_rgb, fusion.dc.unit_d):
            grad = unit.offset_net.net[-1].weight.grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(float(grad.abs().sum()), 0)
        for source in (rgb, depth):
            self.assertTrue(torch.isfinite(source.grad).all())
            self.assertGreater(float(source.grad.abs().sum()), 0)

    def test_rpb_global_pair_cutoff_and_private_patch_boundary(self):
        rpb_module = importlib.import_module(model_module._PACKAGE + '.relative_position_bias')
        rpb = rpb_module.ContinuousPositionBias(4, max_pairs=24)
        self.assertEqual(rpb(torch.randn(6, 2), torch.randn(4, 2)).shape, (4, 6, 4))
        self.assertIsNone(rpb(torch.randn(5, 2), torch.randn(5, 2)))
        dc = importlib.import_module(model_module._PACKAGE + '.deformable_cross_attention')
        self.assertIs(self.ds.multihead_attend, model_module.contiguous_multihead_attend)
        self.assertIs(dc.multihead_attend, model_module.contiguous_multihead_attend)
        self.assertIsNot(self.ds.upstream_multihead_attend, self.ds.multihead_attend)
        # This rechecks on-disk source against the receipt after module imports/patching.
        self.assertGreater(len(model_module.upstream_source_sha256()), 10)


@unittest.skipUnless((DEFAULT_CHECKPOINT / 'model.safetensors').exists(),
                     'Pinned local COCO checkpoint unavailable')
class PretrainedIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        before = torch.get_rng_state().clone()
        with patch('torch.cuda.manual_seed_all', side_effect=AssertionError('CPU build touched CUDA RNG')):
            cls.adapter, cls.metadata = model_module.build_trainable(seed=23,
                fusion_checkpointing=True)
        cls.rng_preserved = torch.equal(before, torch.get_rng_state())
        cls.encoder = cls.adapter.model.model.pixel_level_module.encoder

    @classmethod
    def tearDownClass(cls):
        del cls.encoder, cls.adapter

    def test_pretrained_backbones_equal_independent_and_decoder_unchanged(self):
        self.assertTrue(self.rng_preserved)
        rgb = dict(self.encoder.swin_rgb.named_parameters())
        depth = dict(self.encoder.swin_d.named_parameters())
        self.assertEqual(set(rgb), set(depth))
        for name, value in rgb.items():
            self.assertTrue(torch.equal(value, depth[name]), name)
            self.assertNotEqual(value.data_ptr(), depth[name].data_ptr(), name)
            self.assertTrue(value.requires_grad and depth[name].requires_grad)
        state = self.adapter.model.state_dict()
        prefix = 'model.pixel_level_module.encoder.'
        excluded = {'class_predictor.weight', 'class_predictor.bias', 'criterion.empty_weight'}
        checked = 0
        with safe_open(DEFAULT_CHECKPOINT / 'model.safetensors', framework='pt', device='cpu') as original:
            for name in original.keys():
                if not name.startswith(prefix) and name not in excluded:
                    self.assertTrue(torch.equal(original.get_tensor(name), state[name]), name)
                    checked += 1
        self.assertGreater(checked, 100)
        adapter = self.encoder.depth_adapter
        self.assertEqual(int(torch.count_nonzero(adapter.proj.weight)), 0)
        self.assertEqual(int(torch.count_nonzero(adapter.proj.bias)), 0)

    def test_four_scale_shapes_depth_sensitivity_and_decoder_contract(self):
        self.adapter.eval()
        torch.default_generator.manual_seed(41)
        image = torch.randn(1, 7, 64, 64)
        image[:, 6] = 1
        altered = image.clone()
        altered[:, 3:6] = torch.randn_like(altered[:, 3:6])
        with torch.no_grad():
            first = self.encoder(image).feature_maps
            second = self.encoder(altered).feature_maps
            prediction = self.adapter.model(pixel_values=image, return_dict=True)
        self.assertEqual([tuple(x.shape) for x in first],
            [(1, 96, 16, 16), (1, 192, 8, 8), (1, 384, 4, 4), (1, 768, 2, 2)])
        for before, after in zip(first, second):
            self.assertTrue(torch.isfinite(before).all())
            self.assertGreater(float((before - after).abs().max()), 1e-5)
        self.assertEqual(tuple(prediction.class_queries_logits.shape), (1, 100, 2))
        self.assertEqual(tuple(prediction.masks_queries_logits.shape), (1, 100, 16, 16))

    def test_real_depth_helper_batch_alignment_and_gt_input_rejection(self):
        from rgbd_seg.course_depth_data import model_depth_channels
        image = torch.rand(3, 64, 64)
        depth = torch.linspace(1., 2., 64).view(1, 1, 64).expand(1, 64, 64).clone()
        valid = torch.ones_like(depth, dtype=torch.bool)
        valid[:, :5, :7] = False
        depth[~valid] = 0
        sample = {'image': image, 'depth_m': depth, 'depth_valid': valid,
                  'camera_K': torch.tensor([[64., 0., 31.5], [0., 64., 31.5], [0., 0., 1.]])}
        old_size = self.adapter.short_edge, self.adapter.max_size
        self.adapter.short_edge, self.adapter.max_size = 64, 96
        try:
            masks = torch.zeros(2, 64, 64, dtype=torch.uint8)
            masks[0, 10:40, 10:40] = 1
            masks[1, 20:50, 20:50] = 1
            target = {'masks': masks, 'labels': torch.ones(2, dtype=torch.int64)}
            batch, metadata = self.adapter.prepare_batch([sample], [target])
            self.assertEqual(tuple(batch['pixel_values'].shape), (1, 7, 64, 64))
            self.assertEqual(metadata['original_sizes'], [(64, 64)])
            torch.testing.assert_close(batch['pixel_values'][0, 3:],
                                       model_depth_channels(sample, (64, 64)))
            self.assertTrue(torch.equal(batch['pixel_values'][0, 6].bool(), valid[0]))
            self.assertTrue(torch.equal(batch['mask_labels'][0].to(torch.uint8), masks))
            self.assertEqual(int((batch['mask_labels'][0].sum(0) == 2).sum()), 400)
            with self.assertRaises(ValueError):
                self.adapter.prepare_batch([{**sample, 'target': target}])
            self.adapter.eval()
            predictions = self.adapter.predict_sample(sample, score_threshold=0)
            self.assertLessEqual(len(predictions), 100)
            for item in predictions:
                self.assertEqual(set(item), {'category_id', 'score', 'segmentation'})
                self.assertEqual(item['category_id'], 1)
                self.assertEqual(item['segmentation']['size'], [64, 64])
        finally:
            self.adapter.short_edge, self.adapter.max_size = old_size

    def test_bfloat16_native_loss_backward_through_both_branches_and_adapter(self):
        self.adapter.train()
        self.adapter.zero_grad(set_to_none=True)
        torch.default_generator.manual_seed(17)
        inputs = torch.randn(1, 7, 64, 64)
        inputs[:, 6] = 1
        masks = torch.zeros(2, 64, 64)
        masks[0, 10:35, 10:35] = 1
        masks[1, 25:50, 25:50] = 1
        with torch.autocast('cpu', dtype=torch.bfloat16):
            output = self.adapter.model(pixel_values=inputs, mask_labels=[masks],
                class_labels=[torch.zeros(2, dtype=torch.int64)], return_dict=True)
        self.assertEqual(output.loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(output.loss))
        output.loss.backward()
        parameters = {
            'rgb': self.encoder.swin_rgb.embeddings.patch_embeddings.projection.weight,
            'depth': self.encoder.swin_d.embeddings.patch_embeddings.projection.weight,
            'adapter': self.encoder.depth_adapter.proj.weight,
            'classifier': self.adapter.model.class_predictor.weight,
        }
        for name, parameter in parameters.items():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.adapter.model._captured_losses = None
        self.adapter.zero_grad(set_to_none=True)


if __name__ == '__main__':
    unittest.main()
