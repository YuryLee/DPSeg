import copy
import unittest
from unittest.mock import patch

import torch
from torchvision.models.detection import roi_heads as tv_roi_heads

from rgbd_seg.model import build_model
from rgbd_seg.v5_model import PaperModel, build_v5_model, optimizer_groups


class V5ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def tiny(self, variant):
        base = build_model({'mode': 'E2', 'arch': 'tiny', 'size': 64, 'seed': 17})
        return PaperModel(base.core, base.config, base.config, variant,
                          [id(p) for p in base.core.backbone.rgb.parameters()])

    def inputs(self):
        sample = {'image': torch.zeros(3, 64, 64), 'depth': torch.ones(1, 64, 64),
                  'depth_valid': torch.ones(1, 64, 64)}
        mask = torch.zeros(1, 64, 64, dtype=torch.uint8)
        mask[:, 15:41, 10:35] = 1
        target = {'boxes': torch.tensor([[10., 15., 35., 41.]]),
                  'labels': torch.ones(1, dtype=torch.int64), 'masks': mask}
        return [sample], [target]

    def test_loss_variants_preserve_base_losses_and_restore_function(self):
        inputs, targets = self.inputs()
        original = tv_roi_heads.maskrcnn_loss
        results = {}
        for variant in ('baseline', 'bce2', 'bace'):
            model = self.tiny(variant).train()
            torch.manual_seed(99)
            losses = model(inputs, targets)
            results[variant] = {k: v.detach() for k, v in losses.items()}
            sum(losses.values()).backward()
            self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()))
            self.assertIs(tv_roi_heads.maskrcnn_loss, original)
        for variant in ('bce2', 'bace'):
            for key, value in results['baseline'].items():
                torch.testing.assert_close(results[variant][key], value, rtol=0, atol=0)
            self.assertGreater(float(results[variant]['loss_mask_aux']), 0)
        torch.testing.assert_close(results['bce2']['loss_mask_aux'], results['bce2']['loss_mask'])

    def test_exception_restores_loss_function(self):
        model = self.tiny('bace').train()
        original = tv_roi_heads.maskrcnn_loss
        with patch.object(model.core, 'forward', side_effect=RuntimeError('intentional')):
            with self.assertRaisesRegex(RuntimeError, 'intentional'):
                model(*self.inputs())
        self.assertIs(tv_roi_heads.maskrcnn_loss, original)

    def test_inference_loss_variants_are_identical_without_targets(self):
        inputs, _ = self.inputs()
        reference = None
        for variant in ('baseline', 'bce2', 'bace'):
            with torch.inference_mode():
                predictions = self.tiny(variant).eval()(inputs)[0]
            if reference is None:
                reference = predictions
            else:
                for key in reference:
                    torch.testing.assert_close(predictions[key], reference[key], rtol=0, atol=0)

    def test_shared_r50_initialization_rng_and_optimizer_groups(self):
        config = {'mode': 'E2', 'arch': 'r50_v2', 'size': 720, 'seed': 23,
                  'frozen_batch_norm': True, 'trainable_backbone_layers': 3,
                  'zero_init_depth': True}
        baseline, _ = build_v5_model(config, 'baseline')
        rng = torch.get_rng_state().clone()
        original = {k: v.clone() for k, v in baseline.state_dict().items()}
        del baseline
        frequency, _ = build_v5_model(config, 'freqfusion')
        self.assertTrue(torch.equal(torch.get_rng_state(), rng))
        state = frequency.state_dict()
        for key, value in original.items():
            self.assertTrue(torch.equal(state[key], value), key)
        groups, records = optimizer_groups(frequency)
        self.assertEqual(groups[0]['lr'], 0.00002)
        self.assertEqual(groups[1]['lr'], 0.0002)
        self.assertFalse(any('.fusions.' in n for n in records[0]['parameter_names']))
        self.assertTrue(any('.fusions.' in n for n in records[1]['parameter_names']))
        added = sum(p.numel() for p in frequency.core.backbone.rgb.fpn.fusion_parameters())
        self.assertEqual(added, 314070)
        total = sum(p.numel() for p in frequency.parameters() if p.requires_grad)
        self.assertEqual(sum(r['numel'] for r in records), total)


if __name__ == '__main__':
    unittest.main()
