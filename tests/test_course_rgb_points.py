"""CPU contract checks for the RGB base with separate depth mask refinements."""
from types import SimpleNamespace
import unittest

import torch
from torch import nn
from torch.nn import functional as F

from rgbd_seg.course_depth_backbone_points import PointHead, frozen_features, training_batch
from rgbd_seg.course_rgb_points import RGBRefinementAdapter


class TinyContractModel(nn.Module):
    """Small deterministic RGB feature producer with the Swin-L output contract."""

    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(1.))
        self.config = SimpleNamespace(num_labels=1, mask_feature_size=256)
        self.model = SimpleNamespace(pixel_level_module=SimpleNamespace(
            encoder=SimpleNamespace(channels=[192, 384, 768, 1536])))
        self.last_keys = None

    def forward(self, **kwargs):
        self.last_keys = set(kwargs)
        if self.last_keys - {'pixel_values', 'pixel_mask', 'return_dict',
                             'output_hidden_states', 'output_auxiliary_logits'}:
            raise AssertionError('Non-RGB data reached the base model')
        grid = F.avg_pool2d(kwargs['pixel_values'].mean(1, keepdim=True), 4) * self.anchor
        batch = len(grid)
        return SimpleNamespace(encoder_hidden_states=(grid.expand(-1, 192, -1, -1),),
            pixel_decoder_last_hidden_state=grid.expand(-1, 256, -1, -1),
            masks_queries_logits=torch.cat((grid, -grid), dim=1),
            class_queries_logits=grid.new_tensor([[2., -1.], [1., 0.]]).expand(batch, -1, -1))


class RGBPointsTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.adapter = RGBRefinementAdapter(TinyContractModel(), mean=[0., 0., 0.],
            std=[1., 1., 1.], short_edge=32)
        image = torch.linspace(0., 1., 3 * 32 * 32).reshape(3, 32, 32)
        self.sample = dict(image=image, depth_m=torch.ones(1, 32, 32),
                           depth_valid=torch.ones(1, 32, 32, dtype=torch.bool))
        masks = torch.zeros(2, 32, 32, dtype=torch.bool)
        masks[0, 4:20, 4:20] = True
        masks[1, 12:28, 12:28] = True
        self.target = dict(masks=masks, labels=torch.ones(2, dtype=torch.int64))

    def test_only_rgb_reaches_encoder_and_padding_mask(self):
        changed = dict(self.sample, depth_m=torch.full((1, 32, 32), float('nan')),
                       depth_valid=torch.zeros(1, 32, 32, dtype=torch.bool))
        a, size_a = self.adapter.prepare_batch([self.sample])
        b, size_b = self.adapter.prepare_batch([changed])
        direct, direct_size = self.adapter.prepare_batch([self.sample['image']])
        self.assertEqual(set(a), {'pixel_values', 'pixel_mask'})
        self.assertEqual(size_a, size_b)
        self.assertEqual(size_a, direct_size)
        for name in a:
            self.assertTrue(torch.equal(a[name], b[name]))
            self.assertTrue(torch.equal(a[name], direct[name]))
        self.assertTrue(a['pixel_mask'].all())
        before = frozen_features(self.adapter, [self.sample])
        after = frozen_features(self.adapter, [changed])
        self.assertTrue(torch.equal(before[1], after[1]))
        self.assertTrue(torch.equal(before[0].masks_queries_logits, after[0].masks_queries_logits))
        self.assertEqual(before[1].shape, (1, 448, 8, 8))
        self.assertFalse(self.adapter.metadata['rgbd_feature_fusion'])
        self.assertFalse(self.adapter.metadata['encoder_depth_used'])
        self.assertTrue(self.adapter.metadata['depth_used'])

    def test_shared_training_batch_retains_depth_and_only_head_gradients(self):
        def matcher(masks, classes, labels, class_labels):
            self.assertEqual(len(labels), 1)
            self.assertTrue(torch.equal(class_labels[0], torch.zeros(2, dtype=torch.int64)))
            return [(torch.tensor([0, 1]), torch.tensor([0, 1]))]

        torch.manual_seed(17)
        values, counts = training_batch(self.adapter, [self.sample], [self.target], matcher)
        changed = dict(self.sample, depth_m=torch.zeros_like(self.sample['depth_m']),
                       depth_valid=torch.zeros_like(self.sample['depth_valid']))
        torch.manual_seed(17)
        other, _ = training_batch(self.adapter, [changed], [self.target], matcher)
        fine, coarse, depth, truth = values
        self.assertEqual(counts, [2])
        self.assertEqual(fine.shape, (2, 448, 784))
        self.assertEqual(coarse.shape, (2, 1, 784))
        self.assertEqual(depth.shape, (2, 9, 784))
        self.assertEqual(truth.shape, (2, 1, 784))
        self.assertTrue(all(not value.requires_grad for value in values))
        for index in (0, 1, 3):
            self.assertTrue(torch.equal(values[index], other[index]))
        self.assertFalse(torch.equal(depth, other[2]))
        head = PointHead(fine_channels=448, depth_channels=9)
        loss = F.binary_cross_entropy_with_logits(coarse + head(fine, coarse, depth), truth)
        loss.backward()
        self.assertTrue(head.predictor.weight.grad.abs().sum() > 0)
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.adapter.parameters()))

    def test_annotation_intrinsics_and_misalignment_are_rejected(self):
        for extra in ('camera_K', 'targets', 'masks', 'image_id'):
            with self.assertRaises(ValueError):
                self.adapter.prepare_batch([dict(self.sample, **{extra: None})])
        with self.assertRaises(ValueError):
            self.adapter.prepare_batch([dict(self.sample, depth_m=torch.ones(1, 16, 16))])
        with self.assertRaises(ValueError):
            self.adapter.prepare_batch([self.sample, self.sample['image']])

    def test_frozen_features_rejects_a_base_returned_to_training(self):
        self.adapter.train()
        with self.assertRaises(ValueError):
            frozen_features(self.adapter, [self.sample])


if __name__ == '__main__':
    unittest.main()
