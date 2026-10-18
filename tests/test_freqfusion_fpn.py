"""CPU mathematics, source parity, pretrained-state and RNG tests for FreqFusion."""
from collections import OrderedDict
import contextlib
import importlib.util
import io
from pathlib import Path
import types
import unittest
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.ops import FeaturePyramidNetwork
from torchvision.ops.feature_pyramid_network import LastLevelMaxPool
from rgbd_seg.freqfusion_fpn import FreqFusion, FreqFusionFPN, carafe, xavier_init

ROOT = Path(__file__).resolve().parents[1]


def manual_carafe(x, weights, kernel, group=1, up=1):
    """Independent scalar-neighborhood definition with zero outside the image."""
    b, c, h, w = x.shape
    padded = F.pad(x, [kernel//2]*4)
    total = torch.zeros((b, c, h*up, w*up), dtype=x.dtype, device=x.device)
    for iy in range(kernel):
        for ix in range(kernel):
            patch = padded[:, :, iy:iy+h, ix:ix+w]
            enlarged = patch.repeat_interleave(up, 2).repeat_interleave(up, 3)
            total = total + enlarged * weights[:, iy*kernel+ix:iy*kernel+ix+1]
    return total


class FreqFusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_carafe_matches_manual_values_and_gradients(self):
        for up in (1, 2):
            torch.manual_seed(3)
            x = torch.randn(1, 7, 4, 5, dtype=torch.double, requires_grad=True)
            mask = torch.randn(1, 9, 4*up, 5*up, dtype=torch.double).softmax(1).requires_grad_()
            actual = carafe(x, mask, 3, up=up, channel_chunk=3)
            expected = manual_carafe(x, mask, 3, up=up)
            torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
            ga = torch.autograd.grad(actual.square().sum(), (x, mask), retain_graph=True)
            ge = torch.autograd.grad(expected.square().sum(), (x, mask))
            for a, e in zip(ga, ge):
                torch.testing.assert_close(a, e, atol=1e-11, rtol=1e-11)

    def test_zero_padding_not_reflection_and_group_rejection(self):
        x = torch.ones((1, 1, 3, 3))
        mask = torch.full((1, 9, 3, 3), 1/9)
        actual = carafe(x, mask, 3)
        self.assertAlmostEqual(float(actual[0, 0, 0, 0]), 4/9)
        self.assertAlmostEqual(float(actual[0, 0, 1, 1]), 1.)
        with self.assertRaises(ValueError):
            carafe(x, mask, 3, group=2)

    def test_author_root_fallback_interior_agreement(self):
        path = ROOT / 'reports/v5/sources/freqfusion_FreqFusion.py'
        spec = importlib.util.spec_from_file_location('freqfusion_author_fallback', path)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        x = torch.randn((1, 5, 8, 9), requires_grad=True)
        mask = torch.randn((1, 25, 16, 18)).softmax(1).requires_grad_()
        with contextlib.redirect_stdout(io.StringIO()):
            author = source.carafe(x, mask, 5, 1, 2)
        ours = carafe(x, mask, 5, up=2, channel_chunk=2)
        torch.testing.assert_close(ours[:, :, 4:-4, 4:-4], author[:, :, 4:-4, 4:-4])
        ga = torch.autograd.grad(ours[:, :, 4:-4, 4:-4].square().sum(), (x, mask), retain_graph=True)
        gb = torch.autograd.grad(author[:, :, 4:-4, 4:-4].square().sum(), (x, mask))
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b)

    def test_complete_core_matches_pinned_author_with_manual_zero_carafe(self):
        path = ROOT/'reports/v5/sources/freqfusion_mmdetection__mmdet__models__necks__FreqFusion.py'
        code = path.read_text().replace('from mmcv.ops.carafe import normal_init, xavier_init, carafe\n', '')
        reference = types.ModuleType('pinned_author_core')
        reference.__dict__.update(carafe=manual_carafe, xavier_init=xavier_init)
        exec(compile(code, str(path), 'exec'), reference.__dict__)
        kwargs = dict(hr_channels=8, lr_channels=8, compressed_channels=8, feature_resample=True)
        torch.manual_seed(11)
        actual, expected = FreqFusion(**kwargs), reference.FreqFusion(**kwargs)
        expected.load_state_dict(actual.state_dict())
        hr = torch.randn(1, 8, 8, 12, requires_grad=True)
        lr = torch.randn(1, 8, 4, 6, requires_grad=True)
        a, e = actual(hr, lr), expected(hr, lr)
        for ours, author in zip(a, e):
            torch.testing.assert_close(ours, author, rtol=2e-5, atol=2e-6)
        sum(v.square().mean() for v in a).backward()
        sum(v.square().mean() for v in e).backward()
        for (name, p), (_, q) in zip(actual.named_parameters(), expected.named_parameters()):
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0., name)
            torch.testing.assert_close(p.grad, q.grad, rtol=3e-4, atol=2e-6)

    def test_preserves_rng_modules_names_extra_level_and_gradients(self):
        torch.manual_seed(17)
        old = FeaturePyramidNetwork([16, 32, 64, 128], 32, LastLevelMaxPool())
        before = {key: value.clone() for key, value in old.state_dict().items()}
        state = torch.get_rng_state().clone()
        new = FreqFusionFPN(old)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        self.assertIs(new.inner_blocks, old.inner_blocks)
        self.assertIs(new.layer_blocks, old.layer_blocks)
        self.assertIs(new.extra_blocks, old.extra_blocks)
        for key, value in before.items():
            torch.testing.assert_close(value, new.state_dict()[key], rtol=0, atol=0)
        features = OrderedDict((str(i), torch.randn(1, c, 32//2**i, 48//2**i, requires_grad=True)) for i, c in enumerate((16, 32, 64, 128)))
        original, revised = old(features), new(features)
        self.assertEqual(list(original), list(revised))
        self.assertEqual([v.shape for v in original.values()], [v.shape for v in revised.values()])
        sum(v.square().mean() for v in revised.values()).backward()
        self.assertEqual({id(p) for p in new.fusion_parameters()}, {id(p) for p in new.fusions.parameters()})
        for name, p in new.fusions.named_parameters():
            self.assertTrue(p.grad is not None and torch.isfinite(p.grad).all(), name)
            self.assertGreater(float(p.grad.abs().sum()), 0., name)
        with self.assertRaises(ValueError):
            FreqFusionFPN(new)

    def test_fixed_complete_core_parameter_count_and_invalid_geometry(self):
        old = FeaturePyramidNetwork([256, 512, 1024, 2048], 256, LastLevelMaxPool())
        new = FreqFusionFPN(old)
        self.assertEqual(sum(p.numel() for p in new.fusion_parameters()), 314070)
        features = OrderedDict((str(i), torch.randn(1, c, h, w)) for i, (c,h,w) in enumerate([(256,17,24),(512,8,12),(1024,4,6),(2048,2,3)]))
        with self.assertRaisesRegex(ValueError, 'exact 2x'):
            new(features)


if __name__ == '__main__':
    unittest.main()
