"""CPU contract tests; no pretrained download or GPU is used."""
import contextlib
import io
import json
import math
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
from PIL import Image
from rgbd_seg.augment import augment
from rgbd_seg.data import RGBDDataset, dataset_fingerprint, generate_synthetic
from tools import run_v5 as runner


class FakeModel(torch.nn.Module):
    def __init__(self, config, variant):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.4))
        self.other = torch.nn.Parameter(torch.tensor(.2))
        self.base_config = dict(config)
        self.config = {**config, 'paper_variant': variant, 'paper_variant_config': {'fake': True}}

    def forward(self, samples, targets=None):
        if self.training:
            loss = (self.weight - samples[0]['image'].mean()).square() + self.other.square()
            result = {'loss_mask': loss}
            if self.config['paper_variant'] == 'bce2':
                result['loss_mask_aux'] = loss
            return result
        result = []
        for sample in samples:
            h, w = sample['image'].shape[-2:]
            mask = torch.zeros((1, 1, h, w), device=self.weight.device)
            mask[:, :, 3:10, 4:12] = .9
            result.append({'masks': mask, 'scores': torch.tensor([.9], device=mask.device),
                           'labels': torch.tensor([1], device=mask.device),
                           'boxes': torch.tensor([[4., 3., 12., 10.]], device=mask.device)})
        return result


def fake_build(config, variant, init_rgb=None):
    return FakeModel(config, variant), {'sha256': runner.sha256(init_rgb)} if init_rgb else None


def fake_load(path, device='cpu'):
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = FakeModel(checkpoint['base_config'], checkpoint['variant'])
    model.load_state_dict(checkpoint['state_dict'])
    return model.to(device).eval(), checkpoint


def fake_groups(model, lr, backbone_factor):
    groups = [{'params': [model.weight], 'lr': lr * backbone_factor}, {'params': [model.other], 'lr': lr}]
    metadata = [{'name': name, 'parameter_names': [parameter], 'numel': 1, 'lr': group['lr']}
                for name, parameter, group in zip(('rgb', 'other'), ('weight', 'other'), groups)]
    return groups, metadata


class V5RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)

    def test_scheduler_matches_legacy_including_first_update(self):
        parameter = torch.nn.Parameter(torch.tensor(1.))
        opt = torch.optim.AdamW([parameter], lr=.0002)
        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: runner.schedule_multiplier(step, 800))
        for step in range(801):
            legacy = min(1., (step + 1) / 100) * (.1 + .9 * (1 + math.cos(math.pi * step / 800)) / 2)
            self.assertAlmostEqual(opt.param_groups[0]['lr'], .0002 * legacy)
            opt.step()
            scheduler.step()

    def test_sampling_and_augmentation_match_legacy_without_extra_rng_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            generate_synthetic(folder, n_train=3, n_val=0, n_test=0, size=32)
            dataset = RGBDDataset(folder, 'train')
            cached = [dataset[index] for index in range(len(dataset))]
            rng, aug_rng = random.Random(23), random.Random(1023)
            old_rng, old_aug = random.Random(23), random.Random(1023)
            for _ in range(20):
                expected_indices = [old_rng.randrange(len(cached)) for _ in range(2)]
                before = runner.digest_state(old_aug.getstate())
                expected = [augment(*cached[index], old_aug, .35) for index in expected_indices]
                actual, trace = runner.next_batch(cached, dataset.samples, rng, aug_rng)
                self.assertEqual(trace['batch_indices'], expected_indices)
                self.assertEqual(trace['augmentation_rng_before_sha256'], before)
                self.assertEqual(trace['augmentation_rng_after_sha256'], runner.digest_state(old_aug.getstate()))
                for old_pair, new_pair in zip(expected, actual):
                    for old_mapping, new_mapping in zip(old_pair, new_pair):
                        for key in old_mapping:
                            self.assertTrue(torch.equal(old_mapping[key], new_mapping[key]))

    def test_optimizer_groups_reject_duplicate_missing_or_wrong_names(self):
        model = FakeModel({}, 'baseline')
        groups, metadata = fake_groups(model, .0002, .1)
        runner.validate_groups(model, groups, metadata)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            runner.validate_groups(model, groups[:1], metadata[:1])
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            runner.validate_groups(model, groups + groups, metadata + metadata)
        metadata[0]['parameter_names'] = ['other']
        with self.assertRaisesRegex(ValueError, 'names'):
            runner.validate_groups(model, groups, metadata)

    def test_prediction_identity_rejects_reordering_and_wrong_geometry(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            generate_synthetic(root / 'data', n_train=0, n_val=2, n_test=0, size=32)
            dataset = RGBDDataset(root / 'data', 'val')
            rows = [{'sample_id': record['id'], 'image_id': record['image_id'], 'height': 32,
                     'width': 32, 'masks_rle': []} for record in dataset.samples]
            path = root / 'predictions.jsonl'
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            runner.validate_prediction_ids(path, dataset)
            path.write_text(''.join(json.dumps(row) + '\n' for row in reversed(rows)))
            with self.assertRaisesRegex(ValueError, 'IDs/order'):
                runner.validate_prediction_ids(path, dataset)
            rows[0]['width'] = 31
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'geometry'):
                runner.validate_prediction_ids(path, dataset)

    def test_fake_train_eval_roundtrip_cross_dataset_and_receipt_tampering(self):
        with tempfile.TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            root = Path(folder)
            data = root / 'data'
            generate_synthetic(data, n_train=2, n_val=1, n_test=0, size=32)
            config = root / 'config.json'
            config.write_text(json.dumps({'mode': 'E2', 'arch': 'r50_v2', 'seed': 17, 'score_threshold': .01}))
            init, protocol = root / 'init.pt', root / 'protocol.md'
            torch.save({}, init)
            protocol.write_text('Explicit preflight protocol; not a model-quality result.\n')
            common = ['--data', str(data), '--config', str(config), '--init-rgb', str(init),
                      '--steps', '8', '--threads', '2', '--protocol', str(protocol)]
            with patch('rgbd_seg.v5_model.build_v5_model', side_effect=fake_build), \
                 patch('rgbd_seg.v5_model.optimizer_groups', side_effect=fake_groups), \
                 patch('rgbd_seg.v5_model.load_v5_model', side_effect=fake_load):
                for variant in ('baseline', 'bce2'):
                    runner.train(runner.parser().parse_args(['train', *common, '--variant', variant, '--output', str(root / variant)]))
                first, second = [json.loads((root / variant / 'run.json').read_text()) for variant in ('baseline', 'bce2')]
                self.assertEqual(first['phase'], 'preflight')
                self.assertEqual(first['completed_steps'], 8)
                self.assertEqual(first['shared_initialization'], second['shared_initialization'])
                self.assertEqual(first['sampling_trace_sha256'], second['sampling_trace_sha256'])
                loss = json.loads((root / 'bce2/loss.jsonl').read_text().splitlines()[0])
                self.assertEqual(loss['loss'], loss['loss_mask'] + loss['loss_mask_aux'])
                state = torch.load(root / 'baseline/training_state.pt', weights_only=True)
                self.assertFalse(state['resume_supported'])
                with self.assertRaises(FileExistsError):
                    runner.train(runner.parser().parse_args(['train', *common, '--variant', 'baseline', '--output', str(root / 'baseline')]))
                checkpoint_path = root / 'baseline/model.pt'
                cross = root / 'cross'
                generate_synthetic(cross, n_train=0, n_val=0, n_test=1, size=32)
                manifest = json.loads((cross / 'manifest.json').read_text())
                manifest['depth_scale'] = .0001
                manifest['provenance'] = 'public_tless_subset'
                # Two overlapping per-instance masks must stay independent in COCO/area receipt.
                sample = manifest['samples'][0]
                sample.pop('instances')
                sample['instance_masks'] = []
                for index, box in enumerate(((2, 2, 9, 9), (6, 6, 12, 12))):
                    mask = torch.zeros((32, 32), dtype=torch.uint8)
                    x1, y1, x2, y2 = box
                    mask[y1:y2, x1:x2] = 255
                    name = f'mask{index}.png'
                    Image.fromarray(mask.numpy()).save(cross / name)
                    sample['instance_masks'].append({'path': name, 'label': 1})
                (cross / 'manifest.json').write_text(json.dumps(manifest))
                runner.evaluate(runner.parser().parse_args(['eval', '--data', str(cross), '--split', 'test',
                    '--checkpoint', str(checkpoint_path), '--output', str(root / 'eval'), '--threads', '2', '--protocol', str(protocol)]))
                result = json.loads((root / 'eval/evaluation.json').read_text())
                self.assertEqual(len(result['metrics']), 12)
                self.assertEqual(result['training_depth_scale'], .001)
                self.assertEqual(result['evaluation_depth_scale'], .0001)
                self.assertTrue(result['cross_dataset_evaluation'])
                self.assertEqual(result['original_gt_areas']['per_frame'][0]['areas_px'], [49, 36])
                predictions = runner.RLEPredictions(root / 'eval/predictions.jsonl')
                self.assertEqual(tuple(predictions[0]['masks'].shape), (1, 1, 32, 32))
                checkpoint = torch.load(checkpoint_path, weights_only=True)
                ds = RGBDDataset(cross, 'test')
                ds.classes = ['different']
                with self.assertRaisesRegex(ValueError, 'classes'):
                    runner.check_training_receipt(checkpoint_path, checkpoint, ds, dataset_fingerprint(cross))
                ds.classes = ['object']
                protocol.write_text('changed protocol')
                with self.assertRaisesRegex(ValueError, 'protocol changed'):
                    runner.check_training_receipt(checkpoint_path, checkpoint, ds, dataset_fingerprint(cross))


if __name__ == '__main__':
    unittest.main()
