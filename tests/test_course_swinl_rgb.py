"""Protect modality isolation and the matched Large training budget."""
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import torch

from rgbd_seg.course_swinl_rgb import build_trainable
from rgbd_seg.course_training_data import CourseTrainingDataset
from tools.run_course_dafusion import make_optimizer
from tools.run_course_depth_backbone import schedule_factor as rgbd_schedule
from tools.train_course_swinl_rgb import schedule_factor as rgb_schedule
from tools.run_course_swinl_rgb import PHYSICAL_GPUS, PER_RANK_BATCHES, SMOKE_TAIL_IMAGES
from rgbd_seg.course_resolution import rank_batches

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = '/data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip'


class LargeRGBTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(4)
        cls.adapter, cls.metadata = build_trainable(ROOT / 'pretrained/mask2former_swin_large_coco_instance')

    def test_large_backbone_and_fresh_classifier_match_rgbd_contract(self):
        model = self.adapter.model
        encoder = model.model.pixel_level_module.encoder
        self.assertEqual(list(encoder.channels), [192, 384, 768, 1536])
        self.assertEqual([len(x.blocks) for x in encoder.encoder.layers], [2, 2, 18, 2])
        self.assertFalse(any('swin_d' in k or 'fusions' in k for k in model.state_dict()))
        self.assertTrue(self.metadata['initialization']['all_other_tensors_exactly_equal'])
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            classifier = torch.nn.Linear(model.config.hidden_dim, 2)
            model._init_weights(classifier)
        self.assertTrue(torch.equal(classifier.weight, model.class_predictor.weight))
        self.assertTrue(torch.equal(classifier.bias, model.class_predictor.bias))

    def test_real_training_reads_rgb_and_targets_without_depth_pixels(self):
        original = ZipFile.read
        def read(archive, name, *args, **kwargs):
            if str(name).startswith('depth/'):
                raise AssertionError('RGB baseline attempted to read a depth map')
            return original(archive, name, *args, **kwargs)
        with patch.object(ZipFile, 'read', read):
            with CourseTrainingDataset(ARCHIVE, seed=17) as dataset:
                image, target = dataset[0]
                batch, sizes = self.adapter.prepare_batch([image], [target])
        self.assertEqual(tuple(batch['pixel_values'].shape), (1, 3, 1024, 1024))
        self.assertTrue(torch.equal(batch['mask_labels'][0], target['masks'].float()))
        self.assertEqual(set(batch), {'pixel_values', 'pixel_mask', 'mask_labels', 'class_labels'})
        self.assertEqual(sizes['original_sizes'], [(1024, 1024)])
        with self.assertRaises(ValueError):
            self.adapter.prepare_batch([{'image': image, 'depth_m': image[:1]}])

    def test_optimizer_and_schedule_match_large_rgbd(self):
        optimizer, groups = make_optimizer(self.adapter, self.metadata, 2e-4)
        self.assertEqual(sum(g['parameters'] for g in groups), self.metadata['parameters'])
        for group in groups:
            self.assertAlmostEqual(group['base_lr'], 2e-5 if group['backbone'] else 2e-4)
        for step in range(438 * 15):
            self.assertEqual(rgb_schedule(step, 438*15), rgbd_schedule(step, 438*15))
        prior = json.loads((ROOT / 'reports/course_swinl_depth_no_k_v1/swinl_depth_no_k/base15/initialization.json').read_text())
        self.assertEqual(self.metadata['pretrained_sha256'], prior['pretrained_sha256'])
        for key in ('weights', 'no_object_weight', 'train_num_points', 'oversample_ratio',
                    'importance_sample_ratio', 'decoder_layers'):
            self.assertEqual(self.metadata['loss'][key], prior['loss'][key])
        for key in ('mean', 'std', 'shortest_edge', 'longest_edge'):
            self.assertEqual(self.metadata['preprocess'][key], prior['preprocess'][key])
        del optimizer

    def test_engineering_and_full_training_cover_every_image_on_all_ranks(self):
        self.assertEqual(len(PHYSICAL_GPUS), len(PER_RANK_BATCHES))
        self.assertEqual(sum(PER_RANK_BATCHES) * 3, 33)
        for count in (33 + SMOKE_TAIL_IMAGES, 14432):
            order = list(range(count))
            batches = [rank_batches(order, rank=r, batch_sizes=PER_RANK_BATCHES)
                       for r in range(len(PHYSICAL_GPUS))]
            self.assertEqual(len({len(b) for b in batches}), 1)
            self.assertTrue(all(batch for rank in batches for batch in rank))
            merged = [i for rank in batches for batch in rank for i in batch]
            self.assertEqual(sorted(merged), order)


if __name__ == '__main__':
    unittest.main()
