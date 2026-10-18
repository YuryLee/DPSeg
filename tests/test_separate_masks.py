"""Ensure overlapping original instance masks survive the manifest adapter."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

from rgbd_seg.data import RGBDDataset, audit_dataset, dataset_fingerprint
from rgbd_seg.evaluation import evaluate_predictions
from rgbd_seg.instance_metrics import evaluate_extra_predictions


class SeparateMaskTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        Image.fromarray(np.full((10, 12, 3), 100, np.uint8)).save(self.root / 'rgb.png')
        Image.fromarray(np.full((10, 12), 7000, np.uint16)).save(self.root / 'depth.png')
        self.first = np.zeros((10, 12), np.uint8)
        self.second = self.first.copy()
        self.first[1:7, 1:7] = 255
        self.second[4:9, 4:10] = 255
        Image.fromarray(self.first).save(self.root / 'a.png')
        Image.fromarray(self.second).save(self.root / 'b.png')
        self.manifest = {'provenance': 'public_tless_subset', 'depth_scale': .0001,
                         'classes': ['object'], 'samples': [{'id': 'test', 'split': 'test',
                         'rgb': 'rgb.png', 'depth': 'depth.png', 'instance_masks': [
                             {'path': 'a.png', 'label': 1, 'object_id': 3},
                             {'path': 'b.png', 'label': 1, 'object_id': 3}]}]}
        self.save()

    def tearDown(self):
        self.temp.cleanup()

    def save(self):
        (self.root / 'manifest.json').write_text(json.dumps(self.manifest))

    def test_overlap_same_identity_exact_masks_and_perfect_coco(self):
        ds = RGBDDataset(self.root, 'test')
        sample, target = ds[0]
        self.assertIsNone(ds.samples[0]['instance_path'])
        self.assertEqual(len(ds.samples[0]['instance_mask_paths']), 2)
        np.testing.assert_array_equal(target['masks'][0].numpy(), self.first != 0)
        np.testing.assert_array_equal(target['masks'][1].numpy(), self.second != 0)
        self.assertEqual(int((target['masks'].sum(0) > 1).sum()), 9)
        self.assertEqual(target['area'].tolist(), [36., 30.])
        self.assertAlmostEqual(float(sample['depth'][0, 0, 0]), .7, places=6)
        pred = {'masks': target['masks'].float(), 'labels': target['labels'],
                'scores': torch.tensor([.99, .98])}
        self.assertAlmostEqual(evaluate_predictions(ds, [pred])['mask_ap'], 100.)
        with self.assertRaisesRegex(ValueError, 'binary and disjoint'):
            evaluate_extra_predictions(ds, [pred])
        audit = audit_dataset(self.root)
        self.assertEqual(audit['splits']['test']['annotated_images'], 1)
        self.assertEqual(audit['instances'], 2)

    def test_fingerprint_includes_every_separate_mask(self):
        initial = dataset_fingerprint(self.root)
        for name in ['a.png', 'b.png']:
            original = (self.root / name).read_bytes()
            array = np.array(Image.open(self.root / name))
            array[0, 0] = 255
            Image.fromarray(array).save(self.root / name)
            self.assertNotEqual(dataset_fingerprint(self.root), initial)
            (self.root / name).write_bytes(original)
        self.assertEqual(dataset_fingerprint(self.root), initial)

    def test_empty_annotation_list_is_labeled(self):
        self.manifest['samples'][0]['instance_masks'] = []
        self.manifest['samples'][0]['split'] = 'train'
        self.save()
        ds = RGBDDataset(self.root, 'train')
        self.assertEqual(ds[0][1]['masks'].shape, (0, 10, 12))
        self.assertEqual(audit_dataset(self.root)['splits']['train']['annotated_images'], 1)

    def test_paths_labels_and_mutually_exclusive_formats(self):
        item = self.manifest['samples'][0]['instance_masks'][0]
        for path in ['../a.png', str(self.root / 'a.png'), '', None]:
            item['path'] = path
            self.save()
            with self.assertRaises(ValueError):
                RGBDDataset(self.root, 'test')
        item['path'] = 'a.png'
        for label in [0, 2, True, '1']:
            item['label'] = label
            self.save()
            with self.assertRaises(ValueError):
                RGBDDataset(self.root, 'test')
        item['label'] = 1
        self.manifest['samples'][0]['instances'] = 'a.png'
        self.save()
        with self.assertRaises(ValueError):
            RGBDDataset(self.root, 'test')

    def test_nonbinary_empty_wrong_shape_and_color_masks_rejected(self):
        cases = [np.full((10, 12), 2, np.uint8), np.zeros((10, 12), np.uint8),
                 np.ones((9, 12), np.uint8), np.ones((10, 12, 3), np.uint8)]
        for array in cases:
            Image.fromarray(array).save(self.root / 'a.png')
            with self.assertRaises(ValueError):
                RGBDDataset(self.root, 'test')[0]


if __name__ == '__main__':
    unittest.main()
