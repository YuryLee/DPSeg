import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch

from rgbd_seg.uoais_training_data import UOAISTrainingDataset


def encode(mask):
    rle = mask_utils.encode(np.asfortranarray(mask, dtype=np.uint8))
    rle['counts'] = rle['counts'].decode('ascii')
    return rle


class UOAISTrainingDataTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calibration = {
            'camera_K': [[100., 0., 3.5], [0., 101., 2.5], [0., 0., 1.]],
            'image_size': [6, 8], 'depth_scale_m': .001,
            'depth_semantics': 'camera_axis_z', 'invalid_depth_codes': [0, 65535],
            'source': 'synthetic fixture; explicitly not a real UOAIS calibration', 'status': 'assumed'}
        (self.root / 'annotations').mkdir()
        self.visible = np.zeros((6, 8), np.uint8); self.visible[1:4, 2:5] = 1
        self.overlap = np.zeros((6, 8), np.uint8); self.overlap[2:5, 3:6] = 1
        self.amodal = np.ones((6, 8), np.uint8)
        self.documents = {}
        for split in ('train', 'val'):
            images = []
            for image_id in (10, 20):
                rgb = np.full((6, 8, 3), 120, np.uint8)
                depth = np.full((6, 8), 1000, np.uint16)
                depth[0, :3] = [0, 65535, 2500]
                for folder, array in [('rgb', rgb), ('depth', depth)]:
                    (self.root / split / folder).mkdir(parents=True, exist_ok=True)
                    Image.fromarray(array).save(self.root / split / folder / f'{image_id}.png')
                images.append({'id': image_id, 'height': 6, 'width': 8,
                               'file_name': f'rgb/{image_id}.png',
                               'depth_file_name': f'depth/{image_id}.png'})
            annotations = [
                {'id': 1, 'image_id': 10, 'category_id': 7, 'iscrowd': 0,
                 'visible_mask': encode(self.visible), 'segmentation': encode(self.amodal),
                 'area': 48, 'bbox': [0, 0, 8, 6]},
                {'id': 2, 'image_id': 10, 'category_id': 7, 'iscrowd': 0,
                 'visible_mask': encode(self.overlap), 'segmentation': encode(self.amodal)},
                {'id': 3, 'image_id': 10, 'category_id': 7, 'iscrowd': 1,
                 'segmentation': encode(self.amodal)},
                {'id': 4, 'image_id': 20, 'category_id': 7, 'iscrowd': 0,
                 'visible_mask': [], 'segmentation': encode(self.amodal)},
            ]
            self.documents[split] = {'images': images, 'annotations': annotations,
                                     'categories': [{'id': 7, 'name': 'any object'}]}
        self.write()

    def write(self):
        for split, document in self.documents.items():
            path = self.root / 'annotations' / f'coco_anns_uoais_sim_{split}.json'
            path.write_text(json.dumps(document))

    def dataset(self, split='train', calibration=None):
        return UOAISTrainingDataset(self.root, split, calibration or self.calibration)

    def test_visible_targets_are_independent_and_never_amodal(self):
        dataset = self.dataset()
        sample, target = dataset[0]
        self.assertEqual(set(sample), {'image', 'depth_m', 'depth_valid', 'camera_K'})
        self.assertEqual(tuple(sample['image'].shape), (3, 6, 8))
        np.testing.assert_array_equal(target['masks'][0], self.visible)
        np.testing.assert_array_equal(target['masks'][1], self.overlap)
        self.assertTrue((target['masks'].sum(0) > 1).any())
        self.assertEqual(target['labels'].tolist(), [1, 1])
        self.assertEqual(target['area'].tolist(), [9, 9])
        self.assertEqual(target['boxes'].tolist(), [[2, 1, 5, 4], [3, 2, 6, 5]])
        self.assertEqual(target['annotation_ids'].tolist(), [1, 2])
        self.assertEqual(dataset.summary()['crowd_annotations_excluded'], 1)
        self.assertEqual(dataset.summary()['calibration']['status'], 'assumed')

    def test_empty_visible_and_unannotated_images_stay(self):
        dataset = self.dataset()
        self.assertEqual(len(dataset), 2)
        sample, target = dataset[1]
        self.assertEqual(tuple(target['masks'].shape), (0, 6, 8))
        self.assertEqual(tuple(target['boxes'].shape), (0, 4))
        self.assertEqual(target['labels'].dtype, torch.int64)
        self.documents['train']['annotations'] = [a for a in self.documents['train']['annotations'] if a['image_id'] == 10]
        self.write()
        self.assertEqual(self.dataset()[1][1]['masks'].shape[0], 0)

    def test_missing_visible_mask_is_error_even_with_amodal_present(self):
        for missing in ('absent', None):
            annotation = self.documents['train']['annotations'][0]
            if missing == 'absent':
                annotation.pop('visible_mask', None)
            else:
                annotation['visible_mask'] = None
            self.write()
            with self.assertRaisesRegex(ValueError, 'amodal fallback'):
                self.dataset()

    def test_depth_invalid_codes_zeroed_without_masking_targets(self):
        dataset = self.dataset()
        sample = dataset.raw_sample(0)
        self.assertEqual(sample['depth_valid'].dtype, torch.bool)
        self.assertEqual(sample['depth_valid'][0, 0, :3].tolist(), [False, False, True])
        self.assertEqual(sample['depth_m'][0, 0, :3].tolist(), [0., 0., 2.5])
        torch.testing.assert_close(sample['camera_K'], torch.tensor(self.calibration['camera_K']))
        depth_path = self.root / 'train/depth/10.png'
        Image.fromarray(np.zeros((6, 8), np.uint16)).save(depth_path)
        sample, target = dataset[0]
        self.assertFalse(sample['depth_valid'].any())
        self.assertEqual(target['area'].tolist(), [9, 9])

    def test_explicit_calibration_required_and_isolated(self):
        for key in ['camera_K', 'depth_scale_m', 'source', 'status', 'image_size',
                    'depth_semantics', 'invalid_depth_codes']:
            calibration = copy.deepcopy(self.calibration); calibration.pop(key)
            with self.subTest(key=key), self.assertRaises((ValueError, KeyError)):
                self.dataset(calibration=calibration)
        bad = copy.deepcopy(self.calibration); bad['invalid_depth_codes'] = [-1]
        with self.assertRaises(ValueError):
            self.dataset(calibration=bad)
        bad = copy.deepcopy(self.calibration); bad['image_size'] = [8, 6]
        with self.assertRaisesRegex(ValueError, 'resolution'):
            self.dataset(calibration=bad)
        for scale in (1e-50, 1e36):
            bad = copy.deepcopy(self.calibration); bad['depth_scale_m'] = scale
            with self.subTest(scale=scale), self.assertRaisesRegex(ValueError, 'Metric depth'):
                self.dataset(calibration=bad).raw_sample(0)
        dataset = self.dataset()
        self.calibration['camera_K'][0][0] = 1
        self.assertEqual(dataset.raw_sample(0)['camera_K'][0, 0].item(), 100)

    def test_test_split_rejected_and_coco_ids_local_to_split(self):
        with self.assertRaisesRegex(ValueError, 'never test'):
            self.dataset('test')
        train, val = self.dataset('train'), self.dataset('val')
        self.assertEqual(train.records[0]['id'], val.records[0]['id'])
        self.assertNotEqual(train.records[0]['sample_id'], val.records[0]['sample_id'])
        self.assertEqual(set(train.summary()['annotation_receipts']), {'train', 'val'})
        self.documents['val']['images'][0]['split'] = 'test'; self.write()
        with self.assertRaisesRegex(ValueError, 'split mismatch'):
            self.dataset('train')

    def test_author_repeated_image_rows_do_not_repeat_training_samples(self):
        document = self.documents['train']
        first = document['images'][0]
        first['date_captured'] = '2021-07-30 07:50:53.098306'
        document['images'].append(copy.deepcopy(first))
        repeated = copy.deepcopy(first)
        repeated['date_captured'] = '2021-07-30 07:50:53.101892'
        document['images'].append(repeated)
        self.write()
        dataset = self.dataset()
        self.assertEqual(len(dataset), 2)
        self.assertEqual(dataset.raw_target(0)['annotation_ids'].tolist(), [1, 2])
        receipt = dataset.annotation_receipts['train']
        self.assertEqual(receipt['source_image_rows'], 4)
        self.assertEqual(receipt['repeated_image_rows_collapsed'], 2)
        for field, value in [('file_name', 'rgb/20.png'), ('depth_file_name', 'depth/20.png'),
                             ('height', 5), ('width', 7), ('split', 'val'), ('license', 9)]:
            with self.subTest(field=field):
                changed = copy.deepcopy(repeated)
                changed[field] = value
                document['images'][-1] = changed
                self.write()
                with self.assertRaisesRegex(ValueError, 'Conflicting repeated image'):
                    self.dataset()

    def test_conflicting_and_orphan_ids_rejected_before_coco_can_overwrite(self):
        original = copy.deepcopy(self.documents)
        for kind in ('image', 'annotation', 'orphan', 'category'):
            self.documents = copy.deepcopy(original)
            document = self.documents['train']
            if kind == 'image':
                repeated = copy.deepcopy(document['images'][0])
                repeated['width'] += 1
                document['images'].append(repeated)
            elif kind == 'annotation':
                document['annotations'].append(copy.deepcopy(document['annotations'][0]))
            elif kind == 'orphan':
                document['annotations'][0]['image_id'] = 100
            else:
                document['annotations'][0]['category_id'] = 100
            self.write()
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.dataset()

    def test_wrong_split_absolute_missing_and_shared_paths_rejected(self):
        original = self.documents['train']['images'][0]['file_name']
        for path in ('../val/rgb/10.png', str(self.root / 'val/rgb/10.png'), 'rgb/missing.png'):
            self.documents['train']['images'][0]['file_name'] = path; self.write()
            with self.subTest(path=path), self.assertRaises(ValueError):
                self.dataset()
        self.documents['train']['images'][0]['file_name'] = original; self.write()
        val_path = self.root / 'val/rgb/10.png'
        val_path.unlink()
        val_path.symlink_to(self.root / 'train/rgb/10.png')
        with self.assertRaisesRegex(ValueError, 'wrong-split'):
            self.dataset()
        val_path.unlink()
        os.link(self.root / 'train/rgb/10.png', val_path)
        with self.assertRaisesRegex(ValueError, 'shared input'):
            self.dataset()

    def test_image_depth_and_mask_dimensions_checked_without_resize(self):
        dataset = self.dataset()
        Image.fromarray(np.ones((5, 8, 3), np.uint8)).save(self.root / 'train/rgb/10.png')
        with self.assertRaisesRegex(ValueError, 'RGB'):
            dataset.raw_sample(0)
        Image.fromarray(np.ones((6, 8, 3), np.uint8)).save(self.root / 'train/rgb/10.png')
        Image.fromarray(np.ones((6, 8), np.uint8)).save(self.root / 'train/depth/10.png')
        with self.assertRaisesRegex(ValueError, 'uint16 depth'):
            dataset.raw_sample(0)
        self.documents['train']['annotations'][0]['visible_mask'] = encode(np.ones((5, 8), np.uint8))
        self.write()
        with self.assertRaisesRegex(ValueError, 'RLE dimensions'):
            self.dataset().raw_target(0)

    def test_polygons_uncompressed_rle_and_one_pixel_instances(self):
        self.documents['train']['annotations'][0]['visible_mask'] = [[1, 1, 3, 1, 3, 3, 1, 3]]
        self.documents['train']['annotations'][1]['visible_mask'] = {'size': [6, 8], 'counts': [0, 1, 47]}
        self.write()
        target = self.dataset().raw_target(0)
        self.assertEqual(target['area'].tolist(), [4, 1])
        self.assertEqual(target['boxes'][1].tolist(), [0, 0, 1, 1])
        self.documents['train']['annotations'][0]['visible_mask'] = [[0, 0, 1]]; self.write()
        with self.assertRaisesRegex(ValueError, 'polygon'):
            self.dataset().raw_target(0)

    def test_invalid_crowd_ignore_and_train_val_manifest_presence(self):
        for field, value in [('iscrowd', 2), ('ignore', 1)]:
            self.documents['train']['annotations'][0][field] = value; self.write()
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.dataset()
            self.documents['train']['annotations'][0][field] = 0
        self.write()
        (self.root / 'annotations/coco_anns_uoais_sim_val.json').unlink()
        with self.assertRaises(FileNotFoundError):
            self.dataset('train')


if __name__ == '__main__':
    unittest.main()
