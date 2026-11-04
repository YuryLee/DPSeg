import csv
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
import torch

from tools.audit_course_depth_backbone import csv_detections, rescore, checkpoint_architecture


class FinalAuditTests(unittest.TestCase):
    def prediction(self):
        mask = np.zeros((1024, 1024), dtype=np.uint8, order='F')
        mask[3:5, 4:6] = 1
        rle = mask_utils.encode(mask)
        rle['counts'] = rle['counts'].decode('ascii')
        return {'category_id': 1, 'score': .9, 'segmentation': rle}

    def write_predictions(self, folder, *, changed_score=False):
        value = self.prediction()
        with (folder/'submission.csv').open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['image_id', 'PredictionString'])
            writer.writerow([10, json.dumps([value])])
        if changed_score:
            value = {**value, 'score': .8}
        (folder/'predictions.jsonl').write_text(json.dumps({'image_id': 10, 'predictions': [value]})+'\n')

    def test_csv_rescore_perfect_small_instance(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_predictions(folder)
            detections = csv_detections(folder/'submission.csv', folder/'predictions.jsonl', [10])
            coco = COCO()
            coco.dataset = {'info': {}, 'images': [{'id': 10, 'height': 1024, 'width': 1024}],
                'categories': [{'id': 1, 'name': 'component'}], 'annotations': [{
                    'id': 1, 'image_id': 10, 'category_id': 1, 'iscrowd': 0, 'area': 4,
                    'bbox': [4, 3, 2, 2], 'segmentation': self.prediction()['segmentation']}]}
            coco.createIndex()
            actual = rescore(coco, detections, [10], folder/'coco.log')
            self.assertAlmostEqual(actual['mask_ap'], 1.)
            self.assertAlmostEqual(actual['mask_ap_small'], 1.)

    def test_csv_jsonl_score_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_predictions(folder, changed_score=True)
            with self.assertRaises(ValueError):
                csv_detections(folder/'submission.csv', folder/'predictions.jsonl', [10])

    def test_missing_official_image_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write_predictions(folder)
            with self.assertRaises(ValueError):
                csv_detections(folder/'submission.csv', folder/'predictions.jsonl', [10, 11])

    def test_large_backbone_cannot_be_proven_by_metadata_alone(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.pt'
            state = {}
            for branch in ('swin_rgb', 'swin_d'):
                prefix = 'model.model.pixel_level_module.encoder.'+branch+'.'
                state[prefix+'embeddings.patch_embeddings.projection.weight'] = torch.zeros(96, 3, 4, 4)
                for block in range(6):
                    state[prefix+f'encoder.layers.2.blocks.{block}.dummy'] = torch.zeros(1)
            torch.save({'epoch': 15, 'model_metadata': {'depth_encoding': 'xyz',
                'camera_intrinsics_used': True, 'parameters': 1}, 'model': state}, path)
            with self.assertRaises(ValueError):
                checkpoint_architecture(path, 'swinl_xyz')


if __name__ == '__main__':
    unittest.main()
