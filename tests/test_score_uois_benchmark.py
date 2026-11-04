"""Integration checks for saved predictions, original GT and the scoring CLI."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils

from rgbd_seg.uois_benchmark_data import build_manifest
from tools.score_uois_benchmark import main, read_predictions


class ScoreBenchmarkTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for directory in ('image_color', 'disparity', 'annotation'):
            (self.root / directory).mkdir()

    def fixture(self, masks, predict):
        for index, mask in enumerate(masks):
            name = f'{index}.png'
            Image.fromarray(np.zeros((*mask.shape, 3), np.uint8)).save(self.root / 'image_color' / name)
            Image.fromarray(np.full(mask.shape, 1000, np.uint16)).save(self.root / 'disparity' / name)
            Image.fromarray(mask.astype(np.uint16)).save(self.root / 'annotation' / name)
        calibration = {'camera_K': [[100., 0., 3.5], [0., 100., 2.5], [0., 0., 1.]],
                       'depth_scale_m': .001, 'source': 'synthetic scoring test',
                       'status': 'provided', 'image_size': list(masks[0].shape)}
        manifest = build_manifest('osd', self.root, calibration, expected_images=len(masks))
        manifest_path = self.root / 'manifest.json'
        manifest_path.write_text(json.dumps(manifest))
        predictions = []
        for record, mask, enabled in zip(manifest['samples'], masks, predict):
            instances = []
            if enabled:
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle['counts'] = rle['counts'].decode('ascii')
                instances = [{'category_id': 1, 'score': .9, 'segmentation': rle}]
            predictions.append({'image_id': record['id'], 'predictions': instances})
        prediction_path = self.root / 'predictions.jsonl'
        prediction_path.write_text(''.join(json.dumps(row) + '\n' for row in predictions))
        return manifest_path, prediction_path, predictions

    def run_score(self, manifest_path, prediction_path):
        output = self.root / 'scores'
        argv = ['score_uois_benchmark', '--manifest', str(manifest_path),
                '--predictions', str(prediction_path), '--output', str(output),
                '--model-description', 'Synthetic engineering test, no model benchmark']
        with patch('sys.argv', argv), contextlib.redirect_stdout(io.StringIO()):
            main()
        return output, json.loads((output / 'evaluation.json').read_text())

    def test_original_resolution_macro_mean_and_empty_prediction(self):
        mask = np.zeros((6, 8), np.uint8)
        mask[2:4, 3:5] = 1
        manifest, predictions, _ = self.fixture([mask, mask], [True, False])
        output, result = self.run_score(manifest, predictions)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['images'], 2)
        metrics = result['summary']['metrics']
        for name in ('Objects F-measure', 'Boundary F-measure', 'obj_detected_075_percentage'):
            self.assertEqual(metrics[name], .5)
        rows = [json.loads(line) for line in (output / 'per_image.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows), 2)
        protocol = json.loads((output / 'protocol.json').read_text())
        self.assertEqual(protocol['score_comparison'], 'strictly greater than')
        self.assertEqual(len(protocol['manifest_sha256']), 64)

    def test_undefined_native_boundary_is_null_not_zero(self):
        mask = np.ones((6, 8), np.uint8)
        manifest, predictions, _ = self.fixture([mask], [True])
        output, result = self.run_score(manifest, predictions)
        self.assertEqual(result['summary']['metrics']['Objects F-measure'], 1.)
        # Native code returns zero F, but undefined precision/recall, when both
        # full-frame masks have no boundary pixels under its boundary convention.
        self.assertEqual(result['summary']['metrics']['Boundary F-measure'], 0.)
        for name in ('Boundary Precision', 'Boundary Recall'):
            self.assertIsNone(result['summary']['metrics'][name])
            self.assertEqual(result['summary']['undefined_images'][name], 1)
        self.assertNotIn('NaN', (output / 'per_image.jsonl').read_text())

    def test_missing_and_duplicate_predictions_rejected(self):
        mask = np.zeros((6, 8), np.uint8)
        _, path, rows = self.fixture([mask, mask], [False, False])
        ids = [row['image_id'] for row in rows]
        path.write_text(json.dumps(rows[0]) + '\n')
        with self.assertRaisesRegex(ValueError, 'coverage mismatch'):
            read_predictions(path, ids)
        path.write_text((json.dumps(rows[0]) + '\n') * 2)
        with self.assertRaisesRegex(ValueError, 'duplicate image ID'):
            read_predictions(path, ids)


if __name__ == '__main__':
    unittest.main()
