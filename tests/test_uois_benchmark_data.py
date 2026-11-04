import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from rgbd_seg.uois_benchmark_data import BenchmarkDataset, build_manifest


class BenchmarkDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.calibration = {'camera_K': [[100., 0., 3.5], [0., 100., 2.5], [0., 0., 1.]],
                            'depth_scale_m': .001, 'source': 'synthetic unit-test camera',
                            'status': 'provided', 'image_size': [6, 8]}

    def sample(self, root, folders, labels):
        arrays = [np.full((6, 8, 3), 100, np.uint8), np.full((6, 8), 1000, np.uint16), labels]
        for folder, array in zip(folders, arrays):
            (root / folder).mkdir(parents=True, exist_ok=True)
            Image.fromarray(array).save(root / folder / 'frame.png')

    def loader(self, manifest):
        path = self.root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        return BenchmarkDataset(path)

    def test_original_osd_annotation_and_tiny_instance_preserved(self):
        labels = np.zeros((6, 8), np.uint16)
        labels[1, 1] = 7
        self.sample(self.root, ('image_color', 'disparity', 'annotation'), labels)
        (self.root / 'annotation_fixed').mkdir()
        Image.fromarray(np.zeros_like(labels)).save(self.root / 'annotation_fixed/frame.png')
        manifest = build_manifest('osd', self.root, self.calibration, expected_images=1)
        self.assertEqual(manifest['samples'][0]['objects'], [{'label_id': 7, 'area': 1}])
        data = self.loader(manifest)
        self.assertEqual(data.ground_truth(0)[1, 1], 7)
        # An inference call must not open or derive anything from GT.
        (self.root / 'annotation/frame.png').unlink()
        sample = data.inference_sample(0)
        self.assertEqual(set(sample), {'image', 'depth_m', 'depth_valid', 'camera_K'})
        self.assertAlmostEqual(sample['depth_m'][0, 0, 0].item(), 1.)

    def test_ocid_floor_and_table_backgrounds(self):
        labels = np.zeros((6, 8), np.uint16)
        labels[0, :3] = [1, 2, 3]
        for scene in ('floor', 'table'):
            self.sample(self.root / f'ARID20/{scene}/bottom/seq01', ('rgb', 'depth', 'label'), labels)
        manifest = build_manifest('ocid', self.root, self.calibration, expected_images=2)
        data = self.loader(manifest)
        np.testing.assert_array_equal(data.ground_truth(0)[0, :3], [0, 2, 3])
        np.testing.assert_array_equal(data.ground_truth(1)[0, :3], [0, 0, 3])

    def test_missing_pair_rejected_instead_of_sorted_zip_truncation(self):
        self.sample(self.root, ('image_color', 'disparity', 'annotation'), np.zeros((6, 8), np.uint16))
        (self.root / 'disparity/frame.png').rename(self.root / 'disparity/different.png')
        with self.assertRaisesRegex(ValueError, 'filenames'):
            build_manifest('osd', self.root, self.calibration)

    def test_count_and_content_changes_rejected(self):
        self.sample(self.root, ('image_color', 'disparity', 'annotation'), np.zeros((6, 8), np.uint16))
        with self.assertRaisesRegex(ValueError, 'Expected 111'):
            build_manifest('osd', self.root, self.calibration, expected_images=111)
        manifest = build_manifest('osd', self.root, self.calibration, expected_images=1)
        Image.fromarray(np.ones((6, 8), np.uint16)).save(self.root / 'annotation/frame.png')
        with self.assertRaisesRegex(ValueError, 'changed'):
            self.loader(manifest)

    def test_colour_depth_is_rejected(self):
        self.sample(self.root, ('image_color', 'disparity', 'annotation'), np.zeros((6, 8), np.uint16))
        Image.fromarray(np.zeros((6, 8, 3), np.uint8)).save(self.root / 'disparity/frame.png')
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            build_manifest('osd', self.root, self.calibration)


if __name__ == '__main__':
    unittest.main()
