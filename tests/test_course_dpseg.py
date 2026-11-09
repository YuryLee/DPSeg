"""Geometry contracts for the fixed-four-view overlap experiment."""
import unittest

import numpy as np

from rgbd_seg.course_dpseg import OVERLAP_CONFIGS, four_tile_geometry
from rgbd_seg.course_tiled_probability import view_bounds


class FourTileGeometryTest(unittest.TestCase):
    def test_four_distinct_tiles_cover_every_pixel(self):
        for name, config in OVERLAP_CONFIGS.items():
            with self.subTest(name=name):
                tiles = config['bounds'][1:]
                self.assertEqual(len(set(tiles)), 4)
                coverage = np.zeros((1024, 1024), dtype=np.uint8)
                for left, top, right, bottom in tiles:
                    coverage[top:bottom, left:right] += 1
                self.assertGreaterEqual(coverage.min(), 1)
                if name == 'overlap00':
                    self.assertTrue(np.all(coverage == 1))
                else:
                    self.assertEqual(coverage.max(), 4)

    def test_current_geometry_matches_frozen_reference(self):
        self.assertEqual(four_tile_geometry(.4)['bounds'], view_bounds(1024, 1024))

    def test_realized_overlap_is_measured_from_windows(self):
        for config in OVERLAP_CONFIGS.values():
            first, second = config['bounds'][1:3]
            width = first[2] - first[0]
            overlap_px = first[2] - second[0]
            self.assertEqual(overlap_px, config['actual_overlap_px'])
            self.assertEqual(overlap_px / width, config['actual_overlap_ratio'])
            self.assertLess(abs(config['actual_overlap_ratio'] - config['target_overlap_ratio']), .001)

    def test_invalid_or_indistinguishable_windows_rejected(self):
        for value in (-.1, 1., float('nan'), float('inf'), True, .999999):
            with self.subTest(value=value), self.assertRaises(ValueError):
                four_tile_geometry(value)


if __name__ == '__main__':
    unittest.main()
