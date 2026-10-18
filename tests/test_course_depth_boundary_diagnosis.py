import unittest
import numpy as np
from tools.diagnose_course_depth_boundary import adjacent_depth_pairs


class DepthPairsTests(unittest.TestCase):
    def test_vertical_step_and_pair_count(self):
        mask = np.zeros((3, 4), dtype=bool)
        mask[:, :2] = True
        depth = np.where(mask, 1., 2.)
        result = adjacent_depth_pairs(mask, depth, np.ones_like(mask))
        self.assertEqual(result['boundary'][0], 3)
        np.testing.assert_array_equal(result['boundary'][1], np.ones(3))
        self.assertEqual(result['interior'][0], 7)
        np.testing.assert_array_equal(result['interior'][1], np.zeros(7))

    def test_invalid_edge_is_excluded(self):
        mask = np.array([[True, False]], dtype=bool)
        result = adjacent_depth_pairs(mask, np.array([[1., 0.]]), mask.copy())
        self.assertEqual(result['boundary'][0], 1)
        self.assertEqual(result['boundary'][1].size, 0)
        self.assertEqual(result['interior'][0], 0)

    def test_flat_depth_has_no_boundary_signal(self):
        mask = np.zeros((3, 3), dtype=bool)
        mask[1, 1] = True
        result = adjacent_depth_pairs(mask, np.ones((3, 3)), np.ones_like(mask))
        self.assertEqual(result['boundary'][0], 4)
        np.testing.assert_array_equal(result['boundary'][1], np.zeros(4))
        self.assertEqual(result['interior'][0], 0)


if __name__ == '__main__':
    unittest.main()
