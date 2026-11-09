"""CPU regression checks for guard priority and the unchanged score cap."""

import unittest

import numpy as np
import torch

from rgbd_seg.course_guarded_merge import VARIANTS, identity, select_candidates
from rgbd_seg.course_tiled_probability import Candidate, encode_mask, merge_candidates


def rectangle_candidate(x, y, width, height, score, view, query):
    mask = np.zeros((1024, 1024), dtype=bool)
    mask[y:y + height, x:x + width] = True
    bounds = {
        0: (0, 0, 1024, 1024),
        1: (0, 0, 640, 640),
        2: (384, 0, 1024, 640),
    }[view]
    return Candidate(score, query, view, bounds, torch.empty(0), encode_mask(mask))


class GuardedMergeTop100Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.whole = rectangle_candidate(100, 100, 50, 50, .7, 0, 0)
        cls.duplicate = rectangle_candidate(100, 100, 50, 50, .99, 1, 0)
        # Respect the real per-view limit: 91 candidates in view 1 and 10 in view 2.
        cls.small = [rectangle_candidate(200 + (i % 10) * 15, 200 + (i // 10) * 15,
                                        5, 5, .8, 1, i + 1) for i in range(90)]
        cls.small += [rectangle_candidate(700 + i * 15, 200, 5, 5, .8, 2, i)
                      for i in range(10)]

    def test_guard_can_suppress_duplicate_then_be_lost_to_score_cap(self):
        candidates = [self.whole, self.duplicate, *self.small]
        original_scores = [c.score for c in candidates]
        legacy, _ = merge_candidates(candidates)
        self.assertIn(identity(self.duplicate), {identity(c) for c in legacy})

        for variant in VARIANTS:
            with self.subTest(variant=variant):
                selected, stats = select_candidates(candidates, variant)
                # Guard changes conflict selection, without reserving a final slot.
                self.assertEqual([identity(c) for c in selected],
                                 [identity(c) for c in self.small])
                self.assertEqual([c.score for c in selected], [.8] * 100)
                self.assertEqual(stats['protected_whole_candidates'], 1)
                self.assertEqual(stats['protected_whole_selected'], 0)
                self.assertEqual(stats['protected_whole_lost_to_top100'], 1)
                self.assertEqual(stats['top100_removed'], 1)
                self.assertEqual(stats['selected_whole'], 0)
        self.assertEqual([c.score for c in candidates], original_scores)

    def test_guard_survives_when_one_final_slot_is_available(self):
        candidates = [self.whole, self.duplicate, *self.small[:-1]]
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                selected, stats = select_candidates(candidates, variant)
                self.assertEqual(len(selected), 100)
                self.assertIs(selected[-1], self.whole)
                self.assertNotIn(identity(self.duplicate), {identity(c) for c in selected})
                self.assertEqual(stats['protected_whole_candidates'], 1)
                self.assertEqual(stats['protected_whole_selected'], 1)
                self.assertEqual(stats['protected_whole_lost_to_top100'], 0)
                self.assertEqual(stats['top100_removed'], 0)


if __name__ == '__main__':
    unittest.main()
