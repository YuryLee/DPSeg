import copy
import unittest

from tools.check_course_prediction_reload import compare_predictions


class PredictionReloadTests(unittest.TestCase):
    def setUp(self):
        self.reference = [{'category_id': 1, 'score': .8,
                           'segmentation': {'size': [1024, 1024], 'counts': 'example'}}]

    def test_only_small_score_roundoff_is_allowed(self):
        actual = copy.deepcopy(self.reference)
        actual[0]['score'] += 1e-7
        self.assertLess(compare_predictions(self.reference, actual), 1e-6)
        actual[0]['score'] += 1e-4
        with self.assertRaises(ValueError):
            compare_predictions(self.reference, actual)

    def test_mask_category_count_changes_are_rejected(self):
        variants = [[], [dict(self.reference[0], category_id=2)],
                    [dict(self.reference[0], segmentation={'size': [1024, 1024], 'counts': 'changed'})]]
        for actual in variants:
            with self.subTest(actual=actual), self.assertRaises(ValueError):
                compare_predictions(self.reference, actual)

    def test_nonfinite_and_out_of_range_scores_are_rejected(self):
        for score in (float('nan'), float('inf'), -1, 2):
            with self.subTest(score=score), self.assertRaises(ValueError):
                compare_predictions(self.reference, [dict(self.reference[0], score=score)])


if __name__ == '__main__':
    unittest.main()
