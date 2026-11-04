import unittest
from tools.evaluate_uoais_system import STAGES, select_stage


class SystemSelectionTests(unittest.TestCase):
    def summary(self):
        return {'scope': 'full_validation', 'images': 5000,
                'metrics': {name: {'Objects F-measure': .5, 'Boundary F-measure': .4} for name in STAGES}}

    def test_ties_prefer_simpler_pipeline(self):
        result = self.summary()
        self.assertEqual(select_stage(result), 'rgbd_fusion')
        result['metrics']['depth_point']['Objects F-measure'] = .6
        self.assertEqual(select_stage(result), 'depth_point')
        result['metrics']['depth_point_tiled']['Objects F-measure'] = .6
        result['metrics']['depth_point_tiled']['Boundary F-measure'] = .5
        self.assertEqual(select_stage(result), 'depth_point_tiled')

    def test_test_scores_and_partial_validation_cannot_select(self):
        for scope, images in [('final_test', 5000), ('engineering_smoke', 8), ('full_validation', 4999)]:
            result = self.summary(); result.update(scope=scope, images=images)
            with self.subTest(scope=scope, images=images), self.assertRaises(ValueError):
                select_stage(result)

    def test_undefined_primary_metric_cannot_select(self):
        result = self.summary()
        result['metrics']['depth_point']['Objects F-measure'] = None
        with self.assertRaises(ValueError):
            select_stage(result)


if __name__ == '__main__':
    unittest.main()
