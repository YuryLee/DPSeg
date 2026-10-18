import copy
from pathlib import Path
import unittest
from tools.select_course_interventions import check_baseline_settings, choose_method, read


def metrics(small, overall):
    return {'mask_ap_small':small, 'mask_ap':overall}


class InterventionSelectionTests(unittest.TestCase):
    def setUp(self):
        self.gates = {m:{'APsmall_gain_at_least':.005, 'AP_drop_at_most':.005}
                      for m in ('crop', 'focal')}
        self.baseline = metrics(.46, .79)

    def test_both_conditions_are_required(self):
        result = choose_method(self.baseline, {'crop':metrics(.464, .80),
                                'focal':metrics(.48, .78)}, self.gates)
        self.assertEqual(result['recommended_method_for_next_stage'], 'native')
        self.assertFalse(result['comparisons']['crop']['adopt'])
        self.assertFalse(result['comparisons']['focal']['adopt'])

    def test_best_eligible_small_ap_then_overall_ap_wins(self):
        result = choose_method(self.baseline, {'crop':metrics(.48, .79),
                                'focal':metrics(.49, .79)}, self.gates)
        self.assertEqual(result['recommended_method_for_next_stage'], 'focal')
        result = choose_method(self.baseline, {'crop':metrics(.49, .80),
                                'focal':metrics(.49, .79)}, self.gates)
        self.assertEqual(result['recommended_method_for_next_stage'], 'crop')

    def test_invalid_scores_are_rejected(self):
        for score in (None, float('nan'), float('inf'), -1, 2):
            with self.subTest(score=score), self.assertRaises((ValueError, TypeError)):
                choose_method(self.baseline, {'crop':metrics(score, .79),
                              'focal':metrics(.49, .79)}, self.gates)

    def test_real_frozen_protocols_match_baseline_settings(self):
        root = Path(__file__).resolve().parents[1]/'reports'
        baseline = read(root/'course_resolution_ddp/protocol.json')
        for method in ('crop', 'focal'):
            check_baseline_settings(read(root/f'course_crop_focal/protocol_{method}_v1.json'), baseline)

    def test_optimizer_and_evaluation_changes_are_rejected(self):
        root = Path(__file__).resolve().parents[1]/'reports'
        baseline = read(root/'course_resolution_ddp/protocol.json')
        protocol = read(root/'course_crop_focal/protocol_crop_v1.json')
        for group, key, value in (('training', 'lr', .1), ('evaluation', 'max_detections', 200)):
            changed = copy.deepcopy(protocol)
            changed[group][key] = value
            with self.subTest(group=group, key=key), self.assertRaises(ValueError):
                check_baseline_settings(changed, baseline)


if __name__ == '__main__':
    unittest.main()
