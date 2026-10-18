import copy
import unittest

from tools.report_course_longer_system import comparison_rows


class SystemReportTests(unittest.TestCase):
    def fixtures(self):
        candidates = [
            {'family': 'selected_base', 'variant': 'native', 'mask_ap_small': .65, 'mask_ap': .85},
            {'family': 'original_base_extended_heads', 'variant': 'tile', 'mask_ap_small': .64, 'mask_ap': .86},
            {'family': 'historical', 'variant': 'old', 'mask_ap_small': .63, 'mask_ap': .86}]
        decision = {'candidates': candidates, 'selected': candidates[0]}
        reports = {}
        for i, candidate in enumerate(candidates):
            variant = candidate['variant']
            reports[candidate['family']] = {
                'val': {variant: {key: candidate[key] for key in ('mask_ap_small', 'mask_ap')}},
                'test': {variant: {'mask_ap_small': .40 + i * .1, 'mask_ap': .8}},
                'csvs': {variant: {'path': f'/example/{variant}.csv'}}}
        return decision, reports

    def test_larger_test_score_does_not_replace_locked_winner_or_csv(self):
        decision, reports = self.fixtures()
        rows, chosen = comparison_rows(decision, reports)
        self.assertGreater(rows[-1]['test_APs'], chosen['test_APs'])
        self.assertEqual(chosen['family'], 'selected_base')
        self.assertEqual(chosen['csv']['path'], '/example/native.csv')

    def test_missing_csv_or_changed_validation_cannot_produce_complete_report(self):
        decision, reports = self.fixtures()
        incomplete = copy.deepcopy(reports)
        incomplete['original_base_extended_heads']['csvs'].clear()
        with self.assertRaisesRegex(ValueError, 'missing'):
            comparison_rows(decision, incomplete)
        reports['selected_base']['val']['native']['mask_ap_small'] = .66
        with self.assertRaisesRegex(ValueError, 'locked selection'):
            comparison_rows(decision, reports)


if __name__ == '__main__':
    unittest.main()
