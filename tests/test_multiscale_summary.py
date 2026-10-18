import copy
import tempfile
import unittest
from pathlib import Path
import json
from tools.summarize_multiscale_trial import gate_development, audit


def rows_for(gains=(3., 3., 3.), overall=(-1., -1., -1.)):
    rows = []
    for index, seed in enumerate((17, 23, 41)):
        for method, gain, latency in [('full', 0., 10.), ('highres', gains[index], 20.), ('crop_fuse', gains[index], 30.)]:
            rows.append({'dataset': 'graspnet', 'split': 'val', 'seed': seed, 'mode': 'E2', 'method': method,
                         'metrics': {'mask_ap': 50. + (overall[index] if method != 'full' else 0.), 'mask_ap_small': 10. + gain},
                         'timing': {'mean_ms': latency}})
    return rows


class GateTests(unittest.TestCase):
    def test_tie_uses_latency_and_keeps_full_default(self):
        result = gate_development(rows_for())
        self.assertEqual(result['preferred_optional_mode'], 'highres')
        self.assertEqual(result['demo_default'], 'full')
        self.assertTrue(result['methods']['highres']['passed'])

    def test_mean_gain_is_not_enough_without_two_positive_seeds(self):
        result = gate_development(rows_for((8., 0., -1.)))
        self.assertIsNone(result['preferred_optional_mode'])
        self.assertFalse(result['methods']['highres']['conditions']['positive_small_seeds_at_least_2'])

    def test_overall_ap_guardrail_and_exact_small_boundary(self):
        self.assertTrue(gate_development(rows_for((2., 2., 2.)))['methods']['highres']['passed'])
        self.assertFalse(gate_development(rows_for(overall=(-1.01, -1.01, -1.01)))['methods']['highres']['passed'])

    def test_test_data_never_changes_selection(self):
        rows = rows_for()
        baseline = gate_development(rows)
        fake_test = copy.deepcopy(rows)
        for row in fake_test:
            row['split'] = 'test'
            row['metrics'] = {'mask_ap': -1e6, 'mask_ap_small': 1e6}
        self.assertEqual(baseline, gate_development(rows + fake_test))

    def test_incomplete_matrix_rejected_before_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/'matrix.json').write_text(json.dumps({'status': 'running'}))
            with self.assertRaisesRegex(ValueError, 'not completed'):
                audit(directory, .05)


if __name__ == '__main__':
    unittest.main()
