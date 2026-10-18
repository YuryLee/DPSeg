import json
from pathlib import Path
import tempfile
import unittest

from tools.lock_course_longer_system import inspect_evaluation, rank_all, sha, verify_lock


class GlobalSelectionTests(unittest.TestCase):
    def test_both_families_required_and_ap_cannot_override_aps(self):
        old = {'old': {'mask_ap_small': .62, 'mask_ap': .85}}
        families = {'selected_base': {'native': {'mask_ap_small': .64, 'mask_ap': .80}},
                    'original_base_extended_heads': {'tile': {'mask_ap_small': .63, 'mask_ap': .90}}}
        self.assertEqual(rank_all(families, old)[0]['family'], 'selected_base')
        with self.assertRaisesRegex(ValueError, 'Both planned'):
            rank_all({'selected_base': families['selected_base']}, old)
        families['selected_base']['native']['mask_ap_small'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid'):
            rank_all(families, old)

    def test_exact_tie_retains_historical_candidate(self):
        values = {'tile': {'mask_ap_small': .62, 'mask_ap': .85}}
        self.assertEqual(rank_all({'selected_base': values, 'original_base_extended_heads': values}, values)[0]['family'],
                         'historical')

    def test_rejects_partial_validation_or_changed_scored_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            ids = list(range(2584))
            prediction = folder / 'predictions.jsonl'
            prediction.write_text(''.join(json.dumps({'image_id': n, 'predictions': []}) + '\n' for n in ids))
            metrics = {'mask_ap_small': 0., 'mask_ap': 0.}
            evaluation = {'status': 'completed', 'split': 'val', 'images': 2584, 'metrics': metrics,
                          'iou_type': 'segm', 'max_detections': [1, 10, 100], 'area_ranges': [[0, 1e10], [0, 1024]],
                          'ground_truth_area': 'original annotation area preserved',
                          'iou_thresholds': [.5 + .05 * i for i in range(10)],
                          'predictions_sha256': sha(prediction), 'predicted_instances': 0}
            (folder / 'evaluation.json').write_text(json.dumps(evaluation))
            inspect_evaluation(folder, ids, metrics, {})
            with self.assertRaisesRegex(ValueError, 'Incomplete'):
                inspect_evaluation(folder, ids[:2], metrics, {})
            prediction.write_text(prediction.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'changed'):
                inspect_evaluation(folder, ids, metrics, {})

    def test_test_gate_rejects_mutated_validation_evidence(self):
        with tempfile.TemporaryDirectory() as temp:
            evidence, decision = Path(temp) / 'metrics.json', Path(temp) / 'selection.json'
            evidence.write_text('{}')
            values = {'tile': {'mask_ap_small': .62, 'mask_ap': .85}}
            candidates = rank_all({'selected_base': values, 'original_base_extended_heads': values}, values)
            decision.write_text(json.dumps({'status': 'locked', 'scope': 'both_families_full_validation',
                'selection_split': 'val', 'test_used_for_selection': False, 'selected_before_new_test': True,
                'candidates': candidates, 'selected': candidates[0],
                'artifact_sha256': {str(evidence): sha(evidence)}}))
            verify_lock(decision)
            tampered = json.loads(decision.read_text())
            tampered['selected'] = candidates[-1]
            decision.write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'ranking changed'):
                verify_lock(decision)
            tampered['selected'] = candidates[0]
            decision.write_text(json.dumps(tampered))
            evidence.write_text('{"changed": true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                verify_lock(decision)


if __name__ == '__main__':
    unittest.main()
