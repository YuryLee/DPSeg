"""Synthetic logs test integrity checks, without claiming mask-score accuracy."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from rgbd_seg.uoais_point_validation_audit import ARMS, METRICS, audit_point_epoch


class PairedPointAuditTests(unittest.TestCase):
    def fixture(self, folder):
        rows = []
        for sample, value in zip('abcd', [.2, .4, .6, .8]):
            row = {'sample_id': sample, 'arms': {}}
            for arm, offset in zip(ARMS, [0, .05, .1]):
                metrics = {key: value + offset for key in METRICS}
                if sample == 'a' and arm == 'point':
                    metrics['Boundary Precision'] = None
                row['arms'][arm] = {'predictions': [], 'metrics': metrics}
            rows.append(row)
        summary = {'scope': 'full_validation', 'epoch': 1, 'images': 4,
                   'test_used': False, 'selection_metric': 'Objects F-measure',
                   'precision': 'float32', 'metrics': {}, 'undefined_images': {}}
        for arm, mean in zip(ARMS, [.5, .55, .6]):
            summary['metrics'][arm] = {key: mean for key in METRICS}
            summary['undefined_images'][arm] = {}
        summary['metrics']['point']['Boundary Precision'] = None
        summary['undefined_images']['point']['Boundary Precision'] = 1
        self.write(folder, rows, summary)
        return rows, summary

    def write(self, folder, rows, summary):
        (folder / 'summary.json').write_text(json.dumps(summary))
        for rank in (0, 1):
            (folder / f'predictions_rank{rank}.jsonl').write_text(
                ''.join(json.dumps(row) + '\n' for row in rows[rank::2]))

    def test_all_arms_and_native_undefined_mean_are_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp); self.fixture(folder)
            result = audit_point_epoch(folder, set('abcd'), 1)
            self.assertEqual(result['images'], 4)
            self.assertEqual([r['rows'] for r in result['files']], [2, 2])
            self.assertAlmostEqual(result['recomputed_metrics']['point']['Objects F-measure'], .55)
            self.assertIsNone(result['recomputed_metrics']['point']['Boundary Precision'])
            self.assertEqual(result['undefined_images']['point'], {'Boundary Precision': 1})
            delta = result['deltas']['point_depth_minus_point_percentage_points']
            self.assertAlmostEqual(delta['Objects F-measure'], 5.)
            self.assertIsNone(delta['Boundary Precision'])
            self.assertIn('not_RLE_GT_rescoring', result['scope'])

    def test_missing_extra_duplicate_samples_and_extra_shard_are_rejected(self):
        for change in ['missing', 'extra', 'duplicate', 'extra_shard']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp); rows, summary = self.fixture(folder)
                if change == 'missing': rows.pop()
                elif change == 'extra':
                    extra = copy.deepcopy(rows[0]); extra['sample_id'] = 'e'; rows.append(extra)
                elif change == 'duplicate': rows[2]['sample_id'] = 'a'
                self.write(folder, rows, summary)
                if change == 'extra_shard': (folder / 'predictions_rank2.jsonl').write_text('')
                with self.assertRaises(ValueError): audit_point_epoch(folder, set('abcd'), 1)

    def test_changed_summary_and_undefined_counts_are_rejected(self):
        for change in ['mean', 'undefined_count', 'undefined_mean']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp); rows, summary = self.fixture(folder)
                if change == 'mean': summary['metrics']['point_depth']['Objects F-measure'] = .7
                elif change == 'undefined_count': summary['undefined_images']['point'] = {}
                else: summary['metrics']['point']['Boundary Precision'] = .6
                self.write(folder, rows, summary)
                with self.assertRaises(ValueError): audit_point_epoch(folder, set('abcd'), 1)

    def test_missing_arm_and_invalid_per_image_metrics_are_rejected(self):
        for change in ['arm', 'infinite', 'undefined_selection', 'boolean']:
            with self.subTest(change=change), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp); rows, summary = self.fixture(folder)
                if change == 'arm': del rows[0]['arms']['point']
                elif change == 'infinite': rows[0]['arms']['native']['metrics']['Objects Precision'] = float('inf')
                elif change == 'boolean': rows[0]['arms']['native']['metrics']['Objects Precision'] = True
                else: rows[0]['arms']['point']['metrics']['Objects F-measure'] = None
                self.write(folder, rows, summary)
                with self.assertRaises(ValueError): audit_point_epoch(folder, set('abcd'), 1)

    def test_smoke_wrong_epoch_and_wrong_precision_are_rejected(self):
        for field, value in [('scope', 'engineering_smoke'), ('epoch', 2), ('precision', 'float16')]:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as temp:
                folder = Path(temp); rows, summary = self.fixture(folder)
                summary[field] = value; self.write(folder, rows, summary)
                with self.assertRaises(ValueError): audit_point_epoch(folder, set('abcd'), 1)


if __name__ == '__main__':
    unittest.main()
