"""Integrity checks for the final comparison consumer, without model loading."""
import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest

from tools.report_uoais_benchmark import (
    COUNTS, METRICS, STAGES, Evidence, audit_system, completed_budget,
    mean_metrics, readiness, records, selected_stage, write_report,
    checkpoint_metadata, verify_checkpoint_selection, verify_point_checkpoints,
)


class ReportIntegrityTests(unittest.TestCase):
    def test_checkpoint_must_be_the_best_epoch_with_earliest_ties(self):
        history = [{'epoch': epoch, **{key: value for key in METRICS}}
                   for epoch, value in [(1, .8), (2, .9), (3, .9)]]
        saved = {'scope': 'full_train_val', 'family': 'dafusion', 'protocol_sha256': 'protocol',
                 'source_sha256': {'source': 'hash'}, 'selection': {
                     'scope': 'full_validation', 'images': 5000, 'test_used': False,
                     'epoch': 2, 'metrics': {key: .9 for key in METRICS}}}
        self.assertEqual(verify_checkpoint_selection(saved, history, 'protocol', {'source': 'hash'},
                                                    family='dafusion'), 2)
        for epoch in [1, 3]:
            wrong = copy.deepcopy(saved)
            wrong['selection']['epoch'] = epoch
            with self.assertRaisesRegex(ValueError, 'validation winner'):
                verify_checkpoint_selection(wrong, history, 'protocol', {'source': 'hash'}, family='dafusion')
        point = copy.deepcopy(saved)
        point['selected_arm'] = 'point_depth'
        point['selection']['metrics'] = {'point_depth': point['selection']['metrics']}
        paired_history = [{'epoch': v['epoch'], 'metrics': {'point_depth': {k: v[k] for k in METRICS}}}
                          for v in history]
        self.assertEqual(verify_checkpoint_selection(point, paired_history, 'protocol', {'source': 'hash'},
                                                    arm='point_depth'), 2)

    def test_checkpoint_metadata_reader_excludes_tensor_payloads(self):
        import torch
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'best.pt'
            torch.save({'model': {'weight': torch.zeros(2)}, 'scope': 'full_train_val',
                        'selection': {'epoch': 2}, 'family': 'dafusion'}, path)
            value = checkpoint_metadata(path)
            self.assertNotIn('model', value)
            self.assertEqual(value['selection'], {'epoch': 2})
            self.assertEqual(value['family'], 'dafusion')

    def test_both_point_controls_have_independent_winners_and_the_same_base(self):
        import torch
        history = []
        for epoch, plain, depth in [(1, .9, .8), (2, .8, .95), (3, .85, .95)]:
            history.append({'scope': 'full_validation', 'images': 5000, 'test_used': False,
                            'epoch': epoch, 'metrics': {
                                'point': {key: plain for key in METRICS},
                                'point_depth': {key: depth for key in METRICS}}})
        run = {'protocol_sha256': 'protocol', 'source_sha256': {'source': 'hash'},
               'state': {'validation': history}}
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            saved = {}
            for arm, epoch in [('point', 1), ('point_depth', 2)]:
                saved[arm] = {'heads': {'weight': torch.zeros(2)}, 'scope': 'full_train_val',
                              'selected_arm': arm, 'base_checkpoint_sha256': 'base',
                              'protocol_sha256': 'protocol', 'source_sha256': {'source': 'hash'},
                              'selection': copy.deepcopy(history[epoch - 1])}
                torch.save(saved[arm], folder / f'best_{arm}.pt')
            result = verify_point_checkpoints(folder, run, 'base', Evidence())
            self.assertEqual({k: v['selected_epoch'] for k, v in result.items()},
                             {'point': 1, 'point_depth': 2})
            wrong = copy.deepcopy(saved['point'])
            wrong['selection'] = history[1]
            torch.save(wrong, folder / 'best_point.pt')
            with self.assertRaisesRegex(ValueError, 'validation winner'):
                verify_point_checkpoints(folder, run, 'base', Evidence())
            wrong = copy.deepcopy(saved['point']); wrong['base_checkpoint_sha256'] = 'other-base'
            torch.save(wrong, folder / 'best_point.pt')
            with self.assertRaisesRegex(ValueError, 'another base'):
                verify_point_checkpoints(folder, run, 'base', Evidence())

    def test_completed_label_requires_the_actual_full_budget(self):
        run = {'status': 'training_completed', 'scope': 'full_train_val', 'state': {
            'epoch': 20, 'global_step': 112500, 'batch_in_epoch': 0,
            'validation': [{'epoch': i, 'scope': 'full_validation', 'images': 5000,
                            'test_used': False} for i in range(1, 21)]}}
        self.assertTrue(completed_budget(run, 20, 112500))
        for field, value in [('global_step', 112499), ('epoch', 19), ('batch_in_epoch', 1)]:
            wrong = copy.deepcopy(run)
            wrong['state'][field] = value
            self.assertFalse(completed_budget(wrong, 20, 112500))
        wrong = copy.deepcopy(run)
        wrong['state']['validation'][-1]['images'] = 8
        self.assertFalse(completed_budget(wrong, 20, 112500))

    def test_macro_mean_and_undefined_metrics_are_preserved(self):
        rows = [{key: value for key in METRICS} for value in (.2, .8)]
        self.assertEqual(mean_metrics(rows)['Objects F-measure'], .5)
        rows[0]['Boundary F-measure'] = None
        self.assertIsNone(mean_metrics(rows)['Boundary F-measure'])
        rows[1]['Objects F-measure'] = 1.2
        with self.assertRaises(ValueError):
            mean_metrics(rows)

    def test_exact_image_coverage_rejects_missing_extra_and_duplicates(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'rows.jsonl'
            for ids in [('a',), ('a', 'a'), ('a', 'c')]:
                path.write_text(''.join(json.dumps({'image_id': key}) + '\n' for key in ids))
                with self.assertRaises(ValueError):
                    list(records([path], {'a', 'b'}, Evidence()))

    def test_selection_is_validation_only_and_uses_declared_ties(self):
        summary = {'scope': 'full_validation', 'images': 5000, 'metrics': {
            stage: {'Objects F-measure': .8, 'Boundary F-measure': .7} for stage in STAGES}}
        self.assertEqual(selected_stage(summary), 'rgbd_fusion')
        summary['metrics']['depth_point']['Boundary F-measure'] = .75
        self.assertEqual(selected_stage(summary), 'depth_point')
        summary['scope'] = 'final_test'
        with self.assertRaises(ValueError):
            selected_stage(summary)

    def test_saved_masks_are_rescored_and_changed_scores_or_gt_are_rejected(self):
        import numpy as np
        from pycocotools import mask as mask_util
        from rgbd_seg.uois_benchmark_metrics import evaluate_label_maps

        truth = np.zeros((16, 16), dtype=np.int32)
        truth[3:8, 4:9] = 1
        rle = mask_util.encode(np.asfortranarray(truth.astype(np.uint8)))
        rle['counts'] = rle['counts'].decode('ascii')
        stage = {'predictions': [{'score': .9, 'segmentation': rle}],
                 **evaluate_label_maps(truth, truth)}

        class Data:
            records = [{'id': 'a'}, {'id': 'b'}]
            def ground_truth(self, index):
                return truth.copy() if index == 0 else np.zeros_like(truth)

        with tempfile.TemporaryDirectory() as folder:
            folder = Path(folder)
            rows = [{'image_id': key, 'stages': {name: copy.deepcopy(stage) for name in STAGES}}
                    for key in ('a', 'b')]
            empty_stage = {'predictions': [], **evaluate_label_maps(np.zeros_like(truth), np.zeros_like(truth))}
            rows[1]['stages'] = {name: copy.deepcopy(empty_stage) for name in STAGES}
            summary = {'status': 'completed', 'images': 2,
                       'metrics': {name: stage['metrics'] for name in STAGES}}
            (folder / 'summary.json').write_text(json.dumps(summary))
            for rank, row in enumerate(rows):
                (folder / f'predictions_rank{rank}.jsonl').write_text(json.dumps(row) + '\n')
            metrics, diagnostics = audit_system(folder, {'a', 'b'}, Evidence(), Data())
            self.assertEqual(metrics['rgbd_fusion']['Objects F-measure'], 1.)
            self.assertEqual(diagnostics['rgbd_fusion']['both_empty_images'], 1)
            self.assertEqual(diagnostics['rgbd_fusion']['empty_gt_images'], 1)
            self.assertEqual(diagnostics['rgbd_fusion']['empty_prediction_images'], 1)
            rows[0]['stages']['rgbd_fusion']['metrics']['Objects F-measure'] = .5
            (folder / 'predictions_rank0.jsonl').write_text(json.dumps(rows[0]) + '\n')
            with self.assertRaisesRegex(ValueError, 'Metric mismatch'):
                audit_system(folder, {'a', 'b'}, Evidence(), Data())
            rows[0]['stages']['rgbd_fusion']['metrics']['Objects F-measure'] = 1.
            (folder / 'predictions_rank0.jsonl').write_text(json.dumps(rows[0]) + '\n')
            original = Data.ground_truth
            Data.ground_truth = lambda self, index: np.roll(truth, 2, axis=0)
            with self.assertRaisesRegex(ValueError, 'Metric mismatch'):
                audit_system(folder, {'a', 'b'}, Evidence(), Data())
            Data.ground_truth = original

    def test_missing_ocbd_cannot_produce_a_full_report(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            check = readiness(root, root)
            self.assertEqual(check['required_benchmarks'], COUNTS)
            self.assertFalse(check['ready_for_verification'])
            self.assertIn(str(root / 'ocbd_manifest.json'), check['missing_files'])
            self.assertIn(str(root / 'uoais_dpseg_point_5ep/best_point.pt'), check['missing_files'])
            with self.assertRaises(ValueError):
                write_report(root / 'output', {'results': {'osd': {}, 'ocid': {}}})
            self.assertFalse((root / 'output').exists())

    def test_csv_keeps_fractions_and_markdown_uses_percent(self):
        metrics = {key: .5 for key in METRICS}
        result = {'results': {split: {'images': count, 'dafusion': metrics,
                  'dpseg': {stage: metrics for stage in STAGES},
                  'selected_dpseg_stage': 'depth_point'} for split, count in COUNTS.items()},
                  'paper_reported_percent_separate_from_local_measurements':
                      {split: [80.] * 7 for split in COUNTS}, 'paper_source': 'https://example.invalid/paper'}
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder) / 'report'
            write_report(out, result)
            with (out / 'comparison.csv').open() as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 15)
            self.assertTrue(all(float(row['Objects F-measure']) == .5 for row in rows))
            self.assertEqual(sum(row['model'] == 'DPSeg' and row['selected'] == 'True' for row in rows), 3)
            self.assertIn('50.00', (out / 'comparison.md').read_text())
            with self.assertRaises(ValueError):
                write_report(out, result)


if __name__ == '__main__':
    unittest.main()
