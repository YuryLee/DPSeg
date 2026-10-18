"""Adversarial checks for the independent audit; no real artifacts or GPU."""
import copy
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import unittest

import torch

from rgbd_seg.augment import augment
from tools import audit_training_methods as audit


class TrainingAuditTests(unittest.TestCase):
    def fixture(self, variant='bce2', steps=3):
        records = [{'id': 'a'}, {'id': 'b'}]
        traces, _, _ = audit.replay_sampling(17, records, [(1, 1), (2, 0)], steps=steps)
        rows = []
        for i, trace in enumerate(traces):
            losses = {key: .25 for key in audit.LOSS_KEYS}
            if variant in ('bce2', 'bace'):
                losses['loss_mask_aux'] = .25
            rows.append({**trace, **losses, 'loss': sum(losses.values()), 'grad_norm': 9.,
                         'elapsed_s': i + 1.,
                         'lr_used': [base * audit.expected_schedule(i) for base in (.00002, .0002)],
                         'lr_next': [base * audit.expected_schedule(i + 1) for base in (.00002, .0002)]})
        return rows, traces

    def test_valid_trace_and_preclip_norm_above_five(self):
        rows, traces = self.fixture()
        result = audit.audit_loss_trace(rows, traces, traces, 'bce2', steps=3)
        self.assertTrue(result['bce2_auxiliary_exactly_equals_mask_every_step'])

    def test_bce2_mismatch_rejected_even_if_total_is_adjusted(self):
        rows, traces = self.fixture()
        rows[1]['loss_mask_aux'] += .1
        rows[1]['loss'] += .1
        with self.assertRaisesRegex(ValueError, 'BCE2 auxiliary'):
            audit.audit_loss_trace(rows, traces, traces, 'bce2', steps=3)

    def test_schedule_shift_missing_steps_and_nonfinite_rejected(self):
        rows, traces = self.fixture()
        for corruption in ('schedule', 'nan', 'missing'):
            damaged = copy.deepcopy(rows)
            if corruption == 'schedule':
                damaged[0]['lr_used'] = damaged[0]['lr_next']
            elif corruption == 'nan':
                damaged[0]['loss_mask'] = float('nan')
            else:
                damaged.pop()
            with self.assertRaises(ValueError):
                audit.audit_loss_trace(damaged, traces, traces, 'bce2', steps=3)

    def test_sampling_or_rng_change_rejected(self):
        rows, traces = self.fixture()
        changed_indices = [1 - value for value in traces[1]['batch_indices']]
        for key, value in [('batch_indices', changed_indices), ('augmentation_rng_after_sha256', 'wrong')]:
            damaged = copy.deepcopy(traces)
            damaged[1][key] = value
            with self.assertRaisesRegex(ValueError, 'Sampling/augmentation'):
                audit.audit_loss_trace(rows, damaged, traces, 'bce2', steps=3)

    def test_rng_replay_matches_actual_augmentation(self):
        torch.set_num_threads(2)
        sample = {'image': torch.rand(3, 32, 32), 'depth': torch.ones(1, 32, 32),
                  'depth_valid': torch.ones(1, 32, 32)}
        masks = torch.zeros(2, 32, 32, dtype=torch.uint8)
        masks[0, 2:9, 3:10] = 1
        masks[1, 12:24, 14:25] = 1
        target = {'masks': masks, 'boxes': torch.tensor([[3., 2., 10., 9.], [14., 12., 25., 24.]]),
                  'area': masks.flatten(1).sum(1).float(), 'labels': torch.ones(2, dtype=torch.int64),
                  'iscrowd': torch.zeros(2, dtype=torch.int64)}
        records = [{'id': 'a'}, {'id': 'b'}, {'id': 'c'}]
        rows, expected_sampling, expected_augmentation = audit.replay_sampling(23, records, [(2, 2)] * 3, steps=30)
        sampler, augmenter = random.Random(23), random.Random(1023)
        for row in rows:
            indices = [sampler.randrange(3) for _ in range(2)]
            self.assertEqual(indices, row['batch_indices'])
            self.assertEqual(audit.state_digest(augmenter.getstate()), row['augmentation_rng_before_sha256'])
            for _ in indices:
                augment(sample, target, augmenter, .35)
            self.assertEqual(audit.state_digest(augmenter.getstate()), row['augmentation_rng_after_sha256'])
        self.assertEqual(sampler.getstate(), expected_sampling)
        self.assertEqual(augmenter.getstate(), expected_augmentation)

    def test_hash_cache_rejects_modified_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'file'
            path.write_text('first')
            audit.sha(path)
            previous = path.stat().st_mtime_ns
            path.write_text('other')
            os.utime(path, ns=(previous + 100000, previous + 100000))
            with self.assertRaisesRegex(ValueError, 'changed during audit'):
                audit.sha(path)
            audit._HASHES.pop(path.resolve())

    def test_missing_matrix_is_incomplete_not_pass(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'audit.json'
            result = subprocess.run([sys.executable, str(audit.ROOT / 'tools/audit_training_methods.py'),
                '--training-root', str(Path(folder) / 'training'), '--evaluation-root', str(Path(folder) / 'evaluation'),
                '--output', str(output)], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 2, result.stderr)
            receipt = json.loads(output.read_text())
            self.assertEqual(receipt['status'], 'incomplete')
            self.assertEqual(len(receipt['missing']), 40)
            self.assertNotIn('gate', receipt)

    def test_supervisor_nonzero_exit_is_rejected_despite_completed_status(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trains = [(seed, variant, root / f'train/{seed}/{variant}') for seed in audit.SEEDS for variant in audit.VARIANTS]
            evaluations = [('graspnet', seed, variant, split, root / f'eval/{seed}/{variant}/{split}')
                           for seed in audit.SEEDS for variant in audit.VARIANTS for split in ('val', 'test')]
            evaluations += [('tless', 17, variant, 'test', root / f'tless/{variant}') for variant in audit.VARIANTS]
            outputs = [(path, 'train') for *_, path in trains] + [(path, 'eval') for *_, path in evaluations]
            ledger = {'status': 'completed', 'source_sha256': audit.FROZEN_SOURCES,
                      'protocol_sha256': audit.PROTOCOL_SHA, 'physical_gpu': 5,
                      'expected_trainings': 12, 'expected_evaluations': 28,
                      'jobs': [{'id': str(i), 'output': str(path), 'kind': kind,
                                'status': 'completed', 'returncode': -9} for i, (path, kind) in enumerate(outputs)]}
            path = root / 'ledger.json'
            path.write_text(json.dumps(ledger))
            with self.assertRaisesRegex(ValueError, 'Unsuccessful process'):
                audit.audit_supervisor(path, trains, evaluations)

    def test_completed_quiet_evaluation_accepts_zero_last_poll_bytes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            trains = [(seed, variant, root / f'train/{seed}/{variant}') for seed in audit.SEEDS for variant in audit.VARIANTS]
            evaluations = [('graspnet', seed, variant, split, root / f'eval/{seed}/{variant}/{split}')
                           for seed in audit.SEEDS for variant in audit.VARIANTS for split in ('val', 'test')]
            evaluations += [('tless', 17, variant, 'test', root / f'tless/{variant}') for variant in audit.VARIANTS]
            outputs = [(path, 'train') for *_, path in trains] + [(path, 'eval') for *_, path in evaluations]
            jobs = []
            for i, (output, kind) in enumerate(outputs):
                output.mkdir(parents=True)
                command = ['python', 'tools/run_v5.py', kind]
                receipt = {'command': command[1:], 'variant': 'baseline'}
                lines = []
                if kind == 'train':
                    losses = [{'step': step} for step in range(1, 801)]
                    (output / 'loss.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in losses))
                    lines = [losses[step - 1] for step in [1, *range(25, 801, 25)]]
                    final = {'output': str(output), 'exit_status': 'completed', 'phase': 'formal', 'steps': 800, 'variant': 'baseline'}
                else:
                    receipt['metrics'] = {'fixture_only': 1}
                    final = {'output': str(output), 'metrics': receipt['metrics']}
                (output / ('run.json' if kind == 'train' else 'evaluation.json')).write_text(json.dumps(receipt))
                log = output / 'stdout.log'
                log.write_text(''.join(json.dumps(row) + '\n' for row in [*lines, final]))
                jobs.append({'id': str(i), 'kind': kind, 'output': str(output), 'status': 'completed', 'returncode': 0,
                             'timeout_s': 1800 if kind == 'train' else 900, 'duration_s': 3, 'pid': i + 1,
                             'started_at_utc': '2026-10-02T12:00:00+00:00', 'last_alive_at_utc': '2026-10-02T12:00:02+00:00',
                             'ended_at_utc': '2026-10-02T12:00:03+00:00', 'command': command, 'log': str(log), 'log_bytes': 0})
            ledger = {'status': 'completed', 'source_sha256': audit.FROZEN_SOURCES, 'protocol_sha256': audit.PROTOCOL_SHA,
                      'physical_gpu': 5, 'expected_trainings': 12, 'expected_evaluations': 28, 'jobs': jobs}
            path = root / 'ledger.json'
            path.write_text(json.dumps(ledger))
            checked = audit.audit_supervisor(path, trains, evaluations)
            self.assertEqual(checked['status'], 'pass')
            self.assertEqual(len(checked['jobs']), 40)
            for cached_path in list(audit._HASHES):
                if cached_path.is_relative_to(root):
                    audit._HASHES.pop(cached_path)


if __name__ == '__main__':
    unittest.main()
