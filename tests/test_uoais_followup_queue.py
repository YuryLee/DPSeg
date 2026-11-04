"""Prevent premature or unsupported follow-up GPU jobs."""
import json
from pathlib import Path
import tempfile
import unittest

from tools.queue_uoais_followup import base_ready, digest


class FollowupPrerequisiteTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        protocol = {'status': 'frozen', 'training': {'epochs': 20}}
        (self.base / 'protocol.json').write_text(json.dumps(protocol))
        self.run = {'family': 'dpseg', 'scope': 'full_train_val', 'status': 'training',
                    'protocol_sha256': digest(self.base / 'protocol.json'),
                    'hardware': [{'pid': 123}, {'pid': 456}]}

    def check(self, alive=False):
        (self.base / 'run.json').write_text(json.dumps(self.run))
        return base_ready(self.base, 'dpseg', lambda pid, base: alive)

    def completed(self):
        self.run['status'] = 'training_completed'
        self.run['state'] = {'epoch': 20, 'global_step': 112500, 'batch_in_epoch': 0,
                             'validation': [{'epoch': i, 'scope': 'full_validation', 'images': 5000,
                                             'test_used': False, 'metrics': {'Objects F-measure': .5}}
                                            for i in range(1, 21)]}
        (self.base / 'best.pt').touch()

    def test_live_training_waits_without_launching(self):
        self.assertEqual(self.check(alive=True),
                         (False, {'reason': 'base_training_running', 'worker_pids': [123, 456]}))

    def test_incomplete_dead_run_fails_instead_of_restarting(self):
        with self.assertRaisesRegex(RuntimeError, 'no longer live'):
            self.check()

    def test_complete_run_waits_for_worker_cleanup(self):
        self.completed()
        self.assertFalse(self.check(alive=True)[0])
        self.assertTrue(self.check(alive=False)[0])

    def test_smoke_cannot_authorize_formal_followup(self):
        self.completed()
        self.run['scope'] = 'engineering_smoke'
        with self.assertRaisesRegex(ValueError, 'full UOAIS'):
            self.check()

    def test_partial_or_test_based_validation_cannot_authorize_followup(self):
        self.completed()
        for field, bad, original in [('images', 8, 5000), ('test_used', True, False)]:
            self.run['state']['validation'][-1][field] = bad
            with self.assertRaisesRegex(ValueError, 'validation history'):
                self.check()
            self.run['state']['validation'][-1][field] = original

    def test_missing_checkpoint_or_failure_record_prevents_launch(self):
        self.completed()
        (self.base / 'best.pt').unlink()
        with self.assertRaisesRegex(ValueError, 'checkpoint is missing'):
            self.check()
        (self.base / 'failure_rank1.json').write_text('{}')
        with self.assertRaisesRegex(RuntimeError, 'failure record'):
            self.check()


if __name__ == '__main__':
    unittest.main()
