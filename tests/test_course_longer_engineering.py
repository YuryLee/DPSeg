"""Engineering completion must not accept live jobs or reset optimizer state."""
import copy
import unittest

import torch

from tools.audit_course_longer_engineering import check_exit, check_optimizer


class EngineeringTests(unittest.TestCase):
    def test_process_exit_and_engineering_scope_are_required(self):
        protocol = {'arm': 'test'}
        control = {'status': 'completed', 'scope': 'engineering_smoke', 'exit_code': 0,
                   'active_process_alive': False,
                   'trials': [{'arm': 'test', 'status': 'completed', 'exit_code': 0,
                               'protocol_sha256': 'protocol'}]}
        run = {'status': 'completed', 'scope': 'engineering_smoke',
               'arguments': {'smoke_steps': 2, 'smoke_tail_images': 11,
                             'smoke_dense': True, 'resume': False, 'resolution': 1024}}
        check_exit(control, run, protocol, 'protocol')
        for target, key, value in [('control', 'status', 'running'),
                                   ('control', 'active_process_alive', True),
                                   ('control', 'exit_code', 1),
                                   ('run', 'scope', 'full_train_val')]:
            c, r = copy.deepcopy(control), copy.deepcopy(run)
            (c if target == 'control' else r)[key] = value
            with self.subTest(target=target, key=key), self.assertRaises(ValueError):
                check_exit(c, r, protocol, 'protocol')
        for key, value in [('smoke_steps', 1), ('smoke_tail_images', 0), ('smoke_dense', False)]:
            r = copy.deepcopy(run)
            r['arguments'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                check_exit(control, r, protocol, 'protocol')

    def test_adamw_must_continue_cumulative_steps_and_update_moments(self):
        before = {'state': {7: {'step': torch.tensor(6570.), 'exp_avg': torch.zeros(2),
                                'exp_avg_sq': torch.ones(2)}},
                  'param_groups': [{'params': [7]}]}
        after = copy.deepcopy(before)
        after['state'][7]['step'] += 2
        after['state'][7]['exp_avg'] += .1
        self.assertEqual(check_optimizer(before, after, 6570), 1)
        for value in (2., 6571., 6573., float('nan')):
            wrong = copy.deepcopy(after)
            wrong['state'][7]['step'] = torch.tensor(value)
            with self.subTest(step=value), self.assertRaises(ValueError):
                check_optimizer(before, wrong, 6570)
        for name, value in [('exp_avg', torch.tensor([float('nan'), 0.])),
                            ('exp_avg_sq', torch.ones(3))]:
            wrong = copy.deepcopy(after)
            wrong['state'][7][name] = value
            with self.subTest(moment=name), self.assertRaises(ValueError):
                check_optimizer(before, wrong, 6570)
        unchanged = copy.deepcopy(before)
        unchanged['state'][7]['step'] += 2
        with self.assertRaises(ValueError):
            check_optimizer(before, unchanged, 6570)


if __name__ == '__main__':
    unittest.main()
