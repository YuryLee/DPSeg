import copy
import unittest

import torch

from rgbd_seg.course_continuation_state import validated_parent_state


def fixture(schema, epochs, epoch_offset=0, update_offset=0):
    protocol = {'schema': schema, 'source_sha256': {'runtime': 'unchanged'},
                'hardware': {'physical_gpus': ['2']},
                'training': {'epochs': epochs, 'effective_batch_size': 33,
                             'images_per_epoch': 14432, 'updates_per_epoch': 438,
                             'total_updates': epochs * 438}}
    if epoch_offset:
        protocol['continuation'] = {'parent_epoch': epoch_offset,
                                    'parent_optimizer_steps': update_offset}
    history = [{'epoch': epoch_offset + i, 'global_step': i * 438, 'training_images': 14432,
                'metrics': {'mask_ap': .7 + .001 * i, 'mask_ap_small': .4 + .001 * i}}
               for i in range(1, epochs + 1)]
    checkpoint = {'protocol_sha256': 'parent-sha', 'source_sha256': protocol['source_sha256'],
                  'model_metadata': {'architecture': 'same'}, 'rank_rng': [{'state': 'present'}],
                  'state': {'epoch': epochs, 'global_step': epochs * 438,
                            'images_in_epoch': 0, 'stage': 'training', 'val_results': history,
                            'best_epoch': epoch_offset + epochs,
                            'best_key': [.4 + .001 * epochs, .7 + .001 * epochs]},
                  'optimizer': {'param_groups': [{'params': [0]}],
                                'state': {0: {'step': torch.tensor(update_offset + epochs * 438),
                                              'exp_avg': torch.ones(2), 'exp_avg_sq': torch.ones(2)}}}}
    return protocol, checkpoint


class ContinuationStateTests(unittest.TestCase):
    def check(self, protocol, checkpoint, epoch):
        return validated_parent_state(checkpoint, protocol, 'parent-sha', {'architecture': 'same'}, epoch)

    def test_cumulative_offsets_survive_multiple_continuations(self):
        cases = [('course_resolution_ddp_v1', 5, 0, 0, 5, 2190),
                 ('course_longer_training_v1', 10, 5, 2190, 15, 6570),
                 ('course_longer_extension_v1', 5, 15, 6570, 20, 8760)]
        for schema, epochs, offset, steps, total_epoch, total_steps in cases:
            with self.subTest(schema=schema):
                p, c = fixture(schema, epochs, offset, steps)
                result = self.check(p, c, total_epoch)
                self.assertEqual(result['parent_epoch'], total_epoch)
                self.assertEqual(result['parent_optimizer_steps'], total_steps)
                self.assertEqual(result['selection']['epoch'], total_epoch)

    def test_incomplete_or_miscounted_state_cannot_start_extension(self):
        p, original = fixture('course_longer_training_v1', 10, 5, 2190)
        mutations = [lambda c: c['state'].update(images_in_epoch=33),
                     lambda c: c['state'].update(stage='validation'),
                     lambda c: c['state'].update(epoch=15),
                     lambda c: c['state'].update(global_step=6570),
                     lambda c: c['optimizer']['state'][0].update(step=torch.tensor(4380)),
                     lambda c: c['state']['val_results'][-1].update(epoch=10),
                     lambda c: c['state'].update(best_epoch=14),
                     lambda c: c.update(protocol_sha256='changed'),
                     lambda c: c['optimizer']['state'][0]['exp_avg'].fill_(float('nan'))]
        for mutate in mutations:
            c = copy.deepcopy(original)
            mutate(c)
            with self.assertRaises(ValueError):
                self.check(p, c, 15)

    def test_equal_score_retains_earlier_epoch(self):
        p, c = fixture('course_longer_training_v1', 10, 5, 2190)
        c['state']['val_results'][-2]['metrics'] = copy.deepcopy(c['state']['val_results'][-1]['metrics'])
        with self.assertRaisesRegex(ValueError, 'not the validation-selected best'):
            self.check(p, c, 15)


if __name__ == '__main__':
    unittest.main()
