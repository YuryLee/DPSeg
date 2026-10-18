"""GT-only selection and complete-matrix safeguards for the static v5 gallery."""
import copy
import unittest
import torch
from tools import build_training_gallery as gallery


class FixtureDataset:
    def __init__(self, entries):
        self.samples = [{'id': name, 'image_id': index + 1, 'source_scene': 1, 'source_frame': index}
                        for index, (name, _) in enumerate(entries)]
        self.targets = [{'masks': torch.ones((1, 1, area), dtype=torch.uint8)} for _, area in entries]

    def __getitem__(self, index):
        return {}, self.targets[index]


class TrainingGalleryTests(unittest.TestCase):
    def test_selection_uses_gt_all_development_and_lexical_external_tie(self):
        data = {('graspnet', 'val'): FixtureDataset([(f'v{i}', 2000) for i in range(4)]),
                ('graspnet', 'test'): FixtureDataset([('not-small', 2000), ('small', 1024)]),
                ('tless', 'test'): FixtureDataset([('z', 16), ('b', 32), ('a', 16)])}
        selected = gallery.select_cases(data)
        self.assertEqual([item['sample_id'] for item in selected], ['v0', 'v1', 'v2', 'v3', 'small', 'a'])
        self.assertEqual(selected[-1]['small_gt_indices'], [0])
        self.assertTrue(all(not item['small_gt_indices'] for item in selected[:4]))

    def test_selection_never_silently_drops_additional_test_cases(self):
        data = {('graspnet', 'val'): FixtureDataset([(f'v{i}', 2000) for i in range(4)]),
                ('graspnet', 'test'): FixtureDataset([('a', 1), ('b', 2)]),
                ('tless', 'test'): FixtureDataset([('t', 3)])}
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            gallery.select_cases(data)

    def test_summary_requires_full_matrix_hashes_and_same_gate(self):
        receipts = [{'dataset': name, 'seed': seed, 'variant': variant, 'split': split,
                     'evaluation_sha256': f'{name}-{seed}-{variant}-{split}', 'protocol_sha256': 'protocol'}
                    for name, seeds, splits in (('graspnet', (17, 23, 41), ('val', 'test')), ('tless', (17,), ('test',)))
                    for seed in seeds for variant in gallery.VARIANTS for split in splits]
        development = {'gates': {v: {'passed': False} for v in gallery.VARIANTS[1:]},
                       'rows': [{'variant': v, 'mask_ap_small': {'mean': 3.}} for v in gallery.VARIANTS]}
        summary = {'status': 'complete_and_verified', 'counts': {'completed_training': 12, 'expected_training': 12,
                   'completed_evaluations': 28, 'expected_evaluations': 28}, 'evaluations': copy.deepcopy(receipts),
                   'protocol': {'sha256': 'protocol'}, 'development_gate': {'status': 'complete',
                   'choices': {v: {'passed': False, 'mean_val_ap_small': 3.} for v in gallery.VARIANTS[1:]}}}
        gallery.validate_summary(summary, receipts, development)
        partial = copy.deepcopy(summary)
        partial['status'] = 'partial_no_final_gate'
        with self.assertRaisesRegex(ValueError, 'complete'):
            gallery.validate_summary(partial, receipts, development)
        tampered = copy.deepcopy(summary)
        tampered['evaluations'][0]['evaluation_sha256'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'disagree'):
            gallery.validate_summary(tampered, receipts, development)
        tampered = copy.deepcopy(summary)
        tampered['development_gate']['choices']['bace']['passed'] = True
        with self.assertRaisesRegex(ValueError, 'decisions differ'):
            gallery.validate_summary(tampered, receipts, development)


if __name__ == '__main__':
    unittest.main()
