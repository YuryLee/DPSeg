import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tools.evaluate_multiscale import RLEPredictions, encode_prediction, ensure_empty, write_new, check_run, sha256


class RunnerIntegrityTests(unittest.TestCase):
    def test_rle_roundtrip_keeps_original_coordinates_and_empty(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'predictions.jsonl'
            mask = torch.zeros((2, 1, 13, 17), dtype=torch.bool)
            mask[0, 0, 3:7, 2:6] = True
            mask[1, 0, 5:9, 4:8] = True  # Deliberately overlapping, independently preserved.
            pred = {'masks': mask, 'scores': torch.tensor([.8, .3]), 'labels': torch.tensor([1, 1]),
                    'boxes': torch.tensor([[2., 3., 6., 7.], [4., 5., 8., 9.]])}
            empty = {key: value[:0] for key, value in pred.items()}
            path.write_text('\n'.join(json.dumps(encode_prediction(p, str(i), i+1, 13, 17)) for i, p in enumerate((pred, empty))) + '\n')
            cache = RLEPredictions(path)
            self.assertEqual(len(cache), 2)
            self.assertTrue(torch.equal(cache[0]['masks'], mask))
            self.assertTrue(torch.equal(cache[0]['scores'], pred['scores']))
            self.assertEqual(tuple(cache[1]['masks'].shape), (0, 1, 13, 17))
            self.assertEqual(len(list(cache)), 2)

    def test_existing_outputs_are_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            path = ensure_empty(directory)
            write_new(path / 'x.json', {'original': True})
            with self.assertRaises(FileExistsError):
                ensure_empty(directory)
            with self.assertRaises(FileExistsError):
                write_new(path / 'x.json', {'original': False})
            self.assertEqual(json.loads((path / 'x.json').read_text()), {'original': True})

    def test_run_receipt_rejects_altered_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            path.write_bytes(b'changed checkpoint')
            (path.parent / 'run.json').write_text(json.dumps({'exit_status': 'completed', 'checkpoint_sha256': '0'*64}))
            with self.assertRaisesRegex(ValueError, 'checkpoint SHA256'):
                check_run(path, {}, None, 'irrelevant')

    def test_run_receipt_rejects_configuration_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            path.write_bytes(b'unchanged checkpoint')
            (path.parent / 'run.json').write_text(json.dumps({'exit_status': 'completed', 'checkpoint_sha256': sha256(path), 'config': {'seed': 17}}))
            with self.assertRaisesRegex(ValueError, 'config'):
                check_run(path, {'config': {'seed': 23}}, None, 'irrelevant')


if __name__ == '__main__':
    unittest.main()
