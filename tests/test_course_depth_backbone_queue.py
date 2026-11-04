import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from tools.queue_course_depth_backbone import completed_base, check_results


class CompletionChecks(unittest.TestCase):
    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def test_exit_zero_smoke_is_not_a_completed_formal_base(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write(folder/'run.json', {'status': 'completed', 'scope': 'engineering_smoke'})
            with self.assertRaises(ValueError):
                completed_base(folder, {})

    def test_formal_base_requires_all_fifteen_validation_epochs(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            (folder/'final.pt').write_bytes(b'checkpoint')
            digest = hashlib.sha256(b'checkpoint').hexdigest()
            self.write(folder/'run.json', {'status': 'completed', 'scope': 'full_train_val',
                'optimizer_steps': 438*15, 'training_image_exposures': 14432*15,
                'source_sha256': {}, 'final_checkpoint_sha256': digest})
            self.write(folder/'selection.json', {'epochs': [{'epoch': i} for i in range(1, 15)]})
            with self.assertRaises(ValueError):
                completed_base(folder, {'source_sha256': {}})

    def test_final_results_require_full_split_not_a_sample(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            self.write(folder/'comparison.json', {'val': {'native': {
                'scope': 'engineering_smoke', 'full_split_images': 2584}}})
            with self.assertRaises(ValueError):
                check_results(folder)

    def test_full_csv_requires_actual_native_decode_receipt(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            results = {}
            for split, count in (('val', 2584), ('test', 2112)):
                results[split] = {}
                for variant in ('native', 'full'):
                    results[split][variant] = {'scope': 'full_split', 'full_split_images': count,
                        'metrics': {'mask_ap': .5, 'mask_ap_small': .3}}
                    self.write(folder/f'{split}_{variant}'/'evaluation_run.json', {
                        'status': 'completed', 'images': count, 'scope': 'full_split'})
                    if split == 'test':
                        self.write(folder/f'{split}_{variant}'/'submission_validation.json', {
                            'rows': count, 'id_order_verified': True, 'rle_native_decode_verified': False})
            self.write(folder/'comparison.json', results)
            with self.assertRaises(ValueError):
                check_results(folder)


if __name__ == '__main__':
    unittest.main()
