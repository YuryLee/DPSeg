import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rgbd_seg.course_point_refinement import POLICY
from rgbd_seg.course_portable import load_model, predict_variants, source_path, verify_sources


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PortableCourseTests(unittest.TestCase):
    def test_relative_sources_are_verified_in_the_new_project(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'rgbd_seg/example.py'
            path.parent.mkdir()
            path.write_text('original model code\n')
            expected = digest(path)
            result = verify_sources({'rgbd_seg/example.py': expected}, root)
            self.assertEqual(result['rgbd_seg/example.py'], str(path.resolve()))
            path.write_text('modified model code\n')
            with self.assertRaisesRegex(ValueError, 'Training source'):
                verify_sources({'rgbd_seg/example.py': expected}, root)

    def test_external_source_checks_the_installed_copy_even_if_old_file_exists(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            suffix = Path('transformers/models/mask2former/modeling_mask2former.py')
            old, installed = root / 'old' / suffix, root / 'new' / suffix
            for path in (old, installed):
                path.parent.mkdir(parents=True)
                path.write_text('pinned implementation\n')
            expected = digest(old)
            with patch('rgbd_seg.course_portable.importlib.util.find_spec',
                       return_value=SimpleNamespace(origin=str(installed))):
                self.assertEqual(source_path(str(old)), installed.resolve())
                verify_sources({str(old): expected})
                installed.write_text('different installed implementation\n')
                with self.assertRaisesRegex(ValueError, 'Training source'):
                    verify_sources({str(old): expected})

    def test_unrecognized_external_and_escaping_source_paths_are_rejected(self):
        with TemporaryDirectory() as folder:
            with self.assertRaisesRegex(ValueError, 'Unsupported external'):
                source_path('/old/computer/unrelated.py', folder)
            with self.assertRaisesRegex(ValueError, 'leaves the project'):
                source_path('../unrelated.py', folder)

    def test_different_head_base_pair_is_rejected_before_model_construction(self):
        with TemporaryDirectory() as folder:
            root = Path(folder)
            base = root / 'base.pt'
            head = root / 'head.pt'
            base.write_bytes(b'placeholder, never deserialized')
            head.write_bytes(b'placeholder, never deserialized')
            base_protocol = root / 'base.json'
            base_protocol.write_text(json.dumps({'status': 'frozen'}))
            point_protocol = root / 'point.json'
            point_protocol.write_text(json.dumps({
                'status': 'frozen', 'point_policy': POLICY,
                'weights': {'sha256': 'f' * 64},
                'training_protocol': {'sha256': digest(base_protocol)},
            }))
            manifest = root / 'model.json'
            manifest.write_text(json.dumps({
                'schema': 'rgbd-course-model-v1',
                'base': {'sha256': digest(base), 'protocol': 'base.json',
                         'protocol_sha256': digest(base_protocol)},
                'head': {'sha256': digest(head), 'protocol': 'point.json',
                         'protocol_sha256': digest(point_protocol)},
            }))
            with patch('rgbd_seg.course_portable.load_base') as constructor:
                with self.assertRaisesRegex(ValueError, 'different base'):
                    load_model(base_weights=base, head_weights=head,
                               coco_checkpoint=root, manifest=manifest)
                constructor.assert_not_called()

    def test_prediction_interface_rejects_ground_truth_fields(self):
        sample = dict(image=None, depth_m=None, depth_valid=None, camera_K=None, masks=None)
        with self.assertRaisesRegex(ValueError, 'RGB, metric depth'):
            predict_variants(None, None, sample)


if __name__ == '__main__':
    unittest.main()
