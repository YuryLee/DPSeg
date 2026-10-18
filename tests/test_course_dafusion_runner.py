"""CPU checks for DA-Fusion budgets, optimizer grouping, resume and RGB-D eval."""
import gzip
from contextlib import ExitStack
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('course_dafusion_runner', ROOT / 'tools/run_course_dafusion.py')
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class TinyAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.rgb = nn.Linear(2, 2)
        self.depth = nn.Linear(2, 2)
        self.fusion = nn.Linear(2, 2)
        self.query_embed = nn.Embedding(3, 2)
        self.frozen = nn.Parameter(torch.zeros(2), requires_grad=False)


class RunnerTests(unittest.TestCase):
    def test_both_backbones_get_low_lr_and_fusion_gets_full_lr(self):
        adapter = TinyAdapter()
        optimizer, summary = runner.make_optimizer(adapter, {'backbone_parameter_prefixes': ['rgb.', 'depth.']}, .0002)
        grouped = {id(p): group for group in optimizer.param_groups for p in group['params']}
        ids = [id(p) for group in optimizer.param_groups for p in group['params']]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {id(p) for p in adapter.parameters() if p.requires_grad})
        for name, parameter in adapter.named_parameters():
            if not parameter.requires_grad:
                continue
            group = grouped[id(parameter)]
            self.assertAlmostEqual(group['base_lr'], .00002 if name.startswith(('rgb.', 'depth.')) else .0002)
            self.assertEqual(group['weight_decay'], 0. if name.endswith('bias') or 'embed' in name else .05)
        self.assertEqual(sum(row['parameters'] for row in summary), sum(p.numel() for p in adapter.parameters() if p.requires_grad))

    def test_every_declared_backbone_must_match_without_overlap(self):
        for prefixes in ([], ['rgb.', 'missing.'], ['rgb.', 'rgb.'], ['r', 'rgb.'], 'rgb.'):
            with self.subTest(prefixes=prefixes), self.assertRaises(ValueError):
                runner.make_optimizer(TinyAdapter(), {'backbone_parameter_prefixes': prefixes}, .0002)

    def test_gpu_visibility_does_not_touch_reserved_devices(self):
        environment = {}
        self.assertEqual(runner.configure_visibility(environment), 5)
        self.assertEqual(environment['CUDA_VISIBLE_DEVICES'], '5')
        for value in ('0', '1', '2', '3', '', '4,5', '0,5', '-1', 'GPU-unknown'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                runner.configure_visibility({'CUDA_VISIBLE_DEVICES': value})
        self.assertEqual(runner.configure_visibility({'CUDA_VISIBLE_DEVICES': '4'}), 4)

    def test_budget_shuffle_and_lr_remain_identical_to_frozen_rgb_run(self):
        spec = importlib.util.spec_from_file_location('frozen_course_runner', ROOT / 'tools/run_course_finetune.py')
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
        self.assertEqual(14432 // (2 * 4) * 5, 9020)
        for epoch in range(5):
            self.assertEqual(runner.epoch_order(14432, 17, epoch), old.epoch_order(14432, 17, epoch))
        for step in (0, 1, 249, 250, 1803, 9019):
            self.assertEqual(runner.lr_factor(step, 9020), old.lr_factor(step, 9020))
        self.assertEqual(runner.selection_key({'metrics': {'mask_ap_small': .2, 'mask_ap': .6}}), (.2, .6))

    def test_resume_counts_use_optimizer_updates_not_microbatches(self):
        valid = {'epoch': 1, 'images_in_epoch': 2400, 'global_step': 2104,
                 'stage': 'training', 'val_results': [{'epoch': 1}]}
        runner.validate_resume_state(valid, 14432, 8, 5)
        for mutation in ({'images_in_epoch': 2402}, {'global_step': 3004}, {'val_results': []}):
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                runner.validate_resume_state({**valid, **mutation}, 14432, 8, 5)

    def test_resume_rejects_changed_inputs_at_same_paths(self):
        metadata = {'pretrained_sha256': 'old-weights', 'shape': (2, 3)}
        data = {'annotation_member_name_and_bytes_sha256': 'old-annotations'}
        previous = json.loads(json.dumps({'model_metadata': metadata, 'training_data': data}))
        runner.validate_initialization_resume(previous, metadata, data)
        with self.assertRaises(ValueError):
            runner.validate_initialization_resume(previous, {**metadata, 'pretrained_sha256': 'changed'}, data)
        with self.assertRaises(ValueError):
            runner.validate_initialization_resume(previous, metadata,
                {'annotation_member_name_and_bytes_sha256': 'changed'})

    def test_rejected_resume_preserves_previous_run_and_initialization(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            args = SimpleNamespace(model='dafusion', checkpoint=out/'weights', archive=out/'fixture.zip',
                output=out, protocol=None, device='cuda:0', epochs=5, batch_size=2, accumulation=4,
                workers=0, threads=1, seed=17, lr=.0002, precision='bf16', save_every=300,
                timeout_seconds=172800, smoke_steps=1, resume=True)
            metadata = {'backbone_parameter_prefixes': ['rgb.', 'depth.'], 'pretrained_sha256': 'new'}
            previous = {'status': 'failed', 'source_sha256': {'fixture': 'unchanged'}, 'protocol_sha256': None,
                'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                'model_metadata': {**metadata, 'pretrained_sha256': 'old'},
                'training_data': {'annotation_member_name_and_bytes_sha256': 'same'}}
            runner.write_json(out/'run.json', previous)
            runner.write_json(out/'initialization.json', previous['model_metadata'])
            before = {name: (out/name).read_bytes() for name in ('run.json', 'initialization.json')}
            class Dataset:
                closed = False
                def __init__(self, *args, **kwargs): pass
                def __len__(self): return 14432
                def summary(self): return previous['training_data']
                def close(self): Dataset.closed = True
            with ExitStack() as stack:
                stack.enter_context(patch.object(runner.argparse.ArgumentParser, 'parse_args', return_value=args))
                stack.enter_context(patch.object(runner, 'runtime_source_sha256', return_value={'fixture': 'unchanged'}))
                stack.enter_context(patch.object(runner, 'configure_visibility', return_value=5))
                stack.enter_context(patch.object(runner, 'build_adapter', return_value=(TinyAdapter(), metadata)))
                stack.enter_context(patch('rgbd_seg.course_depth_data.CourseDepthTrainingDataset', Dataset))
                stack.enter_context(patch.object(runner.torch, 'set_num_threads'))
                for method in ('set_device', 'set_per_process_memory_fraction', 'manual_seed_all'):
                    stack.enter_context(patch.object(runner.torch.cuda, method))
                for method in ('is_available', 'is_bf16_supported'):
                    stack.enter_context(patch.object(runner.torch.cuda, method, return_value=True))
                stack.enter_context(patch.object(runner.torch.cuda, 'get_device_name', return_value='CPU test fixture'))
                with self.assertRaisesRegex(ValueError, 'Resume model_metadata changed'):
                    runner.main()
            self.assertTrue(Dataset.closed)
            for name, original in before.items():
                self.assertEqual((out/name).read_bytes(), original)
            rejected = list(out.glob('resume_rejected_*.json'))
            self.assertEqual(len(rejected), 1)
            self.assertEqual(json.loads(rejected[0].read_text())['status'], 'failed')


def prediction():
    binary = np.zeros((1024, 1024), dtype=np.uint8)
    binary[10:30, 20:40] = 1
    rle = mask_utils.encode(np.asfortranarray(binary))
    rle['counts'] = rle['counts'].decode('ascii')
    return {'category_id': 1, 'score': .9, 'segmentation': rle}


def fixture(root):
    path = Path(root) / 'rgbd.zip'
    with ZipFile(path, 'w') as archive:
        for split in ('val', 'test'):
            images, annotations = [], []
            for image_id in (101, 102):
                rgb_name, depth_name = f'rgb/{split}/{image_id}.png', f'depth/{split}/{image_id}.png'
                record = {'id': image_id, 'width': 1024, 'height': 1024, 'file_name': rgb_name,
                          'depth_file_name': depth_name, 'depth_scale_m': .0001, 'depth_invalid_code': 0,
                          'depth_semantics': 'camera_axis_z',
                          'camera': {'K': [[900, 0, 511.5], [0, 900, 511.5], [0, 0, 1]],
                                     'resolution': [1024, 1024], 'pixel_center_convention': 'integer'},
                          'support_rle': {'forbidden_input': True}, 'cad_id': 'forbidden',
                          'target_center': ['forbidden']}
                images.append(record)
                annotations.append({'id': image_id, 'image_id': image_id, 'category_id': 1, 'area': 400,
                                    'bbox': [20, 10, 20, 20], 'iscrowd': 0, 'segmentation': prediction()['segmentation']})
                rgb, depth = io.BytesIO(), io.BytesIO()
                Image.new('RGB', (1024, 1024), (image_id, 0, 0)).save(rgb, format='PNG')
                Image.fromarray(np.full((1024, 1024), 10000 + image_id, dtype=np.uint16)).save(depth, format='PNG')
                archive.writestr(rgb_name, rgb.getvalue())
                archive.writestr(depth_name, depth.getvalue())
            archive.writestr(f'annotations/{split}/all.json.gz', gzip.compress(json.dumps({
                'images': images, 'annotations': annotations, 'categories': [{'id': 1, 'name': 'component'}]}).encode()))
        archive.writestr('submission.csv', 'image_id,PredictionString\n102,[]\n101,[]\n')
    return path


class PredictAdapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.child = nn.BatchNorm1d(1)
        self.child.eval()
        self.inputs = []

    def predict_sample(self, sample, *, score_threshold, max_detections):
        assert set(sample) == {'image', 'depth_m', 'depth_valid', 'camera_K'}
        assert not self.training and torch.is_inference_mode_enabled()
        assert sample['image'].shape == (3, 1024, 1024)
        assert sample['depth_m'].shape == (1, 1024, 1024)
        assert sample['depth_valid'].dtype == torch.bool
        assert score_threshold == .01 and max_detections == 100
        image_id = round(float(sample['image'][0, 0, 0]) * 255)
        assert abs(float(sample['depth_m'][0, 0, 0]) - (10000 + image_id) * .0001) < 1e-6
        self.inputs.append(image_id)
        return [prediction()]


class EvaluationTests(unittest.TestCase):
    def test_original_rle_scores_full_val_and_depth_provenance(self):
        from rgbd_seg.course_dafusion_eval import evaluate
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            adapter = PredictAdapter()
            result = evaluate(adapter, path, 'val', Path(folder) / 'val', warmup=1, expected_count=2)
            self.assertEqual(adapter.inputs, [101, 101, 102])
            self.assertAlmostEqual(result['metrics']['mask_ap_small'], 1.)
            self.assertTrue(result['depth_used'])
            self.assertEqual(result['gt_small_including_crowd'], 2)
            self.assertTrue(adapter.training)
            self.assertFalse(adapter.child.training)
            receipt = json.loads((Path(folder) / 'val/evaluation_run.json').read_text())
            self.assertTrue(receipt['depth_used'])
            self.assertIn('course_dafusion_eval.py', receipt['source_sha256'])
            self.assertIn('course_depth_data.py', receipt['source_sha256'])
            self.assertEqual(result['scope'], 'full_split')

    def test_test_prediction_order_and_native_submission_validation(self):
        from rgbd_seg.course_dafusion_eval import evaluate
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            adapter = PredictAdapter()
            evaluate(adapter, path, 'test', Path(folder) / 'test', warmup=0,
                     expected_count=2, export_submission=True)
            self.assertEqual(adapter.inputs, [102, 101])
            receipt = json.loads((Path(folder) / 'test/submission_validation.json').read_text())
            self.assertTrue(receipt['rle_native_decode_verified'])
            self.assertEqual(receipt['rows'], 2)
            self.assertEqual(receipt['summed_instance_area'], 800)

    def test_expected_count_and_callback_failure_prevent_success_receipt(self):
        from rgbd_seg.course_dafusion_eval import evaluate
        with tempfile.TemporaryDirectory() as folder:
            path = fixture(folder)
            adapter = PredictAdapter()
            with self.assertRaises(ValueError):
                evaluate(adapter, path, 'val', Path(folder) / 'bad_count', expected_count=2584)
            self.assertEqual(adapter.inputs, [])
            def stop(update):
                if update['status'] == 'inferring':
                    raise TimeoutError('fixture timeout')
            with self.assertRaises(TimeoutError):
                evaluate(adapter, path, 'val', Path(folder) / 'aborted', warmup=0, progress_callback=stop)
            self.assertTrue(adapter.training)
            self.assertFalse(adapter.child.training)
            self.assertFalse((Path(folder) / 'aborted/evaluation.json').exists())
            receipt = json.loads((Path(folder) / 'aborted/evaluation_run.json').read_text())
            self.assertEqual(receipt['status'], 'failed')


if __name__ == '__main__':
    unittest.main()
