import ast
import gzip
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
import torch.nn.functional as F

from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_training_data import CourseTrainingDataset
from rgbd_seg.course_depth_data import (CourseDepthTrainingDataset, camera_intrinsics,
    collate_fn, depth_to_xyz, load_depth_record, load_inference_sample,
    resize_rgbd_sample, transform_depth, standardize_xyz, model_depth_channels)


def official_standardize_xyz():
    # Execute the author's function unchanged, without importing Detectron2.
    source = Path(__file__).resolve().parents[1]/'third_party/DA-Fusion/dafusion/dafusion/data/depth_xyz.py'
    tree = ast.parse(source.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'standardize_xyz')
    namespace = {'np': np}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), 'exec'), namespace)
    return namespace['standardize_xyz']


def fixture(folder, *, split='train', scale=.001, semantics='camera_axis_z', raw=None):
    path = Path(folder)/'sample.zip'
    raw = np.arange(20, dtype=np.uint16).reshape(4, 5) if raw is None else raw
    rgb = np.stack([np.arange(20, dtype=np.uint8).reshape(4, 5)]*3, axis=-1)
    binary = np.zeros((4, 5), dtype=np.uint8); binary[1:3, 1:4] = 1
    pixel = np.zeros_like(binary); pixel[1, 2] = 1
    record = {'id': 17, 'file_name': f'rgb/{split}/x.png', 'depth_file_name': f'depth/{split}/x.png',
              'height': 4, 'width': 5, 'scene_id': 'x', 'split': split, 'depth_scale_m': scale,
              'depth_invalid_code': 0, 'depth_semantics': semantics,
              'camera': {'K': [[4, 0, 2], [0, 4, 1.5], [0, 0, 1]],
                         'resolution': [5, 4], 'pixel_center_convention': 'integer'},
              'support_rle': 'must never be decoded as input', 'model_ids': ['not-an-input']}
    anns = []
    for i, mask in enumerate([binary, pixel]):
        rle = mask_utils.encode(np.asfortranarray(mask)); rle['counts'] = rle['counts'].decode()
        anns.append({'id': i+1, 'image_id': 17, 'category_id': 1, 'area': int(mask.sum()),
                     'iscrowd': 0, 'bbox': [0, 0, 0, 0], 'segmentation': rle})
    with ZipFile(path, 'w') as z:
        for name, arr in [(record['file_name'], rgb), (record['depth_file_name'], raw)]:
            stream = io.BytesIO(); Image.fromarray(arr).save(stream, format='PNG'); z.writestr(name, stream.getvalue())
        z.writestr(f'annotations/{split}/x.json.gz', gzip.compress(json.dumps({'images': [record], 'annotations': anns,
            'categories': [{'id': 1, 'name': 'component', 'supercategory': 'component'}]}).encode()))
    return path, raw, record


class CourseDepthTests(unittest.TestCase):
    def test_standardization_matches_official_numpy_even_median_and_percentile(self):
        reference = official_standardize_xyz()
        rng = np.random.default_rng(41)
        points = rng.normal(size=(4, 6, 3)).astype(np.float32)
        points[0, 0] = [100, -200, 300]
        valid = np.ones((4, 6), dtype=bool)
        valid[1, :2] = False  # 22 valid values: median averages middle pair.
        points[~valid] = np.nan  # Invalid values never enter the statistics.
        for mask in [valid, np.zeros_like(valid), np.eye(4,6,dtype=bool),
                     np.arange(24).reshape(4,6)==0]:
            with self.subTest(count=int(mask.sum())):
                safe_points = np.nan_to_num(points)
                expected = reference(safe_points, mask)
                actual = standardize_xyz(torch.from_numpy(safe_points).permute(2,0,1), torch.from_numpy(mask)[None])
                np.testing.assert_allclose(actual.permute(1,2,0), expected, rtol=2e-6, atol=2e-6)
                self.assertEqual(int(torch.count_nonzero(actual[:, ~torch.from_numpy(mask)])), 0)
        actual = standardize_xyz(torch.from_numpy(points).permute(2,0,1), torch.from_numpy(valid))
        np.testing.assert_allclose(actual.permute(1,2,0), reference(points,valid), rtol=2e-6, atol=2e-6)
        # A lower-middle torch.median implementation would fail this example.
        two = torch.tensor([[[0., 2.]], [[0., 0.]], [[0., 0.]]])
        torch.testing.assert_close(standardize_xyz(two, torch.ones(1,1,2,dtype=torch.bool))[0,0],
                                   torch.tensor([-1.,1.])/(1+1e-6))

    def test_model_channels_match_official_order_and_training_inference(self):
        reference = official_standardize_xyz()
        with tempfile.TemporaryDirectory() as td:
            path, _, _ = fixture(td)
            with CourseDepthTrainingDataset(path, augment=False) as ds, CourseArchive(path, 'train') as archive:
                sample, _ = ds[0]
                inference = load_inference_sample(archive, 17)
                a = model_depth_channels(sample, (7,9), device='cpu')
                torch.testing.assert_close(a, model_depth_channels(inference, (7,9)), rtol=0, atol=0)
                xyz = depth_to_xyz(sample['depth_m'],sample['depth_valid'],sample['camera_K'])
                for flip in [False, True]:
                    with self.subTest(flip=flip):
                        depth, valid, k = transform_depth(sample['depth_m'],sample['depth_valid'],sample['camera_K'],horizontal_flip=flip)
                        transformed = {**sample, 'depth_m':depth,'depth_valid':valid,'camera_K':k}
                        ref_xyz = xyz.flip(-1) if flip else xyz
                        ref_xyz = F.interpolate(ref_xyz[None],(7,9),mode='bilinear',align_corners=False,antialias=False)[0]
                        if flip: ref_xyz[0] *= -1
                        ref_valid = F.interpolate(valid[None].float(),(7,9),mode='nearest')[0].bool()
                        expected = reference(ref_xyz.permute(1,2,0).numpy(), ref_valid[0].numpy())
                        result = model_depth_channels(transformed,(7,9))
                        np.testing.assert_allclose(result[:3].permute(1,2,0),expected,atol=2e-6,rtol=2e-6)
                        torch.testing.assert_close(result[3],ref_valid[0].float(),rtol=0,atol=0)
                        self.assertEqual(result.dtype,torch.float32)
                        self.assertEqual(tuple(result.shape),(4,7,9))
                zero = {**sample,'depth_m':torch.zeros_like(sample['depth_m']),
                        'depth_valid':torch.zeros_like(sample['depth_valid'])}
                self.assertEqual(int(torch.count_nonzero(model_depth_channels(zero,(3,3)))),0)

    def test_units_invalid_and_no_ground_truth_in_inference(self):
        with tempfile.TemporaryDirectory() as td:
            path, raw, _ = fixture(td)
            with CourseArchive(path, 'train') as archive:
                with patch.object(archive, 'coco_ground_truth', side_effect=AssertionError('GT accessed')):
                    sample = load_inference_sample(archive, 17)
                self.assertEqual(set(sample), {'image', 'depth_m', 'depth_valid', 'camera_K'})
                np.testing.assert_allclose(sample['depth_m'][0], raw*.001, rtol=1e-6)
                np.testing.assert_array_equal(sample['depth_valid'][0], raw>0)
                self.assertEqual(sample['image'].shape, (3, 4, 5))
                with self.assertRaises(ValueError): load_inference_sample(archive, 0)

    def test_flip_exactly_preserves_existing_rgb_targets_and_xyz(self):
        with tempfile.TemporaryDirectory() as td:
            path, _, _ = fixture(td)
            with CourseDepthTrainingDataset(path, seed=17) as ds:
                for ep in range(100):
                    ds.set_epoch(ep)
                    if ds._flip_for(17): break
                self.assertTrue(ds._flip_for(17))
                sample, target = ds[0]
                image0, target0 = CourseTrainingDataset.__getitem__(ds, 0)
                self.assertTrue(torch.equal(image0, sample['image']))
                for key in target: self.assertTrue(torch.equal(target[key], target0[key]), key)
                depth, valid = load_depth_record(ds._handle(), ds.images[0]); k = camera_intrinsics(ds.images[0])
                self.assertTrue(torch.equal(sample['depth_m'], depth.flip(-1)))
                self.assertTrue(torch.equal(sample['depth_valid'], valid.flip(-1)))
                torch.testing.assert_close(depth_to_xyz(sample['depth_m'], sample['depth_valid'], sample['camera_K']), depth_to_xyz(depth, valid, k).flip(-1))
                self.assertEqual(target['area'].tolist(), [6, 1])
                self.assertEqual(int((target['masks'][0]&target['masks'][1]).sum()), 1)
                a, b = collate_fn([(sample, target), (sample, target)])
                self.assertEqual(len(a), 2); self.assertEqual(len(b), 2)

    def test_resize_uses_shared_geometry_and_preserves_invalids(self):
        with tempfile.TemporaryDirectory() as td:
            path, _, _ = fixture(td)
            with CourseDepthTrainingDataset(path, augment=False) as ds:
                sample, target = ds[0]
                out = resize_rgbd_sample(sample, (8, 10))
                torch.testing.assert_close(out['image'], F.interpolate(sample['image'][None], (8, 10), mode='bilinear', align_corners=False, antialias=False)[0], rtol=0, atol=0)
                torch.testing.assert_close(out['depth_m'], sample['depth_m'].repeat_interleave(2,-2).repeat_interleave(2,-1))
                self.assertTrue(torch.equal(out['depth_valid'], sample['depth_valid'].repeat_interleave(2,-2).repeat_interleave(2,-1)))
                torch.testing.assert_close(out['camera_K'], torch.tensor([[8.,0,4.5],[0,8,3.5],[0,0,1]]))
                self.assertTrue(torch.all(out['depth_m'][~out['depth_valid']]==0))
                self.assertEqual(target['masks'].shape, (2,4,5))

    def test_projection_hand_reference_and_all_invalid(self):
        depth = torch.full((1,3,3), 2.); valid = torch.ones_like(depth, dtype=torch.bool)
        k = torch.tensor([[2.,0,1],[0,2,1],[0,0,1]])
        xyz = depth_to_xyz(depth, valid, k)
        torch.testing.assert_close(xyz[:,0,0], torch.tensor([-1.,-1.,2.]))
        torch.testing.assert_close(xyz[:,1,1], torch.tensor([0.,0.,2.]))
        xyz0 = depth_to_xyz(torch.zeros_like(depth), torch.zeros_like(valid), k)
        self.assertEqual(int(torch.count_nonzero(xyz0)), 0)
        out = transform_depth(depth, valid, k, horizontal_flip=True)
        torch.testing.assert_close(depth_to_xyz(*out), xyz.flip(-1))

    def test_val_no_flip_and_seed_contract(self):
        with tempfile.TemporaryDirectory() as td:
            path, raw, _ = fixture(td, split='val')
            with CourseDepthTrainingDataset(path, 'val', augment=True, seed=None) as ds:
                sample, _ = ds[0]
                np.testing.assert_allclose(sample['depth_m'][0], raw*.001, rtol=1e-6)
                self.assertFalse(ds.summary()['rgb_only'])
        with self.assertRaises(ValueError): CourseDepthTrainingDataset('missing', seed=None)
        with self.assertRaises(ValueError): CourseDepthTrainingDataset('missing', split='test')

    def test_bad_units_semantics_shape_dtype_and_camera_rejected(self):
        for kwargs in [{'scale': 0}, {'scale': float('nan')}, {'semantics': 'ray_distance'},
                       {'raw': np.ones((3,5),dtype=np.uint16)}, {'raw': np.ones((4,5),dtype=np.uint8)}]:
            with self.subTest(kwargs=kwargs), tempfile.TemporaryDirectory() as td:
                path, _, record = fixture(td, **kwargs)
                with ZipFile(path) as z, self.assertRaises(ValueError): load_depth_record(z, record)
        with tempfile.TemporaryDirectory() as td:
            _, _, record = fixture(td); record['camera']['pixel_center_convention']='unknown'
            with self.assertRaises(ValueError): camera_intrinsics(record)


REAL_ARCHIVE = Path(os.environ.get('COURSE_DEPTH_ARCHIVE', '/data/yinli/workspace/downloads/buaa_image_segmentation/blender.zip'))


@unittest.skipUnless(os.environ.get('COURSE_DEPTH_REAL_TEST') == '1' and REAL_ARCHIVE.is_file(), 'Opt-in actual train/val ZIP test')
class RealCourseDepthTests(unittest.TestCase):
    def test_actual_train_val_original_alignment_flip_resize(self):
        # Only train/val input samples; no test data or effect-based tuning.
        for split in ['train', 'val']:
            with self.subTest(split=split), CourseDepthTrainingDataset(REAL_ARCHIVE, split, seed=17) as ds:
                for index in [0, 1]:
                    if split == 'train':
                        for ep in range(100):
                            ds.set_epoch(ep)
                            if ds._flip_for(ds.images[index]['id']): break
                    sample, target = ds[index]
                    rgb, reference = CourseTrainingDataset.__getitem__(ds, index)
                    self.assertTrue(torch.equal(sample['image'], rgb))
                    for key in target: self.assertTrue(torch.equal(target[key], reference[key]), key)
                    self.assertEqual(sample['depth_m'].shape, (1,1024,1024))
                    depth, valid = load_depth_record(ds._handle(), ds.images[index])
                    if ds._flip_for(ds.images[index]['id']): depth, valid = depth.flip(-1), valid.flip(-1)
                    self.assertTrue(torch.equal(sample['depth_m'], depth))
                    self.assertTrue(torch.equal(sample['depth_valid'], valid))
                    out = resize_rgbd_sample(sample, (800,800))
                    self.assertEqual(out['image'].shape, (3,800,800))
                    self.assertEqual(out['depth_m'].shape, (1,800,800))
                    self.assertTrue(torch.isfinite(depth_to_xyz(out['depth_m'],out['depth_valid'],out['camera_K'])).all())
                    channels = model_depth_channels(sample,(800,800))
                    self.assertEqual(channels.shape,(4,800,800))
                    self.assertTrue(torch.isfinite(channels).all())


if __name__ == '__main__': unittest.main()
