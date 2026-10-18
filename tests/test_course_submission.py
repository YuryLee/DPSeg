import csv
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from pycocotools import mask as mask_utils
from rgbd_seg.course_submission import SubmissionWriter, validate_csv, _Spec, _checked_runs, _encode_runs


def prediction(mask, score=.8, category=1):
    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    return {'category_id': category, 'score': score,
            'segmentation': {'size': rle['size'], 'counts': rle['counts'].decode('ascii')}}


def write_csv(path, rows, header=('image_id', 'PredictionString')):
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        for image_id, value in rows:
            writer.writerow([image_id, json.dumps(value) if not isinstance(value, str) else value])


class CourseSubmissionTests(unittest.TestCase):
    def test_official_size_empty_and_overlap_roundtrip_no_native_decode_while_writing(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'submission.csv'
            first = np.zeros((1024, 1024), np.uint8)
            first[5:12, 7:15] = 1
            second = first.copy()
            second[7:15, 9:17] = 1
            values = [prediction(first), prediction(second, .3)]
            with patch('rgbd_seg.course_submission.mask_utils.decode', side_effect=AssertionError('write must not native decode')):
                with SubmissionWriter(path, ['0001', '0002']) as writer:
                    writer.write('0001', values)
                    writer.write('0002', [])
            self.assertFalse(writer.receipt['rle_native_decode_verified'])
            checked = validate_csv(path, ['0001', '0002'])
            self.assertEqual(checked['rows'], 2)
            self.assertEqual(checked['predictions'], 2)
            self.assertEqual(checked['empty_images'], 1)
            self.assertEqual(checked['summed_instance_area'], int(first.sum() + second.sum()))
            self.assertTrue(checked['rle_native_decode_verified'])
            self.assertEqual(checked['sha256'], writer.receipt['sha256'])
            self.assertEqual(list(Path(folder).iterdir()), [path])
            with path.open(newline='') as stream:
                rows = list(csv.reader(stream))
            self.assertEqual(rows[2], ['0002', '[]'])
            self.assertEqual(json.loads(rows[1][1]), values)

    def test_compressed_parser_matches_pycocotools_on_random_patterns_and_backslash(self):
        rng = np.random.default_rng(19)
        for height, width in ((1, 1), (4, 16), (23, 31), (64, 64)):
            spec = _Spec(['x'], (1,), (height, width), 100)
            for density in (.001, .1, .5, .99, 1.):
                for _ in range(10):
                    mask = (rng.random((height, width)) < density).astype(np.uint8)
                    mask[0, 0] = 1
                    counts = prediction(mask)['segmentation']['counts']
                    self.assertEqual(_checked_runs(counts, spec), int(mask.sum()))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'escaped.csv'
            counts = _encode_runs([44, 20])
            self.assertIn('\\', counts)
            item = {'category_id': 1, 'score': .1, 'segmentation': {'size': [8, 8], 'counts': counts}}
            with SubmissionWriter(path, [7], image_size=(8, 8)) as writer:
                writer.write(7, [item])
            self.assertEqual(validate_csv(path, [7], image_size=(8, 8))['summed_instance_area'], 20)

    def test_bad_rle_rejected_before_native_decode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'invalid.csv'
            valid = prediction(np.ones((4, 4), np.uint8))
            invalid_counts = ['P', 'PPPPPP0', '/', '\x00', 'é', 'o', _encode_runs([0, 17]),
                              _encode_runs([0, 15]), _encode_runs([16]), 'P00']
            with patch('rgbd_seg.course_submission.mask_utils.decode', side_effect=AssertionError('unsafe native call')):
                for counts in invalid_counts:
                    item = {**valid, 'segmentation': {'size': [4, 4], 'counts': counts}}
                    write_csv(path, [('x', [item])])
                    with self.subTest(counts=repr(counts)), self.assertRaises(ValueError):
                        validate_csv(path, ['x'], image_size=(4, 4))

    def test_missing_unknown_duplicate_order_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / 'submission.csv'
            for rows in ([('a', [])], [('b', []), ('a', [])], [('a', []), ('a', [])], [('a', []), ('unknown', [])]):
                write_csv(path, rows)
                with self.subTest(rows=rows), self.assertRaises(ValueError):
                    validate_csv(path, ['a', 'b'], image_size=(4, 4))
            existing = path.read_bytes()
            with self.assertRaises(FileExistsError):
                SubmissionWriter(path, ['a', 'b'])
            self.assertEqual(path.read_bytes(), existing)
            incomplete = root / 'incomplete.csv'
            with self.assertRaisesRegex(ValueError, 'Missing predictions'):
                with SubmissionWriter(incomplete, ['a', 'b']) as writer:
                    writer.write('a', [])
            self.assertFalse(incomplete.exists())
            self.assertEqual(list(root.iterdir()), [path])
            with self.assertRaisesRegex(ValueError, 'unique'):
                SubmissionWriter(root / 'duplicate.csv', ['1', 1])

    def test_schema_score_labels_size_limits_and_duplicate_json_keys(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            good = prediction(np.ones((4, 4), np.uint8))
            cases = [[{**good, 'score': score}] for score in (-.01, 1.01, float('nan'), float('inf'), True, '0.5')]
            cases += [[{**good, 'category_id': category}] for category in (0, 2, True, 1.0)]
            cases += [[{**good, 'segmentation': {**good['segmentation'], 'size': size}}] for size in ([8, 2], [4.0, 4], [4, True])]
            cases += [[good] * 101, {}, None, [{**good, 'box': [0, 0, 4, 4]}],
                      '[{"category_id":1,"category_id":1,"score":0.8,"segmentation":{"size":[4,4],"counts":"0`0"}}]']
            for value in cases:
                write_csv(path, [('x', value)])
                with self.subTest(value=str(value)[:120]), self.assertRaises(ValueError):
                    validate_csv(path, ['x'], image_size=(4, 4))
            write_csv(path, [('x', [good] * 100)])
            self.assertEqual(validate_csv(path, ['x'], image_size=(4, 4))['predictions'], 100)
            write_csv(path, [('x', [])], header=('PredictionString', 'image_id'))
            with self.assertRaisesRegex(ValueError, 'header'):
                validate_csv(path, ['x'], image_size=(4, 4))

    def test_partial_context_failure_and_publication_race_preserve_existing_file(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'output.csv'
            with self.assertRaises(RuntimeError):
                with SubmissionWriter(path, ['x']) as writer:
                    writer.write('x', [])
                    raise RuntimeError('inference failed')
            self.assertEqual(list(Path(folder).iterdir()), [])
            writer = SubmissionWriter(path, ['x'])
            writer.write('x', [])
            path.write_text('created by someone else')
            with self.assertRaises(FileExistsError):
                writer.close()
            self.assertEqual(path.read_text(), 'created by someone else')
            self.assertEqual(list(Path(folder).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
