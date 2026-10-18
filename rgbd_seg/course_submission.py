"""Streaming course CSV submissions with strict IDs and safe compressed-RLE checks.

No model inference, ground truth, thresholding, sorting, truncation or padding is
performed here. The caller supplies the final predictions in official ID order.
The course class is component=1 and its original images are 1024 by 1024.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
import threading

import numpy as np
from pycocotools import mask as mask_utils

HEADER = ['image_id', 'PredictionString']
_CSV_LIMIT_LOCK = threading.RLock()


def _image_id(value):
    if isinstance(value, bool) or not isinstance(value, (str, Integral)):
        raise ValueError('image_id must be a string or integer')
    value = str(value)
    if not value or any(ord(char) < 32 for char in value):
        raise ValueError('image_id must be nonempty and contain no control characters')
    return value


class _Spec:
    def __init__(self, image_ids, category_ids, image_size, max_detections):
        if isinstance(image_ids, (str, bytes)):
            raise ValueError('image_ids must be an ordered collection, not one string')
        self.ids = tuple(_image_id(value) for value in image_ids)
        if not self.ids or len(self.ids) != len(set(self.ids)):
            raise ValueError('Expected image IDs must be nonempty and unique')
        self.id_set = set(self.ids)
        if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2
                or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in image_size)):
            raise ValueError('image_size must be positive integer (height,width)')
        self.size = tuple(int(x) for x in image_size)
        self.pixels = self.size[0] * self.size[1]
        if self.pixels > 2**31 - 1:
            raise ValueError('Image area exceeds the supported native RLE range')
        self.categories = tuple(category_ids)
        if (not self.categories or any(isinstance(x, bool) or not isinstance(x, Integral) or x <= 0 for x in self.categories)
                or len(set(self.categories)) != len(self.categories)):
            raise ValueError('category_ids must contain unique positive integer labels')
        self.categories = tuple(int(x) for x in self.categories)
        if isinstance(max_detections, bool) or not isinstance(max_detections, Integral) or not 1 <= max_detections <= 100:
            raise ValueError('max_detections must be an integer in [1,100]')
        self.max_detections = int(max_detections)
        # Signed 5-bit groups encode a run or a difference of two runs.
        self.max_groups = (self.pixels.bit_length() + 1 + 4) // 5
        self.max_counts_chars = (self.pixels + 1) * self.max_groups
        # JSON can escape a backslash in counts; allow that expansion and metadata.
        self.max_json_chars = 2 + self.max_detections * (2 * self.max_counts_chars + 256)


def _encode_runs(runs):
    """Reference inverse of COCO maskApi.c rleFrString, using Python integers."""
    encoded = []
    for index, run in enumerate(runs):
        value = run - runs[index - 2] if index > 2 else run
        while True:
            char = value & 0x1f
            value >>= 5
            more = value != (-1 if char & 0x10 else 0)
            if more:
                char |= 0x20
            encoded.append(chr(char + 48))
            if not more:
                break
    return ''.join(encoded)


def _checked_runs(counts, spec):
    """Reject malformed runs before any native pycocotools RLE operation.

    Bounds, termination, sign extension, differential coding, total image area,
    and canonical integer encoding are checked here. Thus native decoding never
    receives truncated codes, negative/overflowing runs or mismatched allocation.
    """
    if not isinstance(counts, str) or not counts or len(counts) > spec.max_counts_chars:
        raise ValueError('RLE counts must be a nonempty bounded compressed ASCII string')
    if any(not 48 <= ord(char) <= 111 for char in counts):
        raise ValueError('RLE counts contains a character outside the COCO ASCII alphabet')
    runs, position, total, area = [], 0, 0, 0
    while position < len(counts):
        value, groups = 0, 0
        while True:
            if position >= len(counts):
                raise ValueError('RLE counts ends inside an unfinished integer')
            char = ord(counts[position]) - 48
            position += 1
            groups += 1
            if groups > spec.max_groups:
                raise ValueError('RLE integer exceeds the image-size bound')
            value |= (char & 0x1f) << (5 * (groups - 1))
            if not char & 0x20:
                if char & 0x10:
                    value |= -1 << (5 * groups)
                break
        if len(runs) > 2:
            value += runs[-2]
        if value < 0 or value > spec.pixels:
            raise ValueError('RLE has a negative or oversized run')
        total += value
        if total > spec.pixels:
            raise ValueError('RLE run total exceeds original image area')
        if len(runs) % 2:
            area += value
        runs.append(value)
        if len(runs) > spec.pixels + 1:
            raise ValueError('RLE contains excessive zero-length runs')
    if total != spec.pixels:
        raise ValueError('RLE run total does not equal original image area')
    if area <= 0:
        raise ValueError('Prediction mask must have nonzero area')
    if _encode_runs(runs) != counts:
        raise ValueError('RLE integers use noncanonical compressed encoding')
    return area


def _predictions(value, spec, *, decode):
    if not isinstance(value, list) or len(value) > spec.max_detections:
        raise ValueError(f'PredictionString must be a list with at most {spec.max_detections} predictions')
    normalized, total_area = [], 0
    for index, prediction in enumerate(value):
        try:
            if not isinstance(prediction, dict) or set(prediction) != {'category_id', 'score', 'segmentation'}:
                raise ValueError('Each prediction must contain exactly category_id, score, segmentation')
            label, score, segmentation = prediction['category_id'], prediction['score'], prediction['segmentation']
            if isinstance(label, bool) or not isinstance(label, Integral) or label not in spec.categories:
                raise ValueError('Unknown or noninteger category_id')
            if isinstance(score, bool) or not isinstance(score, Real) or not math.isfinite(float(score)) or not 0 <= score <= 1:
                raise ValueError('score must be finite and in [0,1]')
            if not isinstance(segmentation, dict) or set(segmentation) != {'size', 'counts'}:
                raise ValueError('segmentation must contain exactly size and counts')
            size = segmentation['size']
            if (not isinstance(size, list) or len(size) != 2
                    or any(isinstance(x, bool) or not isinstance(x, Integral) for x in size)
                    or tuple(size) != spec.size):
                raise ValueError('RLE size must equal the original [height,width]')
            area = _checked_runs(segmentation['counts'], spec)
            if decode:
                binary = mask_utils.decode({'size': list(spec.size), 'counts': segmentation['counts'].encode('ascii')})
                if binary.shape != spec.size or not np.isin(binary, (0, 1)).all() or int(binary.sum()) != area:
                    raise ValueError('Decoded RLE shape/binary values/area differ from checked runs')
                del binary
            normalized.append({'category_id': int(label), 'score': float(score),
                               'segmentation': {'size': list(spec.size), 'counts': segmentation['counts']}})
            total_area += area
        except (ValueError, TypeError, OverflowError) as error:
            raise ValueError(f'Prediction {index}: {error}') from error
    return normalized, total_area


def _check_id(value, row, spec):
    value = _image_id(value)
    if value not in spec.id_set:
        raise ValueError(f'Unknown image_id: {value}')
    if row >= len(spec.ids) or value != spec.ids[row]:
        raise ValueError(f'Duplicate or out-of-order image_id at row {row + 1}: {value}')
    return value


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _receipt(path, spec, rows, predictions, empty, area, decoded):
    return {'path': str(Path(path).resolve()), 'sha256': _sha256(path), 'rows': rows,
            'predictions': predictions, 'empty_images': empty, 'summed_instance_area': area,
            'image_size': list(spec.size), 'category_ids': list(spec.categories),
            'max_detections': spec.max_detections, 'id_order_verified': True,
            'rle_runs_verified': True, 'rle_native_decode_verified': decoded,
            'area_note': 'Sum over instance masks; overlapping pixels can be counted more than once'}


class SubmissionWriter:
    """Write one complete official-ID-ordered image row at a time.

    Example: ``with SubmissionWriter(path, ids) as writer: writer.write(id, [])``.
    An empty list is an explicit no-detection result. Missing rows are errors.
    ``write`` performs pure-Python RLE validation; use ``validate_csv`` once on
    the finished file for native decoding. The final path never overwrites an
    existing file. Incomplete/failed context managers remove their temporary CSV.
    """
    def __init__(self, path, image_ids, *, category_ids=(1,), image_size=(1024, 1024), max_detections=100):
        self.spec = _Spec(image_ids, category_ids, image_size, max_detections)
        self.path = Path(path)
        if self.path.exists() or self.path.is_symlink():
            raise FileExistsError(f'Refusing to overwrite submission: {self.path}')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline='',
                                                   dir=self.path.parent, prefix=f'.{self.path.name}.', suffix='.tmp', delete=False)
        self._temporary = Path(self._stream.name)
        self._writer = csv.writer(self._stream, lineterminator='\n')
        self._writer.writerow(HEADER)
        self.rows = self.predictions = self.empty_images = self.area = 0
        self._closed = False
        self.receipt = None

    def write(self, image_id, predictions):
        if self._closed:
            raise ValueError('SubmissionWriter is closed')
        image_id = _check_id(image_id, self.rows, self.spec)
        predictions, area = _predictions(predictions, self.spec, decode=False)
        self._writer.writerow([image_id, json.dumps(predictions, ensure_ascii=True, allow_nan=False, separators=(',', ':'))])
        self.rows += 1
        self.predictions += len(predictions)
        self.empty_images += not predictions
        self.area += area

    def close(self):
        if self._closed:
            if self.receipt is not None:
                return self.receipt
            raise ValueError('SubmissionWriter was aborted')
        try:
            if self.rows != len(self.spec.ids):
                raise ValueError(f'Missing predictions: wrote {self.rows} of {len(self.spec.ids)} expected image rows')
            self._stream.flush()
            os.fsync(self._stream.fileno())
            self._stream.close()
            # Atomic publication without overwriting a destination created meanwhile.
            os.link(self._temporary, self.path)
            self._temporary.unlink()
            self._closed = True
            self.receipt = _receipt(self.path, self.spec, self.rows, self.predictions, self.empty_images, self.area, False)
            return self.receipt
        except BaseException:
            self.abort()
            raise

    def abort(self):
        if not self._closed:
            self._stream.close()
            self._temporary.unlink(missing_ok=True)
            self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, error_type, error, traceback):
        if error_type is None:
            self.close()
        else:
            self.abort()
        return False


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def validate_csv(path, image_ids, *, category_ids=(1,), image_size=(1024, 1024), max_detections=100):
    """Validate all expected IDs/order and decode each original-size RLE once.

    Returns an audit receipt with file SHA256. Only one image's encoded records
    and one decoded binary mask are retained at a time; no GT is read.
    """
    spec = _Spec(image_ids, category_ids, image_size, max_detections)
    rows = predictions = empty = area = 0
    with _CSV_LIMIT_LOCK:
        old_limit = csv.field_size_limit()
        csv.field_size_limit(max(old_limit, spec.max_json_chars))
        try:
            with Path(path).open('r', encoding='utf-8-sig', newline='') as stream:
                reader = csv.reader(stream, strict=True)
                if next(reader, None) != HEADER:
                    raise ValueError('CSV header must be image_id,PredictionString in this order')
                for row in reader:
                    if len(row) != 2:
                        raise ValueError(f'CSV row {rows + 1} must have exactly two columns')
                    _check_id(row[0], rows, spec)
                    if len(row[1]) > spec.max_json_chars:
                        raise ValueError('PredictionString exceeds the bounded image/count size')
                    try:
                        values = json.loads(row[1], object_pairs_hook=_unique_object)
                    except (ValueError, RecursionError) as error:
                        raise ValueError(f'Invalid PredictionString JSON at row {rows + 1}: {error}') from error
                    values, instance_area = _predictions(values, spec, decode=True)
                    rows += 1
                    predictions += len(values)
                    empty += not values
                    area += instance_area
        except csv.Error as error:
            raise ValueError(f'Malformed submission CSV: {error}') from error
        finally:
            csv.field_size_limit(old_limit)
    if rows != len(spec.ids):
        raise ValueError(f'Missing predictions: found {rows} of {len(spec.ids)} expected image rows')
    return _receipt(path, spec, rows, predictions, empty, area, True)
