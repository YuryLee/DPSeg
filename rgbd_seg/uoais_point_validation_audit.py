"""Read-only integrity checks for saved paired UOAIS point validation results.

These checks recompute saved per-image means. They do not rescore RLE on GT,
run a model, select training settings, or replace the original test evaluation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

ARMS = ('native', 'point', 'point_depth')
METRICS = ('Objects Precision', 'Objects Recall', 'Objects F-measure',
           'Boundary Precision', 'Boundary Recall', 'Boundary F-measure',
           'obj_detected_075_percentage')
TOLERANCE = 2e-12


def require(condition, message):
    if not condition:
        raise ValueError(message)


def file_sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def identity(path):
    stat = Path(path).stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def load_point_validation_cohort(report):
    """Anchor sample IDs to the independently audited complete base validation."""
    report = Path(report).resolve()
    check_path = report / 'uoais_epoch20_validation_comparison_check.json'
    check = json.loads(check_path.read_text())
    require(check['epoch'] == 20 and check['history_rows'] == 40
            and check['images_per_model'] == 5000
            and check['identical_5000_image_and_sample_id_cohorts'] is True,
            'Expected the completed base validation integrity receipt')
    current = next(row for row in check['independently_recomputed_epochs'] if row['family'] == 'dpseg')
    base = report / 'uoais_dpseg_swint_20ep'
    protocol_path = base / 'protocol.json'
    protocol = json.loads(protocol_path.read_text())
    require(file_sha(protocol_path) == check['protocol_sha256'], 'Base validation protocol changed')
    annotation = protocol['annotations']['val']
    require(file_sha(annotation['path']) == annotation['sha256'], 'Original validation annotation changed')
    provenance = {str(check_path): file_sha(check_path), str(protocol_path): file_sha(protocol_path),
                  str(Path(annotation['path']).resolve()): annotation['sha256']}
    expected, image_ids = set(), set()
    require(len(current['files']) == 2, 'Expected both base validation shards')
    for rank, saved in enumerate(current['files']):
        path = base / f'val_epoch20_rank{rank}.jsonl'
        require(saved['rank'] == rank and Path(saved['path']).resolve() == path.resolve(),
                'Base cohort shard path or rank differs')
        before, checksum, count = identity(path), hashlib.sha256(), 0
        with path.open('rb') as stream:
            for raw in stream:
                checksum.update(raw)
                row = json.loads(raw)
                require(row['sample_id'] not in expected and row['image_id'] not in image_ids,
                        'Duplicate base validation identity')
                expected.add(row['sample_id']); image_ids.add(row['image_id']); count += 1
        require(identity(path) == before and checksum.hexdigest() == saved['sha256']
                and count == saved['rows'] == 2500, 'Base cohort file changed or is incomplete')
        provenance[str(path)] = checksum.hexdigest()
    require(len(expected) == len(image_ids) == 5000, 'Expected the complete 5000-image cohort')
    return expected, {'scope': 'sample IDs anchored to independently audited base epoch20',
                      'images': 5000, 'sources_sha256': provenance}


def metric_value(value, *, allow_none=True):
    if value is None:
        require(allow_none, 'Undefined point selection metric')
        return None
    require(isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and -1e-12 <= value <= 1 + 1e-12,
            'Invalid point validation metric')
    return float(value)


def audit_point_epoch(folder, expected_ids, epoch):
    """Check all three arms on exactly the expected samples and both rank files."""
    folder = Path(folder).resolve()
    expected_ids = set(expected_ids)
    require(bool(expected_ids), 'Empty expected point validation cohort')
    summary_path = folder / 'summary.json'
    summary_identity = identity(summary_path)
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    require(summary['scope'] == 'full_validation' and summary['epoch'] == epoch
            and summary['images'] == len(expected_ids) and summary['test_used'] is False
            and summary['selection_metric'] == 'Objects F-measure' and summary['precision'] == 'float32'
            and set(summary['metrics']) == set(summary['undefined_images']) == set(ARMS),
            'Incomplete or mismatched paired point validation summary')
    require({p.name for p in folder.glob('predictions_rank*.jsonl')}
            == {'predictions_rank0.jsonl', 'predictions_rank1.jsonl'},
            'Missing or extra paired point validation shard')
    values = {arm: {key: [] for key in METRICS} for arm in ARMS}
    seen, files = set(), []
    for rank in (0, 1):
        path = folder / f'predictions_rank{rank}.jsonl'
        before, checksum, count = identity(path), hashlib.sha256(), 0
        with path.open('rb') as stream:
            for raw in stream:
                checksum.update(raw)
                row = json.loads(raw)
                sample = row['sample_id']
                require(sample in expected_ids and sample not in seen, 'Unexpected or duplicate point sample')
                require(set(row['arms']) == set(ARMS), 'Missing or extra point validation arm')
                seen.add(sample); count += 1
                for arm in ARMS:
                    require(isinstance(row['arms'][arm]['predictions'], list), 'Missing saved point predictions')
                    for key in METRICS:
                        value = row['arms'][arm]['metrics'][key]
                        values[arm][key].append(metric_value(value, allow_none=key != 'Objects F-measure'))
        require(identity(path) == before, 'Point validation shard changed while reading')
        require(count == len(range(rank, len(expected_ids), 2)), 'Incomplete point validation shard')
        files.append({'rank': rank, 'path': str(path), 'rows': count, 'sha256': checksum.hexdigest()})
    require(seen == expected_ids, 'Missing point validation samples')
    means, undefined, maximum = {}, {}, 0.
    for arm in ARMS:
        require(set(summary['metrics'][arm]) == set(METRICS), 'Wrong point metric set')
        means[arm], undefined[arm] = {}, {}
        for key in METRICS:
            column = values[arm][key]
            missing = sum(value is None for value in column)
            actual = None if missing else math.fsum(column) / len(column)
            saved = metric_value(summary['metrics'][arm][key], allow_none=key != 'Objects F-measure')
            require((actual is None) == (saved is None), 'Undefined point mean was changed')
            if missing:
                undefined[arm][key] = missing
            else:
                difference = abs(actual - saved)
                maximum = max(maximum, difference)
                require(difference < TOLERANCE, 'Point summary differs from per-image mean')
            means[arm][key] = actual
        require(undefined[arm] == summary['undefined_images'][arm], 'Point undefined-value counts differ')
    require(identity(summary_path) == summary_identity, 'Point validation summary changed while reading')
    delta = {}
    for left, right in [('point', 'native'), ('point_depth', 'native'), ('point_depth', 'point')]:
        delta[f'{left}_minus_{right}_percentage_points'] = {
            key: None if means[left][key] is None or means[right][key] is None
            else 100 * (means[left][key] - means[right][key]) for key in METRICS}
    return {'status': 'passed', 'scope': 'paired_full_validation_saved_metric_integrity_not_RLE_GT_rescoring',
            'created_utc': datetime.now(timezone.utc).isoformat(), 'epoch': epoch,
            'images': len(seen), 'duplicates': 0, 'missing': 0, 'extra': 0,
            'arms': list(ARMS), 'files': files, 'summary_path': str(summary_path),
            'summary_sha256': hashlib.sha256(summary_raw).hexdigest(),
            'recomputed_metrics': means, 'undefined_images': undefined,
            'maximum_summary_difference': maximum, 'deltas': delta, 'test_used': False,
            'limitations': 'Saved per-image metrics and sample coverage checked. RLE predictions were not rescored on GT; query identity is checked by the frozen runner before mask restoration.'}
