#!/usr/bin/env python3
"""Read-only audit of two completed course transfer runs; never runs models.

Existing CSV validation receipts bind full native RLE decoding to the exact CSV
SHA. This auditor compares JSONL/CSV semantics and geometry without native decode.
"""
from __future__ import annotations
import argparse
import csv
from collections import Counter
from datetime import datetime, timezone
import hashlib
from itertools import zip_longest
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
METRICS = ('mask_ap', 'mask_ap50', 'mask_ap75', 'mask_ap_small', 'mask_ap_medium', 'mask_ap_large',
           'mask_ar_1', 'mask_ar_10', 'mask_ar_100', 'mask_ar_small', 'mask_ar_medium', 'mask_ar_large')


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def reject_constant(value):
    raise ValueError(f'Nonfinite JSON constant: {value}')


def json_loads(value):
    return json.loads(value, parse_constant=reject_constant)


def load(path):
    return json_loads(Path(path).read_text())


def compare_streams(jsonl_path, csv_path, images):
    """Compare all predictions in order; no COCO/native mask operation here."""
    old_limit = csv.field_size_limit()
    csv.field_size_limit(128 * 1024 * 1024)
    canonical_hash = hashlib.sha256()
    count = rows = empty = 0
    category_instances, category_images, category_above_half = Counter(), Counter(), Counter()
    inference_seconds = []
    try:
        with Path(jsonl_path).open() as raw, Path(csv_path).open(newline='') as submitted:
            reader = csv.DictReader(submitted)
            if reader.fieldnames != ['image_id', 'PredictionString']:
                raise ValueError('CSV columns differ from latest submission contract')
            for index, pair in enumerate(zip_longest(raw, reader)):
                line, row = pair
                if line is None or row is None or index >= len(images):
                    raise ValueError('CSV/JSONL/official image counts differ')
                record = json_loads(line)
                expected = images[index]
                if (type(record.get('image_id')) is not int or record['image_id'] != expected['id']
                        or row['image_id'] != str(expected['id']) or record.get('file_name') != expected['file_name']):
                    raise ValueError(f'ID/order/file_name mismatch at row {index + 1}')
                predictions = record['predictions']
                submitted_predictions = json_loads(row['PredictionString'])
                if json.dumps(predictions, sort_keys=True, separators=(',', ':'), allow_nan=False) != json.dumps(submitted_predictions, sort_keys=True, separators=(',', ':'), allow_nan=False):
                    raise ValueError(f'CSV and canonical JSONL predictions differ at row {index + 1}')
                if not isinstance(predictions, list) or len(predictions) > 100:
                    raise ValueError('Invalid instance count')
                original = record.get('original_category_ids')
                if not isinstance(original, list) or len(original) != len(predictions) or any(type(x) is not int or not 1 <= x <= 90 for x in original):
                    raise ValueError('Original foreground categories are missing or invalid')
                seconds = record.get('inference_seconds')
                if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
                    raise ValueError('Nonpositive/nonfinite per-image time')
                for prediction in predictions:
                    if set(prediction) != {'category_id', 'score', 'segmentation'}:
                        raise ValueError('Prediction must have only category_id, score, segmentation; no bbox area override')
                    if type(prediction['category_id']) is not int or prediction['category_id'] != 1:
                        raise ValueError('Category mapping is not component=1')
                    score = prediction['score']
                    if type(score) not in (int, float) or not math.isfinite(score) or not .01 <= score <= 1:
                        raise ValueError('Invalid score or inconsistent threshold')
                    rle = prediction['segmentation']
                    if (not isinstance(rle, dict) or set(rle) != {'size','counts'}
                            or rle['size'] != [expected['height'], expected['width']]
                            or any(type(v) is not int for v in rle['size'])
                            or not isinstance(rle['counts'], str) or not rle['counts']):
                        raise ValueError('Invalid original-resolution compressed RLE structure')
                category_instances.update(original)
                category_images.update(set(original))
                category_above_half.update(label for label, prediction in zip(original, predictions) if prediction['score'] >= .5)
                inference_seconds.append(seconds)
                canonical_hash.update(json.dumps({'image_id': expected['id'], 'predictions': predictions}, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()+b'\n')
                count += len(predictions)
                empty += not predictions
                rows += 1
        if rows != len(images):
            raise ValueError('Missing official image rows')
    finally:
        csv.field_size_limit(old_limit)
    import numpy as np
    return {'rows': rows, 'predictions': count, 'empty_images': empty,
            'timing_from_jsonl': {'mean_ms': float(np.mean(inference_seconds)*1000),
                                  'median_ms': float(np.median(inference_seconds)*1000),
                                  'p95_ms': float(np.percentile(inference_seconds,95)*1000),
                                  'fps': rows/sum(inference_seconds)},
            'original_coco_category_distribution': [
                {'category_id': label, 'instances': category_instances[label],
                 'images': category_images[label], 'instances_score_ge_0_5': category_above_half[label],
                 'fraction_of_all_emitted_instances': category_instances[label]/count if count else None}
                for label in sorted(category_instances)],
            'original_category_note': 'Counts among emitted predictions (score >= 0.01), before mapping to component=1. COCO labels are out-of-domain predictions, not course GT semantic identities. Additional >=0.5 counts describe display visibility only; no filtering or metric change.',

            'canonical_semantics_sha256': canonical_hash.hexdigest(),
            'id_file_order_verified': True, 'csv_jsonl_semantics_equal': True,
            'geometry_checked': [1024, 1024], 'native_rle_decode_repeated': False}



def execution_checks(first, second, deviation, protocol_sha):
    """Check recorded execution evidence; this is not runtime instrumentation."""
    errors, warnings = [], []
    def require(condition, message):
        if not condition:
            errors.append(message)
    first_start = datetime.fromisoformat(first['started_at_utc'])
    first_finish = datetime.fromisoformat(first['finished_at_utc'])
    second_start = datetime.fromisoformat(second['started_at_utc'])
    second_finish = datetime.fromisoformat(second['finished_at_utc'])
    overlap = first_start < second_finish and second_start < first_finish
    require(first_start <= second_start, 'Mask2Former began before MaskRCNN')
    evidence = {'whole_process_intervals_overlap': overlap,
                'timing_comparison_controlled': not overlap,
                'inference_nonoverlap_basis': 'Completed whole-process intervals',
                'runtime_instrumentation_performed': False}
    if deviation is None:
        require(not overlap, 'Whole-process intervals overlap without a prior execution deviation receipt')
        return evidence, errors, warnings
    evidence.update(timing_comparison_controlled=False,
                    inference_nonoverlap_basis='Prior validating_submission snapshot and frozen runner ordering; recorded evidence only',
                    execution_deviation_disclosed=True)
    require(deviation.get('original_protocol_sha256') == protocol_sha, 'Execution deviation references a different protocol')
    require(deviation.get('frozen_inference_and_evaluation_settings_changed') is False, 'Execution deviation does not preserve frozen settings')
    require(deviation.get('timing_comparison_controlled') is False, 'Execution deviation must disclose uncontrolled timing comparison')
    recorded = datetime.fromisoformat(deviation['recorded_at_utc'])
    require(first_start <= recorded < second_start, 'Execution deviation was not recorded after MaskRCNN began and before Mask2Former began')
    snapshot = deviation['maskrcnn_run_snapshot']
    snapshot_bytes = (json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False)+'\n').encode()
    require(hashlib.sha256(snapshot_bytes).hexdigest() == deviation.get('maskrcnn_run_snapshot_sha256'), 'Execution snapshot SHA mismatch')
    require(snapshot.get('status') == 'validating_submission' and snapshot.get('images_completed') == 2112,
            'Execution snapshot does not record completed 2112-image inference before validation')
    require(snapshot.get('expected_images') == 2112 and snapshot.get('scope') == 'full_test' and snapshot.get('model') == 'maskrcnn',
            'Execution snapshot scope differs')
    # Every recorded field except the moving status must survive unchanged in
    # the final run: model/source hashes, data identity, timing, PID and outputs.
    for key, value in snapshot.items():
        if key != 'status':
            require(first.get(key) == value, f'Final MaskRCNN field differs from execution snapshot: {key}')
    require(snapshot.get('source_sha256') == first.get('source_sha256'), 'Execution snapshot source hashes differ')
    gpu = deviation.get('gpu_before_mask2former_launch', {})
    require(gpu.get('gpu_utilization_percent') == 0 and gpu.get('name') == first.get('device_name'),
            'Prior GPU snapshot does not record idle matching device')
    evidence.update(recorded_at_utc=deviation['recorded_at_utc'],
                    snapshot_sha256=deviation['maskrcnn_run_snapshot_sha256'],
                    snapshot_status=snapshot.get('status'), snapshot_images=snapshot.get('images_completed'))
    warnings.append('Original whole-process sequential execution was relaxed before Mask2Former launch: MaskRCNN CPU validation/evaluation can overlap Mask2Former. Prior snapshot supports sequential inference only as recorded evidence, not independent runtime instrumentation.')
    warnings.append('timing_comparison_controlled=false: background CPU validation can affect Mask2Former preprocessing/postprocessing and resident GPU memory. Latencies are descriptive; do not claim a fair speed advantage.')
    return evidence, errors, warnings


def audit(base):
    from rgbd_seg.course_archive import CourseArchive
    base = Path(base).resolve()
    errors, warnings, missing = [], [], []
    files, runs, details = {}, {}, {}

    def check(condition, message):
        if not condition:
            errors.append(message)

    def hashed(path):
        path = Path(path).resolve()
        digest = sha256(path)
        files[str(path)] = digest
        return digest

    protocol_path, data_path = base/'protocol.json', base/'dataset_audit.json'
    protocol, data = load(protocol_path), load(data_path)
    protocol_sha, data_sha = hashed(protocol_path), hashed(data_path)
    check(protocol.get('expected_images') == 2112 and protocol.get('split') == 'test', 'Protocol test coverage differs')
    check(data.get('status') == 'passed_with_documentation_conflict', 'Dataset audit is not the verified completed audit')
    check(data.get('image_count') == 2112 and data.get('annotations') == 75462 and data.get('small_original_area_inclusive_0_1024') == 9447, 'Dataset audit counts differ')
    check(data.get('original_area_and_iscrowd_preserved') is True and data.get('original_vs_rle_area_mismatch_count') == 0 and data.get('rle_errors') == [], 'Dataset GT area/RLE audit failed')
    archive = Path(protocol['dataset'])
    check(archive.resolve() == Path(data['archive_path']).resolve(), 'Protocol and dataset audit archive paths differ')
    check(archive.stat().st_size == data['archive_bytes'], 'Archive size changed since dataset audit')
    with CourseArchive(archive, split='test') as dataset:
        images = dataset.images
        ground_truth = dataset.coco_ground_truth()
    gt_count = len(ground_truth['annotations'])
    small_count = sum(0 <= a['area'] <= 1024 for a in ground_truth['annotations'])
    check(len(images) == 2112 and gt_count == 75462 and small_count == 9447, 'Independent current ZIP GT counts differ')
    check(all((im['height'], im['width']) == (1024, 1024) for im in images), 'Original image dimensions differ')
    # Recompute the bounded annotation bytes digest, not the 20.55 GB archive.
    from zipfile import ZipFile
    annotation_hash = hashlib.sha256()
    with ZipFile(archive) as source:
        for member in dataset.annotation_members:
            annotation_hash.update(member.encode()+b'\0'+source.read(member))
    check(annotation_hash.hexdigest() == data['annotation_member_name_and_bytes_sha256'], 'Current GT annotation bytes changed')
    warnings.extend([
        'Full 20.55 GB archive SHA is the prior download/protocol receipt claim; this audit rechecks archive size and all test annotation bytes, not all RGB/depth bytes.',
        'Native RLE decoding is trusted only through each exact-file SHA-bound validation receipt; this auditor does not repeat native decoding or COCO evaluation.',
        'README no-test-GT and legacy submission statements conflict with actual archive; latest user CSV-with-JSON compressed-RLE contract is used.',
        'No-training/no-depth and GT-only-for-evaluation are verified against recorded flags and frozen source hashes; no independent runtime instrumentation is claimed.',
        'Different backbones, native postprocessing, and strict/non-strict 0.5 mask ties remain; shared scale is not a controlled architecture comparison.',
    ])
    del ground_truth
    for model in ('maskrcnn', 'mask2former'):
        output = base/model
        required = ['run.json', 'predictions.jsonl', 'submission.csv', 'submission_validation.json', 'evaluation.json']
        absent = [str(output/name) for name in required if not (output/name).is_file()]
        if absent:
            missing.extend(absent)
            continue
        try:
            run, validation, evaluation = (load(output/name) for name in ['run.json','submission_validation.json','evaluation.json'])
            runs[model] = run
            for name in required:
                hashed(output/name)
            check(run.get('status') == 'completed', f'{model}: run incomplete')
            check(run.get('scope') == 'full_test' and run.get('model') == model, f'{model}: wrong run identity/scope')
            check(all(run.get(k) == 2112 for k in ['expected_images','full_test_images','images_completed']), f'{model}: image coverage mismatch')
            check(Path(run['archive']).resolve() == archive.resolve() and run.get('archive_bytes') == data['archive_bytes'], f'{model}: data archive differs')
            check(run.get('protocol_sha256') == protocol_sha and run.get('dataset_audit_sha256') == data_sha, f'{model}: protocol/data audit SHA mismatch')
            required_sources = {'tools/run_course_pretrained.py','rgbd_seg/course_archive.py','rgbd_seg/course_submission.py',f'rgbd_seg/course_{model}.py'}
            check(set(run.get('source_sha256', {})) == required_sources, f'{model}: source coverage incomplete')
            for name, expected in run.get('source_sha256', {}).items():
                check(hashed(ROOT/name) == expected, f'{model}: source changed: {name}')
            for key in ['training_performed','depth_used','test_ground_truth_used_for_selection']:
                check(run.get(key) is False, f'{model}: {key} not explicitly false')
            check(run.get('score_threshold') == .01 and run.get('max_detections') == 100, f'{model}: score/cap differs')
            metadata = run['model_metadata']
            checkpoint = Path(metadata['checkpoint']) if model == 'maskrcnn' else Path(metadata['checkpoint_directory'])/'model.safetensors'
            checkpoint_sha = hashed(checkpoint)
            check(checkpoint_sha == metadata.get('checkpoint_sha256'), f'{model}: checkpoint SHA mismatch')
            if model == 'maskrcnn':
                check(checkpoint.stat().st_size == metadata.get('checkpoint_bytes'), f'{model}: checkpoint size differs')
                check(checkpoint_sha == metadata.get('weight_source',{}).get('pure_state_dict_sha256'), f'{model}: checkpoint provenance hash differs')
            else:
                check(hashed(checkpoint.parent/'source_receipt.json') == metadata.get('source_receipt_sha256'), f'{model}: weight source receipt changed')
                check(metadata.get('loading_info') == {'missing_keys': [], 'unexpected_keys': [], 'mismatched_keys': [], 'error_msgs': []}, f'{model}: checkpoint load not clean')
                check(metadata.get('adapter_sha256') == run['source_sha256'].get(f'rgbd_seg/course_{model}.py'), f'{model}: adapter source hash differs')
            preprocess = metadata['preprocess']
            size = preprocess if model == 'maskrcnn' else preprocess['size']
            check(size.get('shortest_edge') == 800 and size.get('longest_edge') == 1333, f'{model}: preprocessing scale differs')
            post = metadata['postprocess']
            check(post.get('max_detections') == 100 and post.get('score_threshold') == .01 and post.get('mask_threshold') == .5, f'{model}: postprocess differs')
            csv_sha, raw_sha = files[str((output/'submission.csv').resolve())], files[str((output/'predictions.jsonl').resolve())]
            check(run.get('submission_sha256') == csv_sha and run.get('predictions_sha256') == raw_sha, f'{model}: result file SHA mismatch')
            check(validation.get('sha256') == csv_sha and Path(validation['path']).resolve() == (output/'submission.csv').resolve(), f'{model}: validation receipt binds a different CSV')
            for flag in ['id_order_verified','rle_runs_verified','rle_native_decode_verified']:
                check(validation.get(flag) is True, f'{model}: validation {flag} not passed')
            if 'status' in validation:
                check(validation['status'] in ['pass','passed','completed'], f'{model}: validation status failed')
            check(validation.get('rows') == 2112 and validation.get('image_size') == [1024,1024] and validation.get('category_ids') == [1] and validation.get('max_detections') == 100, f'{model}: receipt scope differs')
            stream = compare_streams(output/'predictions.jsonl', output/'submission.csv', images)
            check(stream['predictions'] == run.get('predictions') == validation.get('predictions') == evaluation.get('predicted_instances'), f'{model}: prediction counts differ')
            check(stream['empty_images'] == validation.get('empty_images'), f'{model}: empty image counts differ')
            check(evaluation.get('status') == 'completed' and run.get('evaluation_status') == 'completed', f'{model}: evaluation incomplete')
            check(evaluation.get('images') == 2112 and evaluation.get('gt_instances') == gt_count and evaluation.get('gt_small_including_crowd') == small_count, f'{model}: evaluated GT scope differs')
            check(evaluation.get('split') == 'test' and evaluation.get('iou_type') == 'segm' and evaluation.get('max_detections') == [1,10,100], f'{model}: incorrect evaluation protocol')
            check(evaluation.get('ground_truth_area') == 'original annotation area preserved' and evaluation.get('prediction_area') == 'COCO decoded RLE area; no bbox passed to loadRes', f'{model}: area conventions differ')
            thresholds = evaluation.get('iou_thresholds', [])
            check(len(thresholds) == 10 and all(abs(x-(.5+i*.05))<1e-12 for i,x in enumerate(thresholds)), f'{model}: IoU thresholds differ')
            check(evaluation.get('scale') == '0–1; no multiplication by 100', f'{model}: scale declaration differs')
            metrics = evaluation.get('metrics', {})
            check(set(metrics) == set(METRICS), f'{model}: incomplete 12 COCO metrics')
            for key,value in metrics.items():
                # All current GT bins are populated; unavailable None would be
                # legitimate for absent bins, not silently coerced into zero.
                check(value is not None, f'{model}: unexpected unavailable {key}; all GT area bins are populated')
                if value is not None:
                    check(type(value) in (int,float) and math.isfinite(value) and 0 <= value <= 1, f'{model}: metric {key} not finite 0–1')
            timing = run.get('timing', {})
            check(timing.get('batch') == 1 and timing.get('precision') == 'float32' and timing.get('warmup') == 3 and timing.get('threads') == 4, f'{model}: timing protocol differs')
            for key in ['mean_ms','median_ms','p95_ms','fps']:
                value=timing.get(key)
                check(type(value) in (int,float) and math.isfinite(value) and value>0, f'{model}: invalid timing {key}')
            for key, measured in stream['timing_from_jsonl'].items():
                value = timing.get(key)
                check(type(value) in (int,float) and math.isclose(value, measured, rel_tol=1e-10, abs_tol=1e-8), f'{model}: timing {key} disagrees with per-image records')
            check(str(run.get('device', '')).startswith('cuda:'), f'{model}: run did not use CUDA')
            allocated, reserved = run.get('peak_cuda_allocated_bytes'), run.get('peak_cuda_reserved_bytes')
            check(type(allocated) is int and allocated > 0 and type(reserved) is int and reserved >= allocated, f'{model}: invalid peak CUDA memory metadata')
            start, finish = datetime.fromisoformat(run['started_at_utc']), datetime.fromisoformat(run['finished_at_utc'])
            check(start.tzinfo is not None and finish.tzinfo is not None and finish > start, f'{model}: invalid run interval')
            details[model] = {'output': str(output), 'checkpoint_sha256': checkpoint_sha, 'stream': stream,
                              'metrics': metrics, 'start': run['started_at_utc'], 'finish': run['finished_at_utc'],
                              'original_gt_area_preserved': True, 'native_decode_receipt_sha256': files[str((output/'submission_validation.json').resolve())]}
        except Exception as error:
            errors.append(f'{model}: {type(error).__name__}: {error}')
    execution = {"timing_comparison_controlled": False, "status": "not_yet_checked"}
    if len(details) == 2:
        first, second = runs['maskrcnn'], runs['mask2former']
        for key in ['physical_cuda_visible_devices','device_name','device','threads','protocol_sha256','dataset_audit_sha256']:
            check(first.get(key) == second.get(key) and first.get(key) is not None, f'Paired runs differ in {key}')
        check(first.get('physical_cuda_visible_devices') == str(protocol['compute']['gpu']), 'Runs did not use protocol physical GPU')
        deviation_path = base/'execution_deviation.json'
        deviation = load(deviation_path) if deviation_path.exists() else None
        if deviation is not None:
            hashed(deviation_path)
        execution, execution_errors, execution_warnings = execution_checks(first, second, deviation, protocol_sha)
        errors.extend(execution_errors)
        warnings.extend(execution_warnings)
        for name in ['tools/run_course_pretrained.py','rgbd_seg/course_archive.py','rgbd_seg/course_submission.py']:
            check(first['source_sha256'].get(name) == second['source_sha256'].get(name), f'Shared source differs between runs: {name}')
    status = 'fail' if errors else ('incomplete' if missing or len(details) != 2 else 'pass')
    return {'status': status, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
            'errors': errors, 'warnings': warnings, 'missing': missing, 'models': details,
            'expected_complete_models': 2, 'complete_models_checked': len(details),
            'execution': execution, 'timing_comparison_controlled': execution['timing_comparison_controlled'],
            'official_images': len(images), 'independent_gt_instances': gt_count, 'independent_small_gt': small_count,
            'protocol_sha256': protocol_sha, 'dataset_audit_sha256': data_sha,
            'archive_sha256_from_protocol_not_recomputed': protocol.get('archive_download_sha256'),
            'current_annotation_bytes_sha256': annotation_hash.hexdigest(),
            'actual_file_sha256': files, 'auditor_sha256': sha256(__file__),
            'audit_scope': 'Read-only files, current ZIP annotation metadata, hashes, result structure and semantics; no model, training, download, native RLE decode or COCO rerun.'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, default=ROOT/'reports/course_pretrained')
    parser.add_argument('--output', type=Path, default=ROOT/'reports/course_pretrained/final_audit.json')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f'Refusing to overwrite {args.output}')
    try:
        report=audit(args.base)
    except Exception as error:
        report={'status':'fail','errors':[f'{type(error).__name__}: {error}'], 'warnings':[], 'auditor_sha256':sha256(__file__)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:report.get(k) for k in ['status','complete_models_checked','errors','missing']},ensure_ascii=False))
    return 0 if report['status']=='pass' else 1


if __name__ == '__main__':
    sys.exit(main())
