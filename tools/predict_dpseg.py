#!/usr/bin/env python3
"""Run the current Swin-Large DPSeg on the local course RGB/depth archive.

Inference uses CUDA:0 (select the physical GPU with CUDA_VISIBLE_DEVICES).
The check-manifest command and help do not initialize CUDA.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DEFAULT_MANIFEST = ROOT / 'configs/dpseg_current.json'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    temporary = Path(path).with_suffix(Path(path).suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def verify_reference(item, label):
    path = resolve(item['path'])
    if digest(path) != item['sha256']:
        raise ValueError(f'{label} SHA256 differs: {path}')
    return path


def check_manifest(path, *, verify_weights=False, verify_evidence=False):
    """Verify the current implementation and frozen model lineage on CPU."""
    path = resolve(path)
    manifest = read(path)
    if manifest['schema'] != 'dpseg-current-v1':
        raise ValueError('Unsupported DPSeg manifest schema')
    from rgbd_seg.course_dpseg import DEFAULT_MERGE, DEFAULT_OVERLAP, OVERLAP_CONFIGS
    if (manifest['default']['strategy'] != DEFAULT_MERGE
            or manifest['default']['overlap'] != DEFAULT_OVERLAP
            or manifest['overlap_configs'] != json.loads(json.dumps(OVERLAP_CONFIGS))):
        raise ValueError('Manifest default or geometry differs from the inference module')
    required = {'tools/predict_dpseg.py', 'rgbd_seg/course_dpseg.py',
                'rgbd_seg/course_guarded_merge.py', 'tools/evaluate_course_swinl_removal.py'}
    if not required <= manifest['source_sha256'].keys():
        raise ValueError('Manifest omits required inference sources')
    for name, checksum in manifest['source_sha256'].items():
        source = resolve(name)
        if not source.is_relative_to(ROOT) or digest(source) != checksum:
            raise ValueError(f'Current source SHA256 differs: {name}')
    protocol = read(verify_reference(manifest['protocol'], 'Frozen reference protocol'))
    system = read(verify_reference(manifest['system_protocol'], 'Model protocol'))
    if manifest['system_protocol']['sha256'] != protocol['system_protocol']['sha256']:
        raise ValueError('Model protocol differs from the frozen reference')
    for name, checksum in protocol['new_source_sha256'].items():
        if manifest['source_sha256'].get(name) != checksum:
            raise ValueError(f'Frozen reference source omitted or changed: {name}')
    for name, checksum in system['system_source_sha256'].items():
        if digest(resolve(name)) != checksum:
            raise ValueError(f'Model source SHA256 differs: {name}')
    for name, item in manifest['weights'].items():
        if item['sha256'] != protocol['weights'][name]['sha256']:
            raise ValueError(f'Weight lineage differs: {name}')
        if verify_weights:
            verify_reference(item, f'{name} weights')
    if set(manifest['weights']) != {'base', 'heads'}:
        raise ValueError('Expected base and heads weights')
    if verify_evidence:
        for name, item in manifest['test_evidence']['artifacts'].items():
            verify_reference(item, f'Test evidence {name}')
    return manifest, protocol, dict(status='passed', manifest=str(path),
        manifest_sha256=digest(path), sources_verified=True, weights_verified=verify_weights,
        evidence_files_verified=verify_evidence, archive_sha256_verified=False,
        default=manifest['default'])


def selected_indices(raw, count):
    try:
        values = [int(part.strip()) for part in raw.split(',')]
    except ValueError as error:
        raise ValueError('--indices must be comma-separated integer positions') from error
    if not values or len(set(values)) != len(values) or any(i < 0 or i >= count for i in values):
        raise ValueError('--indices must be unique, nonnegative, in-range positions')
    return sorted(values)


def dataset(args):
    if args.evaluate and not args.all:
        raise ValueError('--evaluate requires --all and an explicitly selected --split')
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Output must be a new directory: {output}')
    manifest, original_plan, receipt = check_manifest(args.manifest)
    overlap = manifest['default']['overlap'] if args.overlap is None else args.overlap
    plan = copy.deepcopy(original_plan)
    for name, override in (('base', args.base_weights), ('heads', args.head_weights)):
        item = copy.deepcopy(manifest['weights'][name])
        if override:
            item['path'] = str(Path(override).expanduser().resolve())
            verify_reference(item, f'{name} weight override')
        item['path'] = str(resolve(item['path']))
        plan['weights'][name] = item
    plan['system_protocol']['path'] = str(resolve(manifest['system_protocol']['path']))
    archive = (Path(args.archive).expanduser().resolve() if args.archive
               else resolve(manifest['dataset']['archive']))
    archive_verified = False
    if args.verify_archive:
        if digest(archive) != manifest['dataset']['archive_sha256']:
            raise ValueError('Archive SHA256 differs from the recorded course dataset')
        archive_verified = True
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_dpseg import predict_dpseg, four_tile_geometry
    from rgbd_seg.course_submission import SubmissionWriter
    from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
    from tools.evaluate_course_swinl_removal import configure_and_load
    with CourseArchive(archive, split=args.split) as data:
        if len(data) != manifest['dataset']['split_counts'][args.split]:
            raise ValueError('Archive image count differs from the official split')
        if args.split == 'test' and data.image_ids != original_plan['image_ids']:
            raise ValueError('Test IDs/order differ from the reference protocol')
        if any((row['height'], row['width']) != (1024, 1024) for row in data.images):
            raise ValueError('DPSeg requires original 1024 by 1024 images')
        indices = list(range(len(data))) if args.all else selected_indices(args.indices, len(data))
        ids = [data.image_ids[i] for i in indices]
        if not torch.cuda.is_available():
            raise RuntimeError('DPSeg inference requires CUDA; use CUDA_VISIBLE_DEVICES to select a GPU')
        output.mkdir(parents=True, exist_ok=False)
        started = time.monotonic()
        run = dict(status='loading', created_at_utc=datetime.now(timezone.utc).isoformat(),
            **{k: v for k, v in receipt.items() if k not in {'status', 'archive_sha256_verified'}},
            archive=str(archive), archive_bytes=archive.stat().st_size,
            archive_sha256_verified=archive_verified, expected_archive_sha256=manifest['dataset']['archive_sha256'],
            split=args.split, full_split=args.all, images=len(ids), image_ids=ids,
            strategy=manifest['default']['strategy'], geometry=four_tile_geometry(overlap),
            is_current_default=(overlap == manifest['default']['overlap']),
            submission_ready=(args.split == 'test' and args.all),
            evaluation_requested=args.evaluate, camera_intrinsics_used=False,
            prediction_input_keys=['image', 'depth_m', 'depth_valid'], weights=plan['weights'],
            source_sha256=manifest['source_sha256'])
        # The archive SHA is optional because the original archive is about 20 GB.
        # Metadata checks above are recorded separately and never imply byte identity.
        try:
            write(output / 'run.json', run)
            write(output / 'manifest_snapshot.json', manifest)
            adapter, heads = configure_and_load(plan)
            run.update(status='predicting', weights_verified=True, gpu=torch.cuda.get_device_name(0))
            write(output / 'run.json', run)
            predicted = 0
            with (output / 'predictions.jsonl').open('x') as stream, \
                    SubmissionWriter(output / 'submission.csv', ids) as writer, torch.inference_mode():
                for position, (index, image_id) in enumerate(zip(indices, ids), 1):
                    sample = load_sample(data, image_id, 'depth')
                    predictions, statistics = predict_dpseg(adapter, heads.depth, sample, overlap=overlap)
                    writer.write(image_id, predictions)
                    stream.write(json.dumps(dict(image_id=image_id,
                        file_name=data.images[index]['file_name'], predictions=predictions,
                        statistics=statistics), allow_nan=False) + '\n')
                    stream.flush()
                    predicted += len(predictions)
                    if position == 1 or position % 25 == 0 or position == len(ids):
                        print(json.dumps(dict(stage='predicting', completed=position,
                            images=len(ids), predicted_instances=predicted)), flush=True)
            validation = validate_csv_fast(output / 'submission.csv', ids)
            write(output / 'csv_validation.json', validation)
            metrics = None
            if args.evaluate:
                evaluation = score_predictions(data, output / 'predictions.jsonl', output)
                evaluation.update(camera_intrinsics_used=False, merge_strategy=run['strategy'],
                    overlap=overlap, manifest_sha256=receipt['manifest_sha256'])
                write(output / 'evaluation.json', evaluation)
                metrics = evaluation['metrics']
            _, _, final_receipt = check_manifest(args.manifest)
            if final_receipt['manifest_sha256'] != receipt['manifest_sha256']:
                raise ValueError('Manifest changed during prediction')
            run.update(status='completed', elapsed_seconds=time.monotonic() - started,
                predicted_instances=predicted, metrics=metrics,
                predictions_sha256=digest(output / 'predictions.jsonl'),
                csv_sha256=digest(output / 'submission.csv'))
            write(output / 'run.json', run)
        except BaseException as error:
            run.update(status='failed', error=f'{type(error).__name__}: {error}',
                elapsed_seconds=time.monotonic() - started)
            write(output / 'run.json', run)
            raise
    print(json.dumps(dict(status='completed', output=str(output), images=len(ids),
        submission_ready=run['submission_ready'], metrics=metrics), ensure_ascii=False))


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest='command', required=True)
    check = commands.add_parser('check-manifest', help='verify source and protocol hashes on CPU')
    check.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    check.add_argument('--verify-weights', action='store_true', help='also hash the model weights')
    check.add_argument('--verify-evidence', action='store_true', help='also hash recorded test artifacts')
    predict = commands.add_parser('dataset', help='predict local course val/test images on CUDA')
    predict.add_argument('--manifest', default=str(DEFAULT_MANIFEST))
    predict.add_argument('--archive', help='local course ZIP; default comes from the manifest')
    predict.add_argument('--verify-archive', action='store_true', help='hash the entire archive before inference')
    predict.add_argument('--split', choices=('val', 'test'), required=True)
    scope = predict.add_mutually_exclusive_group(required=True)
    scope.add_argument('--all', action='store_true', help='predict the complete selected split')
    scope.add_argument('--indices', help='comma-separated zero-based indices, emitted in official order')
    predict.add_argument('--output', required=True, help='new output directory; existing paths are refused')
    predict.add_argument('--overlap', type=float, choices=(0., .2, .4, .6, .8),
                         help='target overlap; default is 0.4, changing this also changes crop size')
    predict.add_argument('--base-weights', help='alternate path to exactly the same base checkpoint bytes')
    predict.add_argument('--head-weights', help='alternate path to exactly the same point-head checkpoint bytes')
    predict.add_argument('--evaluate', action='store_true', help='score the full selected split; requires --all')
    return result


def main():
    arguments = parser().parse_args()
    if arguments.command == 'check-manifest':
        _, _, receipt = check_manifest(arguments.manifest,
            verify_weights=arguments.verify_weights, verify_evidence=arguments.verify_evidence)
        print(json.dumps(receipt, indent=2, ensure_ascii=False))
    else:
        dataset(arguments)


if __name__ == '__main__':
    main()
