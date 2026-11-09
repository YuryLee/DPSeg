#!/usr/bin/env python3
"""Resume the interrupted fixed-overlap comparison without altering its evidence."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import evaluate_course_overlap as original

read, write, digest = original.read, original.write, original.digest
require, now, variants = original.require, original.now, original.variants
SOURCE = 'tools/resume_course_overlap.py'


def complete_rows(path):
    """Preserve all bytes separately; reuse only complete, valid JSONL records."""
    payload = Path(path).read_bytes()
    lines, rows, offset = [], [], 0
    for line in payload.splitlines(keepends=True):
        if not line.endswith(b'\n'):
            break
        require(bool(line.strip()), f'Empty completed JSONL record: {path}')
        rows.append(json.loads(line))
        lines.append(line)
        offset += len(line)
    return lines, rows, dict(bytes=len(payload), complete_rows=len(rows),
                            trailing_bytes=len(payload)-offset)


def verify_evidence(plan):
    original.check_dependencies(plan)
    recovery = plan['recovery']
    for name in ('original_protocol', 'interruption', 'original_evidence_manifest'):
        item = recovery[name]
        require(digest(item['path']) == item['sha256'], f'Recovery evidence changed: {name}')
    require(read(recovery['interruption']['path'])['exit_code'] == 143,
            'Original interruption was not preserved')
    for item in read(recovery['original_evidence_manifest']['path'])['files']:
        require(digest(item['original_path']) == digest(item['copy_path']) == item['sha256'],
                'Original/copied interruption evidence changed')
    for prefix in recovery['prefixes']:
        for item in prefix['artifacts'].values():
            require(digest(item['path']) == item['sha256'], 'Saved recovery prefix changed')


def prepare(source, output):
    require(source != output and not output.exists(), 'Recovery requires a fresh directory')
    plan = read(source/'protocol.json')
    original.check_dependencies(plan)
    original.check_window_configs(plan['overlap_configs'])
    gate = read(source/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['protocol_sha256'] == digest(source/'protocol.json'),
            'Original engineering evidence is invalid')
    require(not (source/'results.json').exists() and not (source/'inferring_exits.json').exists(),
            'Expected the specifically interrupted, uncompleted overlap run')
    paths = [source/f'{name}_rank{rank}.jsonl'
             for rank in range(len(plan['gpus'])) for name in (*variants(), 'checks')]
    total_bytes = sum(path.stat().st_size for path in paths)
    storage = shutil.disk_usage(output.parent)
    require(storage.free >= total_bytes*3 + 4*1024**3,
            'Insufficient space for preserved evidence, prefixes and completed predictions')
    output.mkdir(parents=True)
    backup = output/'evidence/original'
    backup.mkdir(parents=True)
    prefix_dir = output/'evidence/prefixes'
    prefix_dir.mkdir()
    files = []
    for path in [*paths, source/'protocol.json', source/'engineering_gate.json']:
        checksum = digest(path)
        target = backup/path.name
        shutil.copy2(path, target)
        require(digest(path) == digest(target) == checksum, 'Original file changed during backup')
        files.append(dict(original_path=str(path), copy_path=str(target), sha256=checksum,
                          bytes=path.stat().st_size))
    evidence = output/'evidence/original_evidence_manifest.json'
    write(evidence, dict(created_at_utc=now(), files=files))
    interruption = output/'evidence/interruption.json'
    write(interruption, dict(exit_code=143, signal='SIGTERM', cause='unknown external termination',
        original_controller_session_id=75176, original_worker_exit_codes=None,
        observation='The original foreground execution returned exit 143. Host process inspection '
                    'confirmed its six inference workers no longer existed. No worker exit codes were observed.',
        recorded_at_utc=now(), original_status_snapshot=read(source/'status.json'),
        original_inferring_exits_available=False, original_files_modified=False))
    prefixes = []
    for rank in range(len(plan['gpus'])):
        ids = plan['image_ids'][rank::len(plan['gpus'])]
        streams = {name: complete_rows(backup/f'{name}_rank{rank}.jsonl')
                   for name in (*variants(), 'checks')}
        count = min(len(value[1]) for value in streams.values())
        require(0 < count < len(ids), f'Unexpected common prefix size for rank {rank}')
        for name, (_, rows, _) in streams.items():
            require([row['image_id'] for row in rows] == ids[:len(rows)],
                    f'Original {name} rank {rank} is not the official ordered prefix')
        for index, image_id in enumerate(ids[:count]):
            values = {name: streams[name][1][index]['predictions'] for name in variants()}
            audit = streams['checks'][1][index]
            original.check_outputs(values, image_id)
            reference_checked = image_id in (ids[0], ids[-1])
            require(audit['native_exact'] is True and audit['reference40_checked'] is reference_checked
                    and audit['reference40_exact'] is (True if reference_checked else None)
                    and audit['instances'] == {name: len(values[name]) for name in variants()},
                    'Original completed prefix failed its inference audit')
            original.check_runtime_stats(audit['stats'], values, plan, reference_checked)
        artifacts = {}
        for name, (lines, _, details) in streams.items():
            target = prefix_dir/f'{name}_rank{rank}.jsonl'
            with target.open('xb') as stream:
                stream.writelines(lines[:count])
            artifacts[name] = dict(path=str(target), sha256=digest(target))
            details['complete_rows_excluded_from_reuse'] = len(lines)-count
        prefixes.append(dict(rank=rank, images=count, image_ids=ids[:count], artifacts=artifacts,
                             original_streams={name: value[2] for name, value in streams.items()}))
    plan = copy.deepcopy(plan)
    plan['created_at_utc'] = now()
    plan['new_source_sha256'][SOURCE] = digest(ROOT/SOURCE)
    plan['recovery'] = dict(original_run=str(source),
        original_protocol=dict(path=str(source/'protocol.json'), sha256=digest(source/'protocol.json')),
        interruption=dict(path=str(interruption), sha256=digest(interruption)),
        original_evidence_manifest=dict(path=str(evidence), sha256=digest(evidence)),
        prefixes=prefixes, reused_images_per_configuration=sum(row['images'] for row in prefixes),
        remaining_images_per_configuration=len(plan['image_ids'])-sum(row['images'] for row in prefixes),
        policy='Exact complete common prefixes only; all original bytes preserved; independently '
               'recompute prefix endpoints; append missing images under unchanged inference rules',
        storage_free_bytes_before=storage.free)
    verify_evidence(plan)
    write(output/'protocol.json', plan)
    write(output/'status.json', dict(status='prepared', at_utc=now(), recovery=True,
        reused_images=plan['recovery']['reused_images_per_configuration'],
        remaining_images=plan['recovery']['remaining_images_per_configuration']))
    print(json.dumps(read(output/'status.json')), flush=True)


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_dpseg import predict_overlaps
    from tools.check_course_prediction_reload import compare_predictions

    plan = read(output/'protocol.json')
    gate = read(output/'engineering_gate.json')
    require(gate['status'] == 'passed' and gate['protocol_sha256'] == digest(output/'protocol.json'),
            'Recovery engineering gate has not passed')
    verify_evidence(plan)
    prefix = plan['recovery']['prefixes'][rank]
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    count = prefix['images']
    require(prefix['rank'] == rank and prefix['image_ids'] == ids[:count], 'Wrong rank prefix')
    endpoints = list(dict.fromkeys((ids[0], ids[count-1])))
    reference_ids = list(dict.fromkeys((ids[0], ids[-1])))
    required_native_ids = list(dict.fromkeys([*endpoints, *ids[count:]]))
    required_reference_ids = list(dict.fromkeys([*endpoints, ids[-1]]))
    native_reference = original.reference_rows(plan['native_reference'], required_native_ids)
    full_reference = original.reference_rows(plan['full_reference'], required_reference_ids)
    saved = {name: complete_rows(prefix['artifacts'][name]['path'])[1] for name in variants()}
    counts = {name: sum(len(row['predictions']) for row in saved[name]) for name in variants()}
    adapter, heads = original.configure_and_load(plan)
    began, identity = time.monotonic(), []
    with ExitStack() as stack, torch.inference_mode():
        data = stack.enter_context(CourseArchive(plan['archive'], split='test'))
        require(data.image_ids == plan['image_ids'], 'Test IDs/order changed')
        for image_id in endpoints:
            sample = load_sample(data, image_id, 'depth')
            require(set(sample) == {'image', 'depth_m', 'depth_valid'}, 'Unexpected inference input')
            values, stats, native, reference40 = predict_overlaps(adapter, heads.depth, sample,
                                                                include_reference=True)
            original.check_outputs(values, image_id)
            original.check_runtime_stats(stats, values, plan, True)
            index = prefix['image_ids'].index(image_id)
            for name in variants():
                compare_predictions(saved[name][index]['predictions'], values[name], score_tolerance=0.)
            compare_predictions(native_reference.pop(image_id), native, score_tolerance=0.)
            compare_predictions(full_reference.pop(image_id), reference40, score_tolerance=0.)
            identity.append(dict(image_id=image_id, predictions_exact=True, native_exact=True,
                                 reference40_exact=True))
            write(output/f'prefix_identity_rank{rank}.json', dict(status='checking', checks=identity))
            del sample, values, stats, native, reference40
        write(output/f'prefix_identity_rank{rank}.json', dict(status='passed', checks=identity,
              protocol_sha256=digest(output/'protocol.json')))
        streams = {name: stack.enter_context((output/f'{name}_rank{rank}.jsonl').open('xb'))
                   for name in (*variants(), 'checks')}
        for name, stream in streams.items():
            with Path(prefix['artifacts'][name]['path']).open('rb') as before:
                shutil.copyfileobj(before, stream)
            stream.flush()
        for index in range(count, len(ids)):
            image_id = ids[index]
            sample = load_sample(data, image_id, 'depth')
            require(set(sample) == {'image', 'depth_m', 'depth_valid'}, 'Unexpected inference input')
            checked_reference = image_id in reference_ids
            values, stats, native, reference40 = predict_overlaps(adapter, heads.depth, sample,
                                                                include_reference=checked_reference)
            original.check_outputs(values, image_id)
            original.check_runtime_stats(stats, values, plan, checked_reference)
            compare_predictions(native_reference.pop(image_id), native, score_tolerance=0.)
            if checked_reference:
                compare_predictions(full_reference.pop(image_id), reference40, score_tolerance=0.)
            else:
                require(reference40 is None, 'Unexpected unrequested reference inference')
            for name in variants():
                streams[name].write((json.dumps(dict(image_id=image_id, predictions=values[name]),
                                               allow_nan=False)+'\n').encode())
                counts[name] += len(values[name])
            row = dict(image_id=image_id, native_exact=True, reference40_checked=checked_reference,
                reference40_exact=True if checked_reference else None,
                instances={name: len(values[name]) for name in variants()}, stats=stats)
            streams['checks'].write((json.dumps(row, allow_nan=False)+'\n').encode())
            for stream in streams.values():
                stream.flush()
            if index == count or (index+1) % 10 == 0 or index+1 == len(ids):
                progress = dict(status='inferring', rank=rank, images=index+1, expected=len(ids),
                    prefix_images=count, new_images=index+1-count, elapsed_seconds=time.monotonic()-began)
                write(output/f'progress_rank{rank}.json', progress)
                print(json.dumps(progress), flush=True)
            del sample, values, stats, native, reference40
    require(not native_reference and not full_reference, 'Uncompared reference images remain')
    original.check_dependencies(plan)
    write(output/f'rank{rank}.json', dict(status='completed', image_ids=ids, images=len(ids),
        gpu=torch.cuda.get_device_name(0), instances=counts, native_exact=True,
        reference40_exact=True, reference40_image_ids=reference_ids, ground_truth_in_prediction=False,
        artifacts={name: digest(output/f'{name}_rank{rank}.jsonl') for name in variants()},
        checks_sha256=digest(output/f'checks_rank{rank}.jsonl'),
        recovery=dict(prefix_images=count, new_images=len(ids)-count, prefix_identity_checks=identity)))


def run_children(output, phase, specifications, timeout):
    children, logs = [], []
    began, timed_out = time.monotonic(), False
    try:
        for name, arguments, gpu in specifications:
            log = (output/f'{name}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            child = subprocess.Popen([sys.executable, '-u', str(Path(__file__).resolve()),
                '--output', str(output), *arguments], cwd=ROOT, env=env,
                stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT)
            children.append((name, child))
        write(output/'status.json', dict(status=phase, started_at_utc=now(), recovery=True,
            controller_pid=os.getpid(), hard_timeout_seconds=timeout,
            processes=[dict(name=name, pid=child.pid) for name, child in children]))
        reported = set()
        while any(child.poll() is None for _, child in children):
            if time.monotonic()-began > timeout:
                timed_out = True
                print(f'{phase}: declared timeout; terminating only this phase children.', flush=True)
                raise TimeoutError(f'{phase} exceeded declared {timeout}s limit')
            for name, child in children:
                if child.poll() not in (None, 0) and name not in reported:
                    reported.add(name)
                    print(f'{phase}: {name} failed; others continue, no automatic retry.', flush=True)
            time.sleep(3)
        require(all(child.returncode == 0 for _, child in children), f'{phase} did not finish successfully')
    finally:
        if timed_out:
            for _, child in children:
                if child.poll() is None:
                    child.terminate()
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        child.kill()
                        child.wait()
        write(output/f'{phase}_exits.json', {name: child.poll() for name, child in children})
        for log in logs:
            log.close()


def run(output):
    plan = read(output/'protocol.json')
    require(read(output/'status.json')['status'] == 'prepared', 'Recovery already started')
    verify_evidence(plan)
    gpus = plan['gpus']
    capacity = original.gpu_preflight(gpus)
    write(output/'recovery_preflight.json', dict(at_utc=now(), gpus=capacity,
          free_bytes=shutil.disk_usage(output).free, controller_pid=os.getpid()))
    try:
        run_children(output, 'engineering', [('engineering', ['--engineering'], gpus[-1])], 3600)
        original.gpu_preflight(gpus)
        run_children(output, 'inferring', [(f'worker{rank}', ['--worker', str(rank)], gpu)
                                         for rank, gpu in enumerate(gpus)], 21600)
        run_children(output, 'scoring', [(f'score_{name}', ['--score', name], '')
                                       for name in variants()], 7200)
        results = {name: read(output/name/'result.json') for name in variants()}
        verify_evidence(plan)
        write(output/'results.json', dict(status='completed', finished_at_utc=now(), split='test', images=2112,
            protocol_sha256=digest(output/'protocol.json'), variants=results, full_reference=plan['full_reference'],
            training=False, test_used_for_selection=False, default_changed_by_experiment=False,
            default_after_run='overlap40', reference_variant='overlap40', recovery=plan['recovery']))
        write(output/'status.json', dict(status='completed', exit_code=0, finished_at_utc=now(), recovery=True))
        print(json.dumps({key: row['metrics'] for key, row in results.items()}, allow_nan=False), flush=True)
    except BaseException as error:
        write(output/'status.json', dict(status='failed', error=str(error), at_utc=now(), recovery=True))
        raise
    with (output/'summary.log').open('x') as log:
        result = subprocess.run([sys.executable, '-u', str(ROOT/'tools/summarize_course_overlap.py'),
            '--run', str(output), '--docs', str(ROOT/'docs/results'), '--update-readme'],
            cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    write(output/'summary_exit.json', dict(exit_code=result.returncode, at_utc=now()))
    require(result.returncode == 0, 'Inference/scoring completed, but summary audit failed; see summary.log')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run-prepared', action='store_true')
    parser.add_argument('--engineering', action='store_true')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--score')
    args = parser.parse_args()
    output = args.output.resolve()
    if args.engineering:
        original.engineering(output)
    elif args.worker is not None:
        worker(output, args.worker)
    elif args.score:
        require(args.score in variants(), 'Unknown overlap variant')
        original.score(output, args.score)
    elif args.run_prepared:
        run(output)
    else:
        require(args.source is not None, 'Provide the interrupted source run')
        prepare(args.source.resolve(), output)
        if not args.prepare_only:
            run(output)


if __name__ == '__main__':
    main()
