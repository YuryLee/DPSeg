#!/usr/bin/env python3
"""Seven disjoint GPU postprocessing shards, followed by full COCO scoring."""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_depth_record
from rgbd_seg.course_depth_refinement import POLICY, refine_predictions
from rgbd_seg.course_distributed_eval import merge_shards, write_json
from rgbd_seg.course_dafusion_eval import score_predictions
from tools.select_course_resolution import read, sha, require


def checked_protocol(path):
    p = read(path)
    require(p['status'] == 'frozen' and p['policy'] == POLICY, 'Policy not frozen or changed')
    require(p['modes'] == ['spatial', 'depth'] and p['gpus'] == [0, 1, 2, 3, 5, 6, 7],
            'Unexpected modes/devices')
    require(p['scope'] in ('engineering_smoke', 'full_validation'), 'Invalid scope')
    for name, digest in p['source_sha256'].items():
        require(sha(ROOT/name) == digest, f'Frozen source changed: {name}')
    for key in ('decision', 'baseline_predictions', 'baseline_protocol'):
        require(sha(p[key]['path']) == p[key]['sha256'], f'{key} identity changed')
    decision = read(p['decision']['path'])
    require(decision['status'] == 'completed' and decision['recommended_method_for_next_stage'] == 'native'
            and not decision['test_used'], 'Selected comparison differs')
    baseline = decision['baseline']
    folder = Path(baseline['best_checkpoint']).parent
    require(sha(folder/'run.json') == baseline['run_sha256']
            and sha(folder/'selection.json') == baseline['selection_sha256'], 'Baseline changed')
    selected = read(folder/'selection.json')['epochs'][baseline['best_epoch']-1]
    require(Path(selected['output'])/'predictions.jsonl' == Path(p['baseline_predictions']['path']),
            'Not selected baseline predictions')
    original = read(p['baseline_protocol']['path'])
    require(original['archive'] == p['archive'], 'Archive differs from baseline')
    if p['scope'] == 'full_validation':
        evidence = p['engineering']
        require(sha(evidence['path']) == evidence['sha256'], 'Engineering evidence changed')
        smoke = read(evidence['path'])
        require(smoke['status'] == 'completed' and smoke['scope'] == 'engineering_smoke'
                and smoke['images'] == 14 and smoke['world_size'] == 7
                and smoke['source_sha256'] == p['source_sha256']
                and smoke['cpu_gpu_checks_passed'] is True, 'Engineering checks missing')
    return p


def worker(protocol, output, rank):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(protocol['gpus'][rank]), 'Wrong physical GPU')
    require(torch.cuda.device_count() == 1, 'Worker must see exactly one GPU')
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    ids = read(output/'image_ids.json')[rank::7]
    totals = {mode: dict(instances=0, changed_instances=0, added_pixels=0,
                        removed_pixels=0, empty_reverted=0) for mode in protocol['modes']}
    checks = 0
    started = time.monotonic()
    with CourseArchive(protocol['archive'], 'val') as dataset, ExitStack() as stack:
        by_id = dict(zip(dataset.image_ids, dataset.images))
        streams = {mode: stack.enter_context((output/mode/f'predictions_rank{rank}.jsonl').open('x'))
                   for mode in protocol['modes']}
        source = output/f'source_rank{rank}.jsonl'
        source_sha = sha(source)
        with source.open() as stream:
            for index, line in enumerate(stream):
                item = json.loads(line)
                require(item['image_id'] == ids[index], 'Shard image order differs')
                require(item['file_name'] == by_id[item['image_id']]['file_name'], 'RGB identity differs')
                depth, valid = load_depth_record(dataset._image_archive(), by_id[item['image_id']])
                depth, valid = depth[0], valid[0]
                for mode in protocol['modes']:
                    torch.cuda.synchronize()
                    begin = time.perf_counter()
                    predictions, stats = refine_predictions(item['predictions'], depth, valid,
                                                           mode=mode, device='cuda:0')
                    torch.cuda.synchronize()
                    elapsed = time.perf_counter()-begin
                    require([(x['category_id'], x['score']) for x in predictions] ==
                            [(x['category_id'], x['score']) for x in item['predictions']],
                            'Scores/order/count changed')
                    if protocol['scope'] == 'engineering_smoke' and rank == 0:
                        cpu, cpu_stats = refine_predictions(item['predictions'], depth, valid, mode=mode)
                        require(cpu == predictions and cpu_stats == stats, 'Real-image CPU/GPU mismatch')
                        checks += 1
                    result = {**item, 'predictions': predictions, 'refinement_seconds': elapsed}
                    streams[mode].write(json.dumps(result, allow_nan=False)+'\n')
                    for key, value in stats.items():
                        totals[mode][key] += value
                if index == 0 or (index+1) % 25 == 0 or index+1 == len(ids):
                    for output_stream in streams.values():
                        output_stream.flush()
                    write_json(output/f'progress_rank{rank}.json', {'phase': 'refining', 'rank': rank,
                        'images': index+1, 'expected': len(ids), 'elapsed_seconds': time.monotonic()-started})
        require(index+1 == len(ids) and sha(source) == source_sha, 'Shard incomplete or changed')
    write_json(output/f'rank{rank}.json', {'status': 'completed', 'rank': rank, 'pid': os.getpid(),
        'physical_gpu': protocol['gpus'][rank], 'ordered_ids': ids, 'images': len(ids),
        'source_sha256': source_sha, 'stats': totals, 'cpu_gpu_exact_checks': checks,
        'predictions_sha256': {mode: sha(output/mode/f'predictions_rank{rank}.jsonl') for mode in protocol['modes']},
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_seconds': time.monotonic()-started})


def parent(protocol, protocol_path, output):
    require(not output.exists(), 'Output must be new')
    output.mkdir(parents=True)
    for mode in protocol['modes']:
        (output/mode).mkdir()
    report = {'status': 'running', 'scope': protocol['scope'], 'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'protocol_sha256': sha(protocol_path), 'source_sha256': protocol['source_sha256'],
        'world_size': 7, 'test_accessed': False, 'new_model_inference': False,
        'ground_truth_in_refinement': False, 'pid': os.getpid(), 'workers': []}
    write_json(output/'run.json', report)
    children = []
    try:
        with CourseArchive(protocol['archive'], 'val') as dataset:
            require(len(dataset.image_ids) == 2584, 'Full validation metadata missing')
            ids = dataset.image_ids[:14] if protocol['scope'] == 'engineering_smoke' else dataset.image_ids
            report['images'] = len(ids)
            write_json(output/'image_ids.json', ids)
            with ExitStack() as stack:
                streams = [stack.enter_context((output/f'source_rank{rank}.jsonl').open('x')) for rank in range(7)]
                with Path(protocol['baseline_predictions']['path']).open() as source:
                    for index, line in enumerate(source):
                        item = json.loads(line)
                        require(item['image_id'] == dataset.image_ids[index], 'Baseline prediction order differs')
                        if index < len(ids):
                            streams[index % 7].write(line)
                    require(index+1 == 2584, 'Baseline predictions incomplete')
            with ExitStack() as stack:
                for rank, gpu in enumerate(protocol['gpus']):
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), OMP_NUM_THREADS='4',
                               MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
                    log = stack.enter_context((output/f'worker{rank}.log').open('x'))
                    command = [sys.executable, '-u', str(Path(__file__).resolve()), '--protocol', str(protocol_path),
                               '--output', str(output), '--rank', str(rank)]
                    child = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT)
                    children.append(child)
                    report['workers'].append({'rank': rank, 'pid': child.pid, 'physical_gpu': gpu})
                write_json(output/'run.json', report)
                start = time.monotonic()
                while any(child.poll() is None for child in children):
                    require(all(child.poll() in (None, 0) for child in children), 'Worker failed; inspect logs')
                    require(time.monotonic()-start < protocol['worker_timeout_seconds'], 'Worker deadline exceeded')
                    time.sleep(1)
                require(all(child.returncode == 0 for child in children), 'Worker failed')
            shards = [read(output/f'rank{rank}.json') for rank in range(7)]
            for rank, shard in enumerate(shards):
                require(shard['status'] == 'completed' and shard['rank'] == rank
                        and shard['ordered_ids'] == ids[rank::7], 'Shard identity/coverage differs')
                for mode in protocol['modes']:
                    require(sha(output/mode/f'predictions_rank{rank}.jsonl') == shard['predictions_sha256'][mode],
                            'Shard predictions changed')
            check_count = sum(x['cpu_gpu_exact_checks'] for x in shards)
            report['cpu_gpu_exact_checks'] = check_count
            report['cpu_gpu_checks_passed'] = check_count == 4 if protocol['scope'] == 'engineering_smoke' else None
            if protocol['scope'] == 'engineering_smoke':
                require(report['cpu_gpu_checks_passed'], 'Missing CPU/GPU real-image checks')
            report['metrics'] = {}
            report['refinement_stats'] = {}
            for mode in protocol['modes']:
                folder = output/mode
                count = merge_shards([folder/f'predictions_rank{rank}.jsonl' for rank in range(7)],
                                     ids, folder/'predictions.jsonl')
                write_json(output/'progress.json', {'phase': 'scoring', 'mode': mode, 'images': len(ids)})
                scored = score_predictions(dataset, folder/'predictions.jsonl', folder,
                    image_ids=ids if protocol['scope'] == 'engineering_smoke' else None)
                scored.update(scope='engineering_smoke' if protocol['scope'] == 'engineering_smoke' else 'full_split',
                    refinement_mode=mode, new_model_inference=False)
                write_json(folder/'evaluation.json', scored)
                write_json(folder/'evaluation_run.json', {'status': 'completed', 'split': 'val',
                    'scope': scored['scope'], 'images': len(ids), 'predicted_instances': count,
                    'predictions_sha256': sha(folder/'predictions.jsonl'),
                    'evaluation_sha256': sha(folder/'evaluation.json'), 'ground_truth_in_refinement': False})
                report['metrics'][mode] = scored['metrics']
                report['refinement_stats'][mode] = {key: sum(x['stats'][mode][key] for x in shards)
                                                     for key in shards[0]['stats'][mode]}
            require(sha(protocol['baseline_predictions']['path']) == protocol['baseline_predictions']['sha256'],
                    'Baseline predictions changed during run')
            report.update(status='completed', exit_code=0, finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        report.update(status='failed', exit_code=1, error_type=type(error).__name__, error=str(error))
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        raise
    finally:
        write_json(output/'run.json', report)
        write_json(output/'exit.json', {'status': report['status'], 'exit_code': report.get('exit_code', 1),
                   'worker_exit_codes': [p.returncode for p in children], 'protocol_sha256': sha(protocol_path)})
    print(json.dumps({'status': report['status'], 'scope': report['scope'], 'metrics': report['metrics']}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rank', type=int)
    args = parser.parse_args()
    protocol = checked_protocol(args.protocol)
    if args.rank is None:
        parent(protocol, args.protocol.resolve(), args.output.resolve())
    else:
        require(0 <= args.rank < 7, 'Invalid rank')
        worker(protocol, args.output.resolve(), args.rank)


if __name__ == '__main__':
    main()
