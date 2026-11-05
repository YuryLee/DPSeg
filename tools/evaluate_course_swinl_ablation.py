#!/usr/bin/env python3
"""Complete the locked Swin-Large S2/S3 ablation on all official test images."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path = Path(path)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def worker(output, rank):
    import torch
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone import load_sample
    from rgbd_seg.course_depth_backbone_points import PairedPointHeads, predict_paired
    from rgbd_seg.course_tiled_probability import refined_public
    from tools.run_course_depth_backbone_system import load_base, sources
    from tools.check_course_prediction_reload import compare_predictions

    plan = read(output/'protocol.json')
    if digest(__file__) != plan['runner_sha256']:
        raise ValueError('Frozen runner changed')
    system = read(plan['system_protocol'])
    if sources() != system['system_source_sha256']:
        raise ValueError('Frozen inference sources changed')
    for name in ('base', 'heads', 'native_predictions'):
        if digest(plan[name]['path']) != plan[name]['sha256']:
            raise ValueError(f'Frozen {name} changed')
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Leave room for resident services; accuracy runs do not measure speed.
    total = torch.cuda.get_device_properties(0).total_memory
    torch.cuda.set_per_process_memory_fraction(min(1., 18 * 1024**3 / total), 0)
    adapter = load_base(system, Path(plan['base']['path']), torch.device('cuda:0'))
    heads = PairedPointHeads(fine_channels=448)
    saved = torch.load(plan['heads']['path'], map_location='cpu', weights_only=True)
    if saved['epoch'] != 6 or saved['base_checkpoint_sha256'] != plan['base']['sha256']:
        raise ValueError('Point checkpoint differs from fixed epoch6/base15')
    heads.load_state_dict(saved['heads'], strict=True)
    heads.eval().requires_grad_(False).to(adapter.device)
    del saved
    ids = plan['image_ids'][rank::len(plan['gpus'])]
    id_set = set(ids)
    references = {}
    with Path(plan['native_predictions']['path']).open() as stream:
        for line in stream:
            row = json.loads(line)
            if row['image_id'] in id_set:
                references[row['image_id']] = row['predictions']
    if set(references) != id_set:
        raise ValueError('Native reference coverage differs')
    started = time.monotonic()
    max_difference = 0.
    with CourseArchive(plan['archive'], split='test') as data, \
         (output/f'S2_rank{rank}.jsonl').open('x') as f2, \
         (output/f'S3_rank{rank}.jsonl').open('x') as f3, torch.inference_mode():
        if data.image_ids != plan['image_ids']:
            raise ValueError('Official test IDs/order changed')
        for index, image_id in enumerate(ids):
            sample = load_sample(data, image_id, 'depth')
            if set(sample) != {'image', 'depth_m', 'depth_valid'}:
                raise ValueError('Unexpected inference inputs')
            paired = predict_paired(adapter, heads, sample, head_names=('point_depth',))
            max_difference = max(max_difference, compare_predictions(references.pop(image_id),
                [c.public() for c in paired['native']]))
            point = [c.public() for c in paired['point_depth']]
            probability, _ = refined_public(paired['point_depth'], sample, 'probability', adapter.device)
            if [(p['category_id'], p['score']) for p in point] != [
                    (p['category_id'], p['score']) for p in probability]:
                raise ValueError('Probability refinement changed candidate identities or scores')
            for stream, predictions in ((f2, point), (f3, probability)):
                stream.write(json.dumps(dict(image_id=image_id, predictions=predictions), allow_nan=False)+'\n')
            if index == 0 or (index+1) % 25 == 0 or index+1 == len(ids):
                f2.flush()
                f3.flush()
                state = dict(status='inferring', rank=rank, images=index+1, expected=len(ids),
                    elapsed_seconds=time.monotonic()-started, native_max_score_difference=max_difference)
                write(output/f'progress_rank{rank}.json', state)
                print(json.dumps(state), flush=True)
            del paired, sample, point, probability
    write(output/f'rank{rank}.json', dict(status='completed', images=len(ids), image_ids=ids,
        gpu=torch.cuda.get_device_name(0), native_masks_identical=True,
        native_max_score_difference=max_difference,
        artifacts={stage: digest(output/f'{stage}_rank{rank}.jsonl') for stage in ('S2', 'S3')}))


def finalize(output):
    from rgbd_seg.course_archive import CourseArchive
    from rgbd_seg.course_depth_backbone_eval import merge_shards
    from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
    from rgbd_seg.course_submission import SubmissionWriter
    from tools.audit_course_point_refinement import compare_csv_predictions

    plan = read(output/'protocol.json')
    ids = plan['image_ids']
    results = {}
    with CourseArchive(plan['archive'], split='test') as data:
        for stage in ('S2', 'S3'):
            folder = output/stage
            folder.mkdir()
            shards = [output/f'{stage}_rank{rank}.jsonl' for rank in range(len(plan['gpus']))]
            for rank, shard in enumerate(shards):
                receipt = read(output/f'rank{rank}.json')
                if receipt['status'] != 'completed' or receipt['artifacts'][stage] != digest(shard):
                    raise ValueError('Incomplete or changed prediction shard')
            prediction_path = folder/'predictions.jsonl'
            count = merge_shards(shards, ids, prediction_path)
            evaluation = score_predictions(data, prediction_path, folder)
            suffix = 'depth_point' if stage == 'S2' else 'depth_point_probability'
            csv_path = output/'submissions'/f'dpseg_swinl_depth_no_k_{stage.lower()}_{suffix}.csv'
            with SubmissionWriter(csv_path, ids) as writer, prediction_path.open() as stream:
                for line in stream:
                    row = json.loads(line)
                    writer.write(row['image_id'], row['predictions'])
            validation = validate_csv_fast(csv_path, ids)
            binding = compare_csv_predictions(csv_path, prediction_path, ids)
            write(folder/'csv_validation.json', validation)
            results[stage] = dict(metrics=evaluation['metrics'], images=len(ids), predicted_instances=count,
                evaluation=str(folder/'evaluation.json'), predictions=str(prediction_path),
                predictions_sha256=digest(prediction_path), csv=str(csv_path), csv_sha256=digest(csv_path),
                csv_prediction_rows_identical=binding)
            print(json.dumps({'stage': stage, **results[stage]}, ensure_ascii=False), flush=True)
    write(output/'results.json', dict(status='completed', split='test', images=len(ids),
        stages=results, test_used_for_selection=False, protocol_sha256=digest(output/'protocol.json'),
        finished_at_utc=datetime.now(timezone.utc).isoformat()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--run', type=Path, default=ROOT/'reports/course_swinl_depth_no_k_v1/swinl_depth_no_k')
    parser.add_argument('--archive', type=Path, default=ROOT.parent/'downloads/buaa_image_segmentation/blender.zip')
    parser.add_argument('--gpus', default='3,4')
    parser.add_argument('--worker', type=int)
    args = parser.parse_args()
    output = args.output.resolve()
    if args.worker is not None:
        worker(output, args.worker)
        return
    from rgbd_seg.course_archive import CourseArchive
    run = args.run.resolve()
    output.mkdir(parents=True, exist_ok=False)
    with CourseArchive(args.archive, split='test') as data:
        ids = data.image_ids
    if len(ids) != 2112 or len(set(ids)) != 2112:
        raise ValueError('Expected the complete 2112-image test set')
    paths = dict(base=run/'base15/final.pt', heads=run/'points6/heads_epoch_06.pt',
                 native_predictions=run/'final_evaluation/test_native/predictions.jsonl')
    plan = dict(scope='full_test_inference_only', archive=str(args.archive.resolve()), image_ids=ids,
        gpus=args.gpus.split(','), system_protocol=str(run/'protocol_system.json'),
        runner_sha256=digest(__file__), source_sha256=read(run/'protocol_system.json')['system_source_sha256'],
        started_at_utc=datetime.now(timezone.utc).isoformat(), test_used_for_selection=False,
        ground_truth_in_prediction=False, input_keys=['image', 'depth_m', 'depth_valid'],
        precision='float32', tf32=False, max_detections=100, score_threshold=.01,
        stages={'S2': 'whole image + depth point head',
                'S3': 'S2 + fixed depth-guided probability refinement; no tiles'},
        **{name: dict(path=str(path), sha256=digest(path)) for name, path in paths.items()})
    write(output/'protocol.json', plan)
    children, logs = [], []
    try:
        for rank, gpu in enumerate(plan['gpus']):
            log = (output/f'worker{rank}.log').open('x')
            logs.append(log)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='4',
                       MKL_NUM_THREADS='4', OPENBLAS_NUM_THREADS='4')
            command = [sys.executable, str(Path(__file__).resolve()), '--output', str(output), '--worker', str(rank)]
            children.append(subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        write(output/'status.json', dict(status='inferring', pids=[child.pid for child in children]))
        deadline = time.monotonic()+7200
        while any(child.poll() is None for child in children):
            if time.monotonic() > deadline or any(child.poll() not in (None, 0) for child in children):
                raise RuntimeError('Inference worker failed or exceeded two hours; inspect worker logs')
            time.sleep(3)
        codes = [child.returncode for child in children]
        write(output/'workers.json', dict(exit_codes=codes))
        if any(codes):
            raise RuntimeError(f'Worker exits: {codes}')
        write(output/'status.json', dict(status='scoring', exit_codes=codes))
        finalize(output)
        write(output/'status.json', dict(status='completed', exit_code=0))
    except BaseException as error:
        for child in children:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=30)
        write(output/'status.json', dict(status='failed', error=str(error)))
        raise
    finally:
        for log in logs:
            log.close()


if __name__ == '__main__':
    main()
