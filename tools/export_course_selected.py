#!/usr/bin/env python3
"""Export the validation-selected model on seven GPUs, without test scoring."""
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
from rgbd_seg.course_depth_data import load_inference_sample
from rgbd_seg.course_depth_refinement import refine_predictions
from rgbd_seg.course_optimized import load_for_inference
from rgbd_seg.course_distributed_eval import merge_shards, write_json
from rgbd_seg.course_finetune_eval import validate_csv_fast
from rgbd_seg.course_submission import SubmissionWriter
from tools.check_course_prediction_reload import compare_predictions
from tools.select_course_resolution import read, sha, require


def checked_protocol(path):
    p = read(path)
    require(p['status'] == 'frozen' and p['gpus'] == [0,1,2,3,5,6,7], 'Unfrozen export')
    for name, digest in p['source_sha256'].items():
        require(sha(ROOT/name) == digest, f'Export source changed: {name}')
    binding = p['decision']
    require(sha(binding['path']) == binding['sha256'], 'Decision changed')
    d = read(binding['path'])
    require(d['status'] == 'completed' and not d['test_used'] and d['resolution'] == 1024
            and d['selected_method'] in ('native','spatial','depth'), 'Selection incomplete')
    require(sha(d['protocol']['path']) == d['protocol']['sha256']
            and sha(d['run']['path']) == d['run']['sha256'], 'Selection evidence changed')
    require(sha(d['baseline']['best_checkpoint']) == d['baseline']['best_checkpoint_sha256'], 'Weights changed')
    require(sha(d['baseline_protocol']['path']) == d['baseline_protocol']['sha256'], 'Model protocol changed')
    train = read(d['baseline_protocol']['path'])
    require(p['archive'] == train['archive'], 'Data identity differs')
    for name, digest in train['source_sha256'].items():
        require(sha(ROOT/name) == digest, f'Training source changed: {name}')
    ref = p['validation_reference']
    if d['selected_method'] == 'native':
        expected = read(d['protocol']['path'])['baseline_predictions']
    else:
        expected = d['variants'][d['selected_method']]['predictions']
    require(ref == expected and sha(ref['path']) == ref['sha256'], 'Validation reference changed')
    return p, d


def worker(p, d, output, rank):
    require(os.environ.get('CUDA_VISIBLE_DEVICES') == str(p['gpus'][rank])
            and torch.cuda.device_count() == 1, 'Wrong GPU mapping')
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, reload_receipt = load_for_inference(d['baseline']['best_checkpoint'],
                                             d['baseline_protocol']['path'], device='cuda:0')
    require(reload_receipt['epoch'] == d['baseline']['best_epoch'], 'Wrong selected epoch')
    mode = d['selected_method']
    def predict(sample):
        values = model.predict_sample(sample, score_threshold=.01, max_detections=100)
        if mode != 'native':
            values, _ = refine_predictions(values, sample['depth_m'][0], sample['depth_valid'][0],
                                           mode=mode, device='cuda:0')
        return values
    with torch.inference_mode(), CourseArchive(p['archive'], 'val') as validation:
        ids = validation.image_ids[rank::7][:2]
        references = {}
        with Path(p['validation_reference']['path']).open() as stream:
            for line in stream:
                item = json.loads(line)
                if item['image_id'] in ids:
                    references[item['image_id']] = item['predictions']
        require(set(references) == set(ids), 'Missing reload reference')
        first = load_inference_sample(validation, ids[0])
        for _ in range(3):
            predict(first)
        checks = [{'image_id': image_id,
                   'max_score_error': compare_predictions(references[image_id],
                        predict(load_inference_sample(validation, image_id)))} for image_id in ids]
        write_json(output/f'reload_rank{rank}.json', {'status':'passed','rank':rank,
            'physical_gpu':p['gpus'][rank], 'checks':checks, 'reload':reload_receipt})
    started = time.monotonic()
    instances = 0
    with torch.inference_mode(), CourseArchive(p['archive'], 'test') as dataset:
        require(dataset.image_ids == read(output/'image_ids.json'), 'Test metadata order changed')
        indices = list(range(rank,len(dataset),7))
        with (output/f'predictions_rank{rank}.jsonl').open('x') as stream:
            for position, index in enumerate(indices):
                record = dataset.images[index]
                sample = load_inference_sample(dataset, record['id'])
                require(set(sample) == {'image','depth_m','depth_valid','camera_K'}, 'Unexpected inference inputs')
                torch.cuda.synchronize()
                begin = time.perf_counter()
                values = predict(sample)
                torch.cuda.synchronize()
                stream.write(json.dumps({'image_id':record['id'],'file_name':record['file_name'],
                    'predictions':values,'inference_seconds':time.perf_counter()-begin},allow_nan=False)+'\n')
                instances += len(values)
                if position == 0 or (position+1)%25 == 0 or position+1 == len(indices):
                    stream.flush()
                    write_json(output/f'progress_rank{rank}.json', {'phase':'inferring','images':position+1,
                        'expected':len(indices),'instances':instances,'elapsed_seconds':time.monotonic()-started})
        write_json(output/f'rank{rank}.json', {'status':'completed','rank':rank,'physical_gpu':p['gpus'][rank],
            'images':len(indices),'ordered_ids':[dataset.image_ids[i] for i in indices],
            'instances':instances,'predictions_sha256':sha(output/f'predictions_rank{rank}.jsonl'),
            'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
            'elapsed_seconds':time.monotonic()-started,'test_ground_truth_used':False})


def parent(p, d, path, output):
    require(not output.exists(), 'Export directory must be new')
    output.mkdir(parents=True)
    run = {'status':'running','started_at_utc':datetime.now(timezone.utc).isoformat(),
        'pid':os.getpid(),'selected_method':d['selected_method'],'decision':p['decision'],
        'protocol_sha256':sha(path),'source_sha256':p['source_sha256'],'test_ground_truth_used':False,
        'test_scored':False,'workers':[],'world_size':7}
    write_json(output/'run.json',run)
    children = []
    try:
        with CourseArchive(p['archive'],'test') as dataset:
            ids = dataset.image_ids
            require(len(ids) == 2112 and dataset.submission_rows is not None, 'Incomplete official test template')
            write_json(output/'image_ids.json',ids)
        with ExitStack() as stack:
            for rank, gpu in enumerate(p['gpus']):
                env = dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='4',
                           MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4')
                log = stack.enter_context((output/f'worker{rank}.log').open('x'))
                child = subprocess.Popen([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--protocol',str(path),'--output',str(output),'--rank',str(rank)],
                    env=env,stdout=log,stderr=subprocess.STDOUT)
                children.append(child)
                run['workers'].append({'rank':rank,'physical_gpu':gpu,'pid':child.pid})
            write_json(output/'run.json',run)
            began = time.monotonic()
            while any(x.poll() is None for x in children):
                require(all(x.poll() in (None,0) for x in children),'Export worker failed; inspect logs')
                require(time.monotonic()-began < p['worker_timeout_seconds'],'Export worker deadline exceeded')
                time.sleep(1)
            require(all(x.returncode == 0 for x in children),'Export worker failed')
        for rank in range(7):
            receipt = read(output/f'rank{rank}.json')
            require(receipt['status'] == 'completed' and receipt['ordered_ids'] == ids[rank::7]
                    and receipt['predictions_sha256'] == sha(output/f'predictions_rank{rank}.jsonl'),
                    'Test shard incomplete or changed')
            require(read(output/f'reload_rank{rank}.json')['status'] == 'passed','Reload check missing')
        count = merge_shards([output/f'predictions_rank{i}.jsonl' for i in range(7)],ids,output/'predictions.jsonl')
        write_json(output/'progress.json',{'phase':'writing_and_validating_csv','images':len(ids)})
        with SubmissionWriter(output/'submission.csv',ids) as writer, (output/'predictions.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                writer.write(record['image_id'],record['predictions'])
        validation = validate_csv_fast(output/'submission.csv',ids)
        require(validation['rows'] == 2112 and validation['predictions'] == count,'CSV count differs')
        write_json(output/'submission_validation.json',validation)
        run.update(status='completed',exit_code=0,images=len(ids),predicted_instances=count,
            predictions_sha256=sha(output/'predictions.jsonl'),submission_sha256=sha(output/'submission.csv'),
            submission_validation_sha256=sha(output/'submission_validation.json'),
            finished_at_utc=datetime.now(timezone.utc).isoformat())
    except BaseException as error:
        run.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error))
        for child in children:
            if child.poll() is None:
                child.terminate()
        for child in children:
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()
        raise
    finally:
        write_json(output/'run.json',run)
        write_json(output/'exit.json',{'status':run['status'],'exit_code':run.get('exit_code',1),
                                     'worker_exit_codes':[x.returncode for x in children]})
    print(json.dumps({k:run[k] for k in ['status','selected_method','images','predicted_instances','submission_sha256']}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--rank',type=int)
    args = parser.parse_args()
    p,d = checked_protocol(args.protocol)
    if args.rank is None:
        parent(p,d,args.protocol.resolve(),args.output.resolve())
    else:
        require(0 <= args.rank < 7,'Invalid rank')
        worker(p,d,args.output.resolve(),args.rank)


if __name__ == '__main__':
    main()
