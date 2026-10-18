#!/usr/bin/env python3
"""Evaluate trained continuations with frozen tiling; lock selection before test."""
from __future__ import annotations
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import json
import math
import os
import signal
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg.course_depth_data import load_inference_sample
from rgbd_seg.course_optimized import file_sha, load_for_inference, tensor_digest
from rgbd_seg.course_point_refinement import PairedPointHeads
from rgbd_seg.course_point_refinement import predict_paired
from rgbd_seg.course_tiled_point import ARMS as LEGACY_ARMS, predict
from rgbd_seg.course_tiled_probability import predict_variants, refined_public, TILE_POLICY, PROBABILITY_POLICY
from rgbd_seg.course_point_guard import POLICY as GUARD_POLICY
from rgbd_seg.course_distributed_eval import merge_shards, write_json
from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
from rgbd_seg.course_submission import SubmissionWriter
from tools.run_course_point_refinement import checked, load_heads, VARIANTS as WHOLE_ARMS
from tools.check_course_prediction_reload import compare_predictions
from tools.audit_course_point_refinement import compare_csv_predictions

ARMS = ('tiled_probability_depth', 'tiled_point_probability_depth',
        'tiled_point_probability_depth_guard')
ALL_ARMS = (*WHOLE_ARMS,*ARMS)


def read(path):
    return json.loads(Path(path).read_text())


def verify(p):
    for name, digest in p['source_sha256'].items():
        if file_sha(ROOT/name) != digest:
            raise ValueError(f'Source changed: {name}')
    if p['status'] != 'frozen' or p['tile_policy'] != TILE_POLICY or p['probability_policy'] != PROBABILITY_POLICY or p['guard_policy'] != GUARD_POLICY:
        raise ValueError('Inference policies differ from the frozen experiment')
    for name in ('head', 'point_protocol', 'historical_comparison', 'point_validation'):
        if file_sha(p[name]['path']) != p[name]['sha256']:
            raise ValueError(f'Input changed: {name}')


def worker(p, folder, split, rank, smoke):
    if os.environ['CUDA_VISIBLE_DEVICES'] != str(p['gpus'][rank]) or torch.cuda.device_count() != 1:
        raise ValueError('GPU mapping differs')
    torch.set_num_threads(4)
    torch.manual_seed(17)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    pp = checked(p['point_protocol']['path'])
    adapter, receipt = load_for_inference(pp['weights']['path'], pp['training_protocol']['path'], device='cuda:0')
    before = tensor_digest(adapter.state_dict())
    heads = PairedPointHeads().to('cuda:0').eval().requires_grad_(False)
    loaded = load_heads(p['head']['path'], heads, pp)
    if loaded['epoch'] != p['head']['epoch'] or p['head_kind'] not in ('plain','depth'):
        raise ValueError('Wrong head epoch/kind')
    selected_head = heads.plain if p['head_kind']=='plain' else heads.depth
    ids = read(folder/'image_ids.json')[rank::len(p['gpus'])]
    began, diagnostics = time.monotonic(), []
    totals = dict.fromkeys(ALL_ARMS, 0)
    max_score_error = 0.
    fallback = 0
    with CourseArchive(pp['archive'], split) as data, ExitStack() as stack, torch.inference_mode():
        streams = {arm: stack.enter_context((folder/arm/f'predictions_rank{rank}.jsonl').open('x')) for arm in ALL_ARMS}
        for position, image_id in enumerate(ids):
            sample = load_inference_sample(data, image_id)
            legacy, diag = predict(adapter, selected_head, sample, zero_check=smoke)
            outputs = {name:legacy[old] for name,old in zip(ARMS,LEGACY_ARMS)}
            paired=predict_paired(adapter,heads,sample)
            for key,values in paired.items():
                outputs[key]=[c.public() for c in values]
                suffix='native_probability_depth' if key=='native' else key+'_probability_depth'
                outputs[suffix],_=refined_public(values,sample,'probability',adapter.device)
            whole_ids=[(v['category_id'],v['score']) for v in outputs['native']]
            if any([(v['category_id'],v['score']) for v in outputs[a]]!=whole_ids for a in WHOLE_ARMS):
                raise ValueError('Whole-image refinement changed instance score/count/order')
            if position==0 or smoke:
                independent, _ = predict_variants(adapter, sample)
                error = compare_predictions(independent['tiled_probability_depth'], outputs[ARMS[0]])
                max_score_error = max(max_score_error, error)
            fallback += diag['guard']['fallback_instances']
            if position == 0 or smoke:
                diagnostics.append(dict(image_id=image_id, **diag))
            if set(outputs)!=set(ALL_ARMS):
                raise ValueError('Missing whole-image/tiled comparison arm')
            for arm, values in outputs.items():
                totals[arm] += len(values)
                streams[arm].write(json.dumps({'image_id': image_id, 'predictions': values}, allow_nan=False)+'\n')
            if position == 0 or (position+1)%10 == 0 or position+1 == len(ids):
                for stream in streams.values(): stream.flush()
                write_json(folder/f'progress_rank{rank}.json', dict(status='inferring', split=split,
                    rank=rank, images=position+1, expected=len(ids), elapsed_seconds=time.monotonic()-began))
    if tensor_digest(adapter.state_dict()) != before:
        raise ValueError('Frozen base changed')
    write_json(folder/f'rank{rank}.json', dict(status='completed', ordered_ids=ids, diagnostics=diagnostics,
        images=len(ids), instances=totals, max_reference_score_error=max_score_error,
        independent_baseline_checks=len(ids) if smoke else 1, elapsed_seconds=time.monotonic()-began,
        guard_fallback_instances=fallback, reload=receipt,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        prediction_sha256={a:file_sha(folder/a/f'predictions_rank{rank}.jsonl') for a in ALL_ARMS}))


def infer(p, folder, split, *, protocol_path, smoke=False):
    folder.mkdir()
    for arm in ALL_ARMS: (folder/arm).mkdir()
    with CourseArchive(p['archive'], split) as data:
        ids = data.image_ids
    if smoke: ids = [ids[i] for i in (0,97,251,501,777,1024,1450,2583)]
    write_json(folder/'image_ids.json', ids)
    processes = []
    with ExitStack() as stack:
        for rank, gpu in enumerate(p['gpus']):
            log = stack.enter_context((folder/f'worker{rank}.log').open('x'))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--protocol', str(protocol_path), '--worker', '--split', split,
                   '--folder', str(folder), '--rank', str(rank)]
            if smoke: cmd.append('--smoke')
            processes.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True))
        write_json(folder/'workers.json', [{'rank':i,'pid':v.pid,'gpu':p['gpus'][i]} for i,v in enumerate(processes)])
        began=time.monotonic()
        try:
            while any(v.poll() is None for v in processes):
                failed=[(i,v.returncode) for i,v in enumerate(processes) if v.poll() not in (None,0)]
                write_json(folder/'process_status.json', {'phase':split,'checked_at_utc':datetime.now(timezone.utc).isoformat(),
                    'workers':[{'pid':v.pid,'exit_code':v.poll()} for v in processes]})
                if failed: raise RuntimeError(f'Workers failed: {failed}; inspect {folder}')
                if time.monotonic()-began>p['worker_timeout_seconds']:
                    raise TimeoutError('Frozen worker time limit exceeded')
                time.sleep(5)
            if any(v.returncode!=0 for v in processes): raise RuntimeError('Worker failure')
        finally:
            for proc in processes:
                if proc.poll() is None:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try: proc.wait(timeout=30)
                    except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            write_json(folder/'worker_exits.json', [{'pid':v.pid,'exit_code':v.returncode} for v in processes])
    for arm in ALL_ARMS:
        shards = [folder/arm/f'predictions_rank{i}.jsonl' for i in range(len(p['gpus']))]
        for i, path in enumerate(shards):
            r = read(folder/f'rank{i}.json')
            if r['ordered_ids'] != ids[i::len(shards)] or r['prediction_sha256'][arm] != file_sha(path):
                raise ValueError('Shard coverage/integrity differs')
        merge_shards(shards, ids, folder/arm/'predictions.jsonl')
    if smoke:
        views = {v for i in range(len(p['gpus'])) for d in read(folder/f'rank{i}.json')['diagnostics'] for v in d['zero_check_views']}
        if views != {0,1,2,3,4}: raise ValueError(f'Engineering check missing view: {views}')
        write_json(folder/'engineering.json', dict(status='passed', images=len(ids), zero_views=sorted(views),
            independent_baseline_checks=len(ids), score_count_order_preserved=True))
        return None
    metrics = {}
    with CourseArchive(p['archive'], split) as data:
        for arm in ALL_ARMS:
            result = score_predictions(data, folder/arm/'predictions.jsonl', folder/arm)
            metrics[arm] = result['metrics']
    write_json(folder/'metrics.json', metrics)
    return metrics


def rank_candidates(current, historical):
    rows=[]
    for origin,values in [('longer_training',current),('historical',historical)]:
        for variant,metrics in values.items():
            if any(metrics[k] is None or not math.isfinite(metrics[k]) or not 0<=metrics[k]<=1
                   for k in ('mask_ap_small','mask_ap')):
                raise ValueError('Full validation contains an invalid metric')
            rows.append({'origin':origin,'variant':variant,**metrics})
    return sorted(rows,key=lambda r:(-r['mask_ap_small'],-r['mask_ap'],r['origin']!='historical',r['variant']))


def main(p,protocol_path,output,phase):
    from tools.launch_course_longer_training import check_devices
    check_devices([str(gpu) for gpu in p['gpus']])
    output.mkdir(parents=True,exist_ok=True)
    protocol_sha=file_sha(protocol_path)
    run_path=output/f'{phase}_run.json'
    if run_path.exists():
        raise FileExistsError(run_path)
    record={'status':'running','phase':phase,'pid':os.getpid(),
        'started_at_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':protocol_sha}
    write_json(run_path,record)
    try:
        if phase=='val':
            infer(p,output/'smoke','val',protocol_path=protocol_path,smoke=True)
            val=infer(p,output/'val','val',protocol_path=protocol_path)
            historical=read(p['historical_comparison']['path'])['val']
            candidates=rank_candidates(val,historical)
            selection={'status':'locked','selected':candidates[0],'candidates':candidates,
                'selection_split':'val','selected_before_test':True,'test_used':False,
                'protocol_sha256':protocol_sha,'val_metrics_sha256':file_sha(output/'val/metrics.json'),
                'historical_comparison_sha256':p['historical_comparison']['sha256'],
                'head_kind':p['head_kind'],'head_epoch':p['head']['epoch'],
                'base_epoch':p['base_epoch'],'selected_at_utc':datetime.now(timezone.utc).isoformat()}
            write_json(output/'selection.json',selection)
            print(json.dumps({'phase':'val','selected':selection['selected']},ensure_ascii=False),flush=True)
        else:
            completed_val=read(output/'val_run.json')
            if completed_val['status']!='completed' or completed_val['protocol_sha256']!=protocol_sha:
                raise ValueError('Validation phase must have completed successfully')
            selection=read(output/'selection.json')
            if (selection['status']!='locked' or not selection['selected_before_test'] or selection['test_used']
                    or selection['protocol_sha256']!=protocol_sha
                    or selection['val_metrics_sha256']!=file_sha(output/'val/metrics.json')):
                raise ValueError('Test requires an unchanged complete validation selection')
            val=read(output/'val/metrics.json')
            historical=read(p['historical_comparison']['path'])
            if selection['candidates']!=rank_candidates(val,historical['val']):
                raise ValueError('Validation selection does not match recorded metrics')
            selection_sha=file_sha(output/'selection.json')
            test=infer(p,output/'test','test',protocol_path=protocol_path)
            ids=read(output/'test/image_ids.json')
            if len(ids)!=2112:
                raise ValueError('Test split is incomplete')
            csvs={}
            for arm in ALL_ARMS:
                suffix=arm.replace('tiled_point_',f'tiled_point_{p["head_kind"]}_')
                if 'point' in arm:
                    suffix+=f'_head_e{p["head"]["epoch"]:02d}'
                path=output/'test'/arm/f'rgbd_dafusion_swin_t_1024_{p["base_tag"]}_{suffix}.csv'
                predictions=output/'test'/arm/'predictions.jsonl'
                with SubmissionWriter(path,ids) as writer,predictions.open() as source:
                    for line in source:
                        row=json.loads(line)
                        writer.write(row['image_id'],row['predictions'])
                receipt=validate_csv_fast(path,ids)
                compare_csv_predictions(path,predictions,ids)
                csvs[arm]={'path':str(path.resolve()),'sha256':file_sha(path),
                           'validation':receipt,'exact_scored_predictions':True}
            if file_sha(output/'selection.json')!=selection_sha:
                raise ValueError('Selection changed after opening test')
            chosen=selection['selected']
            selected_csv=(csvs[chosen['variant']] if chosen['origin']=='longer_training'
                          else historical['csvs'][chosen['variant']])
            if file_sha(selected_csv['path'])!=selected_csv['sha256']:
                raise ValueError('Selected submission file changed')
            report={'status':'completed','scope':'full_val_then_locked_full_test','selection':selection,
                'val':val,'test':test,'historical_val':historical['val'],'historical_test':historical['test'],
                'csvs':csvs,'selected_csv':selected_csv,
                'protocol_sha256':protocol_sha,'selection_sha256':selection_sha,
                'test_labels_used_for_selection':False,'kaggle_submitted':False,
                'material_passport':{'origin_skill':'academic-research-suite/experiment-agent',
                    'mode':'run','verification_status':'VERIFIED_FULL_SPLITS'},
                'limitations':['One seed; best among evaluated candidates, not a global optimum',
                    'Point head selected on whole-image validation before fixed tiled comparison',
                    'Local full test differs from Kaggle public subset']}
            write_json(output/'comparison.json',report)
            lines=['# 延长训练后的完整系统比较','',
                '使用完整验证集选择方案，然后评测完整本地测试集。分数范围为 0–1。','',
                '| 来源 | 配置 | 验证 AP | 验证 APs | 本地测试 AP | 本地测试 APs |',
                '|---|---|---:|---:|---:|---:|']
            for origin,v,t in [('历史',historical['val'],historical['test']),('本轮',val,test)]:
                for arm,m in v.items():
                    lines.append(f'| {origin} | {arm} | {m["mask_ap"]:.6f} | {m["mask_ap_small"]:.6f} | {t[arm]["mask_ap"]:.6f} | {t[arm]["mask_ap_small"]:.6f} |')
            lines+=['',f'验证集选择：{chosen["origin"]} / {chosen["variant"]}。',
                f'本轮主模型轮次：{p["base_epoch"]}；点头：{p["head_kind"]}，第 {p["head"]["epoch"]} 轮。',
                '测试集成绩未用于重新选择轮次、方法或阈值。本地测试分数不是 Kaggle 公榜分数。',
                f'最佳验证方案提交文件：[CSV]({selected_csv["path"]})','']
            lines.extend(f'- [{arm}]({value["path"]})' for arm,value in csvs.items())
            (output/'report.md').write_text('\n'.join(lines)+'\n')
        verify(p)
        record.update(status='completed',finished_at_utc=datetime.now(timezone.utc).isoformat())
        write_json(run_path,record)
    except BaseException as error:
        record.update(status='failed',error=f'{type(error).__name__}: {error}')
        write_json(run_path,record)
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--phase',choices=['val','test'])
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--split',choices=['val','test'])
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--rank',type=int)
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    protocol=read(args.protocol)
    verify(protocol)
    if args.worker:
        worker(protocol,args.folder,args.split,args.rank,args.smoke)
    else:
        if args.output is None or args.phase is None:
            parser.error('Parent requires --output and --phase')
        main(protocol,args.protocol.resolve(),args.output.resolve(),args.phase)
