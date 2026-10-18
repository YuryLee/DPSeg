#!/usr/bin/env python3
"""Paired frozen tiled+point experiment. Validate before opening test split."""
from __future__ import annotations
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
from rgbd_seg.course_optimized import file_sha, load_for_inference, tensor_digest
from rgbd_seg.course_point_refinement import PairedPointHeads
from rgbd_seg.course_tiled_point import ARMS, predict
from rgbd_seg.course_distributed_eval import merge_shards, write_json
from rgbd_seg.course_dafusion_eval import score_predictions, validate_csv_fast
from rgbd_seg.course_submission import SubmissionWriter
from tools.run_course_point_refinement import checked, load_heads
from tools.check_course_prediction_reload import compare_predictions
from tools.audit_course_point_refinement import compare_csv_predictions

BASE = ROOT/'reports/course_tiled_point_refinement'


def read(path):
    return json.loads(Path(path).read_text())


def verify(p):
    for name, digest in p['source_sha256'].items():
        if file_sha(ROOT/name) != digest:
            raise ValueError(f'Source changed: {name}')
    for name in ('head', 'point_protocol', 'val_reference', 'test_reference'):
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
    if loaded['epoch'] != 2:
        raise ValueError('Wrong head epoch')
    ids = read(folder/'image_ids.json')[rank::len(p['gpus'])]
    wanted = set(ids)
    references = {}
    with Path(p[f'{split}_reference']['path']).open() as stream:
        for line in stream:
            item = json.loads(line)
            if item['image_id'] in wanted:
                references[item['image_id']] = item['predictions']
    if set(references) != wanted:
        raise ValueError('Reference coverage differs')
    began, diagnostics = time.monotonic(), []
    totals = dict.fromkeys(ARMS, 0)
    max_score_error = 0.
    fallback = 0
    with CourseArchive(pp['archive'], split) as data, ExitStack() as stack, torch.inference_mode():
        streams = {arm: stack.enter_context((folder/arm/f'predictions_rank{rank}.jsonl').open('x')) for arm in ARMS}
        for position, image_id in enumerate(ids):
            outputs, diag = predict(adapter, heads.depth, load_inference_sample(data, image_id), zero_check=smoke)
            error = compare_predictions(references[image_id], outputs[ARMS[0]])
            max_score_error = max(max_score_error, error)
            fallback += diag['guard']['fallback_instances']
            if position == 0 or smoke:
                diagnostics.append(dict(image_id=image_id, **diag))
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
        every_baseline_mask_matched_reference=True, elapsed_seconds=time.monotonic()-began,
        guard_fallback_instances=fallback, reload=receipt,
        peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        prediction_sha256={a:file_sha(folder/a/f'predictions_rank{rank}.jsonl') for a in ARMS}))


def infer(p, folder, split, smoke=False):
    folder.mkdir()
    for arm in ARMS: (folder/arm).mkdir()
    with CourseArchive(p['archive'], split) as data:
        ids = data.image_ids
    if smoke: ids = [ids[i] for i in (0,97,251,501)]
    write_json(folder/'image_ids.json', ids)
    processes = []
    with ExitStack() as stack:
        for rank, gpu in enumerate(p['gpus']):
            log = stack.enter_context((folder/f'worker{rank}.log').open('x'))
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--worker', '--split', split,
                   '--folder', str(folder), '--rank', str(rank)]
            if smoke: cmd.append('--smoke')
            processes.append(subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT))
        write_json(folder/'workers.json', [{'rank':i,'pid':v.pid,'gpu':p['gpus'][i]} for i,v in enumerate(processes)])
        while any(v.poll() is None for v in processes):
            failed = [(i,v.returncode) for i,v in enumerate(processes) if v.poll() not in (None,0)]
            if failed: raise RuntimeError(f'Workers failed: {failed}; inspect {folder}')
            time.sleep(5)
        if any(v.returncode != 0 for v in processes): raise RuntimeError('Worker failure')
    for arm in ARMS:
        shards = [folder/arm/f'predictions_rank{i}.jsonl' for i in range(len(p['gpus']))]
        for i, path in enumerate(shards):
            r = read(folder/f'rank{i}.json')
            if r['ordered_ids'] != ids[i::len(shards)] or r['prediction_sha256'][arm] != file_sha(path):
                raise ValueError('Shard coverage/integrity differs')
        merge_shards(shards, ids, folder/arm/'predictions.jsonl')
    if smoke:
        views = {v for i in range(4) for d in read(folder/f'rank{i}.json')['diagnostics'] for v in d['zero_check_views']}
        if views != {0,1,2,3,4}: raise ValueError(f'Engineering check missing view: {views}')
        write_json(folder/'engineering.json', dict(status='passed', images=4, zero_views=sorted(views),
            all_baseline_masks_match=True, score_count_order_preserved=True))
        return None
    metrics = {}
    with CourseArchive(p['archive'], split) as data:
        for arm in ARMS:
            result = score_predictions(data, folder/arm/'predictions.jsonl', folder/arm)
            metrics[arm] = result['metrics']
    write_json(folder/'metrics.json', metrics)
    return metrics


def main(p):
    formal = BASE/'formal_v1'
    formal.mkdir()
    write_json(formal/'run.json', dict(status='running', pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat(),
        protocol_sha256=file_sha(BASE/'protocol.json')))
    infer(p, formal/'smoke', 'val', smoke=True)
    val = infer(p, formal/'val', 'val')
    # Selection is written before any test inference or scoring; no threshold/epoch search.
    selected = max(ARMS, key=lambda a:(val[a]['mask_ap_small'],val[a]['mask_ap']))
    write_json(formal/'selection.json', dict(selected=selected, split='val', metrics=val,
        selected_before_test=True, head_epoch=2, no_new_training=True))
    test = infer(p, formal/'test', 'test')
    csvs = {}
    ids = read(formal/'test'/'image_ids.json')
    for arm, suffix in zip(ARMS[1:], ('tile640_point_depth_probdepth_head_e02',
                                    'tile640_point_depth_probdepth_guard_s1024_iou080_head_e02')):
        pred = formal/'test'/arm/'predictions.jsonl'
        csv_path = formal/'test'/arm/f'rgbd_dafusion_swin_t_1024_e05_{suffix}.csv'
        with SubmissionWriter(csv_path,ids) as writer, pred.open() as stream:
            for line in stream:
                item = json.loads(line)
                writer.write(item['image_id'],item['predictions'])
        receipt = validate_csv_fast(csv_path,ids)
        compare_csv_predictions(csv_path,pred,ids)
        csvs[arm] = dict(path=str(csv_path),sha256=file_sha(csv_path),validation=receipt,
                         exact_scored_predictions=True)
    verify(p)
    report = dict(status='completed', scope='full_val_then_full_local_test', val=val, test=test,
        selected_on_val=selected, csvs=csvs, test_labels_used='final scoring only; no head/threshold tuning',
        kaggle_submitted=False, protocol_sha256=file_sha(BASE/'protocol.json'),
        material_passport=dict(origin_skill='academic-research-suite/experiment-agent',mode='run',
                               verification_status='VERIFIED_PAIRED_OUTPUTS'))
    write_json(formal/'comparison.json',report)
    lines = ['# 切片＋概率深度细化加入深度点头的对照实验','',
        '完整验证集 2,584 张，固定配置后评测完整本地测试集 2,112 张。分数为 0–1。', '',
        '| 配置 | 验证 AP | 验证 APₛ | 本地测试 AP | 本地测试 APₛ |','|---|---:|---:|---:|---:|']
    labels = ('切片＋概率深度细化','再加深度点头 e02','再加固定小目标回退保护')
    for arm,label in zip(ARMS,labels):
        lines.append(f"| {label} | {val[arm]['mask_ap']:.6f} | {val[arm]['mask_ap_small']:.6f} | {test[arm]['mask_ap']:.6f} | {test[arm]['mask_ap_small']:.6f} |")
    lines += ['',f'验证集选择：`{selected}`。', '',
        '复用已训练的 e02 深度点头。整图＋四个 640 窗口的候选筛选、跨视图去重、前100个实例、置信度均固定；只细化已保留实例的边界，再进行原概率深度滤波。',
        '保护规则固定为原预测面积≤1024且旧/新mask IoU<0.8时恢复原概率深度结果，未搜索阈值。',
        '工程检查覆盖整图和全部四个裁剪视图，零输出点头保持原掩码。每一张正式图的基线掩码均与历史结果核对，三个分支实例数量、顺序、置信度一致。',
        '本地测试分数不等同于 Kaggle 公榜分数，本轮未上传 Kaggle。单个已训练点头的组合对照，未作统计显著性判断。','']
    for arm,v in csvs.items(): lines.append(f"- [{arm} CSV]({v['path']})")
    (formal/'report.md').write_text('\n'.join(lines)+'\n')
    write_json(formal/'run.json',dict(status='completed',pid=os.getpid(),report=str(formal/'report.md')))
    print(json.dumps(dict(status='completed',selected=selected,test=test),ensure_ascii=False),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true')
    parser.add_argument('--split',choices=('val','test'))
    parser.add_argument('--folder',type=Path)
    parser.add_argument('--rank',type=int)
    parser.add_argument('--smoke',action='store_true')
    args = parser.parse_args()
    protocol = read(BASE/'protocol.json')
    verify(protocol)
    if args.worker: worker(protocol,args.folder,args.split,args.rank,args.smoke)
    else: main(protocol)
