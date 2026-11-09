#!/usr/bin/env python3
"""Audit and publish current guarded640 independent mask-refinement removals."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.evaluate_course_current_removal import VARIANTS, check_dependencies, check_outputs
from tools.evaluate_course_overlap import read, digest, require, now
from tools.summarize_course_swinl_removal import check_sources, record_digest
from tools.summarize_course_overlap import check_result

METRICS=('mask_ap_small','mask_ar_small','mask_ap','mask_ap50','mask_ap75','mask_ap_medium','mask_ap_large')
LABELS={'full':'当前完整 DPSeg（guarded640）','without_point':'w/o 深度感知点细化',
        'without_probability':'w/o 深度引导概率图细化','without_tiles':'w/o 切片增强推理（已有结果）'}


def audit(run):
    from rgbd_seg.course_archive import CourseArchive
    plan,result,state=(read(run/name) for name in ('protocol.json','results.json','status.json'))
    protocol_sha=digest(run/'protocol.json')
    require(state['status']=='completed' and state['exit_code']==0 and result['status']=='completed'
            and result['protocol_sha256']==protocol_sha and result['split']=='test' and result['images']==2112,
            'Experiment has not completed successfully')
    check_dependencies(plan)
    require(not plan['training'] and not result['training'] and not plan['test_used_for_selection']
            and not result['test_used_for_selection'] and not plan['ground_truth_in_prediction']
            and not plan['default_changed'] and not result['default_changed'] and not plan['speed_benchmark'],
            'Experiment scope changed')
    require(plan['input_keys']==['image','depth_m','depth_valid'] and not plan['camera_intrinsics_used']
            and plan['base_epoch']==15 and plan['head_epoch']==6 and plan['precision']=='float32'
            and plan['tf32'] is False and plan['score_threshold']==.01 and plan['max_detections']==100,
            'Fixed model/input parameters changed')
    for item in [*plan['weights'].values(),plan['system_protocol']]:
        require(digest(item['path'])==item['sha256'],'Checkpoint/system protocol changed')
    system=read(plan['system_protocol']['path'])
    check_sources(system['system_source_sha256'])
    require(system['model']['backbone']=='large' and system['model']['depth_encoding']=='depth'
            and system['base_checkpoint_sha256']==plan['weights']['base']['sha256'], 'Model identity differs')
    require(set(result['variants'])==set(VARIANTS) and result['references']==plan['references'],
            'Missing or changed experiment arms')
    for name in ('reference_protocol','reference_summary'):
        item=plan[name]
        require(digest(item['path'])==item['sha256'],f'{name} changed')
    ref_plan=read(plan['reference_protocol']['path'])
    ref_summary=read(plan['reference_summary']['path'])
    require(ref_summary['status']=='verified' and ref_plan['weights']==plan['weights']
            and ref_summary['protocol']['sha256']==plan['reference_protocol']['sha256'],
            'Current full reference provenance differs')
    promoted=next(row for row in ref_summary['results'] if row['variant']=='global_guard_edge_consensus')
    full=plan['references']['full']
    require(promoted['predictions_sha256']==full['predictions_sha256'] and promoted['metrics']==full['metrics'],
            'Full reference is not the adopted current strategy')
    with CourseArchive(plan['archive'],split='test') as data:
        ids=data.image_ids
    require(ids==plan['image_ids'] and len(ids)==len(set(ids))==2112,'Test coverage differs')
    gate=read(run/'engineering_gate.json')
    require(gate['status']=='passed' and gate['split']=='val' and gate['protocol_sha256']==protocol_sha
            and not gate['ground_truth_used'] and not gate['parameter_changes']
            and [row['image_id'] for row in gate['checks']]==plan['engineering_val_ids']
            and len(gate['checks'])==3,'Engineering gate differs')
    require(all(row['full_without_tiles_native_exact'] and row['score_errors']==
                {'full':0.,'without_tiles':0.,'native':0.} for row in gate['checks']), 'Engineering reproduction failed')
    exits={'engineering':{'engineering':0},
           'inferring':{f'worker{rank}':0 for rank in range(len(plan['gpus']))},
           'scoring':{f'score_{name}':0 for name in VARIANTS}}
    for phase,expected in exits.items():
        require(read(run/f'{phase}_exits.json')==expected,f'{phase} did not finish with actual zero exits')
    semantics,full_validation=check_result(full,ids)
    without_tiles=plan['references']['without_tiles']
    _,tile_validation=check_result(without_tiles,ids,semantics)
    native=plan['references']['native']
    for key in ('evaluation','predictions'):
        require(digest(native[key])==native[key+'_sha256'],'Native reproduction reference changed')
    with Path(full['predictions']).open() as stream:
        full_scores={row['image_id']:[(p['category_id'],p['score']) for p in row['predictions']]
                     for row in map(json.loads,stream)}
    require(set(full_scores)==set(ids),'Full reference coverage differs')
    counts=Counter(); digests={name:{} for name in VARIANTS}; receipts=[]; geometry=Counter()
    for rank in range(len(plan['gpus'])):
        receipt=read(run/f'rank{rank}.json'); rank_ids=ids[rank::len(plan['gpus'])]
        require(receipt['status']=='completed' and receipt['image_ids']==rank_ids and receipt['images']==len(rank_ids)
                and receipt['full_without_tiles_native_exact'] and not receipt['ground_truth_in_prediction'],
                'Invalid rank receipt')
        check_path=run/f'checks_rank{rank}.jsonl'
        require(digest(check_path)==receipt['checks_sha256'],'Per-image audit changed')
        with check_path.open() as stream: checks=[json.loads(line) for line in stream]
        require([row['image_id'] for row in checks]==rank_ids and all(row['full_without_tiles_native_exact'] for row in checks),
                'Full/S3/native reproduction coverage incomplete')
        predictions={}
        for name in VARIANTS:
            path=run/f'{name}_rank{rank}.jsonl'
            require(digest(path)==receipt['artifacts'][name],'Prediction shard changed')
            with path.open() as stream: rows=[json.loads(line) for line in stream]
            require([row['image_id'] for row in rows]==rank_ids,'Prediction shard coverage differs')
            predictions[name]=rows
            count=sum(len(row['predictions']) for row in rows)
            require(count==receipt['instances'][name],'Prediction count differs')
            counts[name]+=count
            for row in rows:
                require(row['image_id'] not in digests[name],'Duplicate prediction image')
                digests[name][row['image_id']]=record_digest(row)
        for index,audit_row in enumerate(checks):
            image_id=audit_row['image_id']; values={name:predictions[name][index]['predictions'] for name in VARIANTS}
            check_outputs(values,audit_row['stats'],image_id,plan)
            require(audit_row['instances']=={name:len(values[name]) for name in VARIANTS},'Audit instance count differs')
            require(all([(p['category_id'],p['score']) for p in values[name]]==full_scores[image_id] for name in VARIANTS),
                    'Removal scores/order differ from the saved current full predictions')
            geometry.update(audit_row['stats']['merging'])
        receipts.append(dict(rank=rank,images=len(rank_ids),gpu=receipt['gpu'],
            path=str(run/f'rank{rank}.json'),sha256=digest(run/f'rank{rank}.json'),checks_sha256=receipt['checks_sha256']))
    rows=[dict(variant='full',label=LABELS['full'],reused=True,**{key:value for key,value in full.items() if key!='variant'})]
    validations={'full':full_validation,'without_tiles':tile_validation}
    for name in VARIANTS:
        row=result['variants'][name]
        require(row==read(run/name/'result.json') and row['variant']==name,'Result receipt differs')
        _,validations[name]=check_result(row,ids,semantics)
        require(row['predicted_instances']==counts[name]==full['predicted_instances'],
                'Removal changed selected instance count')
        with Path(row['predictions']).open() as stream:
            observed=[]
            for item in map(json.loads,stream):
                observed.append(item['image_id'])
                require(record_digest(item)==digests[name][item['image_id']], 'Merged prediction differs from frozen shard')
        require(observed==ids,'Merged prediction order differs')
        rows.append(dict(label=LABELS[name],reused=False,**row))
    rows.append(dict(label=LABELS['without_tiles'],reused=True,**without_tiles))
    for row in rows:
        row['delta_vs_full']={key:row['metrics'][key]-full['metrics'][key] for key in METRICS}
    return dict(status='verified',at_utc=now(),split='test',images=2112,results=rows,phase_exits=exits,
        protocol=dict(path=str(run/'protocol.json'),sha256=protocol_sha),
        full_without_tiles_native_exact_images=2112, candidate_scores_order_exact_images=2112,
        engineering_gate=dict(path=str(run/'engineering_gate.json'),sha256=digest(run/'engineering_gate.json')),
        rank_receipts=receipts,csv_validation=validations,merging_totals=dict(geometry),
        current_default_manifest=plan['current_default_manifest'],checkpoints=plan['weights'],
        source_sha256=plan['new_source_sha256'],system_protocol=plan['system_protocol'],
        reference_protocol=plan['reference_protocol'],reference_summary=plan['reference_summary'],
        comparison_scope=plan['comparison_scope'],test_analysis=plan['test_analysis'],
        training=False,default_changed=False,test_used_for_selection=False,
        publisher_sha256=digest(Path(__file__)),
        publisher_dependencies={name:digest(ROOT/name) for name in ('tools/summarize_course_overlap.py',
            'tools/summarize_course_swinl_removal.py')})


def publish(run,artifact,docs):
    columns=['配置','APₛ','ARₛ','AP','AP50','AP75','APₘ','APₗ']
    table='| '+' | '.join(columns)+' |\n| '+' | '.join(['---']+['---:']*7)+' |\n'
    for row in artifact['results']:
        table+='| '+row['label']+' | '+' | '.join(f"{row['metrics'][key]:.6f}" for key in METRICS)+' |\n'
    full=artifact['results'][0]
    csv_lines='\n'.join(f"- {row['label']}：[CSV]({row['csv']})；{row['predicted_instances']:,} 个预测实例。" for row in artifact['results'])
    text=f'''# 当前 DPSeg 的独立掩码细化消融

{table}
本表从当前完整 DPSeg 分别去除一个模块。完整方法使用 Swin-Large、无内参深度输入、640×640 四切片与整图，以及 `global_guard_edge_consensus` 候选合并。基座为第15轮、点头为第6轮固定权重，两项去除均未重新训练。

去除点细化时，选中的粗掩码直接进入深度引导概率图细化；去除概率图细化时，直接输出点细化后的掩码。两项去除与当前完整方法使用完全相同的候选身份、类别、置信度、排序和每图最多100实例限制，仅掩码细化路径不同。共享前向与参考核验用于节省实验计算，本运行不能作为独立方法的速度测量。

完整方法及无切片行复用已有完整测试集结果。所有2112张均逐图重新精确复现当前完整方法、无切片S3与native S1，掩码、分数和次序一致；三张验证图的独立实现工程核验也全部通过。新推理与评测进程实际退出均为0。

## 评测与范围

完整本地 test=2112；使用统一 COCO 分割 AP，IoU=0.50:0.05:0.95，maxDets=[1,10,100]，小目标面积范围为[0,1024]。GT仅用于评分，预测只接收RGB、米制深度和有效性图。每份CSV均完成2112行覆盖、类别、尺寸、压缩RLE完整解码及与评分预测逐项绑定校验。

这是已多次查看过的本地测试集上的固定协议独立消融，用于分析既定配置下的模块影响；不表示在独立验证集上选出的最优方案，也不代表新增Kaggle评分。当前默认方法未改变。

## 提交文件

{csv_lines}

## 可复核来源

- [本轮协议]({artifact['protocol']['path']})，SHA256：`{artifact['protocol']['sha256']}`。
- [工程核验]({artifact['engineering_gate']['path']})。
- [当前完整版来源协议]({artifact['reference_protocol']['path']})。
- 本轮输出目录：`{run}`。
'''
    targets=[run]
    if docs is not None:
        docs.mkdir(parents=True,exist_ok=True); targets.append(docs)
    for folder in targets:
        stem='summary' if folder==run else 'course_current_removal_swinl'
        (folder/f'{stem}.md').write_text(text)
        (folder/f'{stem}.json').write_text(json.dumps(artifact,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        with (folder/f'{stem}.csv').open('w',encoding='utf-8-sig',newline='') as stream:
            writer=csv.writer(stream,lineterminator='\n')
            writer.writerow(['variant','label',*METRICS,'predicted_instances','reused','submission_csv'])
            for row in artifact['results']:
                writer.writerow([row['variant'],row['label'],*(row['metrics'][key] for key in METRICS),
                    row['predicted_instances'],row['reused'],row['csv']])
    print(table)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--docs',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    publish(run,audit(run),args.docs.resolve() if args.docs else None)


if __name__=='__main__': main()
