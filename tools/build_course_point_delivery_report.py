#!/usr/bin/env python3
"""Build the final point-head report from audited, complete experiment results."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha

LABELS={
    'native':'DA-Fusion 1024 原始预测',
    'native_probability_depth':'已有概率深度细化',
    'point':'普通点头',
    'point_probability_depth':'普通点头＋概率深度细化',
    'point_depth':'深度点头',
    'point_depth_probability_depth':'深度点头＋概率深度细化',
    'point_guard':'深度点头＋概率深度细化＋小掩码保护',
}


def read(path):return json.loads(Path(path).read_text())


def build(output):
    root=ROOT/'reports/course_point_refinement'
    main_root=root/'formal_v1/experiment';guard_root=root/'guard_v1'
    core_audit=read(root/'formal_v1/delivery_audit.json');guard_audit=read(guard_root/'delivery_audit.json')
    if core_audit['status']!='passed' or not core_audit['full_experiment_verified'] or guard_audit['status']!='passed':
        raise ValueError('Both complete experiment audits are required')
    main=read(main_root/'comparison.json');guard=read(guard_root/'test/receipt.json')
    selection=read(guard_root/'selection.json');guard_val=read(selection['validation_receipt']['path'])
    examples_root=root/'formal_v1/examples';examples=read(examples_root/'examples_receipt.json')
    historical_path=ROOT/'reports/course_all_test_comparison/comparison.json';historical=read(historical_path)
    if main['status']!='completed' or main['test']['images']!=2112 or guard['images']!=2112:
        raise ValueError('Complete test data missing')
    if examples['status']!='completed' or examples['selected_checkpoint']!=main['selection']['selected']['checkpoint']:
        raise ValueError('Example checkpoint provenance differs')
    bindings={}
    sources=[main_root/'comparison.json',root/'formal_v1/delivery_audit.json',guard_root/'delivery_audit.json',
             guard_root/'test/receipt.json',guard_root/'selection.json',Path(selection['validation_receipt']['path']),
             examples_root/'examples_receipt.json',historical_path,root/'protocol_full_v1.json',guard_root/'protocol.json',
             root/'formal_v1/independent_reload_diagnostic.json',root/'formal_v1/supervisor.json',guard_root/'supervisor.json']
    for path in sources:bindings[str(path.resolve())]=file_sha(path)
    for name,digest in examples['images_sha256'].items():
        path=examples_root/name
        if file_sha(path)!=digest:raise ValueError('Rendered example changed')
        bindings[str(path.resolve())]=digest
    rows=[];submissions=[]
    for key,label in LABELS.items():
        if key=='point_guard':
            metrics=guard['metrics'];evaluation=guard_root/'test/evaluation.json'
            ms=guard['pipeline_mean_ms_attributed'];csv=guard['submission']
        else:
            metrics=main['test']['metrics'][key];evaluation=main_root/'test'/key/'evaluation.json'
            ms=main['test']['timing_mean_ms'][key];csv=None
            if key.startswith('point'):
                receipt=read(evaluation.parent/'submission_validation.json')
                csv={name:receipt[name] for name in ('path','sha256','rows')}
        if not 0<=metrics['mask_ap']<=1 or not 0<=metrics['mask_ap_small']<=1:
            raise ValueError('Report AP scale invalid')
        row={'configuration':key,'method':label,'test_mask_AP':metrics['mask_ap'],
             'test_mask_APs':metrics['mask_ap_small'],'images':2112,'mean_inference_ms_attributed':ms,
             'evaluation_path':str(evaluation.resolve()),'evaluation_sha256':file_sha(evaluation),
             'submission_csv':None if csv is None else csv['path']}
        bindings[str(evaluation.resolve())]=row['evaluation_sha256']
        if csv is not None:
            if csv['rows']!=2112 or file_sha(csv['path'])!=csv['sha256']:
                raise ValueError('Report submission byte/count binding differs')
            submissions.append({'configuration':key,**csv});bindings[csv['path']]=csv['sha256']
        rows.append(row)
    if len(submissions)!=5:raise ValueError('Expected five point-head submission files')
    history=[]
    for row in historical['rows']:
        if row['images']!=2112 or file_sha(row['evaluation_path'])!=row['evaluation_sha256']:
            raise ValueError('Historical test comparison binding differs')
        bindings[row['evaluation_path']]=row['evaluation_sha256'];history.append(dict(row))
    for row in rows:
        if row['configuration'].startswith('point'):
            item=dict(row);item['configuration']='point_trial_'+row['configuration']+'_head_e02'
            history.append(item)
    if len(history)!=23 or len({row['configuration'] for row in history})!=23:
        raise ValueError('Complete course comparison should contain 23 unique configurations')
    best=max(history,key=lambda row:row['test_mask_APs'])
    baseline=next(row for row in rows if row['configuration']=='native_probability_depth')
    best_new=max((row for row in rows if row['configuration'].startswith('point')),key=lambda row:row['test_mask_APs'])
    trained=main['selection']['selected']['checkpoint']
    if file_sha(trained['path'])!=trained['sha256']:raise ValueError('Final head checkpoint changed')
    bindings[trained['path']]=trained['sha256']
    original_protocol=read(root/'protocol_full_v1.json')
    base=original_protocol['weights'];bindings[base['path']]=base['sha256']
    if file_sha(base['path'])!=base['sha256']:raise ValueError('Frozen base checkpoint changed')
    diagnostic=read(root/'formal_v1/independent_reload_diagnostic.json')
    result={'status':'completed','objective':'尝试一下点细化头','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','mode':'run','verification_status':'VERIFIED_COMPLETE_EXPERIMENT_AND_GUARD'},
        'dataset':{'train_images':14432,'validation_images':2584,'test_images':2112,'classes':['component'],'image_size':[1024,1024]},
        'head_training':{'epochs':2,'gpus':[3,5,6,7],'global_batch_size':8,'seed':17,'base_frozen':True,
            'plain_parameters':222978,'depth_parameters':225282,'selected_checkpoint':trained},
        'point_selection':main['selection'],'guard_selection':selection,
        'validation_at_selected_epoch':main['validation_history'][1]['metrics'],
        'guard_validation_metrics':guard_val['metrics'],'point_test_rows':rows,'submissions':submissions,
        'best_new_test_configuration':best_new['configuration'],
        'test_APs_gain_over_existing_probability_depth':best_new['test_mask_APs']-baseline['test_mask_APs'],
        'test_AP_gain_over_existing_probability_depth':best_new['test_mask_AP']-baseline['test_mask_AP'],
        'all_course_test_configurations':history,'highest_observed_course_test_APs':best,
        'independent_reload':{'images':len(diagnostic['checks']),'masks_exact':diagnostic['validation_predictions_reloaded_exact'],
            'geometry_effect_observed':diagnostic['geometry_effect_observed']},
        'qualitative_examples':{'preselected_images':len(examples['preselected_validation_ids']),'eligible_cases':len(examples['cases']),
                               'receipt':str((examples_root/'examples_receipt.json').resolve())},
        'input_sha256':bindings,
        'limits':['One seed and two head-training epochs; no statistical significance claim',
                  'PointRend-inspired residual point head, not an exact reproduction of its Mask R-CNN configuration',
                  'Plain head also receives RGB-D fused features; explicit metric cues are the extra intervention',
                  'New point heads use extra training; the original base is frozen',
                  'Whole-image point trial; point heads were not combined with historical tile inference',
                  'Local complete-test COCO scores; no new Kaggle public scores in this experiment',
                  'Mixed A800/A100 stage-attributed latency excludes ZIP I/O; guard includes an extra baseline probability-filter branch']}
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'report.json',result)
    gain=result['test_APs_gain_over_existing_probability_depth']
    lines=['# 点细化头实验：完整测试结果与提交文件','','## 结论','',
        f"本轮新增方案中，小目标 APₛ 最高的是 **{best_new['method']}**：AP={best_new['test_mask_AP']:.5f}、APₛ={best_new['test_mask_APs']:.5f}。",
        f"相对已有整图概率深度方案（APₛ={baseline['test_mask_APs']:.5f}），APₛ 增加 **{gain:.5f}**，整体 AP 增加 {result['test_AP_gain_over_existing_probability_depth']:.5f}。普通点头在本轮小目标指标上下降。",'',
        f"此前 **{best['method']}** 的完整测试集 APₛ={best['test_mask_APs']:.5f}，仍高于本轮整图点头。点头与切片的组合需要另做实验。",'',
        '以下主表使用本地 **2112 张完整测试图**，分数范围 0–1；Kaggle 公榜约使用测试集的 30%，这里不把本地分数标成官方公榜分数。', '',
        '## Material Passport','', '- 来源：academic-research-suite / experiment-agent，run。',
        '- 状态：训练、两轮完整验证、完整测试、独立重载、五份 CSV 原生解码与一致性核验均已完成。',
        '- 权重轮次依据验证集选择。测试标注来自实际下载包，仅用于最后独立评分。', '',
        '## 完整测试集对比与 CSV','', '| 配置 | Mask AP | Mask APₛ | 平均推理秒 | 提交文件 |',
        '|---|---:|---:|---:|---|']
    for row in rows:
        link=f"[CSV]({row['submission_csv']})" if row['submission_csv'] else '已有基线'
        lines.append(f"| {row['method']} | {row['test_mask_AP']:.5f} | {row['test_mask_APs']:.5f} | {row['mean_inference_ms_attributed']/1000:.3f} | {link} |")
    lines+=['', '每份 CSV 均包含 2112 行，列为 image_id、PredictionString；每图最多 100 个 component 实例，掩码为 1024×1024 COCO 压缩 RLE。',
        '计时为 batch=1 的预处理、模型、后处理及 CPU RLE 编码，排除 ZIP 读取；设备混合 A800/A100。各方案共享前向计算，表中为分段归因；保护版包含原概率深度分支及实测 CPU 规则时间。', '',
        '## 验证选择与训练设置','',
        '- 固定 DA-Fusion Swin-Tiny、1024 输入、原训练第 5 轮基座；仅训练两个点头。',
        '- 两轮各覆盖 14,432 张训练图，四卡有效 batch=8，seed=17。两个点头共享特征、匹配和采样点。',
        '- 普通点头 222,978 参数；深度点头 225,282 参数。普通点头也读取 RGB-D 融合特征。',
        '- 在每轮完整 2584 张验证图上评分，按预设规则选择第 2 轮，再固定该权重进行测试。',
        '- 采纳门槛：相对原概率深度版，验证 APₛ 增加至少 0.005、整体 AP 下降不超过 0.005。', '',
        '| 第2轮配置 | 验证 AP | 验证 APₛ |','|---|---:|---:|']
    for key,value in result['validation_at_selected_epoch'].items():
        lines.append(f"| {LABELS[key]} | {value['mask_ap']:.5f} | {value['mask_ap_small']:.5f} |")
    lines.append(f"| {LABELS['point_guard']} | {guard_val['metrics']['mask_ap']:.5f} | {guard_val['metrics']['mask_ap_small']:.5f} |")
    lines+=['', '## 方法与诊断','',
        '点头使用 3×256 的 MLP，将 stride4 融合骨干特征、像素解码特征和该实例的粗掩码 logit 组合，在不确定位置重新预测。输出采用残差 logit；零初始化时保持原模型预测。该实现借鉴 [PointRend](https://arxiv.org/abs/1912.08193) 的细粒度特征、逐点 MLP 与细分推理思路，是课程适配。',
        '深度点头额外输入中心有效性、四邻域有符号深度差及邻居有效性，共 9 维。每实例训练 784 点，推理在 512、1024 两级各更新最多 784 点。类别、原置信度和预测数量保持不变。',
        '第一轮配对诊断观察到普通点头扩张并损伤部分小目标；深度点头组合改善了更多中等目标。随后固定一组保护规则：原概率深度预测面积≤1024，且新旧预测 mask IoU<0.8 时，回退到原概率深度掩码。规则只读取预测。阈值在观察保护版评分与任何点头测试评分前固定，未做阈值网格搜索。',
        f"测试集保护规则回退了 {guard['guard_statistics']['fallback_instances']:,} / {guard['guard_statistics']['instances']:,} 个预测。保护版相对深度点头组合的额外测试 APₛ 增益为 {guard['metrics']['mask_ap_small']-main['test']['metrics']['point_depth_probability_depth']['mask_ap_small']:.5f}。", '',
        f"[第一轮几何诊断]({root/'formal_v1/epoch1_geometry_diagnosis/diagnosis.md'})使用固定基线匹配，只解释同一掩码的几何变化，不替代正式 COCO AP 匹配。", '',
        '## 边界示例','',
        f"从预先固定的 {len(examples['preselected_validation_ids'])} 张验证图中，{len(examples['cases'])} 张符合既定小目标展示规则。图中固定同一预测实例索引，红色为 GT，绿色为预测。", '',
        f"![点头边界对比]({examples_root/'point_boundary_examples.png'})", '',
        f"![保护版边界对比]({examples_root/'guard_boundary_examples.png'})", '',
        '此例原概率深度 mask IoU 已高于 0.95，边界继续改善也不会让它新增通过某个课程 AP 阈值。整体收益以完整测试集评分为准。', '',
        '## 交付核验','',
        '- 9 项 CPU 检查、真实数据前后向检查、四卡工程检查通过。',
        '- 两轮训练均无图像遗漏或重复；12 个完整验证评分和 6 个原始完整测试评分正常结束。',
        '- 四卡最终点头权重一致，基座前后权重摘要一致，基座无梯度。',
        '- 独立重载在 4 张固定验证图上准确复现保存的预测；局部深度差消融确实改变点头输出。',
        '- 五份 CSV 逐实例原生解码通过，并与对应评分 JSONL 的每个掩码、置信度、图像 ID 完全一致。',
        '- 固定保护规则在所选完整验证集和完整测试集上逐张重放，结果完全一致。', '',
        f"[原实验审计]({root/'formal_v1/delivery_audit.json'}) · [保护版审计]({guard_root/'delivery_audit.json'}) · [机器可读完整结果]({output/'report.json'})", '',
        '## 范围与后续方向','',
        '这是一个 seed、两轮新增点头训练的结果。点头主要细化已发现实例，无法补出没有生成查询的漏检。普通点头与深度点头的受控对照支持这次显式深度线索的作用；本轮没有比较原模型额外联合训练两轮，也没有验证点头与切片的组合。',
        '本轮点头是整图推理。现有切片方案的小目标指标更高，后续可以固定相同切片与合并规则，检查深度点头的增益能否叠加。', '',
        '## 复现与代码','',
        f"- [实验说明]({ROOT/'docs/course_point_refinement.md'})", 
        f"- [点头实现]({ROOT/'rgbd_seg/course_point_refinement.py'})、[保护规则]({ROOT/'rgbd_seg/course_point_guard.py'})", 
        f"- [原实验配置]({root/'protocol_full_v1.json'})、[保护配置]({guard_root/'protocol.json'})", 
        f"- [选定点头权重]({trained['path']})、[冻结基座权重]({base['path']})", '',
        '## 附录：同一课程测试集的全部23组已完成配置','',
        '该表是描述性汇总。历史单卡、多卡的 batch 和优化步数不同，不能把全部差异归因于一个模块；本轮受控点头对照见主表。', '',
        '| 配置 | Mask AP | Mask APₛ | 评分记录 |','|---|---:|---:|---|']
    for row in history:
        lines.append(f"| {row['method']} | {row['test_mask_AP']:.5f} | {row['test_mask_APs']:.5f} | [评测]({row['evaluation_path']}) |")
    (output/'report.md').write_text('\n'.join(lines)+'\n')
    manifest={'status':'passed','objective':result['objective'],'completed_at_utc':datetime.now(timezone.utc).isoformat(),
        'point_test_configurations':7,'all_course_test_configurations':23,'test_images':2112,'validated_submissions':submissions,
        'source_audits_passed':True,'input_sha256':bindings,
        'delivery_sha256':{str(path.resolve()):file_sha(path) for path in (output/'report.json',output/'report.md')},
        'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_sha(__file__)}}
    write_json(output/'delivery_manifest.json',manifest)
    print(json.dumps({'status':'completed','test_configurations':7,'historical_and_new':23,'csv_files':5,'best_new_APs':best_new['test_mask_APs'],'report':str(output/'report.md')}))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();build(args.output.resolve())


if __name__=='__main__':main()
