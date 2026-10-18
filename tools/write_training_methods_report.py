#!/usr/bin/env python3
"""Write a Chinese v5 report only from the fully audited final 12+28 matrix."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.summarize_training_methods import COCO, EVAL_KEYS, TRAIN_KEYS, VARIANTS, development_gate, describe, require, sha


def read(path):
    return json.loads(Path(path).read_text())


def parse_tests_log(path):
    text=Path(path).read_text()
    matches=list(re.finditer(r'Ran (\d+) tests? in ([\d.]+)s\s*\n\s*OK\s*(?:\n|$)',text))
    require(bool(matches),'Test log lacks a successful unittest summary')
    last=matches[-1]
    require(not re.search(r'FAILED\s*\(',text[last.end():]),'Tests failed after the last successful summary')
    return {'count':int(last.group(1)),'duration_s':float(last.group(2)),'path':str(Path(path).resolve()),'sha256':sha(path)}


def verify_final_summary(summary):
    require(summary.get('status')=='complete_and_verified','Refusing to write completed report from partial/unverified summary')
    require(summary['counts']=={'completed_training':12,'expected_training':12,'completed_evaluations':28,'expected_evaluations':28},'Expected12 completed training and28 evaluations')
    training,rows=summary['training'],summary['evaluations']
    require(len(training)==12 and {(r['seed'],r['variant']) for r in training}==TRAIN_KEYS,'Training matrix coverage differs')
    require(len(rows)==28 and {(r['dataset'],r['seed'],r['variant'],r['split']) for r in rows}==EVAL_KEYS,'Evaluation matrix coverage differs')
    require(summary['development_gate']==development_gate(rows,True),'Recorded development gate differs from recomputed frozen rule')
    require(not summary.get('missing_or_incomplete'),'Summary contains incomplete runs')
    require(sha(summary['protocol']['path'])==summary['protocol']['sha256'],'Protocol changed after summarization')
    for name,digest in summary['source_sha256'].items():
        require(sha(ROOT/name)==digest,f'Source changed after summarization: {name}')
    for row in rows:
        require(sha(row['evaluation_path'])==row['evaluation_sha256'],'Evaluation file changed after summary')
    for run in training:
        require(sha(run['run_path'])==run['run_sha256'],'Training run changed after summary')
        require(Path(run['checkpoint_path']).resolve().is_relative_to((ROOT/'outputs/paper_v5').resolve()),'Report only accepts the final outputs/paper_v5 matrix, not archived attempts')


def msd(stat,digits=2):
    require(stat['n']==3 and stat['missing']==0,'Three available seed values required for report')
    return f"{stat['mean']:.{digits}f} ± {stat['sample_std_ddof1']:.{digits}f}"


def signed_msd(stat):
    require(stat['n']==3 and stat['missing']==0,'Three paired values required')
    return f"{stat['mean']:+.3f} ± {stat['sample_std_ddof1']:.3f}"


def maximum_gib(records,key):
    values=[row[key] for row in records]
    require(all(isinstance(v,(int,float)) and math.isfinite(v) and v>=0 for v in values),'Missing/nonfinite GPU memory evidence')
    return max(values)/2**30


def render_report(summary,tests,attempt,fix,summary_sha,repeat=None):
    """Pure formatting; the CLI validates actual files before calling this."""
    gate=summary['development_gate']; preferred=gate['preferred_optional_variant']
    heading='有方案达到开发集工程门槛' if preferred else '本轮方案均未达到开发集工程门槛'
    conclusion=(f"按预定开发指标排序，**{preferred}** 是当前可选候选。" if preferred else '本轮没有方案达到预先规定的可选门槛。')
    if preferred=='bce2':
        conclusion+='该方案是简单掩码损失加权对照，不能据此认定某篇论文的边界或频率机制有效。'
    strongest=max(gate['choices'],key=lambda v:gate['choices'][v]['mean_val_ap_small'])
    best=gate['choices'][strongest]
    observation=(f"三个改法中，**{strongest}** 的开发APsmall均值最高，为{best['mean_val_ap_small']:.3f}；"
                 f"相对配对基线的平均差为{best['descriptive_delta_pp']['mask_ap_small']['mean']:+.3f}个百分点，"
                 f"{best['positive_small_seeds']}/3个种子为正。该描述不改变预定门槛，也不代表新场景上的稳定优势。")
    lines=['## Material Passport','',
        '- Origin Skill: academic-research-suite / experiment-agent','- Origin Mode: run / validate',
        f"- Origin Date: {datetime.now(timezone.utc).date().isoformat()}",'- Verification Status: ANALYZED',
        '- Version Label: training_methods_v5',
        '- Scope: 固定训练预算的Torchvision/RGB-D工程适配与探索性比较；不宣称完整论文复现。','',
        f'# 论文方法训练对照：{heading}','',
        f"最终 **12/12次训练、28/28组评测** 完整并通过汇总审计。{conclusion}在线默认基线保持不变。候选判断仅使用同seed重训baseline与三种子开发结果，不使用test或T-LESS选择。",'',
        observation,'',
        '## 实际训练了什么','',
        '| 组别 | 实际变化 | 解释范围 |','|---|---|---|',
        '| baseline | 原E2 RGB-D Mask R-CNN，按本轮同卡预算从COCO初值重训。 | 不复用旧轮800步权重作为本轮训练结果。 |',
        '| bce2 | 原mask BCE再加一次；其余损失不变。 | 控制掩码损失权重增加，不是论文边界算式。 |',
        '| bace | 原mask BCE加一次BACE辅助项，scale4，基础与辅助权重均1。 | 只迁移损失，不加入原作者decoder或环境adapter。 |',
        '| freqfusion | FPN三个连接加入完整ALPF、AHPF、偏移采样，共新增314,070参数。 | 保留原侧向/输出卷积与E2深度分支；不是原MMDetection训练复现。 |','',
        'BACE依据 [Pan/Pei原始论文](https://arxiv.org/abs/2504.19643)及核验的[作者BARD-ERA源码](https://github.com/PANpinchi/BARD-ERA/tree/942b5869d0101a39ea53483773e8d90f7252244d)。正式题名与日期见[ACML2025出版记录](https://proceedings.mlr.press/v304/pan26a.html)；最终排版PDF未取得，不声称逐公式核验过该PDF。FreqFusion依据 [Chen等TPAMI2024](https://arxiv.org/abs/2408.12879)及[作者固定commit](https://github.com/Linwei-Chen/FreqFusion/tree/3fb0c70637a3c194fb74294d3ce4681958b26241)的Apache许可子树。版本、读取范围与许可说明分别见[边界方法记录](v5/boundary_method_notes.md)、[FreqFusion审计](v5/freqfusion_implementation_audit.md)。','',
        'BACE实际使用ROIAlign的28×28软目标t与原logits z：T4为4×4最大池化再nearest还原，辅助logit为q=T4(t)+z−T4(z)，用BCEWithLogits(q,t)监督。最大池化非线性，不称线性正交投影；推理仍使用原mask head。FreqFusion使用low5/high3、压缩64通道、偏移groups4及零填充CARAFE，按32通道分块并checkpoint重算；新增参数使用全学习率。','',
        '每组seeds17/23/41、800步、batch2、AdamW，lr0.0002，原RGB参数lr×0.1；初始化、数据、增强与预算按[冻结协议](../docs/training_methods_protocol.md)保持一致。同seed的共享初值摘要、CPU/CUDA初始随机流摘要及800步样本/增强trace一致。方法改变proposal后，内部ROI/RPN采样可能自然分化；匹配初始随机流不代表整个训练轨迹相同。','',
        '## GraspNet：三种子开发与测试成绩','',
        'AP为0–100尺度，下表为三种子均值±样本SD（ddof=1）。SD不是置信区间；三个种子是在同一份数据上重复训练，不是三份独立数据集。','',
        '| 组别 | 开发AP | 开发APsmall | 测试AP | 测试APsmall |','|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        val=summary['graspnet_three_seed_descriptive']['val'][variant]['metrics']; test=summary['graspnet_three_seed_descriptive']['test'][variant]['metrics']
        lines.append(f"| {variant} | {msd(val['mask_ap'])} | {msd(val['mask_ap_small'])} | {msd(test['mask_ap'])} | {msd(test['mask_ap_small'])} |")
    lines+=['','### 预定开发门槛','',
        '相对同seed baseline：APsmall平均增加≥2个百分点，至少2/3种子为正，总体AP平均下降≤1个百分点。多方案通过时依次按开发APsmall均值、总体AP均值降序及开发平均推理时间升序选择。门槛是工程取舍，不是统计显著性。','',
        '| 方案 | 配对ΔAPsmall（百分点） | 配对ΔAP（百分点） | 小目标正增益种子 | 通过 |','|---|---:|---:|---:|---|']
    for variant in VARIANTS[1:]:
        choice=gate['choices'][variant];stats=choice['descriptive_delta_pp']
        lines.append(f"| {variant} | {signed_msd(stats['mask_ap_small'])} | {signed_msd(stats['mask_ap'])} | {choice['positive_small_seeds']}/3 | {'是' if choice['passed'] else '否'} |")
    lines+=['','### BACE 与损失权重对照','',
        '| 划分 | BACE−BCE2 ΔAPsmall（百分点） | BACE−BCE2 ΔAP（百分点） |','|---|---:|---:|']
    for split,label in [('val','开发'),('test','测试')]:
        stats=summary['bace_minus_bce2'][split]['descriptive_delta_pp']
        lines.append(f"| {label} | {signed_msd(stats['mask_ap_small'])} | {signed_msd(stats['mask_ap'])} |")
    bace_gain=summary['bace_minus_bce2']['val']['descriptive_delta_pp']['mask_ap_small']['mean']
    lines+=['',('BACE开发APsmall均值高于BCE2，但差异仅是当前三种子、固定小数据上的描述性证据，不能单独建立边界机制的因果优势。' if bace_gain>0 else 'BACE开发APsmall均值没有高于BCE2，不能把相对baseline的任何收益特指为其边界算式有效。'),'',
        '**数据限制：**训练64帧仅来自2场景；开发只有4个small，GraspNet测试只有1个small。测试APsmall的高数值或大幅变化可能由这一个目标驱动。所有开发/测试集均已反复查看，本轮是探索性复用，没有新的盲测确认集，不作显著性或普遍泛化胜出声明。','',
        '![三种子小目标结果](v5/trial_summary_ap_small.png)','',
        '## T-LESS：独立列出既有压力测试','',
        '仅seed17，100帧、688实例、52个small；相关视角不能当作独立样本。保留重叠GT的独立visible mask，按原图面积计算COCO，不使用会压平重叠GT的UOIS补充指标。输入深度系数为0.0001米，GraspNet为0.001米。这不是未见确认集，也不参与方案选择。','',
        '| 组别 | Mask AP | APsmall | ARsmall | 同卡ms/图 |','|---|---:|---:|---:|---:|']
    for variant in VARIANTS:
        row=next(r for r in summary['evaluations'] if r['dataset']=='tless' and r['variant']==variant)
        m=row['metrics'];lines.append(f"| {variant} | {m['mask_ap']:.3f} | {m['mask_ap_small']:.3f} | {m['mask_ar_small']:.3f} | {row['timing']['mean_ms']:.2f} |")
    lines+=['','## 同卡效率、训练成本和模型大小','',
        f"设备为 **{summary['runtime']['device_name']}**，物理可见卡标识 `{summary['runtime']['cuda_visible_devices']}`，CPU线程{summary['runtime']['threads']}。推理batch1/float32，各设置3次预热，包含内存输入搬运、模型、原图mask二值化及CPU回传，不含读图、RLE、COCO与网页。以下推理时间取GraspNet test三种子均值±样本SD；训练时间取纯训练循环，不含后续权重保存。",'',
        '| 组别 | 800步训练秒数 | 测试ms/图 | 训练峰值已分配GiB | 测试峰值已分配/保留GiB | 纯state_dict权重MB | 参数量 |','|---|---:|---:|---:|---:|---:|---:|']
    for variant in VARIANTS:
        train=[r for r in summary['training'] if r['variant']==variant]
        infer=[r for r in summary['evaluations'] if r['dataset']=='graspnet' and r['split']=='test' and r['variant']==variant]
        times=describe([r['training_duration_s'] for r in train]);latency=describe([r['timing']['mean_ms'] for r in infer])
        parameters={r['parameters'] for r in train};require(len(parameters)==1,'Same variant parameter count differs by seed')
        weights=[r['weight_bytes']/1e6 for r in train]
        weight_text=f'{min(weights):.3f}' if min(weights)==max(weights) else f'{min(weights):.3f}–{max(weights):.3f}'
        lines.append(f"| {variant} | {msd(times)} | {msd(latency)} | {maximum_gib(train,'peak_cuda_allocated_bytes'):.3f} | {maximum_gib(infer,'peak_cuda_allocated_bytes'):.3f}/{maximum_gib(infer,'peak_cuda_reserved_bytes'):.3f} | {weight_text} | {next(iter(parameters)):,} |")
    lines+=['','显存列取对应三次运行的最大峰值，不是均值；已分配量含常驻模型，保留量含缓存，均不包含其他进程。GiB=2³⁰字节，权重MB=10⁶字节；纯state_dict文件大小与含配置的model.pt分开，后者另存JSON/CSV。进程约22GiB上限不等于在课程建议的24GB显卡上验证过速度。FreqFusion计算和反向重算成本应与精度一起判断。','',
        '不同组的total loss不能直接横向排名：BCE2和BACE包含额外项，BACE辅助项本身也可能有非零下界。本报告不以最终total loss大小判断模型优劣；有限loss和梯度仅作为训练完整性检查。','',
        '## 首轮失败与统一重训','',
        f"首轮seed17 baseline/BCE2完成800步后归档且未评分；BACE完成{attempt['formal_failed']['completed_steps']}步，在第{attempt['formal_failed']['failure_step']}次前向因本项目新增的严格目标范围检查中止。诊断定位到ROIAlign的一个float32软目标为 **{fix['observed']['maximum']:.16g}**，比1多一个机器epsilon。问题属于新增范围验证，不是数据标注越界或BACE训练发散。",'',
        f"修复只容许8个dtype机器epsilon的数值误差（float32为{fix['change']['float32_tolerance']:.16g}），目标不clamp，损失公式、权重和scale不变。全部首轮日志、源码和权重保存在[attempt1记录](v5/attempt1/disposition.json)，修复收据见[boundary_roundoff_fix.json](v5/boundary_roundoff_fix.json)。修复后重新冻结协议，最终12组全部从头统一重训；本报告仅使用`outputs/paper_v5`与对应28组新评测，不混入归档模型、旧轮分数或工程冒烟输出。",'',
        '## 核验与可复现材料','',
        f"- 测试日志实际记录 **{tests['count']}项通过**（{tests['duration_s']:.3f}秒），见[测试记录](v5/{Path(tests['path']).name})。测试覆盖软件行为，不等同于模型性能证据。",
        '- 汇总检查每次800步有限loss/梯度、最终checkpoint与文件摘要、原始数据指纹、协议与源码、同seed采样增强trace和共享初始摘要；预测ID、顺序、原图尺寸及RLE尺寸逐项核对。',
        '- 原图COCO十二项、逐种子结果、BACE−BCE2配对差与效率原值完整保留于[summary JSON](v5/trial_summary.json)和[CSV](v5/trial_summary.csv)。不只展示胜出指标。',
        f"- 本报告输入summary SHA256：`{summary_sha}`；冻结协议SHA256：`{summary['protocol']['sha256']}`。",
        '- 初始化摘要和随机流摘要来自训练收据，不能从最终权重反推出全部训练过程；CUDA grid_sample反向不保证逐位确定。独立重放/审计如另有报告，应按其实际检查范围解释。','',
        '## 统计解释检查：11项','',
        '| 检查项 | 本轮处理与剩余限制 |','|---|---|',
        '| Simpson反转 | 开发/测试、四组、三个种子分别报告，不把T-LESS与GraspNet AP合并。 |',
        '| 生态谬误 | 总体均值不代表每个小物体获益，不外推新场景。 |',
        '| Berkson/选择偏差 | 数据为已选子集且反复查看，不声称总体代表性或新的盲测。 |',
        '| 碰撞变量偏差 | 不按检出成功、置信度增益或测试表现筛训练/评测样本；完整矩阵保留。 |',
        '| 基率忽略 | 明列开发4、GraspNet测试1、T-LESS52个small及视角相关性。 |',
        '| 均值回归 | 固定seeds17/23/41，不挑最好一次；仍只有三个种子。 |',
        '| 幸存者偏差 | 首轮失败、两份未评分完整模型与修复过程保留；最终12组统一重训。 |',
        '| 多重寻找 | 三个预定改法与一个baseline都报告，不搜索新阈值，不作显著性声明。 |',
        '| 分析路径自由度 | 正式评分前冻结公式、核参数、预算和门槛；工程修复与首次失败公开。 |',
        '| 相关与因果混淆 | BACE单列对BCE2的差异；FreqFusion包含新参数和学习率安排，不能隔离单个滤波机制。 |',
        '| 反向因果 | 原评分之前固定协议；最终test/T-LESS不反向用于本轮选方案或调参。 |','',
        '课程统一数据与类别仍未确认，当前仅object单类子集，不能当作88类身份分类或正式课程成绩。在线默认基线继续保留；即使达到门槛，也只称当前小数据上的可选候选。','']
    if repeat is not None:
        repeat_text=(f"补充的训练数值观察：[seed17 baseline重复运行收据](v5/correction_baseline_identity.json)显示共享初值、采样及初始CPU/CUDA随机流相同，首步loss一致，从第{repeat['first_nonidentical_total_loss_step']}步开始出现浮点分歧，最终权重最大绝对差为{repeat['max_abs_weight_difference']:.8f}。两次最后50步total loss均值分别为{repeat['mean_final50_total_loss']['attempt1']:.7f}与{repeat['mean_final50_total_loss']['attempt2']:.7f}；这只是同一baseline的训练数值观察，旧模型未评分，不能归因于BACE验证修正，也不构成新旧模型精度比较。固定seed不保证CUDA训练逐位一致。")
        index=lines.index('## 核验与可复现材料')
        lines[index:index]=[repeat_text,'']
    if summary.get('independent_integrity_audit'):
        proof=summary['independent_integrity_audit']
        index=lines.index('## 统计解释检查：11项')
        lines[index:index]=[f"独立完整性审计为pass，报告SHA256为`{proof['report_sha256']}`，见[审计记录](v5/training_methods_audit.json)。其检查范围以原报告为准，不把软件核验状态扩大成论文训练复现或普遍效果证明。",'']
    return '\n'.join(lines)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',default=str(ROOT/'reports/v5/trial_summary.json'))
    parser.add_argument('--tests-log',default=str(ROOT/'reports/v5/unit_tests_corrected.log'))
    parser.add_argument('--attempt-record',default=str(ROOT/'reports/v5/attempt1/disposition.json'))
    parser.add_argument('--fix-receipt',default=str(ROOT/'reports/v5/boundary_roundoff_fix.json'))
    parser.add_argument('--output',default=str(ROOT/'reports/training_methods_result.md'))
    parser.add_argument('--repeat-receipt',default=str(ROOT/'reports/v5/correction_baseline_identity.json'))
    parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args()
    output=Path(args.output)
    require(args.overwrite or not output.exists(),'Refusing to overwrite existing report without --overwrite')
    summary=read(args.summary);verify_final_summary(summary)
    tests=parse_tests_log(args.tests_log);attempt=read(args.attempt_record);fix=read(args.fix_receipt)
    require(attempt['status']=='failed_preserved_unscored' and attempt['model_scores_read'] is False,'Unexpected first-attempt disposition')
    require(sha(ROOT/fix['diagnostic_log'])==fix['diagnostic_log_sha256'],'Failure diagnostic changed')
    require(summary['source_sha256']['rgbd_seg/boundary_loss.py']==fix['change']['after_sha256'],'Final training did not use the corrected source')
    require(not fix['change']['clamp_targets'] and not fix['change']['changes_loss_formula'] and not fix['change']['changes_weights_or_scale'],'Repair scope differs from report')
    repeat=read(args.repeat_receipt)
    require(repeat['same_sampling_sha'] and repeat['same_shared_initialization'] and repeat['same_initial_cpu_rng'] and repeat['same_initial_cuda_rng'] and repeat['all_state_tensors_bitwise_equal'] is False, 'Unexpected baseline repeat evidence')
    text=render_report(summary,tests,attempt,fix,sha(args.summary),repeat)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(text)
    print(json.dumps({'output':str(output.resolve()),'summary_sha256':sha(args.summary),'tests_passed':tests['count']},ensure_ascii=False))


if __name__=='__main__':
    main()
