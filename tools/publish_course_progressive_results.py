#!/usr/bin/env python3
"""Publish the five-stage table only after audited full validation and test."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_progressive_ablation import STAGES

LABELS = ('Mask2Former fine-tune', '+ RGB-D 融合（DA-Fusion课程适配）',
          '+ 深度点细化', '+ 切片推理', '+ 概率图深度细化')
START = '<!-- progressive-ablation:start -->'
END = '<!-- progressive-ablation:end -->'


def read(path):
    return json.loads(Path(path).read_text())


def now():
    return datetime.now(timezone.utc).isoformat()


def audit(evaluation):
    state = read(evaluation/'supervisor.json')
    comparison = read(evaluation/'comparison.json')
    protocol = read(evaluation/'protocol.json')
    if (state['status'] != 'completed' or comparison['status'] != 'completed'
            or comparison['stages'] != list(STAGES) or comparison['test_labels_used_for_selection']
            or comparison['protocol_sha256'] != file_sha(evaluation/'protocol.json')
            or comparison['validation_lock_sha256'] != file_sha(evaluation/'validation_lock.json')):
        raise ValueError('Experiment has not completed its locked full-split comparison')
    from tools.run_course_progressive_ablation import verify
    verify(protocol)
    for name in ('rgbd_val', 'rgb_val', 'all_test'):
        run = read(evaluation/f'{name}_run.json')
        if run['status'] != 'completed': raise ValueError('Missing successful evaluation phase')
    for group, split, stages, count in (('rgbd', 'val', STAGES[1:], 2584),
            ('rgb', 'val', STAGES[:1], 2584), ('all', 'test', STAGES, 2112)):
        folder = evaluation/(f'{group}_val' if split == 'val' else 'test')
        ids = read(folder/'image_ids.json')
        if len(ids) != count or len(set(ids)) != count: raise ValueError('Incomplete evaluation IDs')
        exits = read(folder/'worker_exits.json')
        if any(v['exit_code'] != 0 for v in exits): raise ValueError('Nonzero worker exit')
        for arm in stages:
            result = read(folder/arm/'evaluation.json')
            if (result['predictions_sha256'] != file_sha(folder/arm/'predictions.jsonl')
                    or result['metrics'] != comparison[split][arm]):
                raise ValueError('Published metric differs from scored predictions')
    for arm, spec in comparison['csvs'].items():
        if file_sha(spec['path']) != spec['sha256'] or not spec['exact_scored_predictions']:
            raise ValueError('CSV no longer matches scored predictions')
    return comparison, protocol


def render(comparison, protocol):
    lines = ['### 五步累计消融', '',
        '以下各行累计保留上一行的组件。主表统一使用完整本地测试集2,112张图，分数范围为0–1。', '',
        '| 编号 | 配置（累计添加） | 本地测试 AP | 本地测试 APₛ | APₛ 相对上一组增量 |',
        '|---|---|---:|---:|---:|']
    previous = None
    for i, (arm, label) in enumerate(zip(STAGES, LABELS)):
        metric = comparison['test'][arm]
        delta = '—' if previous is None else f"{metric['mask_ap_small']-previous:+.6f}"
        lines.append(f"| S{i} | {label} | {metric['mask_ap']:.6f} | {metric['mask_ap_small']:.6f} | {delta} |")
        previous = metric['mask_ap_small']
    total = comparison['test'][STAGES[-1]]['mask_ap_small']-comparison['test'][STAGES[0]]['mask_ap_small']
    lines += ['', f'完整系统S4相对RGB基线S0的APₛ绝对变化为 **{total:+.6f}**。负增量同样保留，不能把累计添加组件描述为每一步均有提升。', '',
        '- **统一基座预算：** RGB与RGB-D均从COCO公开权重开始，Swin-Tiny、1024输入、完整训练集、15轮、全局batch33和相同两阶段AdamW学习率计划；主表固定第15轮权重。',
        '- **固定组件：** S1–S4共用第15轮RGB-D基座，S2–S4共用验证集选择的第6轮深度点头；S3关闭概率细化，S4仅增加既定概率图深度细化。',
        '- **评价顺序：** 完成2,584张验证图的五组评价并锁定配置后，再评价完整本地测试集。置信度阈值0.01、每图最多100实例；测试标注不进入推理或参数选择。',
        '- **比较范围：** RGB-D包含额外编码器及DS/DC融合；历史RGB-D与本次RGB的显卡及局部batch划分不同，不能视为逐步梯度完全一致的控制实验。点头另有6轮训练，切片增加推理计算量。', '',
        '完整验证指标、五份CSV本机路径、权重与工程核验见[五步消融报告](docs/results/progressive_ablation.md)，原始数值见[结构化结果](docs/results/progressive_ablation.json)。本表为本地完整测试结果，Kaggle公榜成绩见[历史结果](docs/historical_results.md)。']
    return '\n'.join(lines)


def publish(args):
    evaluation = args.evaluation.resolve()
    comparison, protocol = audit(evaluation)
    readme = ROOT/'README.md'
    if args.expected_readme_sha256 and file_sha(readme) != args.expected_readme_sha256:
        raise ValueError('README changed since the publication plan; preserve user edits for manual reconciliation')
    rows = []
    for i, (arm, label) in enumerate(zip(STAGES, LABELS)):
        row = dict(id=f'S{i}', arm=arm, label=label, val=comparison['val'][arm], test=comparison['test'][arm],
            csv={k:comparison['csvs'][arm][k] for k in ('path', 'sha256', 'exact_scored_predictions')})
        row['test_aps_delta_previous'] = None if i == 0 else row['test']['mask_ap_small']-rows[-1]['test']['mask_ap_small']
        rows.append(row)
    result = dict(status='completed', published_at_utc=now(), rows=rows, train_images=14432,
        val_images=2584, test_images=2112, main_metric='COCO Mask AP_small @[.50:.95], area [0,1024], maxDets100',
        base_epochs=dict(rgb=15, rgbd=15), head_epoch=6, global_batch=33,
        checkpoint_sha256=dict(rgb=protocol['rgb']['weights']['sha256'], rgbd=protocol['inputs']['base']['sha256'],
                               depth_head=protocol['inputs']['head']['sha256']),
        protocol_sha256=comparison['protocol_sha256'], validation_lock_sha256=comparison['validation_lock_sha256'],
        comparison_sha256=file_sha(evaluation/'comparison.json'),
        test_labels_used_for_selection=False, kaggle_submitted=False,
        limitations=protocol['limitations'],
        material_passport=dict(origin='academic-research-suite/experiment-agent', mode='run',
                               verification_status='VERIFIED_FULL_SPLITS'))
    write_json(ROOT/'docs/results/progressive_ablation.json', result)
    body = render(comparison, protocol)
    text = readme.read_text()
    if text.count(START) != 1 or text.count(END) != 1: raise ValueError('Ambiguous README section markers')
    first, tail = text.split(START); _, last = tail.split(END)
    text = first+START+'\n'+body+'\n'+END+last
    text = text.replace('第15轮基座加第6轮点头的切片结果正在补充评测。', '第15轮基座加第6轮点头的完整切片与测试结果见下方五步消融。')
    text = text.replace('五步累计消融正在补齐RGB基线和切片对照；新增结果以实测记录为准。', '五步累计消融已完成完整验证与本地测试；结果见主表。')
    readme.write_text(text)
    report = ['# 五步累计消融', '', '## Material Passport', '',
        '- 来源：ARS experiment-agent，run模式。',
        '- 状态：完整训练、验证、锁定后测试与五份CSV一致性检查完成。',
        '- 范围：单种子课程实验；本地测试分数与Kaggle公榜分别报告。', '',
        body.replace('(docs/results/', '(').replace('(docs/historical_results.md)', '(../historical_results.md)'), '',
        '## 完整验证结果', '', '| 编号 | 配置（累计添加） | 验证AP | 验证APₛ |', '|---|---|---:|---:|']
    for row in rows:
        report.append(f"| {row['id']} | {row['label']} | {row['val']['mask_ap']:.6f} | {row['val']['mask_ap_small']:.6f} |")
    report += ['', '## 测试CSV本机路径', '', '每份文件均包含2,112张图，与该行评分使用的逐图预测完全一致。掩码为1024×1024的COCO压缩RLE，每图最多100实例。', '']
    for row in rows: report.append(f"- **{row['id']} {row['label']}**：`{row['csv']['path']}`")
    report += ['', '## 核验与复现', '',
        '- RGB基线完成5+10轮完整训练和逐轮完整验证；总计6,570次优化器更新。使用固定第15轮，而非依据测试结果选择轮次。',
        '- 新切片实现通过8张真实验证图的旧实现对照，检查五个视图的零残差点头、最终RLE与原实现一致。RGB几何输入不变性检查通过。',
        '- 所有推理分片完整覆盖目标划分，合并后进行COCO评分；实际进程退出状态和权重摘要均保存。',
        '- 概率细化两组共享候选、置信度与顺序。未调整既有阈值、点数、切片和深度细化参数。',
        '- 运行目录：`reports/course_progressive_ablation/v1/`；评价入口：`tools/supervise_course_progressive_ablation.py`；RGB训练入口：`tools/supervise_course_progressive_rgb.py`。', '',
        '结构化指标与权重摘要见[progressive_ablation.json](progressive_ablation.json)。数据、权重、大型预测和CSV由Git忽略，通过课程本机目录共享。', '',
        '### 限制', '']
    report += ['- '+item for item in protocol['limitations']]
    (ROOT/'docs/results/progressive_ablation.md').write_text('\n'.join(report)+'\n')
    if args.commit_and_push:
        paths = ['README.md', 'docs/results/progressive_ablation.md', 'docs/results/progressive_ablation.json']
        subprocess.run(['git', 'add', '--', *paths], cwd=ROOT, check=True)
        subprocess.run(['git', 'commit', '-m', 'Report complete five-stage cumulative ablation and CSV paths'], cwd=ROOT, check=True)
        subprocess.run(['git', 'push', 'origin', 'main'], cwd=ROOT, check=True)
        local = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        remote = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/main'], cwd=ROOT, text=True).split()[0]
        if local != remote: raise ValueError('Remote commit does not match published results')
        write_json(evaluation/'publication.json', dict(status='completed', commit=local,
            remote='https://github.com/YuryLee/DPSeg', published_at_utc=now(),
            readme_sha256=file_sha(readme), comparison_sha256=file_sha(evaluation/'comparison.json')))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, required=True)
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--expected-readme-sha256')
    parser.add_argument('--commit-and-push', action='store_true')
    args = parser.parse_args()
    state_path = args.evaluation/'publication_status.json'
    state = dict(status='running', pid=os.getpid(), started_at_utc=now(), phase='waiting' if args.wait else 'publishing')
    write_json(state_path, state)
    try:
        started = time.monotonic()
        while args.wait:
            evaluation = read(args.evaluation/'supervisor.json')
            if evaluation['status'] == 'failed': raise RuntimeError('Evaluation failed; publication stopped')
            if evaluation['status'] == 'completed': break
            os.kill(evaluation['pid'], 0)
            if time.monotonic()-started > 72*3600: raise TimeoutError('Publication wait expired')
            state.update(last_process_check_at_utc=now(), evaluation_phase=evaluation['phase'])
            write_json(state_path, state); time.sleep(30)
        state['phase'] = 'publishing'; write_json(state_path, state)
        publish(args)
        state.update(status='completed', phase='published', finished_at_utc=now())
    except BaseException as error:
        state.update(status='failed', error_type=type(error).__name__, error=str(error), failed_at_utc=now())
        write_json(state_path, state)
        raise
    write_json(state_path, state)


if __name__ == '__main__':
    main()
