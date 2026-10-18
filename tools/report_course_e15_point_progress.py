#!/usr/bin/env python3
"""Report audited whole-image validation epochs for the user-selected e15 base."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


LABELS = {
    'native': 'DA-Fusion 第15轮',
    'native_probability_depth': '概率图深度细化',
    'point': '普通点头',
    'point_probability_depth': '普通点头＋概率图深度细化',
    'point_depth': '深度点头',
    'point_depth_probability_depth': '深度点头＋概率图深度细化',
}


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] /
                        'reports/course_longer_training/point_e15_v1')
    args = parser.parse_args()
    root = args.root.resolve()
    protocol = read(root / 'protocol.json')
    history = read(root / 'formal/experiment/validation_history.json')
    rows, evidence = [], []
    for entry in history:
        path = root / f"epoch{entry['epoch']:02d}_validation_audit.json"
        # A trainer may finish another epoch while its audit is still pending.
        if not path.exists():
            continue
        audit = read(path)
        if (audit['status'] != 'passed' or audit['scope'] != 'completed_epoch_only'
                or audit['epoch'] != entry['epoch'] or audit['base_epoch'] != 15
                or audit['protocol_sha256'] != sha(root / 'protocol.json')
                or audit['base_checkpoint_sha256'] != protocol['weights']['sha256']
                or audit['training_images'] != 14432 or audit['validation_images'] != 2584
                or audit['checkpoint'] != entry['checkpoint']):
            raise ValueError('Epoch audit scope or provenance differs')
        if sha(audit['checkpoint']['path']) != audit['checkpoint']['sha256']:
            raise ValueError('Audited head checkpoint changed')
        for variant in LABELS:
            for metric in ('mask_ap', 'mask_ap_small'):
                if audit['metrics'][variant][metric] != entry['metrics'][variant][metric]:
                    raise ValueError('History differs from the audited metrics')
        rows.append({'epoch': entry['epoch'], 'checkpoint': audit['checkpoint'],
                     'metrics': audit['metrics']})
        evidence.append({'path': str(path), 'sha256': sha(path)})
    if not rows:
        raise ValueError('No audited full-validation epoch is available')
    rows.sort(key=lambda row: row['epoch'])
    candidates = [dict(epoch=row['epoch'], variant=variant, **metrics)
                  for row in rows for variant, metrics in row['metrics'].items()]
    best = max(candidates, key=lambda row: (row['mask_ap_small'], row['mask_ap'], -row['epoch']))
    baseline = next(row for row in rows if row['epoch'] == best['epoch'])['metrics']['native_probability_depth']
    summary = {'status': 'audited_validation_progress', 'scope': 'whole_image_full_validation',
               'base_epoch': 15, 'planned_epochs': protocol['epochs'],
               'audited_epochs': [row['epoch'] for row in rows], 'validation_images': 2584,
               'epochs': rows, 'best_observed': best,
               'gain_vs_probability_only': {key: best[key] - baseline[key]
                                            for key in ('mask_ap', 'mask_ap_small')},
               'audit_evidence': evidence, 'reporter_sha256': sha(__file__),
               'test_used': False, 'full_run_completion_proven': False}
    (root / 'interim_comparison.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles = [
        ('native', 'DA-Fusion e15', '#777777', '--'),
        ('native_probability_depth', 'Probability refinement', '#333333', '--'),
        ('point_probability_depth', 'Plain point + probability refinement', '#e68624', '-'),
        ('point_depth_probability_depth', 'Depth point + probability refinement', '#287bb5', '-'),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, metric, label in zip(axes, ('mask_ap_small', 'mask_ap'), ('Mask APs', 'Mask AP')):
        for variant, legend, color, linestyle in styles:
            ax.plot([row['epoch'] for row in rows], [row['metrics'][variant][metric] for row in rows],
                    marker='o', color=color, linestyle=linestyle, label=legend)
        ax.set_xlabel('Head training epoch')
        ax.set_ylabel(label)
        ax.set_xticks([row['epoch'] for row in rows])
        ax.set_xlim(rows[0]['epoch'] - .2, rows[-1]['epoch'] + .2)
        ax.grid(alpha=.25)
    fig.suptitle(f"Frozen DA-Fusion epoch 15 | Full validation: 2,584 images | Audited: {len(rows)}/{protocol['epochs']} epochs")
    handles, legends = axes[0].get_legend_handles_labels()
    fig.legend(handles, legends, loc='lower center', ncol=2, fontsize=9)
    fig.subplots_adjust(bottom=.23, top=.89, wspace=.25)
    for suffix in ('png', 'pdf'):
        fig.savefig(root / f'point_learning_curves.{suffix}', dpi=200)
    plt.close(fig)

    lines = ['# 第15轮基础模型上的点头训练：阶段结果', '',
             f"已核验点头轮次：{', '.join(str(row['epoch']) for row in rows)}；计划训练{protocol['epochs']}轮。",
             '基础模型固定为已核验的DA-Fusion第15轮，输入为1024×1024整图。每轮评测全部2,584张验证图，以下分数范围为0–1。', '',
             '| 点头轮次 | 配置 | AP | APₛ |', '|---:|---|---:|---:|']
    for row in rows:
        for variant, metrics in row['metrics'].items():
            lines.append(f"| {row['epoch']} | {LABELS[variant]} | {metrics['mask_ap']:.6f} | {metrics['mask_ap_small']:.6f} |")
    lines += ['', f"当前完整验证集最佳为第{best['epoch']}轮的{LABELS[best['variant']]}：APₛ {best['mask_ap_small']:.6f}、AP {best['mask_ap']:.6f}。",
              f"相对同基础模型单独概率细化，APₛ增加{summary['gain_vs_probability_only']['mask_ap_small']:.6f}。当前选择按APₛ、AP、较早轮次排序。", '',
              '![完整验证集学习曲线](point_learning_curves.png)', '',
              '这是运行中的阶段结果。最终6轮训练覆盖、基础模型冻结、优化器及随机状态仍需整次实验核验。当前表格不包含切片推理或Kaggle成绩。', '',
              '- [结构化对比及审计来源](interim_comparison.json)',
              '- [正式训练协议](protocol.json)',
              '- [点头阶段候选选择](interim_selection.json)']
    (root / 'interim_report.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({'status': summary['status'], 'audited_epochs': summary['audited_epochs'],
                      'best_observed': best, 'report': str(root / 'interim_report.md')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
