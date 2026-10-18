#!/usr/bin/env python3
"""Static figures from verified final course results only; no model execution.

Reads comparison_summary.json plus its SHA-bound training logs. Refuses partial
results, nonfinite numbers, inconsistent checkpoint selection, changed evidence,
or a nonempty figure directory. All AP values stay on the 0–1 scale.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('maskrcnn', 'mask2former')
NAMES = {'maskrcnn': 'Mask R-CNN', 'mask2former': 'Mask2Former'}
COLORS = {'maskrcnn': '#1764AB', 'mask2former': '#C65B22'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path):
    def reject(value):
        raise ValueError(f'Nonfinite JSON constant {value} in {path}')
    return json.loads(Path(path).read_text(), parse_constant=reject)


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def unit_metric(value, description):
    require(finite(value) and 0 <= value <= 1, f'{description} must be an available finite 0–1 metric')
    return float(value)


def load_verified(summary_path, run_root):
    summary_path, run_root = Path(summary_path).resolve(), Path(run_root).resolve()
    summary = read_json(summary_path)
    require(summary.get('status') == 'complete_and_verified', 'Final verified comparison is required; partial results cannot be plotted')
    require(summary.get('schema') == 'course_finetune_comparison_v1', 'Unsupported comparison schema')
    require(set(summary.get('models', {})) == set(MODELS), 'Both completed model results are required')
    require(summary.get('counts') == {'train_per_epoch': 14432, 'epochs': 5, 'val': 2584, 'test': 2112}, 'Expected complete official five-epoch comparison')
    for flag in ['identical_epoch_orders', 'complete_train_val_test', 'validation_only_checkpoint_selection']:
        require(summary.get('checks', {}).get(flag) is True, f'Comparison check not passed: {flag}')
    require(summary.get('metric_scale') == '0–1; delta_percentage_points explicitly multiplies by 100', 'Unexpected metric scale')
    source_hashes = summary.get('source_sha256', {})
    require(bool(source_hashes), 'Summary lacks producing-source hashes')
    for name, expected in source_hashes.items():
        path = (ROOT/name).resolve()
        require(path.is_relative_to(ROOT) and sha256(path) == expected, f'Summary producer source changed: {name}')
    require(sha256(summary['protocol']['path']) == summary['protocol']['sha256'], 'Frozen protocol has changed')
    comparison = {}
    for row in summary['comparison_rows']:
        if row['metric'] == 'mask_ap_small':
            require(row['model'] in MODELS and row['model'] not in comparison, 'Duplicate/unknown APsmall comparison row')
            comparison[row['model']] = row
    require(set(comparison) == set(MODELS), 'Missing final APsmall before/after values')
    data, evidence = {}, {'summary': {'path': str(summary_path), 'sha256': sha256(summary_path)}}
    for name in MODELS:
        result = summary['models'][name]
        folder = run_root/name
        for filename in ['run.json', 'selection.json', 'initialization.json', 'training_log.jsonl']:
            path = folder/filename
            require(sha256(path) == result['receipts_sha256'].get(filename), f'Changed verified input: {path}')
        log_path = folder/'training_log.jsonl'
        rows = [read_json_line(line, log_path) for line in log_path.read_text().splitlines()]
        require(len(rows) == 9020, f'{name}: expected 9,020 optimizer updates')
        for index, row in enumerate(rows, 1):
            require(row.get('global_step') == index and row.get('epoch') == (index-1)//1804+1,
                    f'{name}: missing/duplicated/out-of-order optimizer step')
            require(row.get('images_in_epoch') == (((index-1)%1804)+1)*8
                    and row.get('expected_train_images') == 14432, f'{name}: incomplete training image coverage')
            require(finite(row.get('loss')), f'{name}: nonfinite loss')
        validation = result['validation']
        require([r['epoch'] for r in validation] == [1,2,3,4,5], f'{name}: validation epochs incomplete')
        require(all(r['images'] == 2584 for r in validation), f'{name}: validation is not the complete split')
        val_small = [unit_metric(r['metrics']['mask_ap_small'], f'{name} validation APsmall') for r in validation]
        val_ap = [unit_metric(r['metrics']['mask_ap'], f'{name} validation AP') for r in validation]
        expected_best = max(range(5), key=lambda i: (val_small[i], val_ap[i], -i))+1
        require(result['best_epoch'] == expected_best, f'{name}: selected epoch differs from validation-only rule')
        require(result['test']['images'] == 2112, f'{name}: incomplete final test')
        before = unit_metric(result['zero_shot']['metrics']['mask_ap_small'], f'{name} zero-shot APsmall')
        after = unit_metric(result['test']['metrics']['mask_ap_small'], f'{name} fine-tuned APsmall')
        row = comparison[name]
        require(row['zero_shot'] == before and row['finetuned'] == after and math.isclose(row['delta'], after-before, abs_tol=1e-14),
                f'{name}: comparison row and metrics disagree')
        data[name] = {'steps': [r['global_step'] for r in rows], 'loss': [float(r['loss']) for r in rows],
                      'validation_ap_small': val_small, 'validation_ap': val_ap,
                      'best_epoch': expected_best, 'zero_shot_ap_small': before, 'finetuned_ap_small': after}
        evidence[name] = {'training_log': str(log_path), 'training_log_sha256': sha256(log_path),
                          'test_evaluation_sha256': result['test']['evaluation_sha256'],
                          'zero_shot_evaluation_sha256': result['zero_shot']['evaluation_sha256']}
    return data, evidence, summary


def read_json_line(line, path):
    try:
        value = json.loads(line)
    except ValueError as error:
        raise ValueError(f'Malformed training log {path}') from error
    require(isinstance(value, dict), f'Unexpected log row in {path}')
    return value


def plot(data, evidence, summary, output, window):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    import numpy as np

    output = Path(output).resolve()
    require(not output.exists() or (output.is_dir() and not any(output.iterdir())), 'Figure output must be new or empty; refuse to overwrite existing evidence')
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.dpi': 220})
    artifacts = {}
    ap_axis_limits = {}

    def set_ap_axis(ax, values, name):
        # Preserve the true zero while avoiding an almost empty 0–1 panel.
        padded = min(1.0, max(.01, max(values) * 1.25))
        ticks = MaxNLocator(nbins=5, steps=[1, 2, 2.5, 5, 10]).tick_values(0, padded)
        upper = min(1.0, float(ticks[-1]))
        ax.set_ylim(0, upper)
        ax.set_yticks([float(t) for t in ticks if 0 <= t <= upper])
        ap_axis_limits[name] = [0, upper]

    def save(fig, stem):
        for suffix in ('png', 'pdf'):
            path = output/f'{stem}.{suffix}'
            fig.savefig(path, bbox_inches='tight', facecolor='white')
            artifacts[path.name] = {'sha256': sha256(path), 'bytes': path.stat().st_size}
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.0), sharex=True, sharey=False)
    for ax, name in zip(axes, MODELS):
        row = data[name]
        steps, losses = np.asarray(row['steps']), np.asarray(row['loss'])
        ax.plot(steps, losses, color=COLORS[name], alpha=.16, lw=.55, label='Recorded update loss')
        smoothed = np.convolve(losses, np.ones(window)/window, mode='valid')
        ax.plot(steps[window-1:], smoothed, color=COLORS[name], lw=1.7, label=f'Trailing {window}-update mean')
        ax.set(title=NAMES[name], xlabel='Optimizer update', ylabel='Native training objective')
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
    fig.suptitle('Training loss: separate objectives and independent y-axis scales', fontsize=12)
    fig.text(.5, -.035, 'Loss magnitudes cannot rank these architectures. One seed; 5 complete epochs each.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'training_loss')

    fig, ax = plt.subplots(figsize=(6.8, 4.7))
    for name in MODELS:
        row = data[name]
        ax.plot(range(1,6), row['validation_ap_small'], '-o', color=COLORS[name], label=NAMES[name], lw=1.8, markersize=4)
        for point_epoch, value in enumerate(row['validation_ap_small'], 1):
            other = next(n for n in MODELS if n != name)
            other_value = data[other]['validation_ap_small'][point_epoch-1]
            higher = value > other_value or (value == other_value and name == MODELS[0])
            ax.annotate(f'{value:.4f}', (point_epoch, value),
                        xytext=(0, 8 if higher else -13),
                        textcoords='offset points', ha='center', fontsize=8,
                        color=COLORS[name])
        epoch = row['best_epoch']
        ax.scatter([epoch], [row['validation_ap_small'][epoch-1]], marker='*', s=140, color=COLORS[name], edgecolors='black', linewidths=.5, zorder=4)
    ax.set(xlabel='Completed epoch', ylabel='Validation mask APsmall (0–1)', xticks=range(1,6),
           title='Full validation split: 2,584 images')
    set_ap_axis(ax, [v for name in MODELS for v in data[name]['validation_ap_small']], 'validation_ap_small')
    ax.grid(axis='y', alpha=.25)
    ax.legend(frameon=False)
    fig.text(.5, -.01, 'Stars: checkpoint chosen by validation APsmall, then AP, then earlier epoch.', ha='center', fontsize=9)
    fig.tight_layout()
    save(fig, 'validation_ap_small')

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    x = np.arange(2); width=.34
    before = [data[name]['zero_shot_ap_small'] for name in MODELS]
    after = [data[name]['finetuned_ap_small'] for name in MODELS]
    left = ax.bar(x-width/2, before, width, label='Public COCO weights (no course training)', color='#9BA8B1', edgecolor='black', linewidth=.4)
    right = ax.bar(x+width/2, after, width, label='Fine-tuned on course train split', color='#2679B2', edgecolor='black', linewidth=.4)
    for bars, values in [(left,before),(right,after)]:
        for bar,value in zip(bars,values):
            ax.annotate(f'{value:.6f}', (bar.get_x()+bar.get_width()/2,value), xytext=(0,4), textcoords='offset points', ha='center', va='bottom', fontsize=9)
    ax.set(xticks=x, xticklabels=[NAMES[n] for n in MODELS], ylabel='Test mask APsmall (0–1)',
           title='Final selected checkpoints: all 2,112 test images')
    set_ap_axis(ax, before + after, 'test_ap_small_before_after')
    ax.legend(frameon=False, fontsize=8, loc='upper left')
    ax.grid(axis='y', alpha=.2);ax.set_axisbelow(True)
    fig.text(.5, -.015, 'Single seed; no error bars. Mask2Former before/after also changes resize implementation.', ha='center', fontsize=8.5)
    fig.tight_layout()
    save(fig, 'test_ap_small_before_after')
    receipt = {'status':'completed', 'created_at_utc':datetime.now(timezone.utc).isoformat(),
               'inputs':evidence, 'plotter_sha256':sha256(__file__), 'artifacts':artifacts,
               'metric_scale':'0–1, no percentage multiplication', 'loss_plot_rule':f'All recorded losses plus trailing arithmetic mean of {window} updates; no padding or cross-model normalization; independent panel y scales',
               'ap_axis_rule':'Zero-based, 25% padding above observed maximum, rounded to readable ticks and capped at 1; minimum upper target 0.01',
               'ap_axis_limits':ap_axis_limits,
               'validation_epochs':[1,2,3,4,5], 'model_selection_uses_test':False,
               'real_values':{name:{k:v for k,v in row.items() if k not in ['steps','loss']} for name,row in data.items()},
               'limitations':summary['limitations'], 'no_inference_training_or_mask_decode':True}
    (output/'figure_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return receipt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--summary',type=Path,default=ROOT/'reports/course_finetune/comparison_summary.json')
    parser.add_argument('--run-root',type=Path,default=ROOT/'reports/course_finetune')
    parser.add_argument('--output',type=Path,default=ROOT/'reports/course_finetune/figures')
    parser.add_argument('--loss-window',type=int,default=50,help='Trailing mean window; raw losses remain visible')
    parser.add_argument('--check-only',action='store_true',help='Validate complete real inputs without producing figures')
    args=parser.parse_args()
    require(1 <= args.loss_window <= 9020, 'Loss window must be in [1,9020]')
    data,evidence,summary=load_verified(args.summary,args.run_root)
    if args.check_only:
        print(json.dumps({'status':'verified_inputs','models':list(data),'training_updates_per_model':9020,'figures_created':False}))
        return
    result=plot(data,evidence,summary,args.output,args.loss_window)
    print(json.dumps({'status':result['status'],'output':str(args.output.resolve()),'artifacts':list(result['artifacts'])}))


if __name__=='__main__':
    main()
