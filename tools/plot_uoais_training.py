#!/usr/bin/env python3
"""Render training-loss snapshots and the audited first-epoch validation table."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/dafusion_benchmark_comparison'
MODELS = {
    'dafusion': ('uoais_dafusion_swinl_20ep', 'DA-Fusion (Swin-L)', '#2563eb'),
    'dpseg': ('uoais_dpseg_swint_20ep', 'DPSeg RGB-D base (Swin-T)', '#ea580c'),
}
METRICS = (
    'Objects Precision', 'Objects Recall', 'Objects F-measure',
    'Boundary Precision', 'Boundary Recall', 'Boundary F-measure',
    'obj_detected_075_percentage',
)


def read_training(path, window):
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    incomplete_tail = bool(lines and not lines[-1].endswith(b'\n'))
    if incomplete_tail:
        lines.pop()
    rows = [json.loads(line) for line in lines]
    if not rows or any(not math.isfinite(row['loss']) for row in rows):
        raise ValueError(f'Empty or nonfinite training log: {path}')
    steps = [row['global_step'] for row in rows]
    if steps != sorted(set(steps)) or steps[0] != 1:
        raise ValueError(f'Duplicate, reordered or truncated training history: {path}')
    if any(right != left+1 for left, right in zip(steps, steps[1:])):
        raise ValueError(f'Missing training steps: {path}')
    bins = []
    for start in range(0, len(rows)-window+1, window):
        group = rows[start:start+window]
        bins.append({'step': statistics.mean(row['global_step'] for row in group),
                     'loss': statistics.median(row['loss'] for row in group)})
    if not bins:
        raise ValueError(f'Not enough steps for a {window}-step bin: {path}')
    return {'path': str(path), 'captured_bytes': len(raw),
            'captured_sha256': hashlib.sha256(raw).hexdigest(),
            'incomplete_final_line_ignored': incomplete_tail,
            'records': len(rows), 'last_step': steps[-1], 'last_epoch': rows[-1]['epoch'],
            'omitted_incomplete_bin_steps': len(rows) % window, 'bins': bins}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=REPORT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--window', type=int, default=100)
    args = parser.parse_args()
    if args.window < 1:
        parser.error('--window must be positive')
    out = (args.output or args.reports/'figures').resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(out/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'savefig.facecolor': 'white', 'pdf.fonttype': 42})

    comparison_path = args.reports/'uoais_epoch01_validation_comparison.json'
    comparison_raw = comparison_path.read_bytes()
    comparison = json.loads(comparison_raw)
    if (not comparison['protocol_identical'] or comparison['images_per_model'] != 5000
            or comparison['epoch'] != 1 or comparison['test_used'] is not False):
        raise ValueError('Expected the audited full first-epoch validation comparison')
    snapshots = {}
    batch = None
    for family, (directory, _, _) in MODELS.items():
        run = args.reports/directory
        protocol_raw = (run/'protocol.json').read_bytes()
        if hashlib.sha256(protocol_raw).hexdigest() != comparison['protocol_sha256']:
            raise ValueError('Training protocol differs from the validation comparison')
        protocol = json.loads(protocol_raw)
        current_batch = protocol['training']['global_batch']
        if batch is not None and batch != current_batch:
            raise ValueError('Models have different global batch sizes')
        batch = current_batch
        audit = json.loads((args.reports/comparison['models'][family]['integrity_receipt']).read_text())
        summary_raw = (run/'validation_epoch01.json').read_bytes()
        summary = json.loads(summary_raw)
        if (audit['status'] != 'passed' or audit['images'] != 5000
                or audit['summary_sha256'] != hashlib.sha256(summary_raw).hexdigest()
                or any(abs(summary['metrics'][key]-comparison['models'][family]['metrics'][key]) > 2e-12
                       for key in METRICS)):
            raise ValueError('Validation integrity evidence differs from the table')
        snapshots[family] = read_training(run/'training_log.jsonl', args.window)

    fig, ax = plt.subplots(figsize=(10, 4.8), layout='constrained')
    maximum = 0.
    for family, (_, label, color) in MODELS.items():
        bins = snapshots[family]['bins']
        xx = [row['step']*batch/1000 for row in bins]
        yy = [row['loss'] for row in bins]
        ax.plot(xx, yy, label=label, color=color, linewidth=1.8)
        maximum = max(maximum, max(yy))
    ax.set(title='UOAIS-Sim training progress', xlabel='Training-image presentations (thousands)',
           ylabel=f'Training loss ({args.window}-step bin median)', ylim=(0, maximum*1.08))
    ax.grid(alpha=.2)
    ax.legend(frameon=False)
    fig.suptitle(f'Global batch {batch}; full validation after each 45,000 training images',
                 fontsize=10, color='#475569')
    for suffix in ['png', 'pdf']:
        fig.savefig(out/f'uoais_training_loss_snapshot.{suffix}', dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 2.8), layout='constrained')
    ax.axis('off')
    cells = []
    for family in MODELS:
        row = comparison['models'][family]
        cells.append([MODELS[family][1], *[f'{row["metrics"][key]*100:.2f}' for key in METRICS]])
    table = ax.table(cellText=cells,
                     colLabels=['Model', 'Overlap P', 'Overlap R', 'Overlap F',
                                'Boundary P', 'Boundary R', 'Boundary F', '%75'],
                     cellLoc='center', colWidths=[.27]+[.103]*7, loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1., 1.8)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor('#cbd5e1')
        if row == 0:
            cell.set_facecolor('#e2e8f0')
            cell.set_text_props(weight='bold')
        elif col == 0:
            cell.set_text_props(color=MODELS[list(MODELS)[row-1]][2])
    ax.set_title('Epoch 1 full validation: UOAIS-Sim, 5,000 images, metrics in %', pad=12)
    fig.text(.5, .04, 'RGB-D base systems; validation results. Final OSD/OCID/OCBD comparison remains pending.',
             ha='center', fontsize=9, color='#475569')
    for suffix in ['png', 'pdf']:
        fig.savefig(out/f'uoais_epoch01_validation_table.{suffix}', dpi=200)
    plt.close(fig)

    metadata = {'created_utc': datetime.now(timezone.utc).isoformat(), 'scope': 'training_progress_and_epoch01_validation',
                'global_batch': batch, 'loss_bin_steps': args.window, 'training': snapshots,
                'comparison_path': str(comparison_path),
                'comparison_sha256': hashlib.sha256(comparison_raw).hexdigest(),
                'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (out/'plot_data.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'output': str(out), 'last_steps': {key:value['last_step'] for key,value in snapshots.items()},
                      'files': ['uoais_training_loss_snapshot.png', 'uoais_training_loss_snapshot.pdf',
                                'uoais_epoch01_validation_table.png', 'uoais_epoch01_validation_table.pdf', 'plot_data.json']}))


if __name__ == '__main__':
    main()
