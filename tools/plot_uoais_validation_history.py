#!/usr/bin/env python3
"""Plot completed, audited UOAIS-Sim validation epochs for the two base models."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/dafusion_benchmark_comparison'
MODELS = {
    'dafusion': ('uoais_dafusion_swinl_20ep', 'Swin-L', 'DA-Fusion (Swin-L)', '#2563eb'),
    'dpseg': ('uoais_dpseg_swint_20ep', 'Swin-T', 'DPSeg RGB-D base (Swin-T)', '#ea580c'),
}
METRICS = (
    'Objects Precision', 'Objects Recall', 'Objects F-measure',
    'Boundary Precision', 'Boundary Recall', 'Boundary F-measure',
    'obj_detected_075_percentage',
)
PANELS = (
    ('Objects F-measure', 'Overlap F'),
    ('Boundary F-measure', 'Boundary F'),
    ('obj_detected_075_percentage', '%75 (matched F > 0.75)'),
)


def load_verified_history(report):
    sources = []

    def read(path):
        raw = path.read_bytes()
        sources.append({'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()})
        return json.loads(raw), sources[-1]['sha256']

    history, _ = read(report/'uoais_validation_history.json')
    if history['scope'] != 'full_validation_history_not_final_real_test':
        raise ValueError('Expected validation history, not test or engineering results')
    rows = history['rows']
    if not rows or {row['family'] for row in rows} != set(MODELS):
        raise ValueError('Expected both base models')
    verified = {}
    for family, (directory, backbone, _, _) in MODELS.items():
        protocol, protocol_hash = read(report/directory/'protocol.json')
        if protocol_hash != history['protocol_sha256'] or protocol['status'] != 'frozen':
            raise ValueError('Training protocols differ')
        subset = sorted((row for row in rows if row['family'] == family), key=lambda row: row['epoch'])
        if [row['epoch'] for row in subset] != list(range(1, len(subset)+1)):
            raise ValueError('Duplicate, missing or reordered validation epoch')
        for row in subset:
            epoch = row['epoch']
            if (row['backbone'] != backbone or row['independently_audited'] is not True
                    or row['validation_images'] != 5000
                    or row['train_image_presentations'] != epoch*45000
                    or epoch > protocol['training']['epochs']):
                raise ValueError('History row does not match the formal training scope')
            path = report/directory/f'validation_epoch{epoch:02d}.json'
            summary, summary_hash = read(path)
            audit, _ = read(report/f'uoais_{family}_epoch{epoch:02d}_validation_audit.json')
            if (Path(row['summary_path']).resolve() != path.resolve()
                    or summary_hash != row['summary_sha256']
                    or summary_hash != audit['summary_sha256']
                    or summary['scope'] != 'full_validation' or summary['epoch'] != epoch
                    or summary['images'] != 5000 or summary['test_used'] is not False
                    or summary['score_threshold'] != .5
                    or audit['status'] != 'passed' or audit['family'] != family
                    or audit['images'] != 5000 or audit['test_used'] is not False
                    or audit['expected_unique_images'] != 5000
                    or any(audit[key] != 0 for key in ['duplicates', 'missing', 'extra'])
                    or audit['annotation_sha256'] != protocol['annotations']['val']['sha256']
                    or audit['maximum_summary_difference'] >= 2e-12):
                raise ValueError('Validation sources disagree with the integrity receipt')
            if epoch > 1 and (audit['epoch'] != epoch or audit['protocol_sha256'] != protocol_hash):
                raise ValueError('Audit identifies a different epoch or protocol')
            for key in METRICS:
                value = row[key]
                if (not math.isfinite(value) or not 0 <= value <= 1
                        or abs(value-summary['metrics'][key]) >= 2e-12
                        or abs(value-audit['recomputed_metrics'][key]) >= 2e-12):
                    raise ValueError('History metric differs from its saved validation evidence')
        verified[family] = subset
    return verified, sources, history['protocol_sha256']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports', type=Path, default=REPORT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    rows, sources, protocol_hash = load_verified_history(args.reports.resolve())
    output = (args.output or args.reports/'figures').resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(output/'.matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    plt.rcParams.update({'font.size': 10, 'savefig.facecolor': 'white', 'pdf.fonttype': 42})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    maximum_epoch = max(row['epoch'] for group in rows.values() for row in group)
    shared_epoch = min(group[-1]['epoch'] for group in rows.values())
    for ax, (metric, title) in zip(axes, PANELS):
        values = []
        for family, (_, _, label, color) in MODELS.items():
            group = rows[family]
            yy = [row[metric]*100 for row in group]
            values.extend(yy)
            ax.plot([row['epoch'] for row in group], yy, 'o-', label=label,
                    color=color, linewidth=1.8, markersize=5)
        padding = max(.12, (max(values)-min(values))*.15)
        ax.set(title=title, xlabel='Completed training epoch', ylabel='Validation metric (%)',
               xlim=(.8, maximum_epoch+.2), ylim=(min(values)-padding, min(100, max(values)+padding)))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(alpha=.2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .93),
               ncol=2, frameon=False)
    fig.suptitle('UOAIS-Sim: full validation on the same 5,000 images after each epoch', y=.99)
    fig.subplots_adjust(top=.76, bottom=.24, left=.07, right=.985, wspace=.30)
    fig.text(.5, .105, '45,000 training-image presentations per epoch. Points are completed, audited epochs; no smoothing.',
             ha='center', fontsize=9, color='#475569')
    fig.text(.5, .055, f'Base systems with different backbones and input sizes. Latest shared epoch: {shared_epoch}. '
             'Final real-test comparison remains pending.', ha='center', fontsize=9, color='#475569')
    artifacts = {}
    for suffix in ['png', 'pdf']:
        path = output/f'uoais_validation_history.{suffix}'
        fig.savefig(path, dpi=200)
        artifacts[path.name] = {'bytes': path.stat().st_size,
                               'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    plt.close(fig)
    metadata = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'audited_full_validation_history_not_final_real_test',
        'protocol_sha256': protocol_hash, 'latest_shared_completed_epoch': shared_epoch,
        'rows': rows, 'sources': sources, 'artifacts': artifacts,
        'plot_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (output/'uoais_validation_history_data.json').write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'output': str(output), 'audited_rows': sum(map(len, rows.values())),
                      'latest_shared_completed_epoch': shared_epoch, 'artifacts': artifacts}))


if __name__ == '__main__':
    main()
