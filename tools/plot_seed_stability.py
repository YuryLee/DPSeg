#!/usr/bin/env python3
"""Plot every matched seed; connecting lines pair the same training seed."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', default='reports/v3/seed_stability.json')
    parser.add_argument('--output', default='reports/v3/seed_stability')
    args = parser.parse_args()
    source = Path(args.report)
    report = json.loads(source.read_text())
    if report['matched_seed_count'] < 2:
        raise ValueError('At least two matched seeds are needed for this figure')
    fig, axes = plt.subplots(2, 2, figsize=(10.8, 7.8))
    colors = ['#1976b9', '#c26c22', '#8b50a2', '#368365']
    for row, split in enumerate(('val', 'test')):
        for col, (metric, name) in enumerate((('mask_ap', 'Mask AP'), ('mask_ap_small', 'Mask AP small'))):
            ax = axes[row, col]
            all_values = []
            for i, record in enumerate(report['per_seed']):
                values = [record['methods'][mode]['splits'][split][metric] for mode in ('E0', 'E2')]
                if any(value is None for value in values):
                    continue
                all_values.extend(values)
                ax.plot([0, 1], values, marker='o', lw=1.8, ms=6,
                        color=colors[i % len(colors)], label=f"seed {record['seed']}")
                for x, value in enumerate(values):
                    ax.annotate(f'{value:.2f}', (x, value), xytext=(-35 if x == 0 else 8, (i-1)*11),
                                textcoords='offset points', fontsize=8, color=colors[i % len(colors)])
            ax.set_xticks([0, 1], ['E0: RGB', 'E2: RGB + depth'])
            ax.set_xlim(-.45, 1.45)
            if all_values:
                padding = max(5, (max(all_values)-min(all_values))*.35)
                ax.set_ylim(max(0, min(all_values)-padding), min(100, max(all_values)+padding))
            ax.set_ylabel(f'{name} (0-100)')
            ax.set_title(('Development' if split == 'val' else 'Test') + f' | {name}', loc='left')
            ax.grid(axis='y', alpha=.22)
            ax.spines[['top', 'right']].set_visible(False)
    axes[0, 0].legend(loc='best', frameon=False)
    fig.suptitle('Matched seeds on the fixed GraspNet subset', fontsize=16, y=.98)
    fig.text(.06, .035, 'Each line connects E0 and E2 with the same training seed. Fixed data and 800-step training budget.\n'
             'Development: 4 small instances; test: 1 small instance. This shows seed variation, not uncertainty over new scenes.', fontsize=10)
    fig.tight_layout(rect=(.02, .09, .98, .94), h_pad=3)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for ext in ('png', 'pdf'):
        fig.savefig(output.with_suffix('.'+ext), dpi=220)
    plt.close(fig)
    output.with_name(output.name+'_plot_source.json').write_text(json.dumps({
        'report': str(source.resolve()), 'report_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'matched_seeds': report['matched_seeds'], 'metrics': ['mask_ap', 'mask_ap_small'],
        'splits': ['val', 'test'], 'axes': 'Each panel uses a labelled local vertical range; no error bars or confidence intervals.'
    }, indent=2)+'\n')
    print(output.with_suffix('.png').resolve())


if __name__ == '__main__':
    main()
