#!/usr/bin/env python3
"""Plot measured GraspNet development metrics after verifying paired inputs."""
from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/v2'
MODES = ['E0', 'E2', 'E3']
DATA = {'before': ROOT / 'data/graspnet_subset',
        'after': ROOT / 'data/graspnet_expanded_v2'}
RUNS = {'before': ROOT / 'outputs/graspnet',
        'after': ROOT / 'outputs/graspnet_v2'}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifests = {key: json.loads((path / 'manifest.json').read_text()) for key, path in DATA.items()}
    samples = {key: {sample['id']: sample for sample in manifest['samples'] if sample['split'] == 'val'}
               for key, manifest in manifests.items()}
    ids = sorted(samples['before'])
    assert ids == sorted(samples['after']) and len(ids) == 4
    assert manifests['before']['classes'] == manifests['after']['classes'] == ['object']
    assert manifests['before']['depth_scale'] == manifests['after']['depth_scale']
    proof = {'status': 'verified', 'comparison_split': 'development/val',
             'manifest_sha256': {key: digest(path / 'manifest.json') for key, path in DATA.items()},
             'classes_equal': True, 'depth_scale_equal': True, 'samples': [],
             'train_images': {key: sum(s['split'] == 'train' for s in m['samples']) for key, m in manifests.items()}}
    total_instances = total_small = 0
    for sample_id in ids:
        old, new = samples['before'][sample_id], samples['after'][sample_id]
        assert old['labels'] == new['labels']
        row = {'sample_id': sample_id, 'labels_equal': True, 'files': {}}
        for kind in ['rgb', 'depth', 'instances']:
            old_path, new_path = DATA['before'] / old[kind], DATA['after'] / new[kind]
            old_bytes, new_bytes = old_path.read_bytes(), new_path.read_bytes()
            assert old_bytes == new_bytes
            row['files'][kind] = {'before_path': str(old_path.relative_to(ROOT)),
                                 'after_path': str(new_path.relative_to(ROOT)),
                                 'identical_bytes': True, 'bytes': len(old_bytes),
                                 'sha256': hashlib.sha256(old_bytes).hexdigest()}
        mask = np.array(Image.open(DATA['before'] / old['instances']))
        values, counts = np.unique(mask, return_counts=True)
        areas = [int(area) for label, area in zip(values, counts) if label != 0]
        row.update(shape=list(mask.shape), instances=len(areas), small_instances=sum(area < 1024 for area in areas))
        total_instances += len(areas)
        total_small += row['small_instances']
        proof['samples'].append(row)
    proof.update(images=len(ids), instances=total_instances, small_instances=total_small)
    assert (total_instances, total_small) == (41, 4)
    assert proof['train_images'] == {'before': 8, 'after': 64}

    plotted = {'split': 'development/val', 'units': 'percent', 'sample_ids': ids,
               'rendering': {'library': 'matplotlib', 'version': matplotlib.__version__, 'backend': 'Agg', 'png_dpi': 240},
               'images': len(ids), 'gt_instances': total_instances, 'gt_small_instances': total_small,
               'train_images': proof['train_images'], 'values': [], 'sources': {},
               'interpretation': 'Composite system upgrade; backbone/pretraining/data/resolution/augmentation/training budget changed together. This before/after comparison does not isolate the causal effect of depth fusion.',
               'uncertainty': 'One seed (17), four correlated views from one development scene; no confidence intervals or significance claim.'}
    protocols = []
    for mode in MODES:
        for version in ['before', 'after']:
            folder = RUNS[version] / mode
            ev_path = folder / 'evaluation.json'
            extra_path = ROOT / f'reports/graspnet_extra_{mode}.json' if version == 'before' else folder / 'extra_val.json'
            run_path = folder / 'run.json'
            ev, extra, run = [json.loads(path.read_text()) for path in [ev_path, extra_path, run_path]]
            assert ev['split'] == extra['run']['split'] == 'val'
            assert ev['samples'] == extra['run']['samples'] == 4
            assert ev['manifest_sha256'] == proof['manifest_sha256'][version]
            assert ev['checkpoint_sha256'] == run['checkpoint_sha256'] == extra['run']['input_sha256']
            assert extra['run']['dataset_sha256'] == ev['dataset_sha256']
            assert sorted(row['sample_id'] for row in extra['per_image']) == ids
            assert extra['totals']['gt_objects'] == total_instances
            assert all(row['boundary_tolerance_pixels'] == 5 for row in extra['per_image'])
            assert run['seed'] == 17
            assert run['steps'] == (600 if version == 'before' else 800)
            assert ev['model']['arch'] == ('tiny' if version == 'before' else 'r50_v2')
            assert run['exit_status'] == 'completed'
            protocols.append(extra['protocol'])
            plotted['values'].append({'variant': mode, 'version': version,
                'mask_ap': ev['metrics']['mask_ap'], 'mask_ap_small': ev['metrics']['mask_ap_small'],
                'boundary_f1_macro': extra['metrics_macro']['boundary_f1'],
                'seed': run['seed'], 'steps': run['steps'], 'architecture': ev['model']['arch'],
                'evaluation_short_side': ev['model']['size'], 'checkpoint_sha256': ev['checkpoint_sha256']})
            for kind, path in [('coco', ev_path), ('boundary', extra_path), ('run', run_path)]:
                plotted['sources'][f'{version}_{mode}_{kind}'] = {'path': str(path.relative_to(ROOT)), 'sha256': digest(path)}
    assert all(protocol == protocols[0] for protocol in protocols)
    plotted['boundary_protocol'] = protocols[0]
    proof['metric_source_checks'] = 'COCO/checkpoint/run hashes agree within each model; every Boundary F1 file uses the same protocol and exactly these four sample IDs.'
    (OUT / 'development_comparison_proof.json').write_text(json.dumps(proof, indent=2) + '\n')
    (OUT / 'performance_comparison_values.json').write_text(json.dumps(plotted, indent=2) + '\n')

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.labelsize': 10, 'axes.titlesize': 12,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, (left, right) = plt.subplots(1, 2, figsize=(13.8, 6), gridspec_kw={'width_ratios': [1.45, 1]})
    colors = {'before': '#A1A9B3', 'after': '#0072B2'}
    x = np.arange(3)
    records = {(row['version'], row['variant']): row for row in plotted['values']}
    width = .18
    specifications = [('before', 'mask_ap', '', -1.5), ('after', 'mask_ap', '', -.5),
                      ('before', 'mask_ap_small', '///', .5), ('after', 'mask_ap_small', '///', 1.5)]
    for version, metric, hatch, offset in specifications:
        values = [records[version, mode][metric] for mode in MODES]
        bars = left.bar(x + offset * width, values, width, color=colors[version],
                        edgecolor='#273746', linewidth=.6, hatch=hatch, zorder=3)
        left.bar_label(bars, labels=[f'{value:.2f}' if value else '0' for value in values],
                       padding=4, fontsize=8.5)
    left.set_title('(a) Mask average precision', loc='left', fontweight='bold', pad=13)
    left.legend(handles=[Patch(facecolor=colors[version], edgecolor='#273746', hatch=hatch,
                                label=f'{version.title()} · {label}')
                         for version, label, hatch in [('before', 'AP', ''), ('after', 'AP', ''),
                                                       ('before', r'AP$_s$', '///'), ('after', r'AP$_s$', '///')]],
                loc='upper center', ncol=2, frameon=False, fontsize=9, columnspacing=1.7)
    for offset, version in [(-.18, 'before'), (.18, 'after')]:
        values = [records[version, mode]['boundary_f1_macro'] for mode in MODES]
        bars = right.bar(x + offset, values, .32, color=colors[version], edgecolor='#273746', linewidth=.6, zorder=3)
        right.bar_label(bars, labels=[f'{value:.2f}' for value in values], padding=4, fontsize=9)
    right.set_title('(b) Boundary F1 (image macro mean)', loc='left', fontweight='bold', pad=13)
    right.legend(handles=[Patch(facecolor=colors[v], edgecolor='#273746', label=v.title()) for v in colors],
                 loc='upper center', ncol=2, frameon=False, fontsize=9)
    for ax in (left, right):
        ax.set_ylim(0, 100)
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_ylabel('Score (%)')
        ax.set_xticks(x, ['E0\nRGB', 'E2\nRGB-D sum', 'E3\nRGB-D gate'])
        ax.tick_params(axis='both', length=0, pad=7)
        ax.grid(axis='y', color='#DFE4E8', linewidth=.7, zorder=0)
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_color('#AAB2BA')
    fig.suptitle('GraspNet development set: measured before / after upgrade', x=.07, y=.97,
                 ha='left', fontsize=15, fontweight='bold')
    fig.text(.07, .909, 'Same 4 RGB-D frames and instance masks, verified byte-for-byte · scene 17 · 41 objects, including 4 small',
             ha='left', fontsize=10, color='#45515E')
    fig.text(.07, .137,
             'Before: 8 training images, random Tiny-FPN, 600 steps.   After: 64 training images, pretrained R50-FPN v2, 800 steps.',
             fontsize=9, color='#374151')
    fig.text(.07, .098,
             'Composite upgrade also changes resolution, augmentation and optimization; improvements cannot be attributed to depth fusion alone.',
             fontsize=9, color='#374151')
    fig.text(.07, .059,
             'Single seed (17); development results from one scene. AP$_s$ uses original mask area. Boundary F1: score/mask threshold 0.5, tolerance 5 px.',
             fontsize=9, color='#374151')
    fig.subplots_adjust(left=.07, right=.98, top=.83, bottom=.26, wspace=.22)
    fig.savefig(OUT / 'performance_comparison.png', dpi=240, facecolor='white')
    fig.savefig(OUT / 'performance_comparison.pdf', facecolor='white')
    plt.close(fig)

    lines = ['# Development-set comparison verification', '',
             'The before/after figure compares the same four development samples. All RGB, raw depth, and instance-mask files match byte-for-byte, with identical class mappings and depth scale.', '',
             '| Sample | RGB/depth/mask bytes | Instances | Small |',
             '|---|---|---:|---:|']
    lines.extend(f"| {row['sample_id']} | identical | {row['instances']} | {row['small_instances']} |" for row in proof['samples'])
    lines.extend(['', 'The complete dataset fingerprints differ because the training and test sets were expanded. This does not invalidate the paired development comparison: the four selected input/target samples remain identical.', '',
                  'All plotted values are read directly from recorded COCO and Boundary F1 JSON files. Checkpoint identities agree with their completed training runs, and all six Boundary F1 results use the same metric protocol. Exact file hashes and numeric values are saved in `development_comparison_proof.json` and `performance_comparison_values.json`.', '',
                  'This is a composite system upgrade: training images increase from 8 to 64, the backbone changes from random Tiny-FPN to pretrained R50-FPN v2, and resolution, augmentation, optimization and step budget also change. The comparison establishes a measured improvement of the overall system on this development scene, without isolating the effect of depth fusion. One seed and four small objects do not support a statistical significance claim.', ''])
    (OUT / 'development_comparison_proof.md').write_text('\n'.join(lines))
    print(json.dumps({'figure': str(OUT / 'performance_comparison.png'), 'verified_samples': len(ids), 'values': plotted['values']}, indent=2))


if __name__ == '__main__':
    main()
