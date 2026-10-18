#!/usr/bin/env python3
"""Render the finished test table with separate Latin and CJK font fallback."""
import json
import os
from pathlib import Path
import warnings

os.environ['MPLCONFIGDIR'] = '/tmp/course-test-comparison-mpl'
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/course_all_test_comparison'
font_manager.fontManager.addfont('/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf')
plt.rcParams['font.family'] = ['DejaVu Sans', 'Droid Sans Fallback']
plt.rcParams['axes.unicode_minus'] = False
report = json.loads((OUT/'comparison.json').read_text())
rows = report['rows']
y = np.arange(len(rows))
fig, ax = plt.subplots(figsize=(14, 12), layout='constrained')
best = max(range(len(rows)), key=lambda i: rows[i]['test_mask_APs'])
ax.axhspan(best-.49, best+.49, color='#e7f1e9', zorder=0)
ax.barh(y-.18, [r['test_mask_AP'] for r in rows], height=.33, color='#356990', label='Mask AP（整体）')
ax.barh(y+.18, [r['test_mask_APs'] for r in rows], height=.33, color='#dc8c35', label='Mask APs（小目标）')
for i, row in enumerate(rows):
    ax.text(row['test_mask_AP']+.009, i-.18, f"{row['test_mask_AP']:.4f}", va='center', fontsize=9)
    ax.text(row['test_mask_APs']+.009, i+.18, f"{row['test_mask_APs']:.4f}", va='center', fontsize=9)
ax.set_yticks(y, [r['method'] for r in rows], fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, 1)
ax.set_xticks(np.arange(0, 1.01, .1))
ax.grid(axis='x', alpha=.2)
ax.set_axisbelow(True)
ax.set_title('课程完整测试集：18 个配置的 AP 与 APs', fontsize=17, pad=18)
ax.set_xlabel('COCO Mask AP@[0.50:0.95] · 0–1 · 2,112 张图片', fontsize=11)
ax.legend(loc='lower right', fontsize=11)
for side in ('top', 'right'):
    ax.spines[side].set_visible(False)
with warnings.catch_warnings(record=True) as captured:
    warnings.simplefilter('always')
    fig.savefig(OUT/'comparison.png', dpi=200)
    fig.savefig(OUT/'comparison.pdf')
    missing = [str(w.message) for w in captured if 'Glyph' in str(w.message)]
    if missing:
        raise RuntimeError('Font rendering incomplete: '+str(missing))
plt.close(fig)
print(json.dumps({'status': 'rendered', 'configurations': len(rows), 'missing_glyph_warnings': 0,
                  'png': str(OUT/'comparison.png'), 'pdf': str(OUT/'comparison.pdf')}))
