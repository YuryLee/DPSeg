#!/usr/bin/env python3
"""Read saved fine-tuning progress without touching a running experiment."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

root = Path(__file__).resolve().parents[1] / 'reports/course_finetune'
print(json.dumps({'at_utc': datetime.now(timezone.utc).isoformat(),
                  'free_GiB': round(shutil.disk_usage(root).free / 2**30, 2)}))
for name in ('maskrcnn', 'mask2former'):
    folder = root / name
    run = json.loads((folder / 'run.json').read_text())
    progress = json.loads((folder / 'progress.json').read_text())
    row = {'model': name, 'status': run['status'], 'epoch': run.get('epoch'),
           'stage': progress.get('status'), 'step': progress.get('global_step'),
           'images': progress.get('images_in_epoch', progress.get('images_completed')),
           'total_images': progress.get('expected_train_images', progress.get('expected_images'))}
    selection_path = folder / 'selection.json'
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        row['best_epoch'] = selection['best_epoch']
        row['val_APs'] = [round(e['metrics']['mask_ap_small'], 8) for e in selection['epochs']]
        row['val_AP'] = [round(e['metrics']['mask_ap'], 8) for e in selection['epochs']]
    if run.get('test_metrics'):
        row['test_metrics'] = run['test_metrics']
    if run.get('error'):
        row['error'] = run['error']
    print(json.dumps(row))
