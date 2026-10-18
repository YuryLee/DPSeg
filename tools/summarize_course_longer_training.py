#!/usr/bin/env python3
"""Summarize completed validation epochs without selecting on test results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_optimized import file_sha
from rgbd_seg.course_distributed_eval import write_json


def read(path):
    return json.loads(path.read_text())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'reports/course_longer_training')
    args=parser.parse_args()
    parent_path=ROOT/'reports/course_resolution_ddp/formal_v1/1024/selection.json'
    parent=read(parent_path)
    old=[{'epoch':r['epoch'],'AP':r['metrics']['mask_ap'],'APs':r['metrics']['mask_ap_small']}
         for r in parent['epochs']]
    result={'status':'partial','as_of_utc':datetime.now(timezone.utc).isoformat(),
        'split':'val','test_used':False,'historical_epochs':old,'arms':{}}
    for name in ('lr5e5','lr2e5'):
        folder=args.root/'formal_v1'/name
        if not (folder/'run.json').exists():
            result['arms'][name]={'status':'not_started','epochs':[]}
            continue
        run=read(folder/'run.json')
        arm={'status':run['status'],'epochs':[], 'run':str((folder/'run.json').resolve())}
        if (folder/'selection.json').exists():
            selection=read(folder/'selection.json')
            for entry in selection['epochs']:
                output=Path(entry['output'])
                metrics=read(output/'evaluation.json')
                receipt=read(output/'evaluation_run.json')
                if (receipt['status']!='completed' or receipt['images']!=2584
                        or metrics['scope']!='full_split' or metrics['metrics']!=entry['metrics']
                        or receipt['evaluation_sha256']!=file_sha(output/'evaluation.json')):
                    raise ValueError(f'Incomplete or inconsistent validation: {output}')
                arm['epochs'].append({'epoch':entry['epoch'],'AP':entry['metrics']['mask_ap'],
                    'APs':entry['metrics']['mask_ap_small'],'training_loss':entry['training_loss_mean'],
                    'model_sha256':entry['model_sha256'],'evaluation':str(output/'evaluation.json')})
            arm.update(best_epoch=selection['best_epoch'],best_APs=selection['best_key'][0],
                       best_AP=selection['best_key'][1])
        log=folder/'training_log.jsonl'
        if log.exists():
            rows=[json.loads(line) for line in log.read_text().splitlines()]
            if rows:
                arm['latest_update']=rows[-1]
                arm['median_seconds_per_update_last_20']=statistics.median(
                    r['step_seconds_with_data'] for r in rows[-20:])
        result['arms'][name]=arm
    if all(v['status']=='completed' for v in result['arms'].values()):
        result['status']='base_arms_completed_convergence_review_pending'
    write_json(args.root/'live_summary.json',result)
    if any(v['epochs'] for v in result['arms'].values()):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig,axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
        for axis,metric in zip(axes,('APs','AP')):
            axis.plot([r['epoch'] for r in old],[r[metric] for r in old],color='.4',marker='o',label='Original 5 epochs')
            for name,label in [('lr5e5','Continuation LR 5e-5'),('lr2e5','Continuation LR 2e-5')]:
                values=result['arms'][name]['epochs']
                if values:
                    values=[old[-1],*values]
                    axis.plot([r['epoch'] for r in values],[r[metric] for r in values],marker='o',label=label)
            axis.set(xlabel='Total training epochs',ylabel=f'Validation mask {metric}')
            axis.grid(alpha=.25)
            axis.legend(fontsize=8)
        fig.savefig(args.root/'validation_learning_curves.png',dpi=180)
        fig.savefig(args.root/'validation_learning_curves.pdf')
        plt.close(fig)
    print(json.dumps({'status':result['status'],'path':str(args.root/'live_summary.json'),
        'arms':{k:{'status':v['status'],'validated_epochs':len(v['epochs']),
                   'best_APs':v.get('best_APs')} for k,v in result['arms'].items()}}))


if __name__=='__main__':
    main()
