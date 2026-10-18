#!/usr/bin/env python3
"""Final requirement-by-requirement audit of the completed optimization delivery."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.export_course_selected import checked_protocol
from tools.select_course_resolution import read, sha, require


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--delivery',type=Path,required=True)
    args=parser.parse_args()
    output=args.delivery.resolve(); test=output/'test_v1'
    require(not (output/'completion_audit.json').exists(),'Completion audit already exists')
    p,d=checked_protocol(args.protocol)
    evidence={}
    def record(name,path):
        evidence[name]={'path':str(Path(path).resolve()),'sha256':sha(path)}
        return read(path)
    resolution=record('resolution_decision',ROOT/'reports/course_resolution_ddp/decision.json')
    interventions=record('independent_crop_focal',ROOT/'reports/course_crop_focal/decision.json')
    require(resolution['status']==interventions['status']=='completed'
            and not resolution['test_used'] and not interventions['test_used'],'Earlier stages incomplete')
    require(resolution['selected_resolution']==1024 and interventions['resolution']==1024
            and interventions['recommended_method_for_next_stage']=='native','Stage sequence differs')
    require(interventions['resolution_decision_sha256']==sha(evidence['resolution_decision']['path']),
            'Earlier stage identity differs')
    for stage,decision in [('course_resolution_ddp',resolution),('course_crop_focal',interventions)]:
        terminal=record(stage+'_exit',ROOT/f'reports/{stage}/launch_v1/exit.json')
        require(terminal['status']=='completed' and terminal['exit_code']==0,'Training did not exit0')
        require(sha(evidence[stage+'_exit']['path'])==decision['supervisor_exit_sha256'],'Training exit changed')
    initializations=set()
    for key,trial in {**resolution['trials'],**interventions['trials']}.items():
        folder=Path(trial['best_checkpoint']).parent
        trial_run=record('training_'+key,folder/'run.json')
        selection=read(folder/'selection.json')
        require(sha(folder/'run.json')==trial['run_sha256']
                and sha(folder/'selection.json')==trial['selection_sha256'],'Training evidence changed')
        require(trial_run['status']=='completed' and trial_run['world_size']==7
                and trial_run['effective_batch_size']==33 and trial_run['optimizer_steps']==2190
                and trial_run['training_image_exposures']==72160 and trial_run['epoch']==5
                and len(selection['epochs'])==5 and not trial_run['test_used'],'Training budget incomplete')
        initializations.add(trial_run['initial_model_sha256'])
        for epoch in selection['epochs']:
            val=Path(epoch['output']); receipt=read(val/'evaluation_run.json')
            require(receipt['status']=='completed' and receipt['split']=='val' and receipt['images']==2584
                    and sha(val/'evaluation.json')==receipt['evaluation_sha256']
                    and read(val/'evaluation.json')['metrics']==epoch['metrics'],'Per-epoch validation incomplete')
    require(len(initializations)==1,'Training initializations differ')
    for method in ['crop','focal']:
        audit=record(method+'_actual_training',ROOT/f'reports/course_crop_focal/completed_{method}_actual_training_audit.json')
        require(audit['status']=='passed' and audit['optimizer_updates']==2190
                and audit['training_image_exposures']==72160 and audit['finite_loss_and_gradients']
                and [x['epoch'] for x in audit['counts']]==[1,2,3,4,5]
                and all(x['images']==14432 for x in audit['counts']), 'Actual training incomplete')
        expected=[4243,4180,4255,4258,4205] if method=='crop' else [0]*5
        require([x['cropped'] for x in audit['counts']]==expected,'Actual crop intervention differs')
        folder=ROOT/f'reports/course_crop_focal/formal_v1/{method}'
        for name,digest in audit['actual_forward_log_sha256'].items():
            require(sha(folder/name)==digest,'Actual forward log changed')
    depth=record('depth_decision',p['decision']['path'])
    depth_protocol=record('depth_protocol',depth['protocol']['path'])
    require(depth_protocol['gpus']==[0,1,2,3,5,6,7]
            and depth['selected_method']=='depth' and depth['variants']['depth']['eligible'], 'Final gate not passed')
    training=record('selected_training_protocol',depth['baseline_protocol']['path'])
    require(training['training']['per_rank_batch_sizes']==[6,6,6,6,3,3,3]
            and training['training']['effective_batch_size']==33
            and training['training']['epochs']==5,'Training GPU/batch budget differs')
    run=record('test_run',test/'run.json'); terminal=record('test_exit',test/'exit.json')
    require(run['status']==terminal['status']=='completed' and run['exit_code']==terminal['exit_code']==0
            and terminal['worker_exit_codes']==[0]*7,'Test export not complete')
    require(run['protocol_sha256']==sha(args.protocol) and run['source_sha256']==p['source_sha256']
            and run['selected_method']==depth['selected_method'] and not run['test_ground_truth_used']
            and not run['test_scored'],'Test export identity/scope differs')
    for name,key in [('submission.csv','submission_sha256'),('predictions.jsonl','predictions_sha256'),
                     ('submission_validation.json','submission_validation_sha256')]:
        require(sha(test/name)==run[key],f'Export output changed: {name}')
        evidence[name]={'path':str(test/name),'sha256':run[key]}
    native=read(test/'submission_validation.json')
    require(native['rows']==run['images']==2112 and native['predictions']==run['predicted_instances']
            and native['image_size']==[1024,1024] and native['category_ids']==[1]
            and native['max_detections']==100 and native['sha256']==run['submission_sha256'],
            'CSV native validation scope differs')
    require(all(native[k] for k in ['id_order_verified','rle_runs_verified','rle_native_decode_verified']),
            'Native mask validation missing')
    with ZipFile(p['archive']) as archive: template=archive.read('submission.csv')
    ids=[int(x['image_id']) for x in csv.DictReader(io.StringIO(template.decode('utf-8-sig')))]
    require(len(ids)==len(set(ids))==2112 and ids==read(test/'image_ids.json'),'Official template mismatch')
    csv.field_size_limit(100_000_000)
    rows=count=0
    with (test/'submission.csv').open(newline='') as stream,(test/'predictions.jsonl').open() as raw:
        reader=csv.DictReader(stream)
        require(reader.fieldnames==['image_id','PredictionString'],'CSV columns differ')
        for index,row in enumerate(reader):
            saved=json.loads(next(raw)); values=json.loads(row['PredictionString'])
            require(int(row['image_id'])==saved['image_id']==ids[index]
                    and values==saved['predictions'],'CSV/JSONL semantic mismatch')
            rows+=1;count+=len(values)
        require(next(raw,None) is None and rows==2112 and count==native['predictions'],'CSV coverage differs')
    for rank in range(7):
        reload=record(f'reload_rank{rank}',test/f'reload_rank{rank}.json')
        require(reload['status']=='passed' and len(reload['checks'])==2
                and reload['physical_gpu']==p['gpus'][rank]
                and reload['reload']['checkpoint_sha256']==d['baseline']['best_checkpoint_sha256'],
                'Per-GPU selected-weight reload missing')
    summary=record('comparison_summary',output/'comparison_summary.json')
    require(summary['status']=='completed' and summary['decision']==p['decision']
            and summary['selected_metrics']==d['selected_metrics'],'Report decision mismatch')
    require(summary['script_sha256']==sha(ROOT/'tools/build_course_optimization_delivery.py'),
            'Demo generator changed since rendering')
    gallery=record('gallery',output/'gallery/gallery.json')
    require(gallery['summary_sha256']==sha(output/'comparison_summary.json')
            and gallery['html_sha256']==sha(output/'gallery/index.html') and gallery['split']=='val',
            'Demo artifact changed')
    for name in ['browser_test','zoom_test']:
        browser=record(name,output/f'gallery/browser/{name}.json')
        require(browser['status']=='passed' and not browser['js_errors']
                and not browser['external_requests'],'Browser checks failed')
        require(browser['viewport_widths']==[1500,390,320],'Browser viewport coverage incomplete')
        if name=='browser_test':
            require(len(browser['states'])==18,'Six cases at three display thresholds not checked')
        else:
            require(browser['native_canvas_pixels_unchanged'] and browser['metrics_unchanged'],
                    'Zoom altered underlying image or metrics')
    record('visual_review',output/'visual_review.json')
    for name in ['comparison_report.md','comparison_metrics.png','gallery/index.html']:
        evidence[name]={'path':str(output/name),'sha256':sha(output/name)}
    requirements=[{'requirement':name,'status':'verified','evidence':keys} for name,keys in [
        ('Matched800/1024 resolution experiment and selection',['resolution_decision','course_resolution_ddp_exit']),
        ('Independent crop and Focal trials, full training and validation',['independent_crop_focal','crop_actual_training','focal_actual_training','course_crop_focal_exit']),
        ('Seven available GPUs and global training batch33',['selected_training_protocol']),
        ('Depth-boundary hypothesis, spatial control, full validation and frozen adoption gate',['depth_protocol','depth_decision']),
        ('Selected model test CSV,2112 official IDs,1024 RLE,max100,complete native validation',['test_run','test_exit','submission.csv','submission_validation.json']),
        ('Reloaded selected model on every export GPU',[f'reload_rank{i}' for i in range(7)]),
        ('Current report and offline comparison demo',['comparison_summary','gallery','browser_test','zoom_test'])]]
    result={'status':'completed','finished_at_utc':datetime.now(timezone.utc).isoformat(),
        'requirements':requirements,'evidence':evidence,'script_sha256':sha(__file__),
        'csv_semantic_comparison':{'rows':rows,'instances':count,'all_rows_equal':True,
            'template_sha256':hashlib.sha256(template).hexdigest()},
        'selected_method':d['selected_method'],'validation_metrics':d['selected_metrics'],
        'limits':['Audit executed inline by the implementing assistant, not an independent reviewer',
            'Full native RLE decoding was performed by the export validator; this audit checks its binding and full CSV/JSONL semantic equality',
            'No new Kaggle score; no test-label-based selection; single training seed'],
        'goal_requirements_complete':True}
    (output/'completion_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':'completed','requirements':len(requirements),'csv_rows':rows,'instances':count}))


if __name__=='__main__':main()
