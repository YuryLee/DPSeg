#!/usr/bin/env python3
"""Read actual training processes and progress; never launch or restart jobs."""
import json,os
from pathlib import Path
root=Path(__file__).resolve().parents[1]/'reports/course_longer_training'
active_path=root/'active_base_run.json'
active=json.loads(active_path.read_text()) if active_path.exists() else {
    'control':str(root/'control_v1'),'output':str(root/'formal_v1'),
    'runner':'tools/run_course_longer_training.py'}
control=json.loads((Path(active['control'])/'process.json').read_text())
trial=control['trials'][-1]
pid=trial.get('pid')
try:
    if not isinstance(pid,int) or pid<=0:raise ProcessLookupError()
    os.kill(pid,0)
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes().replace(b'\0',b' ').decode()
    alive=active['runner'] in cmd
except (ProcessLookupError,FileNotFoundError):
    alive=False
p=Path(active['output'])/trial['arm']
r=json.loads((p/'run.json').read_text()) if (p/'run.json').exists() else {'status':'initializing'}
lines=(p/'training_log.jsonl').read_bytes().splitlines(keepends=True) if (p/'training_log.jsonl').exists() else []
last=next((s for s in reversed(lines) if s.endswith(b'\n')),None)
v=json.loads(last) if last is not None else {}
out={'pid':pid,'actual_training_process_alive':alive,'arm':trial['arm'],'phase':r['status'],'epoch':r.get('epoch'),'supervisor_trial_status':trial['status']}
stop_path=root/f"{trial['arm']}_user_stop.json"
if not alive and stop_path.exists() and json.loads(stop_path.read_text()).get('status')=='stopped_by_user':
    out['phase']='stopped_by_user'
out['scope']=r.get('scope')
for output_key,input_key in [('step','global_step'),('training_images','images_in_epoch'),('training_total','expected_train_images'),('loss','loss')]:
    if input_key in v:out[output_key]=v[input_key]
if v and r.get('epoch') != v.get('epoch'):
    # A new epoch is announced before its first update is logged.
    # Do not present the preceding epoch's completed images/loss as the new one.
    out['latest_training_log_epoch']=v.get('epoch')
    out['training_images']=0
    out.pop('loss',None)
val=p/f"val_epoch_{r.get('epoch',0):02d}"
progress=[json.loads(f.read_text()) for f in val.glob('progress_rank*.json')]
if progress:
    out['validation_images']=sum(v['images'] for v in progress)
    out['validation_total']=2584
if (val/'progress.json').exists():
    out['validation_phase']=json.loads((val/'progress.json').read_text())['phase']
if (p/'selection.json').exists():
    selection=json.loads((p/'selection.json').read_text())
    out['best_epoch']=selection['best_epoch']
    out['best_APs']=selection['best_key'][0]
    out['best_AP']=selection['best_key'][1]
    out['completed_validation_epochs']=[v['epoch'] for v in selection['epochs']]
print(json.dumps(out))

q=root/'point_continuation_e05_v1/formal'
if (q/'supervisor.json').exists():
    s=json.loads((q/'supervisor.json').read_text());hp=s.get('pid')
    try:
        if not isinstance(hp,int) or hp<=0:raise ProcessLookupError()
        os.kill(hp,0)
        hc=Path(f'/proc/{hp}/cmdline').read_bytes().decode().replace(chr(0),' ')
        ha='tools/run_course_point_continuation.py' in hc
    except (ProcessLookupError,FileNotFoundError):
        ha=False
    h={'branch':'point_continuation_e05','pid':hp,'actual_training_process_alive':ha,'supervisor_status':s['status']}
    if (q/'experiment/progress_rank0.json').exists():
        z=json.loads((q/'experiment/progress_rank0.json').read_text())
        h.update({k:z[k] for k in ('phase','epoch','epochs','images','expected_images','global_step','point_loss','depth_point_loss') if k in z})
        vf=q/'experiment'/f"val_epoch_{z['epoch']:02d}"
        vp=[json.loads(f.read_text()) for f in vf.glob('progress_rank*.json')]
        if vp:
            h.update(phase='validating',validation_phase='inferring',validation_images=sum(v['images'] for v in vp))
            h['validation_total']=len(json.loads((vf/'image_ids.json').read_text()))
        if (vf/'progress.json').exists():
            stage=json.loads((vf/'progress.json').read_text())
            h.update(phase='validating',validation_phase=stage['phase'])
        if (vf/'comparison.json').exists():
            h.update(phase='validation_complete')
        runfile=q/'experiment/run.json'
        if runfile.exists() and json.loads(runfile.read_text()).get('status')=='completed':
            h['phase']='completed'
    hist=q/'experiment/validation_history.json'
    h['completed_validation_epochs']=[v['epoch'] for v in json.loads(hist.read_text())] if hist.exists() else []
    print(json.dumps(h))

point_pointer=root/'active_point_run.json'
if point_pointer.exists():
    active_point=json.loads(point_pointer.read_text())
    q=Path(active_point['output'])
    if (q/'supervisor.json').exists():
        s=json.loads((q/'supervisor.json').read_text());hp=s.get('pid')
        try:
            if not isinstance(hp,int) or hp<=0:raise ProcessLookupError()
            os.kill(hp,0)
            hc=Path(f'/proc/{hp}/cmdline').read_bytes().decode().replace(chr(0),' ')
            ha='tools/run_course_longer_point.py' in hc
        except (ProcessLookupError,FileNotFoundError):
            ha=False
        h={'branch':active_point['name'],'base_epoch':active_point['base_epoch'],
           'pid':hp,'actual_training_process_alive':ha,'supervisor_status':s['status'],
           'gpus':s['gpus'],'scope':s['scope']}
        progress=[json.loads(f.read_text()) for f in (q/'experiment').glob('progress_rank*.json')]
        if progress:
            epoch=max(v['epoch'] for v in progress)
            current=[v for v in progress if v['epoch']==epoch]
            h.update(phase='training',epoch=epoch,epochs=current[0]['epochs'],
                     training_images=sum(v['images'] for v in current),
                     training_total=sum(v['expected_images'] for v in current),
                     progress_ranks_reported=len(current),world_size=len(s['gpus']),
                     global_step=max(v['global_step'] for v in current))
            vf=q/'experiment'/f'val_epoch_{epoch:02d}'
            vp=[json.loads(f.read_text()) for f in vf.glob('progress_rank*.json')]
            if vp:
                h.update(phase='validating',validation_phase='inferring',
                         validation_images=sum(v['images'] for v in vp))
                h['validation_total']=len(json.loads((vf/'image_ids.json').read_text()))
            if (vf/'progress.json').exists():
                h.update(phase='validating',validation_phase=json.loads((vf/'progress.json').read_text())['phase'])
            if (vf/'comparison.json').exists():h['phase']='validation_complete'
        runfile=q/'experiment/run.json'
        if runfile.exists() and json.loads(runfile.read_text()).get('status')=='completed':h['phase']='completed'
        hist=q/'experiment/validation_history.json'
        h['completed_validation_epochs']=[v['epoch'] for v in json.loads(hist.read_text())] if hist.exists() else []
        print(json.dumps(h))
