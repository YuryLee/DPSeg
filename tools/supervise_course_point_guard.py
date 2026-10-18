#!/usr/bin/env python3
"""Finish one fixed guard extension after the live original point-head run."""
from __future__ import annotations

import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_distributed_eval import write_json
from rgbd_seg.course_optimized import file_sha


def read(path):return json.loads(Path(path).read_text())


def now():return datetime.now(timezone.utc).isoformat()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--original-supervisor',type=Path,required=True)
    parser.add_argument('--timeout-hours',type=float,default=4.)
    args=parser.parse_args()
    root=args.protocol.resolve().parent;experiment=args.original_supervisor.resolve().parent/'experiment'
    state_path=root/'supervisor.json'
    if state_path.exists():raise FileExistsError('A guard supervisor already exists; inspect its live handle before acting')
    state={'status':'waiting_original_validation_selection','pid':os.getpid(),'created_at_utc':now(),
        'protocol':str(args.protocol.resolve()),'protocol_sha256':file_sha(args.protocol),
        'original_supervisor':str(args.original_supervisor.resolve()),'timeout_hours':args.timeout_hours,
        'source_sha256':{str(Path(__file__).relative_to(ROOT)):file_sha(__file__)},'children':[]}
    write_json(state_path,state)
    print(json.dumps({'status':state['status'],'pid':os.getpid(),'state':str(state_path)}),flush=True)
    started=time.monotonic();missing_owner_ticks=0

    def alive():
        nonlocal missing_owner_ticks
        if time.monotonic()-started>args.timeout_hours*3600:
            raise TimeoutError('Bounded guard supervision deadline reached')
        original=read(args.original_supervisor)
        if original['status'] in ('failed','timed_out'):raise RuntimeError('Original point-head experiment failed')
        if original['status']!='completed':
            try:os.kill(original['supervisor_pid'],0);missing_owner_ticks=0
            except ProcessLookupError:missing_owner_ticks+=1
            if missing_owner_ticks>=3:raise RuntimeError('Original supervisor no longer live and no completion receipt is available')
        state['last_original_state']=original['status'];state['last_checked_at_utc']=now()
        state['elapsed_seconds']=time.monotonic()-started
        write_json(state_path,state)
        return original

    def await_file(path):
        while not path.exists():alive();time.sleep(20)

    def run_stage(input_path,output_path):
        command=[sys.executable,str(ROOT/'tools/run_course_point_guard.py'),'--protocol',str(args.protocol.resolve()),
                 '--input',str(input_path),'--output',str(output_path)]
        log=root/f'{output_path.name}.log'
        with log.open('x') as stream:
            proc=subprocess.Popen(command,cwd=ROOT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2'},
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            record={'pid':proc.pid,'command':command,'log':str(log),'started_at_utc':now()}
            state['children'].append(record);write_json(state_path,state)
            began=time.monotonic()
            while proc.poll() is None:
                if time.monotonic()-began>30*60:
                    os.killpg(proc.pid,signal.SIGTERM)
                    try:proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                    raise TimeoutError('Guard stage exceeded its fixed 30-minute timeout')
                alive();time.sleep(20)
            record.update(exit_code=proc.wait(),finished_at_utc=now());write_json(state_path,state)
            if record['exit_code']!=0:raise RuntimeError(f'Guard stage failed; see {log}')

    try:
        await_file(experiment/'selection.json')
        original_selection=read(experiment/'selection.json');checkpoint=original_selection['selected']['checkpoint']
        epoch=checkpoint['epoch'];val_input=experiment/f'val_epoch_{epoch:02d}'
        val_output=root/f'val_epoch_{epoch:02d}'
        if (val_output/'receipt.json').exists():
            existing=read(val_output/'receipt.json')
            if existing['checkpoint']!=checkpoint or existing['guard_protocol_sha256']!=state['protocol_sha256']:
                raise ValueError('Existing guard validation belongs to another checkpoint/protocol')
        else:
            if val_output.exists():raise RuntimeError('Partial guard validation exists; no automatic retry')
            state['status']='scoring_selected_epoch_validation';write_json(state_path,state)
            run_stage(val_input,val_output)
        validation=read(val_output/'receipt.json')
        selection={'status':'frozen','selection_split':'val','checkpoint':checkpoint,
            'guard_protocol_sha256':state['protocol_sha256'],
            'original_selection_sha256':file_sha(experiment/'selection.json'),
            'validation_receipt':{'path':str((val_output/'receipt.json').resolve()),'sha256':file_sha(val_output/'receipt.json')},
            'validation_eligible':validation['validation_eligible'],
            'guard_thresholds_searched':False,'test_used_for_selection':False,'created_at_utc':now()}
        if (root/'selection.json').exists():raise FileExistsError('Guard selection is already frozen')
        write_json(root/'selection.json',selection)
        state['status']='waiting_original_completed_test';write_json(state_path,state)
        while alive()['status']!='completed':time.sleep(20)
        state['status']='scoring_guard_test';write_json(state_path,state)
        run_stage(experiment/'test',root/'test')
        receipt=read(root/'test/receipt.json')
        csv=receipt['submission']
        if receipt['checkpoint']!=checkpoint or csv['rows']!=2112 or file_sha(csv['path'])!=csv['sha256']:
            raise ValueError('Guard test checkpoint/CSV binding differs')
        artifacts=[args.protocol,root/'selection.json',val_output/'receipt.json',root/'test/receipt.json',
            root/'test/evaluation.json',root/'test/submission_validation.json',root/'test/predictions.jsonl',Path(csv['path'])]
        state.update(status='completed',exit_code=0,finished_at_utc=now(),
            artifact_sha256={str(path.resolve()):file_sha(path) for path in artifacts})
    except BaseException as error:
        state.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error),finished_at_utc=now())
        raise
    finally:
        write_json(state_path,state);write_json(root/'exit.json',{'status':state['status'],'exit_code':state.get('exit_code',1)})
    print(json.dumps({'status':state['status'],'selected_head_epoch':epoch,'validation_eligible':selection['validation_eligible']}),flush=True)


if __name__=='__main__':main()
