#!/usr/bin/env python3
"""Bounded supervision for longer point training and complete validation only."""
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


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--timeout-hours',type=float,default=24.)
    args=parser.parse_args()
    protocol=json.loads(args.protocol.read_text())
    if protocol.get('evaluation_policy')!='validation_only':
        raise ValueError('Only validation-only protocols are accepted')
    from tools.launch_course_longer_training import check_devices
    check_devices([str(v) for v in protocol['gpus']])
    output=args.output.resolve()
    output.mkdir(parents=True,exist_ok=False)
    command=[sys.executable,'-m','torch.distributed.run','--standalone',
        f"--nproc_per_node={len(protocol['gpus'])}",str(ROOT/'tools/run_course_longer_point.py'),
        '--protocol',str(args.protocol.resolve()),'--output',str(output/'experiment')]
    env={**os.environ,'CUDA_VISIBLE_DEVICES':','.join(map(str,protocol['gpus'])),
         'OMP_NUM_THREADS':'4','PYTHONUNBUFFERED':'1','TOKENIZERS_PARALLELISM':'false'}
    state={'status':'starting','supervisor_pid':os.getpid(),'created_at_utc':datetime.now(timezone.utc).isoformat(),
        'protocol':str(args.protocol.resolve()),'protocol_sha256':file_sha(args.protocol),
        'command':command,'gpus':protocol['gpus'],'timeout_hours':args.timeout_hours,
        'scope':protocol['scope'],'log':str(output/'training.log')}
    started=time.monotonic()
    with (output/'training.log').open('x') as log:
        proc=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(status='running',pid=proc.pid)
        write_json(output/'supervisor.json',state)
        print(json.dumps({'status':'running','supervisor_pid':os.getpid(),'child_pid':proc.pid,'output':str(output)}),flush=True)
        while proc.poll() is None:
            state.update(elapsed_seconds=time.monotonic()-started,last_process_alive_at_utc=datetime.now(timezone.utc).isoformat())
            files=list((output/'experiment').glob('progress_rank*.json'))
            state['progress_files']={str(path.relative_to(output)):path.stat().st_mtime_ns for path in files}
            write_json(output/'supervisor.json',state)
            if time.monotonic()-started>args.timeout_hours*3600:
                state['status']='hard_timeout_terminating'
                write_json(output/'supervisor.json',state)
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=30)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                break
            time.sleep(30)
        code=proc.wait()
    state.update(exit_code=code,elapsed_seconds=time.monotonic()-started,
                 finished_at_utc=datetime.now(timezone.utc).isoformat())
    if code==0:
        final=json.loads((output/'experiment/comparison.json').read_text())
        if final['status']!='completed':
            raise RuntimeError('Process exited successfully without a completed experiment report')
        required=[output/'experiment/comparison.json',output/'experiment/comparison.md']
        if final['test_used'] or len(final['validation_history'])!=protocol['epochs']:
            raise RuntimeError('Final validation history is incomplete or test was used')
        expected=2584 if protocol['scope']=='full_experiment' else 2*len(protocol['gpus'])
        for entry in final['validation_history']:
            if entry['split']!='val' or entry['images']!=expected:
                raise RuntimeError('Validation coverage differs')
            checkpoint=Path(entry['checkpoint']['path'])
            if file_sha(checkpoint)!=entry['checkpoint']['sha256']:
                raise RuntimeError('Point checkpoint changed')
            required.append(checkpoint)
        if (output/'experiment/test').exists():
            raise RuntimeError('Validation-only run produced a test directory')
        state.update(status='completed',artifact_sha256={str(path):file_sha(path) for path in required})
    else:
        state['status']='failed' if state['status']!='hard_timeout_terminating' else 'timed_out'
    write_json(output/'supervisor.json',state)
    write_json(output/'exit.json',{'status':state['status'],'exit_code':code})
    print(json.dumps({'status':state['status'],'exit_code':code,'seconds':state['elapsed_seconds']}),flush=True)
    sys.exit(code if code else 0)


if __name__=='__main__':main()
