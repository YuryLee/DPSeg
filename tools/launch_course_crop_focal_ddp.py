#!/usr/bin/env python3
"""Run frozen crop-only then Focal-only trials, with bounded process supervision."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.launch_course_resolution_ddp import now, write_json


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_protocols(paths):
    protocols = [json.loads(path.read_text()) for path in paths]
    if [p['intervention']['method'] for p in protocols] != ['crop','focal']:
        raise ValueError('Provide exactly two independent protocols: crop, then focal')
    for protocol in protocols:
        if protocol['status'] != 'frozen':
            raise ValueError('Formal launch requires frozen protocols after GPU engineering checks')
        for field in ('resolution_selection','engineering_validation','parent_resolution_protocol'):
            reference = protocol[field]
            if sha(reference['path']) != reference['sha256']:
                raise ValueError(f'{field} identity changed')
        decision = json.loads(Path(protocol['resolution_selection']['path']).read_text())
        if (decision['status'] != 'completed' or decision['selected_resolution'] not in (800,1024)
                or decision['selected_resolution'] != protocol['selected_resolution']
                or decision['protocol_sha256'] != protocol['parent_resolution_protocol']['sha256']):
            raise ValueError('Completed resolution selection does not match the intervention')
        engineering = json.loads(Path(protocol['engineering_validation']['path']).read_text())
        method = protocol['intervention']['method']
        if (engineering['status'] != 'passed'
                or engineering.get('scope') != 'seven_gpu_crop_focal_smoke'
                or engineering.get('source_sha256') != protocol['source_sha256']
                or engineering.get('resolution') != protocol['selected_resolution']
                or engineering.get('world_size') != len(protocol['hardware']['physical_gpus'])
                or engineering.get('per_rank_batch_sizes') != protocol['training']['per_rank_batch_sizes']
                or engineering.get('methods',{}).get(method,{}).get('status') != 'passed'):
            raise ValueError('GPU engineering validation has not passed')
        for name,expected in protocol['source_sha256'].items():
            path = Path(name) if Path(name).is_absolute() else ROOT/name
            if sha(path) != expected:
                raise ValueError(f'Frozen source changed: {name}')
    for field in ('hardware','archive','checkpoint','input_sha256','resolution_selection','selected_resolution','evaluation'):
        if protocols[0][field] != protocols[1][field]:
            raise ValueError(f'Independent arms have different {field}')
    settings = [{k:v for k,v in p['training'].items() if k != 'augmentation'} for p in protocols]
    if settings[0] != settings[1]:
        raise ValueError('Independent arms have different training budgets or optimizer settings')
    return protocols


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocols',type=Path,nargs=2,required=True,metavar=('CROP','FOCAL'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--control',type=Path,required=True)
    parser.add_argument('--supervise',action='store_true')
    args = parser.parse_args()
    protocols = load_protocols(args.protocols)
    hashes = [sha(path) for path in args.protocols]
    if not args.supervise:
        if args.output.exists() and any(args.output.iterdir()):
            raise FileExistsError(args.output)
        args.control.mkdir(parents=True,exist_ok=False)
        command = [sys.executable,'-u',str(Path(__file__).resolve()),'--protocols',
            *[str(path.resolve()) for path in args.protocols],
            '--output',str(args.output.resolve()),'--control',str(args.control.resolve()),'--supervise']
        with (args.control/'supervisor.log').open('x') as stream:
            process = subprocess.Popen(command,cwd=ROOT,stdin=subprocess.DEVNULL,
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
        receipt = {'supervisor_pid':process.pid,'command':command,'started_at_utc':now(),
                   'protocol_sha256':hashes}
        write_json(args.control/'supervisor_start.json',receipt)
        print(json.dumps(receipt),flush=True)
        return 0
    record = {'status':'starting','supervisor_pid':os.getpid(),'started_at_utc':now(),
              'protocol_sha256':hashes,'trials':[]}
    code = 1
    try:
        for path,protocol,expected_hash in zip(args.protocols,protocols,hashes):
            if sha(path) != expected_hash:
                raise ValueError('Protocol changed while supervisor was running')
            method = protocol['intervention']['method']
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(protocol['hardware']['physical_gpus']),
                OMP_NUM_THREADS=str(protocol['training']['threads']),PYTHONUNBUFFERED='1')
            command = [sys.executable,'-u','-m','torch.distributed.run','--standalone','--nnodes=1',
                f'--nproc-per-node={len(protocol["hardware"]["physical_gpus"])}','--max-restarts=0',
                str(ROOT/'tools/run_course_crop_focal_ddp.py'),'--protocol',str(path.resolve()),
                '--output',str(args.output.resolve()/method),'--resolution',str(protocol['selected_resolution'])]
            with (args.control/f'console_{method}.log').open('x') as stream:
                process = subprocess.Popen(command,cwd=ROOT,env=env,stdin=subprocess.DEVNULL,
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                trial = {'method':method,'pid':process.pid,'command':command,
                         'started_at_utc':now(),'status':'running'}
                record['trials'].append(trial)
                record.update(status='running',active_method=method)
                write_json(args.control/'process.json',record)
                try:
                    code = process.wait(timeout=protocol['limits']['external_trial_seconds'])
                except subprocess.TimeoutExpired:
                    trial['hard_timeout'] = True
                    os.killpg(process.pid,signal.SIGTERM)
                    try:
                        code = process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL)
                        code = process.wait()
                trial.update(status='completed' if code == 0 else 'failed',exit_code=code,finished_at_utc=now())
                write_json(args.control/'process.json',record)
                if code:
                    break
    except BaseException as error:
        code = 1
        record['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        record.update(status='completed' if code == 0 else 'failed',exit_code=code,finished_at_utc=now())
        write_json(args.control/'exit.json',record)
    return code


if __name__ == '__main__':
    raise SystemExit(main())
