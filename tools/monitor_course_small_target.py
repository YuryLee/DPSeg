#!/usr/bin/env python3
"""Read-only host-process and artifact observation for the existing trial."""
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def probe(pid, command):
    try:
        os.kill(pid,0)
        directory=Path('/proc')/str(pid)
        arguments=(directory/'cmdline').read_bytes().rstrip(b'\0').split(b'\0')
        state=next(line.split()[1] for line in (directory/'status').read_text().splitlines()
                   if line.startswith('State:'))
        matched=arguments==[value.encode() for value in command]
        return {'pid':pid,'live':matched and state!='Z','command_matches':matched,'process_state':state}
    except (ProcessLookupError,FileNotFoundError):return {'pid':pid,'live':False,'handle_missing':True}
    except OSError as error:return {'pid':pid,'live':None,'observation_error':str(error)}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT/'reports/course_small_target')
    args=parser.parse_args();p=args.root.resolve()
    launch=json.loads((p/'launch_v1/supervisor_start.json').read_text())
    process=json.loads((p/'launch_v1/process.json').read_text())
    result={'observed_at_utc':datetime.now(timezone.utc).isoformat(),
        'supervisor':probe(launch['supervisor_pid'],launch['command']),
        'training':probe(process['pid'],process['command'])}
    for label,path in [('exit',p/'launch_v1/exit.json'),('progress',p/'formal_v1/progress.json')]:
        if path.exists():
            value=json.loads(path.read_text())
            result[label]={k:value[k] for k in ('status','phase','exit_code','error','epoch','global_step',
                'total_steps','images_in_epoch','images_completed','expected_images','loss','elapsed_seconds') if k in value}
    path=p/'formal_v1/selection.json'
    if path.exists():
        value=json.loads(path.read_text())
        result['validation']={'completed_epochs':len(value['epochs']),
            'best_epoch':value['best_epoch'],'best_key':value['best_key']}
    print(json.dumps(result))


if __name__=='__main__':main()
