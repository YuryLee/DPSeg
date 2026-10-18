#!/usr/bin/env python3
"""One bounded delayed submission for the already-authorized spatial ablation."""
import argparse
from datetime import datetime,timedelta,timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time

from kaggle_batch_records import BATCH,MANIFEST,DailyQuota,poll_scores,read,save,submit_one,verified_item,write
from kaggle_mcp_batch_client import Client


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--not-before',required=True)
    args=parser.parse_args();due=datetime.fromisoformat(args.not_before)
    assert due.tzinfo is not None
    folder=BATCH/'deferred_spatial';folder.mkdir(exist_ok=True)
    lock=(folder/'worker.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    item,path=verified_item(read(MANIFEST),'spatial_1024')
    token_path=Path.home()/'.kaggle/access_token'
    token_digest=hashlib.sha256(token_path.read_bytes()).hexdigest()
    state={'status':'waiting_daily_quota','pid':os.getpid(),'key':'spatial_1024',
        'not_before_utc':due.isoformat(),'csv_path':str(path),'csv_sha256':item['sha256'],
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'quota_checks':0,
        'scope':'One fixed previously authorized CSV; no training or parameter changes',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
            [Path(__file__),Path(__file__).with_name('kaggle_batch_records.py'),
             Path(__file__).with_name('kaggle_mcp_batch_client.py')]}}
    manifest=read(MANIFEST);target=next(x for x in manifest['items'] if x['key']=='spatial_1024')
    target.update(status='waiting_daily_quota',not_before_utc=due.isoformat());save(manifest)
    write(folder/'run.json',state)
    deadline=due+timedelta(hours=36)
    try:
        while datetime.now(timezone.utc)<deadline:
            if (folder/'CANCEL').exists():
                state.update(status='cancelled',exit_code=0);break
            remaining=(due-datetime.now(timezone.utc)).total_seconds()
            if remaining>0:
                time.sleep(min(30,remaining));continue
            if hashlib.sha256(token_path.read_bytes()).hexdigest()!=token_digest:
                raise RuntimeError('Kaggle credential changed; revalidation required')
            client=Client();state['quota_checks']+=1
            try:
                reference=submit_one('spatial_1024',client)
            except DailyQuota:
                due=datetime.now(timezone.utc)+timedelta(hours=1)
                state.update(status='waiting_daily_quota',not_before_utc=due.isoformat())
                write(folder/'run.json',state);continue
            state.update(status='submitted_waiting_score',submission_id=reference)
            write(folder/'run.json',state)
            for _ in range(60):
                manifest=poll_scores(client);target=next(x for x in manifest['items'] if x['key']=='spatial_1024')
                if target['status']=='complete':
                    state.update(status='completed',exit_code=0,public_score=target['public_score']);break
                if target['status']=='failed':raise RuntimeError('Kaggle evaluation failed')
                time.sleep(20)
            else:raise RuntimeError('Submission accepted; scoring still pending after20minutes')
            break
        else:raise RuntimeError('Daily quota did not recover within the bounded waiting window')
        state['finished_at_utc']=datetime.now(timezone.utc).isoformat()
    except BaseException as error:
        state.update(status='failed',exit_code=1,error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write(folder/'run.json',state)
        write(folder/'exit.json',{'status':state['status'],'exit_code':state.get('exit_code',1)})


if __name__=='__main__':main()
