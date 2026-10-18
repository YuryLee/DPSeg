"""Submit an explicitly selected, hash-verified tiled comparison to Kaggle."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.kaggle_mcp_batch_client import Client, KaggleError


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def save(path, state):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n')
    temp.replace(path)


def submit(path):
    state = json.loads(path.read_text())
    client = Client()
    for item in state['items']:
        csv = Path(item['path'])
        if sha(csv) != item['sha256']:
            raise KaggleError('CSV hash differs from verified experiment')
        existing = client.submissions()
        matches = [s for s in existing if s.get('file_name')==csv.name
                   and 'sha256='+item['sha256'] in s.get('description','')]
        if matches:
            item.update(submission_id=matches[0]['ref'],status=matches[0]['status'],
                        public_score=matches[0].get('public_score'))
            save(path,state)
            print(json.dumps(dict(configuration=item['configuration'],reused_existing=True,
                                  submission_id=item['submission_id'])),flush=True)
            continue
        if item.get('status') in ('submitting','submission_outcome_unknown') or item.get('submission_id'):
            raise KaggleError('Unresolved prior submission; refusing duplicate')
        today = datetime.now(timezone.utc).date()
        count = sum(datetime.fromisoformat(s['date'].replace('Z','+00:00')).date()==today for s in existing)
        if count >= state['max_daily_submissions']:
            item['status']='waiting_daily_quota';save(path,state)
            raise KaggleError('Daily submission quota reached')
        item['status']='uploading';save(path,state)
        upload = client.start_upload(csv)
        blob = client.upload_file(csv,upload,item['sha256'])
        item['status']='submitting';save(path,state)
        try:
            result = client.submit(blob,item['description']+' | sha256='+item['sha256'])
        except Exception:
            item['status']='submission_outcome_unknown';save(path,state)
            raise
        if not result.get('ref'):
            item['status']='submission_outcome_unknown';save(path,state)
            raise KaggleError('Submission reference missing; reconcile before retry')
        item.update(status='submitted_waiting_score',submission_id=int(result['ref']),
                    submitted_at_utc=datetime.now(timezone.utc).isoformat())
        save(path,state)
        print(json.dumps(dict(configuration=item['configuration'],submission_id=item['submission_id'])),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest',type=Path,required=True)
    args = parser.parse_args()
    try:
        submit(args.manifest)
    except KaggleError as error:
        print(str(error),file=sys.stderr)
        raise SystemExit(1)
