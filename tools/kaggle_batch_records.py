"""Named CSV ledger, idempotent submission, and public-score reporting."""
import csv
from datetime import datetime,timezone
import hashlib
import json
import os
from pathlib import Path

from kaggle_mcp_batch_client import Client,KaggleError

ROOT=Path(__file__).resolve().parents[1]
BATCH=ROOT/'reports/kaggle_batch_20261016'
MANIFEST=BATCH/'submission_manifest.json'
KEYS={'native_800','native_1024','crop_1024','focal_1024','spatial_1024','depth_1024'}


class DailyQuota(KaggleError):pass


def read(path):return json.loads(Path(path).read_text())


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def write(path,value):
    path=Path(path);tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def refresh_files(manifest):
    for item in manifest['items']:
        if item['status']=='awaiting_export' and Path(item['export_receipt']).exists():
            receipt=read(item['export_receipt'])
            if receipt['status']=='completed':
                item.update(status='ready',sha256=receipt['csv_sha256'],instances=receipt['instances'])


def render(manifest):
    values={'native_800':.3523167534655871,'native_1024':.46666994930213684,
            'crop_1024':.4533542864295693,'focal_1024':.45207942255550043,
            'spatial_1024':.46229864998901693,'depth_1024':.5256361837733781}
    labels={'awaiting_export':'等待 CSV','ready':'CSV 校验通过','uploading':'上传中','submitting':'提交中',
            'submitted_waiting_score':'已提交，评分中','complete':'已完成',
            'already_submitted_exact_match':'已提交，文件一致','waiting_daily_quota':'等待每日配额',
            'submission_outcome_unknown':'需核对提交状态','failed':'提交失败'}
    lines=['本轮六组配置的 Kaggle 提交记录。验证集 APₛ 与 Kaggle 公榜分数对应不同图片范围，分别列出。',
           '', '每日最多 5 次提交；已验证内容相同的深度版复用原提交记录。', '',
           '| 配置 | 验证 APₛ | Kaggle 公榜 | 状态 | 提交 ID | CSV 文件 |',
           '|---|---:|---:|---|---|---|']
    rows=[]
    for item in sorted(manifest['items'],key=lambda x:['native_800','native_1024','crop_1024','focal_1024','spatial_1024','depth_1024'].index(x['key'])):
        score=item.get('public_score');score_text='' if score is None else f'{float(score):.5f}'
        lines.append(f'| {item["label"]} | {values[item["key"]]:.6f} | {score_text or "待评分"} | '
            f'{labels.get(item["status"],item["status"])} | {item.get("submission_id","")} | '
            f'[{item["filename"]}]({item["csv_path"]}) |')
        rows.append({'configuration':item['key'],'filename':item['filename'],'validation_APsmall':values[item['key']],
            'kaggle_public_score':score_text,'status':item['status'],'submission_id':item.get('submission_id',''),
            'csv_path':item['csv_path']})
    lines+=['','可提交的文件位于 `submissions/`；`results.csv` 是汇总表。',
            '',f'更新时间（UTC）：{datetime.now(timezone.utc).isoformat()}','']
    (BATCH/'results.md').write_text('\n'.join(lines))
    with (BATCH/'results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def save(manifest):
    manifest['updated_at_utc']=datetime.now(timezone.utc).isoformat()
    refresh_files(manifest);write(MANIFEST,manifest);render(manifest)


def verified_item(manifest,key):
    if manifest['competition']!='rgb-d' or {x['key'] for x in manifest['items']}!=KEYS:
        raise KaggleError('Unexpected authorized submission batch')
    item=next(x for x in manifest['items'] if x['key']==key)
    receipt=read(item['export_receipt']); validation=read(receipt['validation_path']);path=Path(item['csv_path'])
    if (not path.resolve().is_relative_to((BATCH/'submissions').resolve())
            or path.name!=item['filename'] or receipt['status']!='completed'
            or receipt['images']!=2112 or validation['rows']!=2112
            or not validation['rle_native_decode_verified']
            or sha(path)!=receipt['csv_sha256'] or receipt['csv_sha256']!=validation['sha256']):
        raise KaggleError('CSV scope,identity,or native RLE validation differs')
    item.update(sha256=receipt['csv_sha256'],instances=receipt['instances'])
    return item,path


def poll_scores(client=None):
    client=client or Client();manifest=read(MANIFEST);refresh_files(manifest)
    for item in manifest['items']:
        if item['status']=='submitted_waiting_score':
            info=client.get_submission(item['submission_id'])
            if info.get('status')=='COMPLETE':
                item.update(status='complete',public_score=float(info['public_score']))
            elif info.get('status') in ['ERROR','FAILED','INVALID','CANCELLED']:
                item.update(status='failed',kaggle_status=info['status'])
    save(manifest);return manifest


def submit_one(key,client=None):
    client=client or Client();manifest=read(MANIFEST);refresh_files(manifest)
    item,path=verified_item(manifest,key)
    if item.get('submission_id'):return item['submission_id']
    existing=client.submissions()
    fingerprint='sha256='+item['sha256']
    matches=[r for r in existing if r.get('file_name')==item['filename'] and fingerprint in r.get('description','')]
    if matches:
        item.update(status='submitted_waiting_score',submission_id=matches[0]['ref']);save(manifest)
        return matches[0]['ref']
    if item['status']=='submission_outcome_unknown':
        raise KaggleError('Previous submission outcome unresolved; refusing duplicate')
    today=datetime.now(timezone.utc).date()
    count=sum(datetime.fromisoformat(r['date'].replace('Z','+00:00')).date()==today for r in existing)
    if count>=manifest['max_daily_submissions']:
        item['status']='waiting_daily_quota';save(manifest);raise DailyQuota('Daily submission quota reached')
    item['status']='uploading';save(manifest)
    upload=client.start_upload(path)
    state=BATCH/'.upload_state';state.mkdir(mode=0o700,exist_ok=True);state.chmod(0o700)
    private=state/(key+'.json')
    fd=os.open(private,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600)
    with os.fdopen(fd,'w') as stream:json.dump({'key':key,'upload':upload,'sha256':item['sha256']},stream)
    blob=client.upload_file(path,upload,item['sha256'])
    description=item['description']+' | sha256='+item['sha256']
    item.update(status='submitting',submitted_description=description);save(manifest)
    try:
        result=client.submit(blob,description)
    except Exception as error:
        message=str(error).lower()
        if ('daily' in message or 'per day' in message) and ('limit' in message or 'maximum' in message or 'exceed' in message):
            item['status']='waiting_daily_quota';save(manifest);raise DailyQuota('Kaggle rejected daily quota') from None
        item['status']='submission_outcome_unknown';save(manifest);raise
    if not result.get('ref'):
        item['status']='submission_outcome_unknown';save(manifest)
        raise KaggleError('Submission response missing reference; reconcile before retrying')
    item.update(status='submitted_waiting_score',submission_id=int(result['ref']))
    item['submitted_at_utc']=datetime.now(timezone.utc).isoformat();save(manifest)
    private.write_text(json.dumps({'key':key,'uploaded':True,'sha256':item['sha256'],
                                  'submission_id':item['submission_id']})+'\n')
    return item['submission_id']
