#!/usr/bin/env python3
"""Fetch a pinned public GraspNet subset via HF mirror, with no proxy."""
import hashlib
import json
import os
from pathlib import Path
import time

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/graspnet_download'
DEST.mkdir(parents=True,exist_ok=True)
for key in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
    os.environ.pop(key,None)
os.environ['NO_PROXY']='*'
os.environ['HF_ENDPOINT']='https://hf-mirror.com'
os.environ['HF_XET_CACHE']=str(DEST/'xet_cache')
os.environ['HF_HUB_CACHE']=str(DEST/'hub_cache')
os.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'
os.environ['SSL_CERT_FILE']='/etc/ssl/certs/ca-certificates.crt'
os.environ['REQUESTS_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'
os.environ['RUST_LOG']='error'

import requests
from huggingface_hub import configure_http_backend
from huggingface_hub.file_download import xet_get
from huggingface_hub.utils import XetFileData


def session():
    s=requests.Session(); s.trust_env=False
    s.verify='/etc/ssl/certs/ca-certificates.crt'
    s.headers['User-Agent']='Mozilla/5.0'
    return s


def main():
    configure_http_backend(backend_factory=session)
    repo='Litian2002/graspnet1b-rsi-exam'
    base='https://hf-mirror.com'
    s=session()
    meta=s.get(f'{base}/api/datasets/{repo}',timeout=20); meta.raise_for_status()
    revision=meta.json()['sha']
    # Public metadata inspected before download; full SHA256 is required.
    expected='5343d6c7f693c7f08440862bc7a692242042911a94acb27f865831aab4d6aaed'
    size=925060136
    target=DEST/'agent_bundle.tar.gz'
    receipt={'repository':repo,'revision':revision,'mirror':base,'proxy':False,
             'file':'agent_bundle.tar.gz','expected_bytes':size,'expected_sha256':expected,
             'official_source':'https://graspnet.net/datasets.html','repack':'third-party scene/view subset',
             'status':'started','started_at':time.time()}
    report=ROOT/'reports/graspnet_download.json'
    report.write_text(json.dumps(receipt,indent=2))
    for filename in ('NOTICE','agent_manifest.json'):
        r=s.get(f'{base}/datasets/{repo}/resolve/{revision}/{filename}',timeout=20);r.raise_for_status()
        (DEST/filename).write_bytes(r.content)
    print(json.dumps({'mirror':base,'bytes':size,'proxy':False,'transport':'mirror Xet storage protocol'}),flush=True)
    try:
        if not target.exists():
            xet_get(incomplete_path=target.with_suffix('.partial'),
                    xet_file_data=XetFileData(file_hash='aebdd90a7f02b0be99d10edece4bee965b5cdb2dea4bb551a7965017377f42e4',
                        refresh_route=f'{base}/api/datasets/{repo}/xet-read-token/{revision}'),
                    headers={'User-Agent':'Mozilla/5.0'},expected_size=size,displayed_filename='GraspNet public subset')
            target.with_suffix('.partial').rename(target)
        digest=hashlib.file_digest(target.open('rb'),'sha256').hexdigest()
        if target.stat().st_size!=size or digest!=expected: raise ValueError('Size/SHA256 mismatch; do not extract')
        receipt.update(status='verified',actual_sha256=digest,actual_bytes=target.stat().st_size,finished_at=time.time())
        print(json.dumps({'status':'verified','bytes':size,'sha256':digest}),flush=True)
    except Exception as exc:
        receipt.update(status='failed',error_type=type(exc).__name__)
        raise
    finally:
        report.write_text(json.dumps(receipt,indent=2))


if __name__=='__main__':main()
