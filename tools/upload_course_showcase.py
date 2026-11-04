#!/usr/bin/env python3
"""Upload the verified case package to the project's Hugging Face repository."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path('outputs/DPSeg_case_demo_v1.zip'))
    parser.add_argument('--demo-dir',type=Path,default=Path('outputs/course_showcase_test_v1'))
    parser.add_argument('--repo',default='Lemma456/DPSeg')
    args=parser.parse_args()
    # Follow the user's direct-connection requirement; never disable TLS verification.
    for name in ('HTTP_PROXY','HTTPS_PROXY','ALL_PROXY','http_proxy','https_proxy','all_proxy'):
        os.environ.pop(name,None)
    from huggingface_hub import HfApi, CommitOperationAdd
    receipt=json.loads(args.archive.with_suffix('.manifest.json').read_text())
    with args.archive.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    assert digest==receipt['sha256']
    assert json.loads((args.demo_dir/'browser/check.json').read_text())['status']=='passed'
    api=HfApi(endpoint='https://huggingface.co')
    info=api.model_info(args.repo)
    assert not info.private
    files=[args.archive,args.archive.with_suffix('.manifest.json')]
    files += [args.demo_dir/name for name in ('README.md','case_index.csv','overview.png')]
    prefix='demos/course_showcase_v1/'
    commit=api.create_commit(repo_id=args.repo,repo_type='model',parent_commit=info.sha,
        operations=[CommitOperationAdd(path_in_repo=prefix+p.name,path_or_fileobj=str(p)) for p in files],
        commit_message='Add 12 verified GT and S0-S4 qualitative cases with offline demo')
    remote=api.get_paths_info(args.repo,[prefix+args.archive.name],repo_type='model',revision=commit.oid)[0]
    assert remote.size==args.archive.stat().st_size
    assert remote.lfs and remote.lfs.sha256==digest
    result=dict(status='uploaded_and_verified',repository=args.repo,commit=commit.oid,
        bytes=remote.size,sha256=digest,
        download=f'https://huggingface.co/{args.repo}/resolve/{commit.oid}/{prefix}{args.archive.name}')
    (args.demo_dir/'hf_upload.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__=='__main__':
    main()
