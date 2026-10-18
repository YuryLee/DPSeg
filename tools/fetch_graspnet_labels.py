#!/usr/bin/env python3
"""Walk a remote uncompressed tar and fetch selected original label PNGs."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import re
import tarfile
import time
from PIL import Image
from xet_range import RangeReader


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--raw',default='data/graspnet_raw');p.add_argument('--scenes',default='0,1,2')
    p.add_argument('--max-frames',type=int,default=4);p.add_argument('--budget-mb',type=int,default=3072)
    p.add_argument('--window-mb',type=int,default=4);p.add_argument('--resume',action='store_true')
    p.add_argument('--include-inputs',action='store_true',help='Fetch original RGB/depth/meta as well as labels at fixed frames 0,64,128,192')
    p.add_argument('--auto-scenes',type=int,default=0,help='Take first N official training scenes encountered in archive order')
    a=p.parse_args(); raw=Path(a.raw); wanted=set()
    selected=[]
    def add_scene(scene):
        selected.append(scene)
        base=raw/f'scenes/scene_{scene:04d}/realsense'
        if a.include_inputs:
            for frame in [0,64,128,192][:a.max_frames]:
                for kind in ['rgb','depth','label','meta']:
                    suffix='mat' if kind=='meta' else 'png'
                    wanted.add(str((base/kind/f'{frame:04d}.{suffix}').relative_to(raw)))
        else:
            for f in sorted((base/'rgb').glob('*.png'))[:a.max_frames]:
                wanted.add(str((base/'label'/f.name).relative_to(raw)))
    if a.auto_scenes and not a.include_inputs:raise ValueError('Auto scenes requires original inputs')
    if not a.auto_scenes:
        for scene in map(int,a.scenes.split(',')):add_scene(scene)
    if not wanted and not a.auto_scenes:raise ValueError('No requested RGB frames exist')
    report=Path('reports/graspnet_labels_download.json')
    previous=json.loads(report.read_text()) if a.resume and report.exists() else {}
    if a.auto_scenes:
        for scene in previous.get('selected_scenes',[]):add_scene(scene)
    found=previous.get('files',{});wanted-=set(found)
    off=previous.get('next_offset',0);started=time.time(); scene_seen=None
    repo='DravenALG/GraspNet-1Billion';rev='1f9db26044cd2a7e629bbb465be3665364cbf2bf'
    full_size=142475571200;file_hash='051a80be3773d2bf40b98f0121f0fd20e5038dc2728d65fb40f887ada2aa1a69'
    reader=RangeReader(repo,rev,file_hash,full_size,max_download_bytes=a.budget_mb*1024**2)
    window=b'';window_start=-1;count=0;longname=None
    def read(start,n):
        nonlocal window,window_start
        if start<window_start or start+n>window_start+len(window):
            window_start=start
            window=reader.read_range(start,min(max(n,a.window_mb*1024**2),full_size-start))
        return window[start-window_start:start-window_start+n]
    def save(status):
        report.write_text(json.dumps({'status':status,'repository':repo,'revision':rev,'proxy':False,
            'archive_size':full_size,'next_offset':off,'files':found,'remaining':sorted(wanted),
            'selected_scenes':selected,
            'downloaded_this_run':reader.downloaded_bytes,'elapsed_s':time.time()-started,
            'integrity':'Pinned archive Xet identity, TLS verification, tar checksums, PNG verification and local per-file SHA256; complete archive SHA256 not recomputed'},indent=2))
    try:
        while (wanted or (a.auto_scenes and len(selected)<a.auto_scenes)) and off<full_size:
            header=read(off,512)
            if header==b'\0'*512:break
            info=tarfile.TarInfo.frombuf(header,encoding='utf-8',errors='strict')
            payload=off+512;next_offset=payload+((info.size+511)//512)*512
            if info.type==tarfile.GNUTYPE_LONGNAME:
                longname=read(payload,info.size).rstrip(b'\0').decode()
            elif info.type in (tarfile.XHDTYPE,tarfile.XGLTYPE):
                # Standard GraspNet paths fit USTAR; reject path overrides rather than misread.
                text=read(payload,info.size).decode()
                match=re.search(r'\d+ path=([^\n]+)',text)
                if match:longname=match[1]
            else:
                name=(longname or info.name).lstrip('./');longname=None
                index=name.find('scenes/scene_')
                relative=name[index:] if index>=0 else name
                m=re.search(r'scene_(\d{4})/',relative)
                scene=int(m.group(1)) if m else None
                if a.auto_scenes and scene is not None and 0<=scene<100 and scene not in selected and len(selected)<a.auto_scenes:
                    add_scene(scene)
                if scene is not None and scene!=scene_seen:
                    scene_seen=scene;print(json.dumps({'archive_scene':scene,'offset_gb':round(off/1e9,3),'download_mb':round(reader.downloaded_bytes/1e6,1),'remaining':len(wanted)}),flush=True)
                if relative in wanted:
                    if not info.isfile() or not 0<info.size<10_000_000:raise ValueError('Unexpected label entry')
                    data=read(payload,info.size)
                    if relative.endswith('.png'):
                        with Image.open(io.BytesIO(data)) as im:
                            im.verify()
                    target=raw/relative;target.parent.mkdir(parents=True,exist_ok=True)
                    if target.exists() and target.read_bytes()!=data:raise ValueError('Existing label differs')
                    target.write_bytes(data)
                    found[relative]={'sha256':hashlib.sha256(data).hexdigest(),'bytes':len(data),'tar_offset':payload}
                    wanted.remove(relative)
                    print(json.dumps({'fetched':relative,'remaining':len(wanted)}),flush=True)
            off=next_offset;count+=1
            if count%50==0:save('running')
        save('completed' if not wanted else 'incomplete')
        if wanted:raise RuntimeError('Archive scan ended before all labels were found')
    except BaseException:
        save('incomplete');raise
    finally:reader.close()


if __name__=='__main__':main()
