#!/usr/bin/env python3
"""Local RGB-D inference demo. All predictions come from loaded checkpoints."""
import argparse
import base64
import io
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import time
from http.server import BaseHTTPRequestHandler,HTTPServer
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image
from rgbd_seg.data import RGBDDataset,load_rgbd,dataset_fingerprint
from rgbd_seg.model import load_model
from rgbd_seg.render import rgb_image,png_uri,depth_image,overlay


class Demo:
    def __init__(self,data,runs,device='cpu',split='val'):
        if split not in ('val','test'): raise ValueError('Demo split must be val or test')
        self.dataset=RGBDDataset(data,split); self.device=device; self.split=split
        self.models={}; self.info_models=[]
        manifest=self.dataset.manifest
        fingerprint=dataset_fingerprint(data)
        manifest_hash=hashlib.sha256((Path(data)/'manifest.json').read_bytes()).hexdigest()
        self.manifest_sha256=manifest_hash; self.dataset_sha256=fingerprint
        provenance=None
        for mode in ('E0','E2','E3'):
            folder=Path(runs)/mode
            model,ckpt=load_model(folder/'model.pt',device)
            checkpoint_hash=hashlib.sha256((folder/'model.pt').read_bytes()).hexdigest()
            if ckpt['config']['mode']!=mode: raise ValueError('Checkpoint mode mismatch')
            if ckpt['classes']!=manifest['classes']: raise ValueError('Checkpoint class mapping mismatch')
            if ckpt['provenance']!=manifest['provenance']: raise ValueError('Checkpoint and dataset provenance differ')
            if provenance and provenance!=ckpt['provenance']: raise ValueError('Models have mixed provenance')
            provenance=ckpt['provenance']; self.models[mode]=model
            report_name='evaluation.json' if split=='val' else 'evaluation_test.json'
            report_path=folder/report_name
            report=json.loads(report_path.read_text()) if report_path.exists() else {}
            if report and (report.get('checkpoint_sha256')!=checkpoint_hash
                           or report.get('manifest_sha256')!=manifest_hash
                           or report.get('dataset_sha256')!=fingerprint
                           or report.get('provenance')!=manifest['provenance']
                           or report.get('split')!=split):
                raise ValueError(f'{mode}: {report_name} does not match checkpoint/dataset/split; rerun evaluation')
            extra_path=folder/f'extra_{split}.json'
            extra=json.loads(extra_path.read_text()) if extra_path.exists() else {}
            extra_run=extra.get('run',{})
            if extra and (extra_run.get('input_type')!='checkpoint'
                          or extra_run.get('input_sha256')!=checkpoint_hash
                          or extra_run.get('dataset_sha256')!=fingerprint
                          or extra_run.get('provenance')!=manifest['provenance']
                          or extra_run.get('split')!=split):
                raise ValueError(f'{mode}: extra_{split}.json does not match checkpoint/dataset/split; evaluate directly from checkpoint')
            run_path=folder/'run.json'
            run=json.loads(run_path.read_text()) if run_path.exists() else {}
            run_metadata=None
            if run and run.get('checkpoint_sha256')==checkpoint_hash:
                run_metadata={key:run.get(key) for key in ('run_id','steps','batch_size','seed','device','selection','exit_status')}
                initialization=run.get('rgb_initialization')
                if isinstance(initialization,dict):
                    initialization={key:value for key,value in initialization.items()
                                    if key not in ('copied_keys','copied','skipped_keys','loaded_keys')}
                run_metadata['rgb_initialization']=initialization
            self.info_models.append({'id':mode,'name':{'E0':'RGB Mask R-CNN','E2':'RGB-D 相加融合','E3':'有效性门控融合'}[mode],
                'arch':ckpt['config']['arch'],'config':ckpt['config'],'run_metadata':run_metadata,
                'checkpoint_sha256':checkpoint_hash,
                'parameters':sum(p.numel() for p in model.parameters()),
                'checkpoint_mb':(folder/'weights.pt').stat().st_size/1024**2 if (folder/'weights.pt').exists() else (folder/'model.pt').stat().st_size/1024**2,
                'metrics':report.get('metrics',{'mask_ap':None,'mask_ap_small':None}),
                'metrics_split':split,'extra_metrics':extra.get('metrics_macro'),
                'extra_protocol':extra.get('protocol'),'latency_ms':report.get('timing',{}).get('mean_ms')})
        self.provenance=provenance
        summary={'split':split,'images':len(self.dataset),'annotated_images':0,'instances':0,'small_instances':0,
                 'scene_ids':sorted({record['source_scene'] for record in self.dataset.samples if 'source_scene' in record}),
                 'split_sizes':{key:sum(record['split']==key for record in manifest['samples']) for key in ('train','val','test')},
                 'area_reference':'original-resolution masks; small area < 32^2 pixels'}
        self.info_samples=[]
        for index,record in enumerate(self.dataset.samples):
            sample_info={'id':record['id'],'name':record['id'],'instances':None,'small_instances':None}
            if 'source_scene' in record and 'source_frame' in record:
                sample_info['name']=f"场景 {record['source_scene']:03d} · 帧 {record['source_frame']:04d}"
            if record['instance_path'] is not None:
                _,target=self.dataset[index]
                summary['annotated_images']+=1
                summary['instances']+=len(target['labels'])
                small_count=int((target['area']<32**2).sum())
                summary['small_instances']+=small_count
                sample_info.update(instances=len(target['labels']),small_instances=small_count)
                sample_info['name']+=f' · {small_count} 个小目标'
            self.info_samples.append(sample_info)
        self.dataset_summary=summary

    def info(self):
        return {'provenance':self.provenance,'models':self.info_models,'split':self.split,
                'manifest_sha256':self.manifest_sha256,'dataset_sha256':self.dataset_sha256,
                'dataset_summary':self.dataset_summary,
                'samples':self.info_samples,
                'classes':self.dataset.classes,'class_scheme':self.dataset.manifest.get('class_scheme'),
                'depth_scale':self.dataset.depth_scale}

    def predict(self,sample,gt,threshold,input_source):
        w,h=rgb_image(sample).size
        if w*h>2_000_000: raise ValueError('演示输入最多 200 万像素，请先对齐缩小 RGB、深度与标注')
        image=rgb_image(sample)
        panels={'rgb':png_uri(image),'depth':png_uri(depth_image(sample)),
                'gt':png_uri(overlay(image,gt)) if gt is not None else None}
        predictions={}; timing={}
        with torch.inference_mode():
            for mode,model in self.models.items():
                if self.device.startswith('cuda'): torch.cuda.synchronize()
                start=time.perf_counter()
                moved={k:v.to(self.device) for k,v in sample.items()}
                raw=model([moved])[0]
                if self.device.startswith('cuda'): torch.cuda.synchronize()
                timing[mode]=1000*(time.perf_counter()-start)
                keep=raw['scores']>=threshold
                pred={k:v[keep].cpu().numpy() for k,v in raw.items()}
                masks=pred['masks'][:,0]>=.5
                panels[mode]=png_uri(overlay(image,masks,pred['boxes'],pred['labels'],pred['scores'],self.dataset.classes))
                predictions[mode]=[{'label':self.dataset.classes[int(label)-1],'score':float(score),
                    'box':box.tolist(),'area':int(mask.sum())} for label,score,box,mask in zip(pred['labels'],pred['scores'],pred['boxes'],masks)]
                from pycocotools import mask as mask_utils
                for item,mask in zip(predictions[mode],masks):
                    rle=mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    item['mask_rle']={'size':rle['size'],'counts':rle['counts'].decode('ascii')}
        return {'provenance':self.provenance,'split':self.split,'input_source':input_source,'width':w,'height':h,'panels':panels,
            'predictions':predictions,'timing_ms':timing,'depth_valid_fraction':float(sample['depth_valid'].mean()),
            'display_threshold':threshold,'mask_threshold':.5,'depth_visualization':'per-image 1–99% range; holes are dark; not raw depth'}

    def infer(self,request):
        threshold=float(request.get('threshold',.35))
        if not math.isfinite(threshold) or not 0<=threshold<=1: raise ValueError('阈值须位于 0–1')
        if 'sample_id' in request:
            index=next((i for i,s in enumerate(self.dataset.samples) if s['id']==request['sample_id']),None)
            if index is None: raise ValueError('样例不存在')
            sample,target=self.dataset[index]
            gt=target['masks'].numpy() if self.dataset.samples[index]['instance_path'] else None
            return self.predict(sample,gt,threshold,request['sample_id'])
        suffix=Path(request.get('depth_name','depth.png')).suffix.lower()
        if suffix not in ('.png','.npy'): raise ValueError('深度格式仅支持原始 16 位 PNG 或 NPY')
        def decode(value):
            if not isinstance(value,str): raise ValueError('文件内容须为 base64')
            if value.startswith('data:'): value=value.split(',',1)[1]
            return base64.b64decode(value,validate=True)
        with tempfile.TemporaryDirectory(prefix='rgbd_upload_') as tmp:
            rgb=Path(tmp)/'rgb.png'; depth=Path(tmp)/('depth'+suffix)
            rgb.write_bytes(decode(request['rgb'])); depth.write_bytes(decode(request['depth']))
            with Image.open(rgb) as im:
                if im.width*im.height>2_000_000: raise ValueError('图像最多 200 万像素')
            sample=load_rgbd(rgb,depth,float(request['depth_scale']))
            gt=None
            if request.get('gt'):
                with Image.open(io.BytesIO(decode(request['gt']))) as im: ids=np.array(im)
                if ids.ndim!=2 or ids.shape!=tuple(sample['depth'].shape[-2:]) or ids.dtype.kind not in 'ui':
                    raise ValueError('真值须为与 RGB 等大的单通道整数实例 ID 图')
                unique=np.unique(ids); unique=unique[unique>0]
                if len(unique)>500: raise ValueError('演示最多展示 500 个真值实例')
                gt=[ids==i for i in unique]
            return self.predict(sample,gt,threshold,'uploaded')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',default='data/synthetic'); p.add_argument('--runs',default='outputs/demo')
    p.add_argument('--port',type=int,default=8876); p.add_argument('--host',default='127.0.0.1')
    p.add_argument('--split',choices=('val','test'),default='val')
    p.add_argument('--device',default='cpu'); p.add_argument('--threads',type=int,default=2)
    args=p.parse_args(); torch.set_num_threads(args.threads)
    demo=Demo(args.data,args.runs,args.device,split=args.split)
    html=Path(__file__).resolve().parents[1]/'web/index.html'
    class Handler(BaseHTTPRequestHandler):
        def send(self,status,payload,kind='application/json; charset=utf-8'):
            body=payload if isinstance(payload,bytes) else json.dumps(payload,ensure_ascii=False,allow_nan=False).encode()
            self.send_response(status); self.send_header('Content-Type',kind); self.send_header('Content-Length',str(len(body)))
            self.send_header('Cache-Control','no-store'); self.send_header('X-Content-Type-Options','nosniff')
            self.end_headers(); self.wfile.write(body)
        def do_GET(self):
            if self.path=='/api/info': self.send(200,demo.info())
            elif self.path in ('/','/index.html'): self.send(200,html.read_bytes(),'text/html; charset=utf-8')
            elif self.path=='/paper-trial':
                trial=Path(__file__).resolve().parents[1]/'reports/v4/gallery/index.html'
                if trial.is_file(): self.send(200,trial.read_bytes(),'text/html; charset=utf-8')
                else: self.send(404,{'error':'论文改进试验页面尚未生成'})
            else: self.send(404,{'error':'Not found'})
        def do_POST(self):
            if self.path!='/api/infer': self.send(404,{'error':'Not found'}); return
            try:
                size=int(self.headers.get('Content-Length',0))
                if not 0<size<=32*1024*1024: raise ValueError('请求大小须小于 32 MB')
                request=json.loads(self.rfile.read(size))
                if not isinstance(request,dict): raise ValueError('请求须为 JSON 对象')
                self.send(200,demo.infer(request))
            except (ValueError,KeyError,OSError,TypeError) as exc:
                self.send(400,{'error':str(exc)})
            except Exception as exc:
                print(f'Inference error: {type(exc).__name__}: {exc}',flush=True)
                self.send(500,{'error':'推理失败，详见本地服务日志'})
    server=HTTPServer((args.host,args.port),Handler)
    url=f'http://{args.host}:{server.server_address[1]}'
    report=Path(__file__).resolve().parents[1]/'reports/server.json'
    report.write_text(json.dumps({'url':url,'pid':os.getpid(),'provenance':demo.provenance,'split':demo.split}))
    print(f'RGB-D demo listening at {url} | {demo.provenance}',flush=True)
    server.serve_forever()


if __name__=='__main__': main()
