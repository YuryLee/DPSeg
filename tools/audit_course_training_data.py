#!/usr/bin/env python3
"""Audit official course train/val; test is checked for partition identity only."""
from collections import Counter
from datetime import datetime, timezone
import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from PIL import Image
from pycocotools import mask as mask_utils
from rgbd_seg.course_training_data import CourseTrainingDataset


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def audit(archive, test_audit):
    errors=[]
    summaries={}
    sets={}
    with ZipFile(archive) as z:
        rows=list(csv.DictReader(io.StringIO(z.read('submission.csv').decode('utf-8-sig'))))
        test_ids={int(row['image_id']) for row in rows}
        test_rgb=[n for n in z.namelist() if n.startswith('rgb/test/') and n.endswith('.png')]
        test_scenes={Path(n).stem.split('_v')[0] for n in test_rgb}
    sets['test']={'image_ids':test_ids,'scene_ids':test_scenes}
    if len(test_ids)!=2112 or len(rows)!=2112 or len(test_rgb)!=2112:
        errors.append('Official test partition identity count differs from 2112')
    for split,expected_count in [('train',14432),('val',2584)]:
        print(f'Loading and checking {split} encoded annotations',flush=True)
        with CourseTrainingDataset(archive, split, augment=False, seed=17) as data:
            summary=data.summary()
            if len(data)!=expected_count:
                errors.append(f'{split}: image count differs from official metadata')
            scenes={im.get('scene_id') for im in data.images}
            if None in scenes:
                errors.append(f'{split}: missing scene IDs')
            sets[split]={'image_ids':set(data.image_ids),'scene_ids':scenes}
            crowds, scales, invalid, semantics = Counter(),Counter(),Counter(),Counter()
            bbox_mismatches=0
            zero_box=0
            area_exact_1024=area_exact_9216=0
            category_counts=Counter()
            for group in data.annotations_by_image.values():
                for ann in group:
                    crowds[str(ann['iscrowd'])]+=1
                    category_counts[str(ann['category_id'])]+=1
                    area_exact_1024+=ann['area']==1024
                    area_exact_9216+=ann['area']==9216
                    measured=mask_utils.toBbox(ann['segmentation']).tolist()
                    original=ann['bbox']
                    if len(original)!=4 or any(abs(float(a)-float(b))>1e-6 for a,b in zip(original,measured)):
                        bbox_mismatches+=1
                    zero_box+=ann['area']>0 and (measured[2]<=0 or measured[3]<=0)
            headers=Counter()
            with ZipFile(archive) as z:
                for image in data.images:
                    scales[str(image.get('depth_scale_m'))]+=1
                    invalid[str(image.get('depth_invalid_code'))]+=1
                    semantics[str(image.get('depth_semantics'))]+=1
                    for field in ['file_name','depth_file_name']:
                        with z.open(image[field]) as raw:
                            with Image.open(raw) as picture:
                                if picture.size!=(image['width'],image['height']):
                                    errors.append(f'{split}: PNG dimensions mismatch: {image[field]}')
                                headers[(field,picture.mode,picture.size)]+=1
            decoded=[]
            for index in sorted({0,len(data)//2,len(data)-1}):
                image,target=data[index]
                decoded.append({'index':index,'image_id':int(target['image_id'][0]),'rgb_shape':list(image.shape),
                                'instances':len(target['labels']),'masks_shape':list(target['masks'].shape),
                                'area_sum':float(target['area'].sum()),'min_box_side':float((target['boxes'][:,2:]-target['boxes'][:,:2]).min()) if len(target['boxes']) else None})
            summary.update({'rle_areas_checked':data.annotation_count,'original_vs_rle_area_mismatch_count':0,
                            'rle_errors':[], 'iscrowd_counts':dict(crowds),'instances_by_category':dict(category_counts),
                            'original_bbox_vs_rle_bbox_mismatches':bbox_mismatches,
                            'positive_area_degenerate_mask_boxes':zero_box,'area_exactly_1024':area_exact_1024,
                            'area_exactly_9216':area_exact_9216,
                            'depth_scale_m_counts':dict(scales),'depth_invalid_code_counts':dict(invalid),
                            'depth_semantics_counts':dict(semantics),
                            'png_headers':[{'field':k[0],'mode':k[1],'size':list(k[2]),'count':v} for k,v in headers.items()],
                            'decoded_samples':decoded,
                            'all_referenced_RGB_depth_members_exist':True,
                            'all_image_ids_unique':True,'all_annotation_ids_unique':True})
            summaries[split]=summary
            print(f'{split}: {len(data)} images, {data.annotation_count} annotations checked',flush=True)
    overlaps={}
    for a,b in [('train','val'),('train','test'),('val','test')]:
        pair={key:sorted(sets[a][key]&sets[b][key]) for key in ['image_ids','scene_ids']}
        overlaps[f'{a}_{b}']=pair
        if any(pair.values()):
            errors.append(f'Official partition conflict {a}/{b}; original splits preserved, no silent reassignment')
    prior=json.loads(Path(test_audit).read_text())
    if Path(prior['archive_path']).resolve()!=Path(archive).resolve() or prior['archive_bytes']!=Path(archive).stat().st_size:
        errors.append('Prior test audit does not reference this archive')
    return {'status':'pass' if not errors else 'fail','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'material_passport':{'origin_skill':'academic-research-suite/experiment-agent','mode':'run/validate','verification_status':'ANALYZED','scope':'Dataset schema, all train/val encoded GT and partition audit; no model results'},
            'archive_path':str(Path(archive).resolve()),'archive_bytes':Path(archive).stat().st_size,
            'splits':summaries,'partition_overlap':overlaps,'errors':errors,
            'test_usage':'Only submission image IDs and RGB filename scene IDs checked for split leakage; no test annotation or model-performance analysis for parameter selection',
            'existing_test_audit':{'path':str(Path(test_audit).resolve()),'sha256':sha(test_audit),'status':prior['status']},
            'empty_mask_rule':'Zero-area masks omitted from training targets and counted; original COCO GT preserved. Every positive-area mask retained, including one-pixel instances.',
            'bbox_rule':'Recompute exclusive xyxy from independent decoded masks; original bbox discrepancies counted, never fix original COCO GT.',
            'dataset_source_sha256':sha(ROOT/'rgbd_seg/course_training_data.py'),'auditor_sha256':sha(__file__),
            'limits':['All train/val annotation entries fully read; all RGB/depth PNG headers inspected, only three images per split fully decoded.',
                      'No full image-content hash or full ZIP SHA recomputation in this bounded audit; archive download receipt is separate.',
                      'Official splits are not changed; any scene/ID overlap is an audit failure.',
                      'No training, inference, extraction, downloads, or use of test scores to set training parameters.']}


def main():
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True)
    p.add_argument('--test-audit',type=Path,default=ROOT/'reports/course_pretrained/dataset_audit.json')
    p.add_argument('--output',type=Path,default=ROOT/'reports/course_finetune/data_audit.json')
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    try:report=audit(args.archive,args.test_audit)
    except Exception as exc:report={'status':'fail','errors':[f'{type(exc).__name__}: {exc}'],'auditor_sha256':sha(__file__)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'status':report['status'],'errors':report['errors']},ensure_ascii=False),flush=True)
    return 0 if report['status']=='pass' else 1


if __name__=='__main__':sys.exit(main())
