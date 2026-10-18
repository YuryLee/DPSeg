#!/usr/bin/env python3
"""Size/IoU breakdown for one completed DDP arm, using saved validation masks."""
import argparse
import contextlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
# Reuse the historical area ranges and COCO precision aggregation unchanged.
from tools.diagnose_course_small_target_validation import RANGES, average_valid
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from rgbd_seg.course_archive import CourseArchive
from rgbd_seg import course_submission as submission
from tools.run_course_dafusion import now, sha256, write_json


def require(value,message):
    if not value:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    folder = args.run.resolve()
    run = json.loads((folder/'run.json').read_text())
    protocol = json.loads(args.protocol.read_text())
    require(run['status']=='completed' and run['scope']=='full_train_val','Only completed formal arms accepted')
    require(run['protocol_sha256']==sha256(args.protocol),'Protocol identity differs')
    require(run['test_used'] is False,'Unexpected test selection')
    require(run['source_sha256']==protocol['source_sha256'],'Training source identity differs')
    selected = json.loads((folder/'selection.json').read_text())
    require(selected['test_used'] is False,'Unexpected test selection')
    best = max(selected['epochs'],key=lambda r:(r['metrics']['mask_ap_small'],r['metrics']['mask_ap'],-r['epoch']))
    require(best['epoch']==selected['best_epoch']==run['best_epoch'],'Selected epoch differs')
    path = Path(best['output']).resolve()
    require(path.parent==folder,'Selected validation outside run')
    reference = json.loads((path/'evaluation.json').read_text())
    receipt = json.loads((path/'evaluation_run.json').read_text())
    require(receipt['status']=='completed' and receipt['split']=='val'
            and receipt['scope']=='full_split' and receipt['images']==2584,'Incomplete validation')
    require(reference['metrics']==best['metrics'],'Selected score identity differs')
    require(sha256(path/'evaluation.json')==receipt['evaluation_sha256'],'Score changed')
    require(sha256(path/'predictions.jsonl')==receipt['predictions_sha256'],'Predictions changed')
    require(not args.output.exists(),'Output already exists')
    args.output.mkdir(parents=True)
    torch.set_num_threads(4)
    report = {'status':'running','started_at_utc':now(),'run':str(folder),
        'scope':'Descriptive full-validation size/IoU breakdown; no method selection',
        'new_inference':False,'test_accessed':False,'gpu_used':False,
        'selected_epoch':best['epoch'],'model_sha256':best['model_sha256'],
        'area_ranges':dict(RANGES),'protocol_sha256':sha256(args.protocol),
        'predictions_sha256':receipt['predictions_sha256'],
        'evaluation_sha256':receipt['evaluation_sha256'],
        'script_sha256':sha256(__file__),
        'helper_sha256':sha256(ROOT/'tools/diagnose_course_small_target_validation.py')}
    write_json(args.output/'diagnostics.json',report)
    try:
        with CourseArchive(protocol['archive'],'val') as dataset:
            truth = dataset.coco_ground_truth()
            truth.setdefault('info',{})
            require(len(dataset.image_ids)==2584 and len(truth['annotations'])==73421,'Validation GT coverage differs')
            require(all(a['area']>0 and a['area']==int(a['area']) for a in truth['annotations']),'Bins require positive integer areas')
            counts = {label:sum(lo<=a['area']<=hi for a in truth['annotations']) for label,(lo,hi) in RANGES}
            require(counts['small']==10999,'Small GT coverage differs')
            spec = submission._Spec(dataset.image_ids,(1,),(1024,1024),100)
            predictions = []
            count = 0
            with (path/'predictions.jsonl').open() as stream:
                for index,line in enumerate(stream):
                    record = json.loads(line,object_pairs_hook=submission._unique_object)
                    submission._check_id(record['image_id'],index,spec)
                    values,_ = submission._predictions(record['predictions'],spec,decode=False)
                    predictions.extend({'image_id':record['image_id'],**p} for p in values)
                    count += 1
            require(count==2584 and len(predictions)==reference['predicted_instances'],'Prediction coverage differs')
            write_json(args.output/'progress.json',{'phase':'scoring','images':count,'predictions':len(predictions)})
            with (args.output/'coco.log').open('x') as log,contextlib.redirect_stdout(log):
                coco = COCO();coco.dataset=truth;coco.createIndex()
                evaluator = COCOeval(coco,coco.loadRes(predictions),'segm')
                evaluator.params.imgIds=dataset.image_ids
                evaluator.params.catIds=[1]
                evaluator.params.maxDets=[100]
                evaluator.params.areaRng=[bounds for _,bounds in RANGES]
                evaluator.params.areaRngLbl=[label for label,_ in RANGES]
                evaluator.evaluate();evaluator.accumulate()
            bins = {}
            for index,(label,_) in enumerate(RANGES):
                precision=evaluator.eval['precision'][:,:,:,index,0]
                recall=evaluator.eval['recall'][:,:,index,0]
                bins[label]={'ap':average_valid(precision),'ar':average_valid(recall),
                    'ap_by_iou':[average_valid(row) for row in precision],
                    'recall_by_iou':[average_valid(row) for row in recall]}
            require(abs(bins['small']['ap']-reference['metrics']['mask_ap_small'])<1e-12,'Small AP did not reproduce saved evaluation')
            require(abs(bins['small']['ar']-reference['metrics']['mask_ar_small'])<1e-12,'Small AR did not reproduce saved evaluation')
            require(sha256(path/'predictions.jsonl')==receipt['predictions_sha256'],'Input predictions changed during diagnosis')
            require(not torch.cuda.is_initialized(),'Unexpected CUDA initialization')
            report.update(status='passed',finished_at_utc=now(),images=count,gt_counts=counts,bins=bins,
                iou_thresholds=evaluator.params.iouThrs.tolist(),standard_small_ap_ar_reproduced=True,
                interpretation='Descriptive only. Sub-bin AP values must not be averaged to reconstruct standard APsmall.')
            print(json.dumps({'status':'passed','bins':bins,'gt_counts':counts}),flush=True)
    except BaseException as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise
    finally:
        write_json(args.output/'diagnostics.json',report)


if __name__ == '__main__':
    main()
