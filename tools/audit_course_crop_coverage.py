#!/usr/bin/env python3
"""Pre-run crop frequency from train annotations and the deterministic coin."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from rgbd_seg.course_crop import CROP_POLICY
from rgbd_seg.course_depth_data import CourseDepthTrainingDataset
from tools.run_course_dafusion import now, sha256, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    if protocol['intervention']['method'] != 'crop' or protocol['intervention']['policy'] != CROP_POLICY:
        raise ValueError('Coverage audit requires the implemented crop policy')
    args.output.mkdir(parents=True,exist_ok=False)
    seed,epochs = protocol['training']['seed'],protocol['training']['epochs']
    with CourseDepthTrainingDataset(protocol['archive'],split='train',augment=True,seed=seed) as dataset:
        eligible = {image_id:sum(0<a['area']<=CROP_POLICY['small_original_area_max']
                    for a in dataset.annotations_by_image[image_id]) for image_id in dataset.image_ids}
        counts, first_epoch = [],{}
        with (args.output/'expected_training_crops.csv').open('x',newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['epoch','image_id','eligible_small_instances','expected_crop'])
            for epoch in range(epochs):
                cropped = 0
                for image_id in dataset.image_ids:
                    key = f'course_crop512_v1:{seed}:{epoch}:{image_id}'.encode()
                    rng = random.Random(int.from_bytes(hashlib.sha256(key).digest(),'big'))
                    applied = bool(eligible[image_id] and rng.random()<CROP_POLICY['probability_on_eligible_images'])
                    cropped += applied
                    writer.writerow([epoch+1,image_id,eligible[image_id],int(applied)])
                    if epoch == 0:
                        first_epoch[image_id] = applied
                counts.append({'epoch':epoch+1,'cropped':cropped,'whole_image':len(dataset)-cropped,
                               'fraction_of_all_images':cropped/len(dataset)})
        actual = json.loads((ROOT/'reports/course_crop_focal/real_data_check_v2.json').read_text())
        for sample in actual['samples']:
            if first_epoch[sample['image_id']] != sample['crop_info']['applied']:
                raise AssertionError('Independent coin calculation differs from actual data check')
        result = {'status':'completed','created_at_utc':now(),
            'scope':'Expected crop frequency from complete train metadata, not an executed training receipt',
            'train_images':len(dataset),'annotations':dataset.annotation_count,
            'eligible_images':sum(n>0 for n in eligible.values()),'small_instances':sum(eligible.values()),
            'seed':seed,'epochs':epochs,'counts':counts,'predicted_cropped_image_exposures':sum(c['cropped'] for c in counts),
            'total_image_exposures':len(dataset)*epochs,
            'actual_examples_checked':len(actual['samples']),
            'annotation_sha256':dataset.annotation_sha256,
            'policy':CROP_POLICY,'protocol_path':str(args.protocol.resolve()),'protocol_sha256':sha256(args.protocol),
            'expected_manifest_sha256':sha256(args.output/'expected_training_crops.csv'),
            'source_sha256':{name:sha256(ROOT/name) for name in ('tools/audit_course_crop_coverage.py','rgbd_seg/course_crop.py')},
            'limitations':['Does not measure retained pixels, partial-instance counts, or AP; actual training audit remains required',
                           'Input resolution is still pending; crop application decisions do not depend on that choice']}
        write_json(args.output/'coverage.json',result)
        print(json.dumps({key:result[key] for key in ('train_images','eligible_images','small_instances','counts','predicted_cropped_image_exposures','total_image_exposures')}),flush=True)


if __name__ == '__main__':
    main()
