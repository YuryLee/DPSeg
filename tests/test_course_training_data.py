import gzip
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from rgbd_seg.course_training_data import CourseTrainingDataset, collate_fn


class CourseTrainingDataTests(unittest.TestCase):
    def make_archive(self, folder, *, split='train', bad_area=False, missing_rgb=False):
        path=Path(folder)/'data.zip'
        rgb=np.zeros((8,10,3),dtype=np.uint8);rgb[:,0,0]=255
        first=np.zeros((8,10),dtype=np.uint8);first[1:4,2:6]=1
        second=np.zeros_like(first);second[2:5,4:8]=1
        pixel=np.zeros_like(first);pixel[7,9]=1
        masks=[first,second,pixel,np.zeros_like(first)]
        annotations=[]
        for i,mask in enumerate(masks):
            rle=mask_utils.encode(np.asfortranarray(mask));rle['counts']=rle['counts'].decode()
            annotations.append({'id':100+i,'image_id':17,'category_id':1,'segmentation':rle,
                                'area':int(mask.sum())+(1 if bad_area and i==0 else 0),
                                'bbox':[-20,-30,-1,-1], 'iscrowd':0})
        record={'id':17,'file_name':f'rgb/{split}/0017_v000.png','depth_file_name':f'depth/{split}/0017_v000.png',
                'height':8,'width':10,'scene_id':'0017','split':split,'depth_scale_m':.0001,'depth_invalid_code':0}
        with ZipFile(path,'w') as z:
            z.writestr(f'annotations/{split}/scene.json.gz',gzip.compress(json.dumps({'images':[record],
                'annotations':annotations,'categories':[{'id':1,'name':'component','supercategory':'component'}]}).encode()))
            if not missing_rgb:
                b=io.BytesIO();Image.fromarray(rgb).save(b,format='PNG');z.writestr(record['file_name'],b.getvalue())
            b=io.BytesIO();Image.fromarray(np.ones((8,10),dtype=np.uint16)).save(b,format='PNG');z.writestr(record['depth_file_name'],b.getvalue())
        return path,masks,annotations

    def test_overlap_original_area_one_pixel_and_empty_rule(self):
        with tempfile.TemporaryDirectory() as folder:
            path,masks,annotations=self.make_archive(folder)
            with CourseTrainingDataset(path,augment=False,seed=17) as dataset:
                image,target=dataset[0]
                self.assertEqual(tuple(image.shape),(3,8,10))
                self.assertEqual(image.dtype,torch.float32)
                self.assertEqual(target['masks'].dtype,torch.uint8)
                self.assertEqual(tuple(target['masks'].shape),(3,8,10))
                self.assertEqual(int((target['masks'][0]&target['masks'][1]).sum()),4)
                self.assertEqual(target['boxes'].tolist(),[[2,1,6,4],[4,2,8,5],[9,7,10,8]])
                self.assertEqual(target['area'].tolist(),[12,12,1])
                self.assertEqual(target['labels'].tolist(),[1,1,1])
                self.assertEqual(dataset.summary()['zero_area_annotations_excluded_from_targets'],1)
                self.assertEqual(dataset.coco_ground_truth()['annotations'],annotations)
                self.assertEqual(len(list(Path(folder).iterdir())),1)

    def test_flip_is_aligned_and_independent_of_model_rng(self):
        with tempfile.TemporaryDirectory() as folder:
            path,masks,_=self.make_archive(folder)
            with CourseTrainingDataset(path,augment=True,seed=17) as dataset:
                for epoch in range(100):
                    dataset.set_epoch(epoch)
                    if dataset._flip_for(17):break
                self.assertTrue(dataset._flip_for(17))
                image,target=dataset[0]
                self.assertEqual(image[0,0,-1].item(),1)
                np.testing.assert_array_equal(target['masks'][0].numpy(),masks[0][:,::-1])
                self.assertEqual(target['boxes'].tolist(),[[4,1,8,4],[2,2,6,5],[0,7,1,8]])
                torch.manual_seed(874);torch.rand(999)
                image2,target2=dataset[0]
                self.assertTrue(torch.equal(image,image2))
                self.assertTrue(torch.equal(target['masks'],target2['masks']))

    def test_val_never_augments_and_collate_preserves_instances(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_,_=self.make_archive(folder,split='val')
            with CourseTrainingDataset(path,'val',augment=True,seed=17) as dataset:
                self.assertFalse(dataset._flip_for(17))
                images,targets=collate_fn([dataset[0],dataset[0]])
                self.assertIsInstance(images,list);self.assertIsInstance(targets,list)
                self.assertEqual(len(targets[0]['masks']),3)

    def test_bad_area_missing_file_and_test_split_rejected(self):
        for field in ('bad_area','missing_rgb'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as folder:
                path,_,_=self.make_archive(folder,**{field:True})
                with self.assertRaises(ValueError):CourseTrainingDataset(path)
        with self.assertRaises(ValueError):CourseTrainingDataset('unused.zip','test')

    def test_no_eager_mask_decode_and_zip_context_closes(self):
        with tempfile.TemporaryDirectory() as folder:
            path,_,_=self.make_archive(folder)
            with patch('rgbd_seg.course_training_data.mask_utils.decode',side_effect=AssertionError('eager decode')):
                dataset=CourseTrainingDataset(path,augment=False)
            with dataset:
                dataset.load_rgb(0);handle=dataset._archive
                dataset.load_rgb(0);self.assertIs(handle,dataset._archive)
            self.assertIsNone(dataset._archive)
            self.assertIsNone(handle.fp)


if __name__=='__main__':unittest.main()
