import unittest

import numpy as np
from pycocotools import mask as mask_utils

from rgbd_seg.course_point_guard import guard_predictions


def prediction(mask,score=.7):
    rle=mask_utils.encode(np.asfortranarray(mask,dtype=np.uint8))
    return {'category_id':1,'score':score,'segmentation':{'size':rle['size'],'counts':rle['counts'].decode('ascii')}}


class PointGuardTests(unittest.TestCase):
    def test_small_large_and_single_pixel_masks_keep_correct_identity_and_order(self):
        small=np.zeros((1024,1024),dtype=np.uint8);small[10:20,10:20]=1
        expanded=small.copy();expanded[8:23,8:23]=1
        large=np.zeros_like(small);large[100:150,100:150]=1
        other=np.zeros_like(small);other[500:550,500:550]=1
        one=np.zeros_like(small);one[0,0]=1
        shifted=np.zeros_like(small);shifted[0,1]=1
        old=[prediction(small,.8),prediction(large,.7),prediction(one,.6)]
        new=[prediction(expanded,.8),prediction(other,.7),prediction(shifted,.6)]
        values,stats=guard_predictions(old,new)
        self.assertEqual(values,[old[0],new[1],old[2]])
        self.assertEqual(stats['fallback_instances'],2)
        self.assertEqual(stats['small_baseline_instances'],2)

    def test_iou_at_threshold_is_accepted_and_original_inputs_are_preserved(self):
        old=np.zeros((1024,1024),dtype=np.uint8);old[0,:10]=1
        new=old.copy();new[0,8:10]=0
        baseline=[prediction(old)];candidate=[prediction(new)]
        values,stats=guard_predictions(baseline,candidate)
        self.assertEqual(values,candidate)
        self.assertEqual(stats['fallback_instances'],0)
        self.assertEqual(int(mask_utils.area(baseline[0]['segmentation'])),10)

    def test_changed_score_or_count_is_rejected(self):
        mask=np.ones((1024,1024),dtype=np.uint8)
        with self.assertRaises(ValueError):guard_predictions([prediction(mask)],[prediction(mask,.6)])
        with self.assertRaises(ValueError):guard_predictions([prediction(mask)],[])


if __name__=='__main__':unittest.main()
