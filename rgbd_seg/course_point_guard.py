"""One fixed prediction-only guard for the trained point-head experiment."""
from __future__ import annotations

from pycocotools import mask as mask_utils

POLICY={
    'baseline':'native_probability_depth',
    'candidate':'point_depth_probability_depth',
    'small_predicted_area_max':1024,
    'minimum_old_new_mask_iou':.8,
    'trigger':'baseline predicted area <=1024 and old/new mask IoU strictly <0.8',
    'fallback':'entire original probability-depth-refined mask for that prediction',
    'scores_and_order':'unchanged original class-times-mask-quality score and query order',
    'scope':'one fixed guard; no threshold grid; inference uses predictions only',
}


def guard_predictions(baseline,candidate):
    if len(baseline)!=len(candidate):raise ValueError('Guard inputs must have identical prediction counts')
    values=[]
    statistics={'instances':len(baseline),'small_baseline_instances':0,'fallback_instances':0,
                'fallback_old_area_sum':0,'fallback_new_area_sum':0}
    for old,new in zip(baseline,candidate):
        if (old['category_id'],old['score'])!=(new['category_id'],new['score']):
            raise ValueError('Guard inputs must preserve instance identities/scores')
        if old['segmentation']['size']!=new['segmentation']['size']:
            raise ValueError('Guard mask sizes differ')
        area=int(mask_utils.area(old['segmentation']))
        use_old=False
        if area<=POLICY['small_predicted_area_max']:
            statistics['small_baseline_instances']+=1
            iou=float(mask_utils.iou([old['segmentation']],[new['segmentation']],[0])[0,0])
            use_old=iou<POLICY['minimum_old_new_mask_iou']
            if use_old:
                statistics['fallback_instances']+=1
                statistics['fallback_old_area_sum']+=area
                statistics['fallback_new_area_sum']+=int(mask_utils.area(new['segmentation']))
        chosen=old if use_old else new
        values.append({'category_id':new['category_id'],'score':new['score'],
                       'segmentation':dict(chosen['segmentation'])})
    return values,statistics
