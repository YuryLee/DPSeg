"""Explicit local COCO initialization; never downloads or executes checkpoint code."""
import hashlib
from pathlib import Path
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights


def initialize_rgb(model, path, foreground_mean=False):
    source=torch.load(path,map_location='cpu',weights_only=True)
    for key in ('state_dict','model'):
        if isinstance(source,dict) and key in source and isinstance(source[key],dict):
            source=source[key]
    if not isinstance(source,dict) or not all(torch.is_tensor(v) for v in source.values()):
        raise ValueError('Expected a local tensor state_dict')
    target=model.core.state_dict();copied=[];adapted=[]
    names=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1.meta['categories']
    foreground=[i for i,name in enumerate(names) if i and name!='N/A']
    for key,value in source.items():
        key=key.removeprefix('core.')
        mapped='backbone.rgb.'+key[len('backbone.'):] if key.startswith('backbone.') and not key.startswith('backbone.rgb.') else key
        if mapped not in target:
            continue
        if target[mapped].shape==value.shape:
            target[mapped]=value;copied.append(mapped)
        elif foreground_mean and model.config['num_classes']==2:
            if key.startswith(('roi_heads.box_predictor.cls_score.','roi_heads.mask_predictor.mask_fcn_logits.')) and value.shape[0]==91:
                merged=torch.stack([value[0],value[foreground].mean(0)])
            elif key.startswith('roi_heads.box_predictor.bbox_pred.') and value.shape[0]==364:
                by_class=value.reshape(91,4,*value.shape[1:])
                merged=torch.stack([by_class[0],by_class[foreground].mean(0)]).reshape(8,*value.shape[1:])
            else:
                continue
            if merged.shape==target[mapped].shape:
                target[mapped]=merged;adapted.append(mapped)
    if not copied:
        raise ValueError('No compatible RGB tensors in initialization checkpoint')
    model.core.load_state_dict(target)
    return {'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'copied_keys':copied,'adapted_keys':adapted,
            'head_initialization':'COCO background plus mean of 80 valid foreground heads' if foreground_mean else 'shape-compatible tensors; incompatible heads remain randomly initialized'}
