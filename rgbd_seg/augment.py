"""Joint RGB/depth/mask augmentation driven only by training annotations."""
import torch
from torchvision.transforms import functional as TF


def augment(sample, target, rng, crop_probability=0.0):
    sample={k:v.clone() for k,v in sample.items()}
    target={k:v.clone() if torch.is_tensor(v) else v for k,v in target.items()}
    h,w=sample['image'].shape[-2:]
    if len(target['masks']) and rng.random()<crop_probability:
        # Prefer small/medium training instances, but keep contextual pixels.
        candidates=torch.where(target['area']<96**2)[0].tolist() or list(range(len(target['masks'])))
        chosen=rng.choice(candidates);x1,y1,x2,y2=target['boxes'][chosen].tolist()
        ch=min(h,max(256,round(h*rng.uniform(0.5,0.8)),round((y2-y1)*1.5)))
        cw=min(w,max(256,round(w*rng.uniform(0.5,0.8)),round((x2-x1)*1.5)))
        top=max(0,min(h-ch,round((y1+y2-ch)/2+rng.uniform(-0.1,0.1)*ch)))
        left=max(0,min(w-cw,round((x1+x2-cw)/2+rng.uniform(-0.1,0.1)*cw)))
        sample={k:v[:,top:top+ch,left:left+cw] for k,v in sample.items()}
        masks=target['masks'][:,top:top+ch,left:left+cw];area=masks.flatten(1).sum(1)
        keep=area>0
        for key in ('labels','iscrowd'):
            if key in target:target[key]=target[key][keep]
        target['masks']=masks[keep];target['area']=area[keep].float()
        boxes=[]
        for mask in target['masks']:
            yy,xx=torch.where(mask>0);boxes.append([xx.min(),yy.min(),xx.max()+1,yy.max()+1])
        target['boxes']=torch.tensor(boxes,dtype=torch.float32).reshape(-1,4)
        h,w=ch,cw
    if rng.random()<0.5:
        sample={k:v.flip(-1) for k,v in sample.items()}
        target['masks']=target['masks'].flip(-1)
        boxes=target['boxes'].clone();boxes[:,[0,2]]=w-target['boxes'][:,[2,0]];target['boxes']=boxes
    image=TF.adjust_brightness(sample['image'],rng.uniform(0.85,1.15))
    image=TF.adjust_contrast(image,rng.uniform(0.85,1.15))
    sample['image']=TF.adjust_saturation(image,rng.uniform(0.85,1.15)).clamp(0,1)
    return sample,target
