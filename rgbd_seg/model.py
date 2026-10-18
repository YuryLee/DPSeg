"""Explicit RGB-D path through Mask R-CNN; no GT is required at inference."""
from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import TwoMLPHead, FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNHeads, MaskRCNNPredictor
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.ops import MultiScaleRoIAlign, FeaturePyramidNetwork, FrozenBatchNorm2d


def freeze_batch_norm(module):
    """Preserve pretrained running statistics under small-batch fine-tuning."""
    for name, child in module.named_children():
        if isinstance(child, nn.BatchNorm2d):
            frozen = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                for key in ('weight', 'bias', 'running_mean', 'running_var'):
                    getattr(frozen, key).copy_(getattr(child, key))
            setattr(module, name, frozen)
        else:
            freeze_batch_norm(child)


def block(a, b, stride=1):
    return nn.Sequential(nn.Conv2d(a,b,3,stride,1), nn.GroupNorm(4,b), nn.ReLU())


class TinyFPN(nn.Module):
    """CPU smoke backbone. This is explicitly not ResNet-50."""
    out_channels = 32

    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(block(3,16,2), block(16,32,2), block(32,32))
        self.stage = nn.Sequential(block(32,64,2), block(64,64))
        self.fpn = FeaturePyramidNetwork([32,64],32)

    def forward(self,x):
        p2 = self.stem(x)
        return self.fpn(OrderedDict([('0',p2),('1',self.stage(p2))]))


class FusionBackbone(nn.Module):
    def __init__(self, rgb, mode, zero_init_depth=False):
        super().__init__()
        self.rgb, self.mode, self.out_channels = rgb, mode, rgb.out_channels
        c = self.out_channels
        if mode != 'E0':
            self.depth_encoder = nn.Sequential(block(2,16,2),block(16,32,2),block(32,32))
            self.depth_down = block(32,32,2)
            self.projections = nn.ModuleList([nn.Conv2d(32,c,1),nn.Conv2d(32,c,1)])
            if zero_init_depth:
                for projection in self.projections:
                    nn.init.zeros_(projection.weight)
                    nn.init.zeros_(projection.bias)
            if mode == 'E3':
                self.gates = nn.ModuleList([nn.Conv2d(2*c+1,1,1) for _ in range(2)])

    def forward(self, x):
        features = self.rgb(x[:,:3])
        if self.mode == 'E0':
            return features
        d2 = self.depth_encoder(x[:,3:5])
        ds = [d2, self.depth_down(d2)]
        for i,key in enumerate(['0','1']):
            p = features[key]
            q = self.projections[i](F.interpolate(ds[i],size=p.shape[-2:],mode='bilinear',align_corners=False))
            v = F.interpolate(x[:,4:5],size=p.shape[-2:],mode='nearest')
            if self.mode == 'E3':
                g = torch.sigmoid(self.gates[i](torch.cat([p,q,v],1)))
                q = v*g*q
            features[key] = p+q
        return features


class RGBDTransform(GeneralizedRCNNTransform):
    def normalize(self,image):
        return torch.cat([super().normalize(image[:3]),image[3:]],0)

    def resize(self,image,target=None):
        rgb,target = super().resize(image[:3],target)
        # Nearest-neighbor retains depth holes and uses the same geometric scale.
        dv = F.interpolate(image[None,3:],size=rgb.shape[-2:],mode='nearest')[0]
        return torch.cat([rgb,dv],0),target


class RGBDMaskRCNN(nn.Module):
    def __init__(self,core,config):
        super().__init__()
        self.core, self.config = core, dict(config)

    def forward(self,samples,targets=None):
        packed=[]
        for s in samples:
            image,depth,valid=s['image'],s['depth'],s['depth_valid']
            if image.ndim!=3 or image.shape[0]!=3 or depth.shape!=(1,*image.shape[-2:]) or valid.shape!=depth.shape:
                raise ValueError('RGB, depth and validity must be aligned CHW tensors')
            v=(valid>0.5)&torch.isfinite(depth)&(depth>0)
            d=torch.where(v,depth,torch.zeros_like(depth)).clamp(0,self.config['max_depth_m'])/self.config['max_depth_m']
            packed.append(torch.cat([image,d,v.to(image.dtype)],0))
        return self.core(packed,targets)


def build_model(config):
    config = {'mode':'E3','arch':'tiny','num_classes':2,'size':128,'seed':17,'max_depth_m':5.0,**config}
    if config['mode'] not in ('E0','E2','E3') or config['arch'] not in ('tiny','r50','r50_v2'):
        raise ValueError('Expected mode E0/E2/E3 and arch tiny/r50/r50_v2')
    if config['max_depth_m']<=0 or config['num_classes']<2:
        raise ValueError('Invalid maximum depth or class count')
    torch.manual_seed(config['seed'])
    n=config['num_classes']; size=config['size']
    if config['arch'] in ('r50','r50_v2'):
        # No implicit network download; optional local RGB checkpoint loaded by CLI.
        builder=maskrcnn_resnet50_fpn_v2 if config['arch']=='r50_v2' else maskrcnn_resnet50_fpn
        core=builder(weights=None,weights_backbone=None,num_classes=n,min_size=size,
                     max_size=config.get('max_size',round(size*1.67)),
                     box_score_thresh=config.get('score_threshold',0.05),
                     box_detections_per_img=config.get('max_detections',100))
        if config.get('frozen_batch_norm',False):
            freeze_batch_norm(core)
        layers=config.get('trainable_backbone_layers',5)
        if not 0<=layers<=5:
            raise ValueError('trainable_backbone_layers must be in [0, 5]')
        prefixes=['layer4','layer3','layer2','layer1','conv1'][:layers]
        for name,parameter in core.backbone.body.named_parameters():
            parameter.requires_grad_(any(name.startswith(prefix) for prefix in prefixes))
        if 'anchor_sizes' in config:
            sizes=config['anchor_sizes']
            if len(sizes)!=5 or any(v<=0 for v in sizes):
                raise ValueError('Five positive R50 pyramid anchor sizes are required')
            core.rpn.anchor_generator=AnchorGenerator(tuple((v,) for v in sizes),((0.5,1.,2.),)*5)
    else:
        c=32
        core=MaskRCNN(TinyFPN(),num_classes=None,min_size=size,max_size=round(size*1.5),
            rpn_anchor_generator=AnchorGenerator(((8,16,32),(32,64,96)),((0.5,1.,2.),)*2),
            rpn_pre_nms_top_n_train=256,rpn_pre_nms_top_n_test=192,
            rpn_post_nms_top_n_train=128,rpn_post_nms_top_n_test=96,rpn_batch_size_per_image=128,
            box_roi_pool=MultiScaleRoIAlign(['0','1'],5,2),
            box_head=TwoMLPHead(c*5*5,128),box_predictor=FastRCNNPredictor(128,n),
            box_batch_size_per_image=64,box_detections_per_img=40,box_score_thresh=0.01,
            mask_roi_pool=MultiScaleRoIAlign(['0','1'],14,2),
            mask_head=MaskRCNNHeads(c,[32,32],1),mask_predictor=MaskRCNNPredictor(32,32,n))
    # Construct shared RGB/RPN/RoI weights before variant-specific branches.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config['seed']+1)
        core.backbone=FusionBackbone(core.backbone,config['mode'],config.get('zero_init_depth',False))
    t=core.transform
    core.transform=RGBDTransform(t.min_size,t.max_size,t.image_mean,t.image_std,size_divisible=32)
    if 'train_scales' in config:
        scales=config['train_scales']
        if not scales or any(not isinstance(v,int) or v<=0 for v in scales) or scales[-1]!=size:
            raise ValueError('Positive train_scales must end with the configured evaluation size')
        core.transform.min_size=tuple(scales)
    return RGBDMaskRCNN(core,config)


def load_model(path,device='cpu'):
    obj=torch.load(path,map_location='cpu',weights_only=True)
    model=build_model(obj['config'])
    model.load_state_dict(obj['state_dict'],strict=True)
    return model.to(device).eval(),obj
