import random
import unittest
import torch
from torch import nn
from torchvision.ops import FrozenBatchNorm2d
from rgbd_seg.model import build_model, freeze_batch_norm
from rgbd_seg.augment import augment


class TransferTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)

    def test_zero_depth_residual_preserves_rgb_and_starts_learning(self):
        for mode in ['E2','E3']:
            model=build_model({'mode':mode,'zero_init_depth':True}).core.backbone
            x=torch.rand(1,5,64,64);x[:,4]=1
            rgb=model.rgb(x[:,:3]);fused=model(x)
            for k in rgb:torch.testing.assert_close(rgb[k],fused[k],rtol=0,atol=0)
            sum(v.square().mean() for v in fused.values()).backward()
            self.assertGreater(model.projections[0].weight.grad.abs().sum().item(),0)

    def test_frozen_bn_preserves_statistics_and_eval_output(self):
        net=nn.Sequential(nn.BatchNorm2d(4)).eval();x=torch.rand(3,4,8,8)
        with torch.no_grad():net[0].running_mean.fill_(2);net[0].running_var.fill_(3);expected=net(x)
        freeze_batch_norm(net);self.assertIsInstance(net[0],FrozenBatchNorm2d)
        net.train();torch.testing.assert_close(net(x),expected)
        torch.testing.assert_close(net[0].running_mean,torch.full((4,),2.))

    def test_joint_augmentation_copies_and_recomputes_masks(self):
        mask=torch.zeros(2,400,600,dtype=torch.uint8);mask[0,50:80,50:80]=1;mask[1,280:390,450:570]=1
        depth=mask.sum(0).float()[None]+1
        s={'image':depth.repeat(3,1,1)/3,'depth':depth,'depth_valid':torch.ones_like(depth)}
        t={'boxes':torch.tensor([[50.,50.,80.,80.],[450.,280.,570.,390.]]),'labels':torch.tensor([1,1]),'masks':mask,'area':mask.flatten(1).sum(1).float(),'iscrowd':torch.tensor([0,0])}
        a,b=augment(s,t,random.Random(8),1.)
        self.assertEqual(tuple(s['image'].shape),(3,400,600))
        self.assertTrue(torch.equal(t['masks'],mask))
        torch.testing.assert_close(a['depth'][0]-1,b['masks'].sum(0).float())
        torch.testing.assert_close(b['area'],b['masks'].flatten(1).sum(1).float())
        for m,box in zip(b['masks'],b['boxes']):
            yy,xx=torch.where(m>0);self.assertEqual(box.tolist(),[float(xx.min()),float(yy.min()),float(xx.max()+1),float(yy.max()+1)])

if __name__=='__main__':unittest.main()
