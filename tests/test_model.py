import unittest
import torch
from rgbd_seg.model import build_model


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def test_validity_and_sensitivity(self):
        m=build_model({'mode':'E3'}).eval().core.backbone
        x=torch.rand(1,5,64,64); x[:,4]=0
        with torch.no_grad():
            rgb=m.rgb(x[:,:3]); out=m(x)
            for k in rgb: torch.testing.assert_close(rgb[k],out[k],rtol=0,atol=0)
            x[:,4]=1; a=m(x)['0']; x[:,3]*=0.3; b=m(x)['0']
            self.assertGreater((a-b).abs().max().item(),1e-5)

    def test_backward_empty_and_reload(self):
        m=build_model({'mode':'E3','size':64})
        s={'image':torch.rand(3,64,64),'depth':torch.ones(1,64,64),'depth_valid':torch.ones(1,64,64)}
        masks=torch.zeros(1,64,64,dtype=torch.uint8); masks[:,12:25,15:30]=1
        t={'boxes':torch.tensor([[15.,12.,30.,25.]]),'labels':torch.ones(1,dtype=torch.long),'masks':masks}
        losses=m([s],[t]); loss=sum(losses.values()); self.assertTrue(torch.isfinite(loss)); loss.backward()
        grad=m.core.backbone.depth_encoder[0][0].weight.grad
        self.assertIsNotNone(grad); self.assertGreater(grad.abs().sum().item(),0)
        empty={'boxes':torch.zeros(0,4),'labels':torch.zeros(0,dtype=torch.long),'masks':torch.zeros(0,64,64,dtype=torch.uint8)}
        self.assertTrue(torch.isfinite(sum(m([s],[empty]).values())))
        copy=build_model(m.config); copy.load_state_dict(m.state_dict()); m.eval(); copy.eval()
        with torch.no_grad():
            a=m([s])[0]; b=copy([s])[0]
            torch.testing.assert_close(a['masks'],b['masks'],rtol=0,atol=0)

    def test_shared_initialization(self):
        a=build_model({'mode':'E2'}).state_dict(); b=build_model({'mode':'E3'}).state_dict()
        for k in a: torch.testing.assert_close(a[k],b[k],rtol=0,atol=0)

    def test_resize_validity(self):
        m=build_model({'size':64}); x=torch.rand(5,48,60); x[4]=(x[4]>.5).float()
        resized,_=m.core.transform.resize(x)
        self.assertTrue(set(resized[4].unique().tolist())<={0.,1.})

if __name__=='__main__': unittest.main()
