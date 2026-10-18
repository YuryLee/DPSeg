import gc
from pathlib import Path
import unittest

import torch

from rgbd_seg.course_focal import MaskFocalLoss
from rgbd_seg.course_optimized import build_trainable, FAMILIES, tensor_digest


class SharedConstructionTests(unittest.TestCase):
    def test_independent_methods_preserve_initial_weights_and_inference_resolution(self):
        torch.set_num_threads(2)
        checkpoint = Path(__file__).resolve().parents[1]/'pretrained/mask2former_swin_tiny_coco_instance'
        initial = None
        for method in ('native','crop','focal'):
            model,metadata = build_trainable(checkpoint,method=method,resolution=800,device='cpu',seed=17)
            actual = tensor_digest(model.state_dict())
            if initial is None:
                initial = actual
            self.assertEqual(actual,initial)
            self.assertEqual(model.short_edge,800)
            self.assertEqual(metadata['preprocess']['shortest_edge'],800)
            self.assertEqual(metadata['experiment_family'],FAMILIES[method])
            self.assertEqual('training_crop_policy' in metadata['preprocess'],method=='crop')
            self.assertEqual(isinstance(model.model.criterion,MaskFocalLoss),method=='focal')
            self.assertEqual(metadata['fusion']['rpb_active_at_input'],[False,False,True,True])
            del model
            gc.collect()

    def test_unknown_or_combined_method_is_rejected(self):
        with self.assertRaisesRegex(ValueError,'Unknown independent intervention'):
            build_trainable('unused',method='crop+focal',resolution=1024)


if __name__ == '__main__':
    unittest.main()
