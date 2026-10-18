"""Semantic checks for independent crop and Focal experiments (CPU only)."""
import copy
from pathlib import Path
import random
import types
import unittest
from unittest.mock import patch

import torch
from transformers import Mask2FormerConfig
from transformers.models.mask2former.modeling_mask2former import (
    Mask2FormerLoss, sigmoid_cross_entropy_loss,
)

from rgbd_seg.course_crop import choose_crop, crop_sample_target, CourseCropTrainingDataset
from rgbd_seg.course_depth_data import depth_to_xyz, transform_depth, CourseDepthTrainingDataset
from rgbd_seg.course_focal import MaskFocalLoss, focal_mask_loss, apply_focal
from rgbd_seg.course_resolution import local_num_masks


def example():
    h, w = 12, 14
    image = torch.arange(3*h*w, dtype=torch.float32).reshape(3,h,w)
    depth = torch.arange(h*w, dtype=torch.float32).reshape(1,h,w)/100+1
    valid = torch.ones_like(depth, dtype=torch.bool)
    valid[:,4,5] = False
    depth[~valid] = 0
    sample = dict(image=image, depth_m=depth, depth_valid=valid,
                  camera_K=torch.tensor([[20.,0,7],[0,30,6],[0,0,1]]))
    masks = torch.zeros(4,h,w,dtype=torch.uint8)
    masks[0,3,4] = 1  # single-pixel anchor
    masks[1,2:6,3:7] = 1  # overlaps anchor and crosses crop boundaries
    masks[2,0,0] = 1  # fully outside
    masks[3,4:6,5:7] = 1  # overlaps second instance
    target = dict(masks=masks, labels=torch.ones(4,dtype=torch.long),
        area=masks.flatten(1).sum(1).float(), boxes=torch.zeros(4,4),
        iscrowd=torch.zeros(4,dtype=torch.long), image_id=torch.tensor([42]))
    return sample, target


class CropTests(unittest.TestCase):
    def test_aligned_crop_keeps_overlap_partial_and_one_pixel(self):
        sample, target = example()
        before = copy.deepcopy((sample,target))
        cropped, labels = crop_sample_target(sample,target,(3,4,5,5,0))
        for key in ('image','depth_m','depth_valid'):
            self.assertTrue(torch.equal(cropped[key], sample[key][...,3:8,4:9]))
        self.assertEqual(labels['area'].tolist(), [1.,9.,4.])
        self.assertEqual(labels['original_area'].tolist(), [1.,16.,4.])
        self.assertEqual(labels['boxes'].tolist(), [[0,0,1,1],[0,0,3,3],[1,1,3,3]])
        self.assertTrue(bool(labels['masks'][0,0,0] & labels['masks'][1,0,0]))
        self.assertEqual(labels['crop_info']['truncated_instances'], 1)
        for actual, old in zip((sample,target), before):
            for key in actual:
                self.assertTrue(torch.equal(actual[key], old[key]), key)

    def test_crop_preserves_camera_xyz_before_and_after_flip(self):
        for flip in (False,True):
            sample,target = example()
            if flip:
                sample['depth_m'],sample['depth_valid'],sample['camera_K'] = transform_depth(
                    sample['depth_m'],sample['depth_valid'],sample['camera_K'],horizontal_flip=True)
                sample['image'] = sample['image'].flip(-1)
                target['masks'] = target['masks'].flip(-1)
            expected = depth_to_xyz(sample['depth_m'],sample['depth_valid'],sample['camera_K'])
            result,_ = crop_sample_target(sample,target,(3,4,5,5,None))
            observed = depth_to_xyz(result['depth_m'],result['depth_valid'],result['camera_K'])
            torch.testing.assert_close(observed, expected[:,3:8,4:9])

    def test_deterministic_anchor_and_no_global_rng_consumption(self):
        _,target = example()
        py_state,torch_state = random.getstate(),torch.get_rng_state()
        windows = []
        for epoch in range(20):
            a = choose_crop(target,seed=17,epoch=epoch,size=5,probability=1)
            self.assertEqual(a,choose_crop(target,seed=17,epoch=epoch,size=5,probability=1))
            top,left,h,w,anchor = a
            self.assertGreater(target['masks'][anchor,top:top+h,left:left+w].sum(),0)
            windows.append(a)
        self.assertGreater(len(set(windows)),1)
        self.assertEqual(random.getstate(),py_state)
        self.assertTrue(torch.equal(torch.get_rng_state(),torch_state))
        self.assertIsNone(choose_crop(target,seed=17,epoch=0,size=5,probability=0))
        target['area'].fill_(2000)
        self.assertIsNone(choose_crop(target,seed=17,epoch=0,size=5,probability=1))

    def test_dataset_only_augmented_training_uses_crop(self):
        dataset = CourseCropTrainingDataset.__new__(CourseCropTrainingDataset)
        dataset.seed,dataset.epoch = 17,0
        sample,target = example()
        with patch.object(CourseDepthTrainingDataset,'__getitem__',return_value=(sample,target)):
            with patch('rgbd_seg.course_crop.choose_crop',return_value=(3,4,5,5,0)) as choose:
                for split,augment in (('val',True),('train',False),('test',False)):
                    dataset.split,dataset.augment = split,augment
                    actual = dataset[0]
                    self.assertIs(actual[0],sample)
                    self.assertIs(actual[1],target)
                choose.assert_not_called()
                dataset.split,dataset.augment = 'train',True
                transformed,labels = dataset[0]
                self.assertEqual(tuple(transformed['image'].shape),(3,5,5))
                self.assertTrue(labels['crop_info']['applied'])
                choose.assert_called_once()


class FocalTests(unittest.TestCase):
    def test_gamma_zero_matches_native_loss_and_gradient_exactly(self):
        torch.manual_seed(17)
        logits = torch.randn(3,32,requires_grad=True)
        labels = torch.rand_like(logits)  # native point sampling yields soft targets
        a = sigmoid_cross_entropy_loss(logits,labels,3)
        b = focal_mask_loss(logits,labels,3,gamma=0)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(torch.autograd.grad(a,logits)[0],torch.autograd.grad(b,logits)[0]))

    def test_modulation_and_finite_extreme_gradients(self):
        p = torch.tensor([.99,.9,.5,.1],dtype=torch.float64)
        logits = torch.logit(p)[:,None].requires_grad_()
        labels = torch.ones_like(logits)
        actual = focal_mask_loss(logits,labels,4)
        expected = (-(1-p).square()*p.log()).mean()
        torch.testing.assert_close(actual,expected)
        grad = torch.autograd.grad(actual,logits)[0]
        self.assertLess(abs(grad[0].item()),abs(grad[-1].item()))
        extreme = torch.tensor([[-100.,100.,-100.,100.,0.]],requires_grad=True)
        loss = focal_mask_loss(extreme,torch.tensor([[0.,1.,1.,0.,.4]]),1)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(torch.autograd.grad(loss,extreme)[0]).all())

    def test_full_criterion_gamma_zero_preserves_matching_points_aux_and_gradients(self):
        config = Mask2FormerConfig(num_labels=1,train_num_points=32)
        weights = dict(loss_cross_entropy=2,loss_mask=5,loss_dice=5)
        native = Mask2FormerLoss(config,weights)
        native.get_num_masks = types.MethodType(local_num_masks,native)
        focal = MaskFocalLoss(config,weights,gamma=0)
        torch.manual_seed(19)
        masks = [torch.randint(0,2,(n,12,12)).float() for n in (2,3)]
        classes = [torch.zeros(n,dtype=torch.long) for n in (2,3)]
        outputs = [dict(masks_queries_logits=torch.randn(2,5,8,8,requires_grad=True),
                        class_queries_logits=torch.randn(2,5,2,requires_grad=True)) for _ in range(3)]
        parameters = [t for output in outputs for t in output.values()]
        results = []
        for criterion in (native,focal):
            torch.manual_seed(23)
            with patch.object(criterion.matcher,'forward',wraps=criterion.matcher.forward) as matcher:
                losses = criterion(**outputs[0],mask_labels=masks,class_labels=classes,
                                   auxiliary_predictions=outputs[1:])
                self.assertEqual(matcher.call_count,3)
            gradients = torch.autograd.grad(sum(losses.values()),parameters)
            results.append((losses,gradients,torch.get_rng_state()))
        self.assertEqual(focal.mask_loss_calls,3)
        self.assertEqual(len(results[0][0]),9)
        for key,value in results[0][0].items():
            torch.testing.assert_close(value,results[1][0][key],rtol=0,atol=0)
        for a,b in zip(results[0][1],results[1][1]):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        self.assertTrue(torch.equal(results[0][2],results[1][2]))

    def test_empty_targets_have_finite_loss_and_gradients(self):
        criterion = MaskFocalLoss(Mask2FormerConfig(num_labels=1,train_num_points=32),{})
        masks = torch.randn(2,5,8,8,requires_grad=True)
        classes = torch.randn(2,5,2,requires_grad=True)
        losses = criterion(masks,classes,[torch.empty(0,12,12)]*2,
                           [torch.empty(0,dtype=torch.long)]*2)
        self.assertEqual(losses['loss_mask'].item(),0)
        self.assertEqual(losses['loss_dice'].item(),0)
        for gradient in torch.autograd.grad(sum(losses.values()),(masks,classes)):
            self.assertTrue(torch.isfinite(gradient).all())

    def test_apply_preserves_all_model_tensors_and_preprocessing(self):
        from rgbd_seg.course_resolution import build_trainable
        from tools.run_course_resolution_ddp import model_digest
        root = Path(__file__).resolve().parents[1]
        adapter,metadata = build_trainable(root/'pretrained/mask2former_swin_tiny_coco_instance',
                                           device='cpu',seed=17,resolution=1024)
        digest = model_digest(adapter)
        count = sum(p.numel() for p in adapter.parameters())
        preprocess = copy.deepcopy(metadata['preprocess'])
        adapter,updated = apply_focal(adapter,metadata)
        self.assertEqual(model_digest(adapter),digest)
        self.assertEqual(sum(p.numel() for p in adapter.parameters()),count)
        self.assertEqual(updated['preprocess'],preprocess)
        self.assertIsInstance(adapter.model.criterion,MaskFocalLoss)
        self.assertNotIn('focal_policy',metadata['loss'])


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
