"""CPU tests: native losses, overlap fidelity, gradients and state reload."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from transformers import Mask2FormerConfig, SwinConfig

from rgbd_seg.course_finetune_mask2former import (
    BACKBONE_PREFIX, CourseTrainableMask2Former, LossReportingMask2Former,
    build_trainable, component_predictions,
)
from rgbd_seg.course_mask2former import DEFAULT_CHECKPOINT


def tiny_adapter():
    # Same HF architecture/loss path, smaller dimensions for an actual CPU
    # backward pass. This is not a quality test of the 47M-parameter checkpoint.
    config = Mask2FormerConfig(
        backbone_config=SwinConfig(embed_dim=32, depths=[1, 1, 1, 1],
            num_heads=[1, 2, 4, 8], window_size=2, drop_path_rate=0,
            out_features=["stage1", "stage2", "stage3", "stage4"]),
        feature_size=32, mask_feature_size=32, hidden_dim=32,
        encoder_feedforward_dim=64, dim_feedforward=64,
        encoder_layers=1, decoder_layers=2, num_attention_heads=4,
        num_queries=4, num_labels=1, train_num_points=32, dropout=0)
    return CourseTrainableMask2Former(LossReportingMask2Former(config),
        mean=[.485, .456, .406], std=[.229, .224, .225], short_edge=64, max_size=96)


def sample():
    image = torch.rand(3, 64, 64)
    masks = torch.zeros(2, 64, 64, dtype=torch.uint8)
    masks[0, 10:40, 10:40] = 1
    masks[1, 20:50, 20:50] = 1
    return image, {"masks": masks, "labels": torch.ones(2, dtype=torch.int64)}


class CourseFinetuneMask2FormerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.default_generator.manual_seed(17)

    def test_independent_overlapping_targets_and_normalization(self):
        adapter = tiny_adapter()
        image, target = sample()
        batch, sizes = adapter.prepare_batch([image], [target])
        self.assertEqual(batch["mask_labels"][0].shape, (2, 64, 64))
        self.assertTrue(torch.equal(batch["mask_labels"][0].to(torch.uint8), target["masks"]))
        self.assertEqual(int((batch["mask_labels"][0].sum(0) == 2).sum()), 400)
        self.assertEqual(batch["class_labels"][0].tolist(), [0, 0])
        self.assertTrue(bool(batch["pixel_mask"].all()))
        torch.testing.assert_close(batch["pixel_values"][0], (image - adapter.image_mean) / adapter.image_std)
        self.assertEqual(sizes["padded_size"], (64, 64))

    def test_padding_validity_and_mask_coordinates(self):
        adapter = tiny_adapter()
        first = torch.ones(3, 64, 64)
        second = torch.ones(3, 64, 128)
        targets = [{"masks": torch.ones(1, *im.shape[-2:], dtype=torch.uint8),
                    "labels": torch.ones(1, dtype=torch.int64)} for im in (first, second)]
        batch, sizes = adapter.prepare_batch([first, second], targets)
        self.assertEqual(sizes["resized_sizes"], [(64, 64), (48, 96)])
        self.assertEqual(sizes["padded_size"], (64, 96))
        self.assertEqual(batch["pixel_mask"].sum((1, 2)).tolist(), [4096, 4608])
        self.assertEqual(batch["mask_labels"][1].sum().item(), 4608)
        self.assertEqual(batch["pixel_values"][1, :, 48:, :].sum().item(), 0)

    def test_native_weighted_loss_sum_aux_and_gradients(self):
        adapter = tiny_adapter().train()
        image, target = sample()
        total, components = adapter.loss([image], [target])
        self.assertEqual(set(components), {"loss_cross_entropy", "loss_mask", "loss_dice",
            "loss_cross_entropy_0", "loss_mask_0", "loss_dice_0"})
        torch.testing.assert_close(total.detach(), adapter.last_loss_total, rtol=0, atol=0)
        self.assertTrue(all(torch.isfinite(value) for value in components.values()))
        total.backward()
        checked = ["model.class_predictor.weight",
            "model.model.transformer_module.decoder.mask_predictor.mask_embedder.2.0.weight",
            BACKBONE_PREFIX + "embeddings.patch_embeddings.projection.weight"]
        parameters = dict(adapter.named_parameters())
        for name in checked:
            with self.subTest(parameter=name):
                gradient = parameters[name].grad
                self.assertIsNotNone(gradient)
                self.assertTrue(bool(torch.isfinite(gradient).all()))
                self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertIsNone(adapter.model._captured_losses)

    def test_bfloat16_logits_native_losses_are_float32_and_differentiable(self):
        adapter = tiny_adapter()
        masks = torch.randn(1, 4, 16, 16, dtype=torch.bfloat16, requires_grad=True)
        classes = torch.randn(1, 4, 2, dtype=torch.bfloat16, requires_grad=True)
        _, target = sample()
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            losses = adapter.model.get_loss_dict(masks, classes,
                [target["masks"].float()], [torch.zeros(2, dtype=torch.int64)], None)
            total = sum(losses.values())
        self.assertEqual(total.dtype, torch.float32)
        total.backward()
        self.assertTrue(bool(torch.isfinite(masks.grad).all()))
        self.assertTrue(bool(torch.isfinite(classes.grad).all()))
        self.assertGreater(float(masks.grad.abs().sum()), 0)
        self.assertGreater(float(classes.grad.abs().sum()), 0)

    def test_empty_instance_image_has_finite_native_loss(self):
        adapter = tiny_adapter()
        target = {"masks": torch.empty(0, 64, 64, dtype=torch.uint8),
                  "labels": torch.empty(0, dtype=torch.int64)}
        total, _ = adapter.loss([torch.rand(3, 64, 64)], [target])
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()
        self.assertTrue(bool(torch.isfinite(adapter.model.class_predictor.weight.grad).all()))

    def test_one_pixel_instance_lost_to_resize_retains_class_and_finite_loss(self):
        adapter = tiny_adapter()
        image = torch.rand(3, 128, 128)
        masks = torch.zeros(1, 128, 128, dtype=torch.uint8)
        masks[0, 1, 1] = 1  # half-size nearest samples even indices, loses this pixel
        target = {"masks": masks, "labels": torch.ones(1, dtype=torch.int64)}
        batch, metadata = adapter.prepare_batch([image], [target])
        self.assertEqual(batch["mask_labels"][0].shape, (1, 64, 64))
        self.assertEqual(batch["mask_labels"][0].sum().item(), 0)
        self.assertEqual(batch["class_labels"][0].tolist(), [0])
        self.assertEqual(metadata["resized_empty_instances"], [1])
        total, _ = adapter.loss([image], [target])
        self.assertTrue(bool(torch.isfinite(total)))
        total.backward()
        self.assertTrue(bool(torch.isfinite(adapter.model.class_predictor.weight.grad).all()))

    def test_state_reload_keeps_single_image_predictions_identical(self):
        adapter = tiny_adapter().eval()
        image = Image.fromarray(np.random.default_rng(17).integers(0, 256, (64, 64, 3), dtype=np.uint8))
        before = adapter.predict(image, score_threshold=0)
        self.assertGreater(len(before), 0)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "checkpoint.pt"
            torch.save({"model": adapter.state_dict()}, checkpoint)
            restored = tiny_adapter().eval()
            restored.load_state_dict(torch.load(checkpoint, weights_only=True, map_location="cpu")["model"], strict=True)
            after = restored.predict(image, score_threshold=0)
        self.assertEqual(before, after)

    def test_predict_restores_original_size_overlap_and_foreground_label(self):
        classes = torch.tensor([[10., -10.], [9., -9.]])
        masks = torch.ones(2, 2, 2)
        results = component_predictions(classes, masks, padded_size=(64, 96),
            image_size=(48, 96), original_size=(64, 128))
        self.assertEqual(len(results), 2)
        for result in results:
            self.assertEqual(result["category_id"], 1)
            self.assertEqual(result["class_index"], 0)
            decoded = mask_utils.decode(result["segmentation"])
            self.assertEqual(decoded.shape, (64, 128))
            self.assertEqual(int(decoded.sum()), 64 * 128)

    def test_predict_returns_exact_course_submission_fields(self):
        adapter = tiny_adapter().eval()
        image = Image.fromarray(np.random.default_rng(23).integers(0, 256, (64, 64, 3), dtype=np.uint8))
        predictions = adapter.predict(image, score_threshold=0)
        self.assertGreater(len(predictions), 0)
        for prediction in predictions:
            self.assertEqual(set(prediction), {"category_id", "score", "segmentation"})
            self.assertEqual(prediction["category_id"], 1)
            self.assertIsInstance(prediction["score"], float)
            self.assertEqual(set(prediction["segmentation"]), {"size", "counts"})
            self.assertEqual(mask_utils.decode(prediction["segmentation"]).shape, (64, 64))

    def test_bad_inputs_and_semantic_label_ids_are_rejected(self):
        adapter = tiny_adapter()
        image, target = sample()
        for bad in (dict(target, labels=torch.zeros(2, dtype=torch.int64)),
                    dict(target, masks=target["masks"].float()),
                    dict(target, masks=torch.zeros_like(target["masks"]))):
            with self.assertRaises(ValueError):
                adapter.prepare_batch([image], [bad])
        for bad_image in (image.to(torch.uint8), image + 2, image * float("nan")):
            with self.assertRaises(ValueError):
                adapter.prepare_batch([bad_image], [target])
        with self.assertRaises(ValueError):
            adapter.predict(Image.new("RGB", (64, 64)))  # requires eval

    @unittest.skipUnless((DEFAULT_CHECKPOINT / "model.safetensors").exists(), "local pinned checkpoint unavailable")
    def test_real_pretrained_inheritance_and_trainable_parameter_groups(self):
        before_rng = torch.get_rng_state().clone()
        with patch("torch.cuda.manual_seed_all", side_effect=AssertionError("CPU initialization touched CUDA seeds")):
            adapter, metadata = build_trainable(seed=23)
        self.assertTrue(torch.equal(before_rng, torch.get_rng_state()))
        self.assertTrue(adapter.training)
        self.assertTrue(all(parameter.requires_grad for parameter in adapter.parameters()))
        self.assertEqual(adapter.model.class_predictor.out_features, 2)
        self.assertEqual(metadata["backbone_parameter_prefix"], BACKBONE_PREFIX)
        self.assertEqual(adapter.model.criterion.empty_weight.tolist(), [1., torch.tensor(.1).item()])
        self.assertEqual(set(metadata["initialization"]["changed_tensors"]),
                         {"class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"})
        self.assertTrue(metadata["initialization"]["all_other_state_tensors_exactly_equal"])
        self.assertGreater(metadata["initialization"]["shared_tensors_exactly_equal"], 500)
        names = dict(adapter.named_parameters())
        self.assertIn(BACKBONE_PREFIX + "embeddings.patch_embeddings.projection.weight", names)
        self.assertTrue(any(not name.startswith(BACKBONE_PREFIX) for name in names))


if __name__ == "__main__":
    unittest.main()
