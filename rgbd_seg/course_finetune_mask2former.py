"""Train a one-foreground-class Mask2Former with independent instance targets.

Uses the local, pinned HF COCO-instance Swin-Tiny checkpoint and Transformers'
native Hungarian matching, point BCE/Dice, no-object and auxiliary losses.
Only the 81-way class predictor is replaced by a 2-way predictor. The mask
decoder and all backbone weights retain their pretrained values. No augmentation
or ground-truth-dependent selection happens here.
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
from safetensors import safe_open
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import Mask2FormerForUniversalSegmentation
from transformers.models.mask2former.modeling_mask2former import Mask2FormerLoss

from .course_mask2former import DEFAULT_CHECKPOINT, MODEL_ID, REVISION, _sha256


BACKBONE_PREFIX = "model.model.pixel_level_module.encoder."


def _tensor_sha256(tensor):
    tensor = tensor.detach().cpu().contiguous()
    header = json.dumps({"shape": list(tensor.shape), "dtype": str(tensor.dtype)}, sort_keys=True).encode()
    return hashlib.sha256(header + tensor.numpy().tobytes()).hexdigest()


class LossReportingMask2Former(Mask2FormerForUniversalSegmentation):
    """Expose native weighted components without a persistent global patch.

    Only matching/loss arithmetic is promoted to float32 under AMP. Casting
    logits preserves the gradient path to BF16 decoder/classification outputs.
    Sampling counts, Hungarian costs, weights and auxiliary layers are unchanged.
    """

    def get_loss_dict(self, masks_queries_logits, class_queries_logits, mask_labels,
                      class_labels, auxiliary_predictions):
        auxiliary = None if auxiliary_predictions is None else [
            {key: value.float() for key, value in output.items()}
            for output in auxiliary_predictions]
        with torch.autocast(device_type=masks_queries_logits.device.type, enabled=False):
            losses = super().get_loss_dict(
                masks_queries_logits.float(), class_queries_logits.float(),
                [masks.float() for masks in mask_labels], class_labels, auxiliary)
        self._captured_losses = losses
        return losses


def _resize_size(height, width, short_edge, max_size):
    # torchvision's fixed-size detection transform uses this scale and floors
    # the resulting dimensions. Both course models see 1024² -> 800².
    scale = min(short_edge / min(height, width), max_size / max(height, width))
    return max(1, int(height * scale)), max(1, int(width * scale))


@torch.inference_mode()
def component_predictions(class_logits, mask_logits, *, padded_size, image_size,
                          original_size, score_threshold=0.01, max_detections=100):
    """One-class author-style instance inference, keeping overlapping RLEs.

    Restore spatial logits -> padded input -> crop valid input -> original image.
    Category 1 is the course component, HF class 0; class column 1 is no-object.
    """
    if class_logits.ndim != 2 or class_logits.shape[1] != 2:
        raise ValueError("Expected Qx2 foreground/no-object logits")
    if mask_logits.ndim != 3 or mask_logits.shape[0] != class_logits.shape[0]:
        raise ValueError("Mask logits must align with class queries")
    if not torch.isfinite(class_logits).all() or not torch.isfinite(mask_logits).all():
        raise ValueError("Non-finite predictions")
    if not 0 <= score_threshold <= 1 or not math.isfinite(score_threshold):
        raise ValueError("Invalid score threshold")
    if not isinstance(max_detections, int) or not 1 <= max_detections <= 100:
        raise ValueError("max_detections must lie in 1..100")
    for size in (padded_size, image_size, original_size):
        if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
            raise ValueError("Image sizes must contain two positive integers")
    if any(valid > padded for valid, padded in zip(image_size, padded_size)):
        raise ValueError("Valid image size exceeds padded image size")
    if class_logits.shape[0] == 0:
        return []
    scores = class_logits.float().softmax(-1)[:, 0]
    scores, indices = scores.topk(min(max_detections, len(scores)), sorted=False)
    results = []
    for start in range(0, len(indices), 8):
        stop = min(start + 8, len(indices))
        logits = mask_logits[indices[start:stop]].float()[None]
        logits = F.interpolate(logits, size=padded_size, mode="bilinear", align_corners=False)
        logits = logits[:, :, :image_size[0], :image_size[1]]
        logits = F.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)[0]
        masks = logits > 0
        areas = masks.flatten(1).sum(1)
        quality = (logits.sigmoid() * masks).flatten(1).sum(1) / (areas + 1e-6)
        final_scores = scores[start:stop] * quality
        for j in range(stop - start):
            score = float(final_scores[j])
            if not int(areas[j]) or score < score_threshold:
                continue
            rle = mask_utils.encode(np.asfortranarray(masks[j].cpu().numpy(), dtype=np.uint8))
            box = mask_utils.toBbox(rle).tolist()
            rle["counts"] = rle["counts"].decode("ascii")
            results.append({"category_id": 1, "class_index": 0, "category_name": "component",
                "score": score, "segmentation": rle, "area": int(areas[j]), "bbox": box,
                "query_id": int(indices[start + j])})
    results.sort(key=lambda item: (-item["score"], item["query_id"]))
    return results


class CourseTrainableMask2Former(nn.Module):
    """nn.Module loss adapter; runner owns optimizer, augmentations and files."""

    def __init__(self, model, *, mean, std, short_edge=800, max_size=1333, metadata=None):
        super().__init__()
        self.model = model
        if model.config.num_labels != 1:
            raise ValueError("The fine-tuning model must have one foreground class")
        if not 0 < short_edge <= max_size:
            raise ValueError("Invalid resize bounds")
        self.short_edge, self.max_size = int(short_edge), int(max_size)
        self.register_buffer("image_mean", torch.tensor(mean, dtype=torch.float32)[:, None, None], persistent=False)
        self.register_buffer("image_std", torch.tensor(std, dtype=torch.float32)[:, None, None], persistent=False)
        self.metadata = metadata or {}
        self.last_loss_total = None
        self.last_batch_metadata = None

    @property
    def device(self):
        return next(self.model.parameters()).device

    def prepare_batch(self, images, targets=None):
        """Normalize RGB and resize each binary instance separately; never flatten."""
        if not isinstance(images, (list, tuple)) or not images:
            raise ValueError("images must be a nonempty list of RGB tensors")
        if targets is not None and len(targets) != len(images):
            raise ValueError("Image and target counts differ")
        resized, sizes, originals, mask_labels, class_labels = [], [], [], [], []
        resized_empty_instances = []
        # Preprocessing always uses float32 even inside an external AMP context.
        with torch.autocast(device_type=self.device.type, enabled=False):
            for index, image in enumerate(images):
                if not torch.is_tensor(image) or image.ndim != 3 or image.shape[0] != 3:
                    raise ValueError("Each image must be float RGB CHW")
                if min(image.shape[-2:]) <= 0:
                    raise ValueError("Empty image")
                if not image.is_floating_point() or not torch.isfinite(image).all() or image.min() < 0 or image.max() > 1:
                    raise ValueError("Image values must be finite floats in [0,1]")
                height, width = map(int, image.shape[-2:])
                if min(height, width) <= 0:
                    raise ValueError("Empty image")
                size = _resize_size(height, width, self.short_edge, self.max_size)
                image = image.to(self.device, dtype=torch.float32)
                image = F.interpolate(image[None], size=size, mode="bilinear", align_corners=False,
                                      antialias=False)[0]
                resized.append((image - self.image_mean) / self.image_std)
                sizes.append(size)
                originals.append((height, width))
                if targets is None:
                    continue
                target = targets[index]
                masks, labels = target["masks"], target["labels"]
                if masks.ndim != 3 or tuple(masks.shape[-2:]) != (height, width):
                    raise ValueError("Independent instance masks must have shape N x image height x image width")
                if masks.dtype not in (torch.uint8, torch.bool) or not ((masks == 0) | (masks == 1)).all():
                    raise ValueError("Target masks must be independent binary uint8/bool tensors")
                if labels.shape != (len(masks),) or labels.dtype != torch.int64 or (labels != 1).any():
                    raise ValueError("Course labels must be int64 foreground ID 1 for every mask")
                if len(masks) and not masks.flatten(1).any(1).all():
                    raise ValueError("Empty target instances are invalid")
                masks = masks.to(self.device, dtype=torch.float32)
                if len(masks):
                    masks = F.interpolate(masks[:, None], size=size, mode="nearest")[:, 0]
                    # As in the torchvision detector transform, retain original
                    # valid instances even if nearest resizing loses every pixel.
                    # Their class and zero mask stay aligned; never drop labels.
                    resized_empty_instances.append(int((~masks.flatten(1).any(1)).sum()))
                else:
                    masks = torch.empty((0, *size), device=self.device)
                    resized_empty_instances.append(0)
                mask_labels.append(masks)
                class_labels.append(torch.zeros(len(masks), device=self.device, dtype=torch.int64))
            padded = tuple(math.ceil(max(size[axis] for size in sizes) / 32) * 32 for axis in (0, 1))
            pixels = resized[0].new_zeros((len(images), 3, *padded))
            valid = torch.zeros((len(images), *padded), device=self.device, dtype=torch.bool)
            for index, (image, size) in enumerate(zip(resized, sizes)):
                pixels[index, :, :size[0], :size[1]] = image
                valid[index, :size[0], :size[1]] = True
                if targets is not None:
                    mask_labels[index] = F.pad(mask_labels[index], (0, padded[1] - size[1], 0, padded[0] - size[0]))
        batch = {"pixel_values": pixels, "pixel_mask": valid}
        if targets is not None:
            batch.update(mask_labels=mask_labels, class_labels=class_labels)
        return batch, {"resized_sizes": sizes, "original_sizes": originals, "padded_size": padded,
                       "resized_empty_instances": resized_empty_instances}

    def forward(self, images, targets):
        if targets is None:
            raise ValueError("Training forward requires targets; use predict(PIL) for inference")
        batch, self.last_batch_metadata = self.prepare_batch(images, targets)
        self.model._captured_losses = None
        outputs = self.model(**batch, return_dict=True)
        losses = self.model._captured_losses
        self.model._captured_losses = None
        if not losses or outputs.loss is None:
            raise RuntimeError("Native Mask2Former losses were not captured")
        # Keep only a detached diagnostic; no previous-step computation graph.
        self.last_loss_total = outputs.loss.detach()
        return losses

    def loss(self, images, targets):
        losses = self(images, targets)
        return sum(losses.values()), losses

    @torch.inference_mode()
    def predict(self, image, *, score_threshold=0.01, max_detections=100):
        if self.training:
            raise ValueError("Call adapter.eval() before predict")
        if not isinstance(image, Image.Image) or image.mode != "RGB":
            raise ValueError("predict requires a PIL RGB image")
        tensor = torch.from_numpy(np.array(image, copy=True)).permute(2, 0, 1).float().div_(255)
        batch, sizes = self.prepare_batch([tensor])
        outputs = self.model(**batch, return_dict=True)
        predictions = component_predictions(outputs.class_queries_logits[0], outputs.masks_queries_logits[0],
            padded_size=sizes["padded_size"], image_size=sizes["resized_sizes"][0],
            original_size=sizes["original_sizes"][0], score_threshold=score_threshold,
            max_detections=max_detections)
        # The course evaluator requires exactly these submission fields. Keep
        # internal query/area/box diagnostics out of the public prediction API.
        return [{key: item[key] for key in ("category_id", "score", "segmentation")}
                for item in predictions]


def build_trainable(checkpoint_dir=DEFAULT_CHECKPOINT, device="cpu", *, seed=17):
    """Return (trainable adapter, JSON metadata); only classifier is reinitialized."""
    path = Path(checkpoint_dir).resolve()
    receipt_path = path / "source_receipt.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("model_id") != MODEL_ID or receipt.get("revision") != REVISION:
        raise ValueError("Unexpected pretrained model/revision")
    for filename in ("config.json", "preprocessor_config.json", "model.safetensors"):
        if _sha256(path / filename) != receipt["files"][filename]["sha256"]:
            raise ValueError(f"Pretrained file hash mismatch: {filename}")
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed)
        model, loading = LossReportingMask2Former.from_pretrained(path, local_files_only=True,
            use_safetensors=True, torch_dtype=torch.float32, output_loading_info=True)
        if any(loading.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise RuntimeError(f"Pretrained loading mismatch: {loading}")
        if model.config.num_labels != 80:
            raise ValueError("Expected 80-class COCO instance initialization")
        # A fresh two-output classifier; no arbitrary COCO foreground is reused.
        torch.default_generator.manual_seed(seed)
        classifier = nn.Linear(model.config.hidden_dim, 2)
        model._init_weights(classifier)
        model.class_predictor = classifier
        model.config.num_labels = 1
        model.config.id2label, model.config.label2id = {0: "component"}, {"component": 0}
        # This has no trainable parameters; its no-object buffer changes 81 -> 2.
        model.criterion = Mask2FormerLoss(model.config, model.weight_dict)
    model.requires_grad_(True)
    state = model.state_dict()
    changed = {}
    unchanged = []
    allowed = {"class_predictor.weight", "class_predictor.bias", "criterion.empty_weight"}
    with safe_open(path / "model.safetensors", framework="pt", device="cpu") as source:
        if set(source.keys()) != set(state):
            raise RuntimeError("Pretrained/new state key sets differ")
        for key in source.keys():
            old, new = source.get_tensor(key), state[key]
            if torch.equal(old, new):
                unchanged.append(key)
            else:
                if key not in allowed:
                    raise RuntimeError(f"Unexpected pretrained tensor change: {key}")
                changed[key] = {"old_shape": list(old.shape), "new_shape": list(new.shape),
                    "old_sha256": _tensor_sha256(old), "new_sha256": _tensor_sha256(new)}
    if set(changed) != allowed:
        raise RuntimeError("Expected exactly classifier tensors and no-object buffer to change")
    processor_config = json.loads((path / "preprocessor_config.json").read_text())
    hf_source = Path(inspect.getsourcefile(Mask2FormerForUniversalSegmentation))
    metadata = {"model_id": MODEL_ID, "pretrained_revision": REVISION,
        "pretrained_sha256": receipt["files"]["model.safetensors"]["sha256"],
        "source_receipt_sha256": _sha256(receipt_path), "adapter_sha256": _sha256(__file__),
        "transformers_version": transformers.__version__, "hf_modeling_source_sha256": _sha256(hf_source),
        "hf_source": "https://github.com/huggingface/transformers/blob/v4.53.2/src/transformers/models/mask2former/modeling_mask2former.py",
        "initialization": {"seed": seed, "shared_tensors_exactly_equal": len(unchanged),
            "changed_tensors": changed, "classifier_init": "HF native normal std=config.init_std, bias=0; both foreground and no-object fresh",
            "all_other_state_tensors_exactly_equal": True, "preserves_callers_cpu_rng": True},
        "parameters": sum(p.numel() for p in model.parameters()), "backbone_parameter_prefix": BACKBONE_PREFIX,
        "labels": {"dataset_foreground": 1, "hf_foreground": 0, "hf_no_object": 1},
        "preprocess": {"shortest_edge": 800, "longest_edge": 1333, "size_divisibility": 32,
            "rgb_resize": "torch bilinear align_corners=False antialias=False",
            "mask_resize": "nearest per independent binary instance; overlap preserved",
            "instances_lost_to_resize": "retain original class and resulting zero mask, as torchvision transform does; count in last_batch_metadata.resized_empty_instances",
            "mean": processor_config["image_mean"], "std": processor_config["image_std"],
            "input_range": "float RGB [0,1]", "augmentation": "none; runner owns hflip",
            "published_hf_size": processor_config["size"],
            "difference_from_zero_shot": "Uses Torch bilinear without antialias instead of old HF/PIL RGB resize; training and prediction use the same new path",
            "padding": "right/bottom zero normalized image and zero target mask; valid pixel mask supplied",
            "hf_padding_limitation": "HF 4.53.2 accepts but does not use pixel_mask in this model; point loss treats padded targets as zero. Course 1024-square images become 800-square, so no padding is present."},
        "loss": {"weights": model.weight_dict, "no_object_weight": model.config.no_object_weight,
            "train_num_points": model.config.train_num_points, "oversample_ratio": model.config.oversample_ratio,
            "importance_sample_ratio": model.config.importance_sample_ratio,
            "auxiliary_loss": model.config.use_auxiliary_loss, "decoder_layers": model.config.decoder_layers,
            "matching_and_loss_dtype": "float32; native formulas/weights/sampling and auxiliary predictions unchanged; differentiable casts from AMP logits",
            "forward_return": "weighted native components only; sum once, no duplicated total"},
        "postprocess": {"score_threshold": 0.01, "mask_threshold": 0.5, "mask_comparison": ">",
            "max_detections": 100, "category_id": 1, "overlap": "preserved in independent compressed COCO RLE",
            "score": "foreground softmax * mean sigmoid on positive original-resolution mask"},
        "loading_info": loading}
    adapter = CourseTrainableMask2Former(model, mean=processor_config["image_mean"],
        std=processor_config["image_std"], metadata=metadata).to(device).train()
    return adapter, metadata
