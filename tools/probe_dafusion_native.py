#!/usr/bin/env python3
"""Bounded engineering check of the pinned public DA-Fusion implementation.

Uses synthetic inputs and random weights. Produces no benchmark score or checkpoint.
The only runtime compatibility shim restores Pillow's removed LINEAR alias.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--optimizer-step", action="store_true")
    parser.add_argument("--amp", action="store_true", help="Use CUDA fp16 autocast and native GradScaler")
    parser.add_argument("--amp-init-scale", type=float, default=65536.0,
                        help="Explicit GradScaler initial loss scale; PyTorch default is 65536")
    parser.add_argument("--swin-checkpoint", action="store_true", help="Explicitly enable native Swin activation checkpointing")
    args = parser.parse_args()
    if args.size < 64 or args.size > 640 or args.size % 32:
        parser.error("Engineering check is bounded to multiples of 32 between 64 and 640.")
    if args.batch_size not in (1, 2):
        parser.error("Engineering check is bounded to batch size 1 or 2.")
    if args.amp and args.device != "cuda":
        parser.error("AMP check requires CUDA.")
    if not 1.0 <= args.amp_init_scale <= 65536.0:
        parser.error("AMP initial scale must be between 1 and 65536.")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["WANDB_DISABLED"] = "true"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    root = Path(__file__).resolve().parents[1]
    vendor = root / "third_party/DA-Fusion"
    receipt = json.loads((vendor / "source_receipt.json").read_text())
    mismatches = [name for name, info in receipt["files"].items()
                  if sha256(vendor / name) != info["sha256"]]
    if mismatches:
        raise RuntimeError(f"Vendor integrity check failed: {mismatches}")
    sys.path.insert(0, str(vendor / "dafusion"))
    report = {
        "schema": "dafusion-native-engineering-probe-v1",
        "status": "running", "benchmark_result": False,
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "pid": os.getpid(), "python": sys.executable,
        "source_commit": receipt["commit"],
        "verified_source_files": len(receipt["files"]),
        "script_sha256": sha256(__file__),
        "input": {"kind": "synthetic RGB and metric XYZ plus validity", "size": [args.size, args.size], "batch": args.batch_size},
        "precision": "fp16 autocast with GradScaler" if args.amp else "fp32",
        "amp_initial_loss_scale": args.amp_init_scale if args.amp else None,
        "swin_activation_checkpointing": args.swin_checkpoint,
        "weights": "random initialization; no checkpoint loaded",
        "benchmark_scores": None,
        "command": sys.argv,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "network_logging": "WANDB disabled; no download or upload calls",
        "limitations": [
            "Synthetic engineering check is not a benchmark experiment.",
            "Public config specifies Swin-L; the paper does not specify the Swin variant.",
            "Public stage1 uses XYZ+validity and a depth adapter; the paper describes HHA.",
            "A single synthetic step does not establish long-run real-data training stability or accuracy.",
        ],
    }

    def save():
        (args.output / "result.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    started = time.monotonic()
    try:
        import torch
        import MultiScaleDeformableAttention as msda
        from PIL import Image
        report["compatibility_shims"] = []
        if not hasattr(Image, "LINEAR"):
            Image.LINEAR = Image.Resampling.BILINEAR
            report["compatibility_shims"].append("PIL.Image.LINEAR = PIL.Image.Resampling.BILINEAR (process only)")
        import detectron2._C as d2c
        import dafusion
        from detectron2.config import get_cfg
        from detectron2.modeling import build_model
        from detectron2.structures import Instances
        from dafusion.config import add_dafusion_config
        from dafusion.modeling.pixel_decoder.ops.functions.ms_deform_attn_func import (
            MSDeformAttnFunction, ms_deform_attn_core_pytorch,
        )
        report["environment"] = {name: importlib.metadata.version(name) for name in
                                 ("torch", "torchvision", "detectron2", "Pillow", "numpy", "timm", "pycocotools")}
        report["environment"].update({"torch_cuda": torch.version.cuda, "python_version": sys.version,
                                     "msda_extension": msda.__file__, "msda_sha256": sha256(msda.__file__),
                                     "detectron2_extension": d2c.__file__, "detectron2_extension_sha256": sha256(d2c.__file__)})
        torch.set_num_threads(4)
        torch.manual_seed(20261030)
        if args.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is required for the native deformable attention extension.")
            torch.cuda.set_device(0)
            report["environment"]["gpu"] = torch.cuda.get_device_name(0)
            report["environment"]["gpu_capability"] = list(torch.cuda.get_device_capability(0))
            torch.cuda.reset_peak_memory_stats()
        device = torch.device(args.device)
        # Verify the reused compiled extension against the vendored PyTorch reference,
        # including all three differentiable inputs, before exercising the model.
        shapes = torch.tensor([[4, 4], [2, 2]], device=device, dtype=torch.long)
        starts = torch.tensor([0, 16], device=device, dtype=torch.long)
        value = torch.randn(1, 20, 2, 4, device=device, requires_grad=True)
        locations = torch.rand(1, 7, 2, 2, 3, 2, device=device, requires_grad=True)
        weights = torch.rand(1, 7, 2, 2, 3, device=device, requires_grad=True)
        actual = MSDeformAttnFunction.apply(value, shapes, starts, locations, weights, 1)
        expected = ms_deform_attn_core_pytorch(value, shapes, locations, weights)
        grad = torch.randn_like(actual)
        actual_grads = torch.autograd.grad(actual, (value, locations, weights), grad, retain_graph=True)
        expected_grads = torch.autograd.grad(expected, (value, locations, weights), grad)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)
        grad_diffs = []
        for left, right in zip(actual_grads, expected_grads):
            torch.testing.assert_close(left, right, rtol=2e-5, atol=2e-5)
            grad_diffs.append(float((left - right).abs().max()))
        report["msda_parity"] = {"passed": True, "forward_max_abs_error": float((actual - expected).abs().max()),
                                 "backward_max_abs_errors": grad_diffs, "atol": 2e-5, "rtol": 2e-5}
        del value, locations, weights, actual, expected, actual_grads, expected_grads
        print("MSDeformAttn CUDA/PyTorch forward and backward parity passed", flush=True)

        cfg = get_cfg()
        add_dafusion_config(cfg)
        config_path = vendor / "dafusion/configs/dafusion_rgbd_uoais_stage1.yaml"
        cfg.merge_from_file(str(config_path))
        cfg.MODEL.DEVICE = args.device
        cfg.MODEL.WEIGHTS = ""
        if args.swin_checkpoint:
            cfg.MODEL.SWIN.USE_CHECKPOINT = True
        cfg.freeze()
        (args.output / "resolved_config.yaml").write_text(cfg.dump())
        report["config"] = {"path": str(config_path), "sha256": sha256(config_path),
                            "overrides": {"MODEL.DEVICE": args.device, "MODEL.WEIGHTS": ""}}
        if args.swin_checkpoint:
            report["config"]["overrides"]["MODEL.SWIN.USE_CHECKPOINT"] = True
        model = build_model(cfg)
        report["model"] = {"class": type(model).__name__, "parameters": sum(p.numel() for p in model.parameters()),
                           "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                           "swin_embed_dim": cfg.MODEL.SWIN.EMBED_DIM, "swin_depths": list(cfg.MODEL.SWIN.DEPTHS),
                           "fusion_stages": list(model.backbone.fusions.keys()),
                           "ds_modules": sum(type(m).__name__ == "DeformableSelfAttention" for m in model.modules()),
                           "dc_modules": sum(type(m).__name__ == "DeformableCrossAttention" for m in model.modules())}
        save()
        print("Native full Swin-L model constructed", report["model"], flush=True)
        height = width = args.size
        yy, xx = torch.meshgrid(torch.linspace(-0.1, 0.1, height), torch.linspace(-0.1, 0.1, width), indexing="ij")
        depth = torch.stack((xx, yy, torch.ones_like(xx), torch.ones_like(xx)))
        masks = torch.zeros((2, height, width), dtype=torch.bool)
        masks[0, height//4:height//2, width//4:width//2] = True
        masks[1, height//2:3*height//4, width//2:3*width//4] = True
        instances = Instances((height, width))
        instances.gt_classes = torch.zeros(2, dtype=torch.long)
        instances.gt_masks = masks
        sample = {"image": torch.rand(3, height, width) * 255, "depth_image": depth,
                  "height": height, "width": width, "instances": instances}
        samples = [sample] + [{**sample, "image": torch.rand(3, height, width) * 255}
                              for _ in range(args.batch_size - 1)]
        optimizer = None
        if args.optimizer_step:
            from dafusion.train_net import Trainer
            optimizer = Trainer.build_optimizer(cfg, model)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp, init_scale=args.amp_init_scale)
        model.train()
        report["phase"] = "training_forward"
        save()
        with torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            losses = model(samples)
            loss = sum(losses.values())
        if not torch.isfinite(loss):
            raise RuntimeError("Synthetic training loss is nonfinite.")
        report["phase"] = "backward"
        save()
        scaler.scale(loss).backward()
        if optimizer is not None:
            scaler.unscale_(optimizer)
        groups = {"rgb_backbone": "backbone.swin_rgb.", "depth_backbone": "backbone.swin_d.",
                  "ds_rgb": ".ds_rgb.", "ds_depth": ".ds_d.", "dc": ".dc.", "mask_decoder": "sem_seg_head.predictor."}
        gradient_summary = {}
        for name, marker in groups.items():
            grads = [p.grad for pname, p in model.named_parameters() if marker in pname and p.grad is not None]
            if not grads or not all(bool(torch.isfinite(g).all()) for g in grads):
                raise RuntimeError(f"Missing or nonfinite gradients in {name}")
            nonzero = sum(bool(torch.count_nonzero(g)) for g in grads)
            if not nonzero:
                raise RuntimeError(f"All gradients zero in {name}")
            gradient_summary[name] = {"tensors": len(grads), "nonzero_tensors": nonzero, "all_finite": True}
        report["synthetic_backward"] = {"passed": True, "loss_terms": len(losses),
                                         "total_loss": float(loss.detach()), "gradient_groups": gradient_summary}
        print("Forward and backward passed", report["synthetic_backward"], flush=True)
        if args.optimizer_step:
            report["phase"] = "optimizer_step"
            save()
            before = model.sem_seg_head.predictor.class_embed.weight.detach().clone()
            scaler.step(optimizer)
            scaler.update()
            changed = bool(torch.any(before != model.sem_seg_head.predictor.class_embed.weight.detach()))
            if not changed:
                raise RuntimeError("Optimizer step did not change the classifier weight.")
            report["optimizer_step"] = {"passed": True, "class": type(optimizer).__name__,
                                        "parameter_groups": len(optimizer.param_groups), "classifier_weight_changed": changed,
                                        "amp_loss_scale_after_step": scaler.get_scale() if args.amp else None}
            optimizer.zero_grad(set_to_none=True)
            del optimizer
        else:
            model.zero_grad(set_to_none=True)
        model.eval()
        report["phase"] = "inference"
        save()
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=args.amp):
            all_predictions = model([{k: v for k, v in item.items() if k != "instances"} for item in samples])
            prediction = all_predictions[0]["instances"]
        if tuple(prediction.pred_masks.shape[-2:]) != (height, width) or not torch.isfinite(prediction.scores).all():
            raise RuntimeError("Synthetic inference returned invalid dimensions or scores.")
        report["synthetic_inference"] = {"passed": True, "mask_shape": list(prediction.pred_masks.shape),
                                          "output_batch_size": len(all_predictions),
                                          "finite_scores": True, "scores_are_not_accuracy_measurements": True}
        if args.device == "cuda":
            torch.cuda.synchronize()
            report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
            report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
        report["status"] = "passed"
        report["phase"] = "complete"
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        print(json.dumps({"status": report["status"], "result": str(args.output / "result.json")}))
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        try:
            if args.device == "cuda" and torch.cuda.is_initialized():
                report["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
                report["peak_cuda_reserved_bytes"] = torch.cuda.max_memory_reserved()
                report["cuda_memory_summary_on_failure"] = torch.cuda.memory_summary()
        except Exception:
            pass
        report["traceback"] = traceback.format_exc()
        report["elapsed_seconds"] = time.monotonic() - started
        save()
        raise


if __name__ == "__main__":
    main()
