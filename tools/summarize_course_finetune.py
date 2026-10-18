#!/usr/bin/env python3
"""Audit completed course runs and write a comparison; never infer or decode masks."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
MODELS = ("maskrcnn", "mask2former")
DISPLAY_METRICS = ("mask_ap_small", "mask_ap", "mask_ap50", "mask_ap75")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read(path):
    with Path(path).open() as stream:
        return json.load(stream)


def check_hash(path, expected):
    require(isinstance(expected, str) and len(expected) == 64, f"Missing SHA256: {path}")
    actual = sha256(path)
    require(actual == expected, f"SHA256 mismatch: {path}")
    return actual


def metrics(document):
    values = document["metrics"]
    require(len(values) == 12, "Expected all twelve COCO metrics")
    for name, value in values.items():
        require(value is None or (not isinstance(value, bool) and isinstance(value, (float, int))
                                  and math.isfinite(value) and 0 <= value <= 1), f"Invalid 0–1 metric {name}")
    require(all(values.get(name) is not None for name in DISPLAY_METRICS), "Missing comparison/selection metrics")
    return values


def selected_epoch(entries):
    require(len(entries) == 5 and [e["epoch"] for e in entries] == list(range(1, 6)), "Expected five ordered validation epochs")
    return max(entries, key=lambda e: (metrics(e)["mask_ap_small"], metrics(e)["mask_ap"], -e["epoch"]))["epoch"]


def check_sources(sources, *, basename=False):
    require(bool(sources), "Missing source receipt")
    checked = {}
    for name, expected in sources.items():
        path = ROOT / "rgbd_seg" / name if basename else ROOT / name
        require(path.resolve().is_relative_to(ROOT), f"Source outside project: {name}")
        checked[name] = check_hash(path, expected)
    return checked


def check_raw(path, expected_hash, image_ids, *, allow_cleanup=False):
    if not path.exists():
        require(allow_cleanup, f"Missing required predictions: {path}")
        receipt_path = path.parent / "cleanup.json"
        cleanup = read(receipt_path)
        require(cleanup.get("predictions_sha256") == expected_hash and cleanup.get("reason"), "Invalid cleanup receipt")
        require(Path(cleanup["path"]).resolve() == path.resolve(), "Cleanup path differs")
        return {"status": "removed_after_scoring", "sha256_recorded": expected_hash,
                "cleanup_sha256": sha256(receipt_path), "content_reverified": False}
    checksum = check_hash(path, expected_hash)
    rows, predictions = 0, 0
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            require(rows < len(image_ids) and row["image_id"] == image_ids[rows], f"Raw prediction ID/order differs: {path}")
            require(isinstance(row["predictions"], list) and len(row["predictions"]) <= 100, "Invalid prediction count")
            predictions += len(row["predictions"])
            rows += 1
    require(rows == len(image_ids), "Raw prediction image coverage incomplete")
    return {"sha256": checksum, "rows": rows, "predictions": predictions, "content_reverified": True,
            "mask_decode_repeated": False}


def audit_evaluation(folder, split, image_ids, *, allow_cleanup=False, checkpoint_sha=None):
    evaluation, receipt = read(folder / "evaluation.json"), read(folder / "evaluation_run.json")
    require(evaluation["status"] == receipt["status"] == "completed", f"Incomplete evaluation: {folder}")
    require(evaluation["split"] == receipt["split"] == split, "Evaluation split mismatch")
    require(evaluation["images"] == receipt["images"] == len(image_ids), "Evaluation image count mismatch")
    require(evaluation["scope"] == receipt["scope"] == "full_split", "Smoke evaluation cannot enter comparison")
    require(evaluation["full_split_images"] == receipt["full_split_images"] == len(image_ids), "Incomplete split")
    require(evaluation["max_detections"] == [1, 10, 100] and evaluation["iou_type"] == "segm", "Unexpected COCO protocol")
    require(evaluation["ground_truth_area"] == "original annotation area preserved", "GT area protocol differs")
    require(evaluation.get("depth_used") is False and receipt.get("depth_used") is False, "Expected RGB-only evaluation")
    metrics(evaluation)
    check_sources(evaluation["source_sha256"], basename=True)
    require(receipt["source_sha256"] == evaluation["source_sha256"], "Evaluation source receipts differ")
    check_hash(folder / "evaluation.json", receipt["evaluation_sha256"])
    require(receipt["predictions_sha256"] == evaluation["predictions_sha256"], "Raw SHA receipts differ")
    raw = check_raw(folder / "predictions.jsonl", receipt["predictions_sha256"], image_ids, allow_cleanup=allow_cleanup)
    result = {"path": str(folder), "evaluation_sha256": sha256(folder / "evaluation.json"),
              "evaluation_run_sha256": sha256(folder / "evaluation_run.json"), "metrics": evaluation["metrics"],
              "images": len(image_ids), "gt_instances": evaluation["gt_instances"],
              "gt_area_groups_including_crowd": evaluation["gt_area_groups_including_crowd"], "raw": raw}
    if split == "test":
        require(receipt.get("evaluated_checkpoint_sha256") == checkpoint_sha, "Test receipt lacks matching evaluated_checkpoint_sha256")
        validation_path = folder / "submission_validation.json"
        check_hash(validation_path, receipt["submission_validation_sha256"])
        validation = read(validation_path)
        csv_sha = check_hash(folder / "submission.csv", validation["sha256"])
        require(csv_sha == receipt["submission_sha256"], "CSV receipts differ")
        require(validation["rows"] == len(image_ids) and validation["image_size"] == [1024, 1024]
                and validation["category_ids"] == [1] and validation["max_detections"] == 100, "CSV contract differs")
        require(all(validation.get(key) is True for key in
                    ("id_order_verified", "rle_runs_verified", "rle_native_decode_verified")), "CSV validation is incomplete")
        require(validation["predictions"] == raw["predictions"] == evaluation["predicted_instances"], "Prediction counts differ")
        check_sources(validation["source_sha256"], basename=True)
        result.update(submission_sha256=csv_sha, submission_validation_sha256=sha256(validation_path),
                      evaluated_checkpoint_sha256=checkpoint_sha)
    return result


def audit_model(folder, name, protocol_sha, protocol, ids):
    run, selection, initialization = (read(folder / name_) for name_ in ("run.json", "selection.json", "initialization.json"))
    require(run["status"] == "completed" and run["scope"] == "full_train_val_test" and run["model"] == name, "Incomplete/wrong model run")
    require(run["protocol_sha256"] == protocol_sha, "Run protocol differs")
    require(run["depth_used"] is False and run["test_used_for_checkpoint_selection"] is False, "Task/input/selection differs")
    check_sources(run["source_sha256"])
    args = run["arguments"]
    for key in ("epochs", "batch_size", "accumulation", "seed", "lr", "precision", "workers", "threads"):
        require(args[key] == protocol["training"][key], f"Training protocol mismatch: {name}/{key}")
    require(args["epochs"] == 5 and args["smoke_steps"] == 0, "Expected five formal epochs")
    require(run["training_data"]["images"] == 14432 and run["training_images_seen"] == 5 * 14432, "Training coverage differs")
    require(initialization == run["model_metadata"], "Initialization receipt differs from run")
    initial_path = Path(args["checkpoint"])
    initial_sha = initialization["initial_checkpoint_sha256"] if name == "maskrcnn" else initialization["pretrained_sha256"]
    check_hash(initial_path if name == "maskrcnn" else initial_path / "model.safetensors", initial_sha)
    orders = []
    for epoch in range(1, 6):
        path = folder / f"train_order_epoch_{epoch:02d}.json"
        order = read(path)
        require(order["epoch"] == epoch and order["count"] == len(order["image_ids"]) == 14432, "Training order count differs")
        require(len(set(order["image_ids"])) == 14432 and set(order["image_ids"]) == set(ids["train"]), "Training order misses official IDs")
        require(hashlib.sha256(json.dumps(order["image_ids"]).encode()).hexdigest() == order["ordered_image_ids_sha256"], "Training order internal SHA differs")
        orders.append({"epoch": epoch, "sha256": sha256(path), "ordered_image_ids_sha256": order["ordered_image_ids_sha256"]})
    effective_batch = args["batch_size"] * args["accumulation"]
    steps_per_epoch = math.ceil(14432 / effective_batch)
    log_rows = [json.loads(line) for line in (folder / "training_log.jsonl").open()]
    require(len(log_rows) == steps_per_epoch * 5 == run["optimizer_steps"], "Training log step coverage differs")
    for step, row in enumerate(log_rows, 1):
        epoch = (step - 1) // steps_per_epoch + 1
        seen = min(((step - 1) % steps_per_epoch + 1) * effective_batch, 14432)
        require(row["global_step"] == step and row["epoch"] == epoch and row["images_in_epoch"] == seen
                and row["expected_train_images"] == 14432, "Training step/order coverage differs")
        require(math.isfinite(row["loss"]), "Nonfinite training loss")
    best_epoch = selected_epoch(selection["epochs"])
    require(selection["validation_scope"] == "full_2584_images" and selection["test_used_for_selection"] is False, "Selection scope differs")
    require(best_epoch == selection["best_epoch"] == run["best_epoch"], "Checkpoint selection differs from APsmall/AP/early-epoch rule")
    val = []
    for entry in selection["epochs"]:
        epoch, val_folder = entry["epoch"], Path(entry["output"])
        require(val_folder.resolve().parent == folder.resolve(), "Validation output is outside model run")
        require(entry["training_images"] == 14432 and entry["global_step"] == epoch * steps_per_epoch, "Selection training coverage differs")
        result = audit_evaluation(val_folder, "val", ids["val"], allow_cleanup=epoch != best_epoch)
        require(result["metrics"] == entry["metrics"], "Selection metrics differ from evaluated metrics")
        val.append({"epoch": epoch, **result})
    best_entry = selection["epochs"][best_epoch - 1]
    require(selection["best_key"] == [best_entry["metrics"]["mask_ap_small"], best_entry["metrics"]["mask_ap"]], "Best selection key differs")
    checkpoint_sha = check_hash(folder / "best.pt", run["best_checkpoint_sha256"])
    import torch
    checkpoint = torch.load(folder / "best.pt", weights_only=True, map_location="cpu", mmap=True)
    require(checkpoint["epoch"] == best_epoch and checkpoint["selection"] == best_entry
            and checkpoint["model_name"] == name and checkpoint["protocol_sha256"] == protocol_sha
            and checkpoint["source_sha256"] == run["source_sha256"]
            and checkpoint["model_metadata"] == initialization, "Best checkpoint provenance differs")
    del checkpoint
    test_folder = Path(run["test_output"])
    require(test_folder.resolve().parent == folder.resolve(), "Test output is outside model run")
    test_receipt = read(test_folder / "evaluation_run.json")
    require(test_receipt.get("selected_epoch") == best_epoch, "Test receipt selected epoch differs")
    check_hash(test_folder / "evaluation_run.json", run["test_evaluation_run_sha256"])
    test = audit_evaluation(test_folder, "test", ids["test"], checkpoint_sha=checkpoint_sha)
    require(test["metrics"] == run["test_metrics"], "Run test metrics differ")
    return {"model": name, "best_epoch": best_epoch, "best_checkpoint_sha256": checkpoint_sha,
            "initial_checkpoint_sha256": initial_sha, "train_orders": orders, "validation": val, "test": test,
            "training_annotation_sha256": run["training_data"]["annotation_member_name_and_bytes_sha256"],
            "receipts_sha256": {filename: sha256(folder / filename) for filename in
                                ("run.json", "selection.json", "initialization.json", "training_log.jsonl")}}


def summarize(run_root, pretrained_root, protocol_path):
    protocol = read(protocol_path)
    protocol_sha = sha256(protocol_path)
    from rgbd_seg.course_archive import CourseArchive
    ids = {}
    for split, expected in (("train", 14432), ("val", 2584), ("test", 2112)):
        with CourseArchive(protocol["archive"], split) as dataset:
            require(len(dataset) == expected, f"Official {split} count differs")
            ids[split] = list(dataset.image_ids)
    audited = {name: audit_model(run_root / name, name, protocol_sha, protocol, ids) for name in MODELS}
    require(audited["maskrcnn"]["train_orders"] == audited["mask2former"]["train_orders"], "The two models did not consume the same epoch orders")
    require(audited["maskrcnn"]["training_annotation_sha256"] == audited["mask2former"]["training_annotation_sha256"], "Training annotation sources differ")
    comparisons = []
    for name in MODELS:
        previous = pretrained_root / name
        old_run, old = read(previous / "run.json"), read(previous / "evaluation.json")
        require(old_run["status"] == old["status"] == "completed" and old["images"] == 2112 and old["split"] == "test", "Old zero-shot evaluation incomplete")
        metrics(old)
        old_raw = check_raw(previous / "predictions.jsonl", old_run["predictions_sha256"], ids["test"])
        require(old_raw["predictions"] == old["predicted_instances"], "Old prediction counts differ")
        test = audited[name]["test"]
        require(old["gt_instances"] == test["gt_instances"], "Old/new GT populations differ")
        require(old["gt_small_including_crowd"] == test["gt_area_groups_including_crowd"]["small"], "Old/new small GT populations differ")
        for metric in DISPLAY_METRICS:
            comparisons.append({"model": name, "metric": metric, "zero_shot": old["metrics"][metric],
                                "finetuned": test["metrics"][metric],
                                "delta": test["metrics"][metric] - old["metrics"][metric],
                                "delta_percentage_points": 100 * (test["metrics"][metric] - old["metrics"][metric])})
        audited[name]["zero_shot"] = {"metrics": old["metrics"], "raw": old_raw,
                                     "evaluation_sha256": sha256(previous / "evaluation.json")}
    return {"status": "complete_and_verified", "schema": "course_finetune_comparison_v1",
            "status_scope": "Artifact consistency audit; not an independent reproduction",
            "material_passport": {
                "origin_skill": "academic-research-suite/experiment-agent",
                "mode": "run/analyze",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "verification_status": "ANALYZED",
                "scope": "单次训练的结果与产物一致性审计；不声称独立复现",
            },
            "protocol": {"path": str(protocol_path.resolve()), "sha256": protocol_sha},
            "source_sha256": {"tools/summarize_course_finetune.py": sha256(__file__)},
            "counts": {"train_per_epoch": 14432, "epochs": 5, "val": 2584, "test": 2112},
            "models": audited, "comparison_rows": comparisons, "metric_scale": "0–1; delta_percentage_points explicitly multiplies by 100",
            "checks": {"identical_epoch_orders": True, "complete_train_val_test": True,
                       "validation_only_checkpoint_selection": True, "no_mask_decode_repeated": True},
            "limitations": ["两个模型均只使用 RGB；课程 component 单类；不同 backbone 与架构。",
                            "单个种子、各五轮训练；不宣称充分收敛或统计显著。",
                            "此前已查看零样本模型在该测试集上的成绩；本轮五轮训练协议在本轮训练开始前固定。该测试集不是从未查看过的独立留出集。",
                            "Mask2Former 从旧 HF/PIL 插值改为共同 Torch bilinear antialias=False；零样本前后差值包含该预处理差异。",
                            "测试仅用于最终比较，不用于选择轮次；按完整验证集 APs、AP、较早轮次选最佳。",
                            "SHA 与收据核验不重新评分或解码；已清理的非最佳验证 raw 仅核对保留的 SHA/清理记录。",
                            "读取官方 ZIP 元数据核验 ID；本工具不重新计算整个约 20 GB ZIP 的 SHA。"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "reports/course_finetune")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "reports/course_pretrained")
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "comparison_summary.json"
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    result = summarize(args.root.resolve(), args.pretrained.resolve(), args.protocol or args.root / "protocol.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(result, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": result["status"], "output": str(output), "comparison_rows": result["comparison_rows"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
