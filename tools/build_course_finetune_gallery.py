#!/usr/bin/env python3
"""Render only completed, audited fine-tuning results; no inference or placeholders."""
from __future__ import annotations
import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess
import sys
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PIL import Image, ImageDraw
from rgbd_seg.course_archive import CourseArchive
from tools import build_course_pretrained_gallery as previous
from tools.summarize_course_finetune import require, read, sha256, check_hash

TEMPLATE = ROOT / "tools/templates/course-finetuned-inline.html"
RENDERER = Path("/data/yinli/.codex/plugins/cache/openai-bundled/visualize/1.0.49/skills/visualize/scripts/render.py")
DEFAULT_INLINE = Path("/data/yinli/.codex/visualizations/2026/10/01/01a0f8d8-d8a0-7ab3-99e3-d630b1ffcb25/course-finetuned-comparison.html")


def safe_json(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).replace("<", "\\u003c").replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")


def load_predictions(summary, dataset):
    saved, receipts = {}, {}
    for name in previous.MODELS:
        model = summary["models"][name]
        test = model["test"]
        folder = Path(test["path"])
        raw = folder / "predictions.jsonl"
        check_hash(raw, test["raw"]["sha256"])
        check_hash(folder / "evaluation.json", test["evaluation_sha256"])
        check_hash(folder / "evaluation_run.json", test["evaluation_run_sha256"])
        evaluation = read(folder / "evaluation.json")
        receipt = read(folder / "evaluation_run.json")
        require(evaluation["metrics"] == test["metrics"] and evaluation["images"] == len(dataset) == 2112,
                "Gallery must use the complete audited test evaluation")
        require(Path(receipt["archive"]).resolve() == dataset.zip_path, "Gallery ZIP differs from evaluated ZIP")
        saved[name], count = {}, 0
        with raw.open() as stream:
            for index, line in enumerate(stream):
                require(index < len(dataset), "Extra raw image rows")
                record = json.loads(line)
                expected = dataset.images[index]
                require(record["image_id"] == expected["id"] and record["file_name"] == expected["file_name"], "Raw order/identity differs")
                require(len(record["predictions"]) <= 100, "Invalid instance cap")
                if index in previous.INDICES:
                    # The shared renderer expects this compatibility field. In
                    # this run the native trained foreground label itself is 1.
                    record["original_category_ids"] = [p["category_id"] for p in record["predictions"]]
                    saved[name][index] = record
                count += 1
        require(count == len(dataset), "Missing raw image rows")
        receipts[name] = {"folder": str(folder), "metrics": evaluation["metrics"], "evaluated_images": count,
                          "evaluation_sha256": test["evaluation_sha256"], "predictions_sha256": test["raw"]["sha256"],
                          "best_epoch": model["best_epoch"], "best_checkpoint_sha256": model["best_checkpoint_sha256"],
                          "model_metadata": {"name": "Mask R-CNN ResNet50-FPN v2" if name == "maskrcnn" else "Mask2Former Swin-Tiny"}}
    return saved, receipts


def compact_fragment(cases, images):
    template = TEMPLATE.read_text()
    require(template.count("__DATA__") == 1, "Invalid inline template")
    # Only presentation thumbnails change; full gallery keeps detailed images.
    # Small crops stay lossless and retain their pixel dimensions.
    for bound, quality in ((480, 78), (400, 70), (320, 65)):
        packed = {}
        for key, uri in images.items():
            if "_small_" in key:
                packed[key] = uri
                continue
            image = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1]))).convert("RGB")
            image.thumbnail((bound, bound), Image.Resampling.LANCZOS)
            buffer = io.BytesIO()
            image.save(buffer, format="WEBP", quality=quality, method=6)
            packed[key] = "data:image/webp;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
        fragment = template.replace("__DATA__", safe_json({"cases": cases, "images": packed}))
        if len(fragment.encode()) < 1_000_000:
            return fragment, {"full_image_max_side": bound, "webp_quality": quality, "small_crops": "lossless original-size PNG"}
    raise ValueError("Inline exceeds 1 MB; retain full gallery and explicitly revise presentation compression")


def standalone(manifest, images):
    page = previous.HTML
    page = page.replace("课程数据 · 预训练模型直接测试", "课程数据 · 微调后模型对比")
    page = page.replace("FIXED PRETRAINED WEIGHTS · COURSE TEST SET", "FINE-TUNED MODELS · FIXED COURSE TEST CASES")
    page = page.replace("课程数据：两种预训练模型直接测试", "课程数据：五轮微调后的模型对比")
    notice = ("两模型均用完整课程训练集微调五轮，只使用 RGB、component 单类。按完整验证集 APₛ、AP、较早轮次选最佳，再测试一次。"
              "单种子、不同骨干与架构，不宣称充分收敛或统计显著。Mask2Former 相对旧零样本改用共同 Torch bilinear antialias=False 插值，前后差异包含该预处理变化。")
    page = re.sub(r'(<p class="notice">).*?(</p>)', lambda m: m[1] + notice + m[2], page, count=1)
    page = page.replace("course-pretrained-gallery.json", "course-finetuned-gallery.json")
    return page.replace("__DATA__", safe_json({**manifest, "embedded_images": images}))


def browser_check(output, preview):
    """Check real generated content on desktop/mobile, without network or GPU."""
    from playwright.sync_api import sync_playwright
    check_dir = output / "browser"
    check_dir.mkdir(exist_ok=True)
    manifest = read(output / "gallery.json")
    errors, external, checked = [], [], []
    with sync_playwright() as pw:
        browser = pw.chromium.launch(executable_path="/usr/bin/google-chrome", headless=True, args=["--no-sandbox"])
        context = browser.new_context(viewport={"width": 1400, "height": 1000}, offline=True)
        page = context.new_page()
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda req: external.append(req.url) if req.url.startswith(("http:", "https:")) else None)
        page.goto((output / "index.html").as_uri())
        for case in manifest["cases"]:
            page.locator("#case").select_option(case["key"])
            page.wait_for_function("Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)")
            require(page.locator("#full img").count() == 4, "Missing full gallery panels")
            require(page.locator("#small img").count() == (4 if case["small_detail"] else 0), "Small crop panel state differs")
        page.locator("#zoom").evaluate("el=>{el.value='2';el.dispatchEvent(new Event('input',{bubbles:true}))}")
        require(page.locator("#zoom-value").inner_text() == "2.00×", "Gallery zoom failed")
        page.locator("#reset").click()
        page.screenshot(path=str(check_dir / "gallery-desktop.png"), full_page=True)
        for width in (390, 320):
            page.set_viewport_size({"width": width, "height": 1000})
            require(page.evaluate("document.documentElement.scrollWidth<=innerWidth"), "Gallery horizontal overflow")
        page.screenshot(path=str(check_dir / "gallery-mobile320.png"), full_page=True)
        require(not external, f"Standalone gallery made external requests: {external}")
        # The official preview wrapper requests optional tooltip/icon libraries.
        # The fragment uses none of these. Keep the network offline, distinguish
        # these wrapper requests from dependencies introduced by our fragment.
        wrapper_dependencies = set(re.findall(r'<script[^>]+src="([^"]+)"',
            (RENDERER.parent.parent / "assets/visualize.html").read_text()))
        page.goto(preview.as_uri())
        frame = page.frame_locator("iframe")
        frame.locator("#course-finetuned-comparison").wait_for()
        content = next(f for f in page.frames if f.parent_frame is not None)
        for case in manifest["cases"]:
            frame.locator('[data-control="case"]').select_option(case["key"])
            for view in ("full", "small"):
                frame.locator('[data-control="view"]').select_option(view)
                content.wait_for_function("Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)")
                expected = 4 if view == "full" or case["small_detail"] else 0
                require(frame.locator("figure:visible").count() == expected, "Inline visible panels differ")
                checked.append({"case": case["key"], "view": view, "panels": expected})
        frame.locator('[data-control="case"]').select_option("case_0400")
        content.wait_for_function("Array.from(document.images).every(i=>i.complete&&i.naturalWidth>0)")
        for width in (736, 390, 320):
            page.set_viewport_size({"width": width, "height": 2400})
            require(content.evaluate("document.documentElement.scrollWidth<=innerWidth"), "Inline horizontal overflow")
            if width in (736, 390):
                page.screenshot(path=str(check_dir / f"inline-{width}.png"), full_page=True)
        content.evaluate("window.dispatchEvent(new CustomEvent('openai:set_globals',{detail:{globals:{widgetState:{privateContent:{caseKey:'case_0800'},modelContent:{view:'full'}}}}}))")
        require(frame.locator('[data-control="case"]').input_value() == "case_0800"
                and frame.locator('[data-control="view"]').input_value() == "full", "Inline state restore failed")
        unexpected_requests = sorted(set(external) - wrapper_dependencies)
        require(not errors and not unexpected_requests, f"Browser errors/fragment requests: {errors}, {unexpected_requests}")
        browser.close()
    result = {"status": "passed", "checked_states": checked, "widths": [736, 390, 320],
              "js_errors": errors, "gallery_external_requests": [], "fragment_external_requests": unexpected_requests,
              "wrapper_dependency_requests_blocked_offline": sorted(set(external)), "offline": True,
              "state_restore": True, "new_inference": False}
    (check_dir / "browser_test.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")


def build(args):
    summary = read(args.summary)
    require(summary["status"] == "complete_and_verified" and summary["schema"] == "course_finetune_comparison_v1",
            "Generate only after the complete real comparison summary passes")
    require(summary["counts"] == {"train_per_epoch": 14432, "epochs": 5, "val": 2584, "test": 2112}, "Unexpected study scope")
    require(not args.inline.exists() and not args.inline.is_symlink(), "Refusing to overwrite inline output")
    output = previous.ensure_empty(args.output)
    summary_sha = sha256(args.summary)
    with CourseArchive(args.archive, "test") as dataset:
        saved, receipts = load_predictions(summary, dataset)
        by_id = defaultdict(list)
        for annotation in dataset.coco_ground_truth()["annotations"]:
            by_id[annotation["image_id"]].append(annotation)
        images, cases, views = {}, [], []
        for index in previous.INDICES:
            case, full = previous.render_case(dataset, index, by_id[dataset.image_ids[index]], saved, images)
            cases.append(case)
            views.append(full)
        sheet = previous.contact_sheet(cases, views)
        draw = ImageDraw.Draw(sheet)
        draw.rectangle((0, 0, sheet.width, 74), fill="#101824")
        previous.draw_label(draw, (12, 10), "课程数据 · 五轮微调后 · 六个固定案例", size=23, fill="#f0f4f8")
        previous.draw_label(draw, (12, 43), "展示 score ≥ 0.5；评估 score ≥ 0.01；RGB / component；完整验证集选最佳轮次", size=14, fill="#b5c5d8")
        sheet.save(output / "contact_sheet.png", optimize=True)
        annotation_digest, rgb_hashes = hashlib.sha256(), {}
        with ZipFile(args.archive) as archive:
            for name in dataset.annotation_members:
                annotation_digest.update(name.encode() + b"\0" + archive.read(name))
            for case in cases:
                rgb_hashes[case["file_name"]] = hashlib.sha256(archive.read(case["file_name"])).hexdigest()
        manifest = {"status": "completed", "created_at_utc": datetime.now(timezone.utc).isoformat(),
                    "comparison_summary_sha256": summary_sha, "models": receipts, "cases": cases,
                    "fixed_indices_zero_based": list(previous.INDICES), "display_score_threshold": .5,
                    "evaluation_score_threshold": .01, "metrics_scale": "0–1", "selection": "Fixed official indices; no selection by result quality",
                    "annotation_members_sha256": annotation_digest.hexdigest(), "selected_rgb_sha256": rgb_hashes,
                    "small_crop_selection": "Smallest original GT area 0<area<1024; annotation ID tie break; post-evaluation diagnosis only",
                    "small_matching": "Maximum original-image mask IoU among score>=0.5 candidates; GT-assisted, never inference",
                    "limitations": summary["limitations"], "source_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in
                        (Path(__file__), TEMPLATE, Path(previous.__file__))}}
    fragment, compression = compact_fragment(cases, images)
    manifest["inline_compression"] = compression
    manifest["inline_bytes"] = len(fragment.encode())
    manifest["inline_sha256"] = hashlib.sha256(fragment.encode()).hexdigest()
    manifest["contact_sheet_sha256"] = sha256(output / "contact_sheet.png")
    check_hash(args.summary, summary_sha)
    for receipt in receipts.values():
        check_hash(Path(receipt["folder"]) / "predictions.jsonl", receipt["predictions_sha256"])
    (output / "gallery.json").write_text(json.dumps(manifest, ensure_ascii=False, allow_nan=False, indent=2) + "\n")
    (output / "index.html").write_text(standalone(manifest, images))
    args.inline.parent.mkdir(parents=True, exist_ok=True)
    with args.inline.open("x") as stream:
        stream.write(fragment)
    preview = output / "inline-preview.html"
    subprocess.run([sys.executable, str(RENDERER), str(args.inline.resolve()), str(preview.resolve())], check=True)
    if args.browser_check:
        browser_check(output.resolve(), preview.resolve())
    print(json.dumps({"status": "completed", "gallery": str(output.resolve()), "inline": str(args.inline.resolve()),
                      "inline_bytes": len(fragment.encode()), "browser_checked": args.browser_check}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=ROOT / "reports/course_finetune/comparison_summary.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/course_finetune/gallery")
    parser.add_argument("--inline", type=Path, default=DEFAULT_INLINE)
    parser.add_argument("--browser-check", action="store_true")
    parser.add_argument("--check-existing", action="store_true", help="Repeat browser checks only; never regenerate/overwrite result content")
    args = parser.parse_args()
    if args.check_existing:
        manifest = read(args.output / "gallery.json")
        check_hash(args.inline, manifest["inline_sha256"])
        check_hash(args.summary, manifest["comparison_summary_sha256"])
        browser_check(args.output.resolve(), (args.output / "inline-preview.html").resolve())
    else:
        build(args)


if __name__ == "__main__":
    main()
