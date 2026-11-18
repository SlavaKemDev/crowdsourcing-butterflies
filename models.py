"""Sequential local inference, one-factor postprocessing, and Contest export."""

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import shutil
import socket
import time
from datetime import datetime, timezone
from pathlib import Path

from metrics import iou, read_jsonl, validate_box, validate_rows, validate_submission

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

ROOT = Path(__file__).resolve().parent
MODEL_IDS = {
    "dino": "IDEA-Research/grounding-dino-tiny",
    "owlv2": "google/owlv2-base-patch16",
    "qwen": "Qwen/Qwen3-VL-2B-Instruct",
}
MODEL_REVISIONS = {
    "dino": "a2bb814dd30d776dcf7e30523b00659f4f141c71",
    "owlv2": "2a1560802f8cf3c408fec9b809d705f56a2f7146",
    "qwen": "89644892e4d85e24eaac8bacfd4f463576704203",
}
PROMPTS = {
    "dino": "butterfly. moth.",
    "owlv2": "a butterfly|a moth",
    "qwen": (
        "Locate every adult butterfly or moth in this image. Draw one tight bounding box "
        "around all visible parts of each adult, including wings, body, antennae and legs. "
        "For overlapping or cropped adults, include only visible parts. Do not include "
        "other insects, caterpillars, pupae or shadows. Return only a JSON array, with "
        "one object per adult: {\"bbox_2d\": [x_min, y_min, "
        "x_max, y_max]}. Coordinates must be integers normalized to 0..1000. "
        "If there are no adult butterflies or moths, return []."
    ),
}


def xyxy_box(coordinates, width, height):
    if len(coordinates) != 4:
        raise ValueError("expected four xyxy coordinates")
    x1, y1, x2, y2 = [float(v) for v in coordinates]
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        raise ValueError("non-finite model coordinates")
    if x2 <= x1 or y2 <= y1:
        raise ValueError("inverted or empty model box")
    x1, x2 = max(0.0, x1 / width), min(1.0, x2 / width)
    y1, y2 = max(0.0, y1 / height), min(1.0, y2 / height)
    result = dict(left=x1, top=y1, width=x2-x1, height=y2-y1)
    validate_box(result)
    return result


def parse_qwen(text):
    stripped = text.strip()
    if stripped.startswith("```json") and stripped.endswith("```"):
        stripped = stripped[7:-3].strip()
    elif stripped.startswith("```") and stripped.endswith("```"):
        stripped = stripped[3:-3].strip()
    answer = json.loads(stripped)
    if not isinstance(answer, list):
        raise ValueError("Qwen output must be a JSON array")
    candidates = []
    for item in answer:
        if not isinstance(item, dict) or "bbox_2d" not in item:
            raise ValueError("Qwen output has unexpected box format")
        coordinates = item["bbox_2d"]
        if not isinstance(coordinates, list) or not all(type(v) in (int, float) for v in coordinates):
            raise ValueError("Qwen coordinates must be numeric")
        if any(v < 0 or v > 1000 for v in coordinates):
            raise ValueError("Qwen coordinates outside 0..1000")
        # Qwen emits no confidence; equal scores preserve generation order for NMS.
        candidates.append(dict(box=xyxy_box(coordinates, 1000, 1000), score=1.0))
    return candidates


def recover_qwen_prefix(text):
    """Recover complete objects from a response cut inside its next bbox."""
    prefix, _, tail = text.strip().rpartition(',')
    # shortcut: the recorded 768-token failure ends at the opening of a bbox array.
    if "".join(tail.split()) != '{"bbox_2d":[' or not prefix.startswith("```json\n["):
        raise ValueError("unsupported truncated Qwen response")
    candidates = parse_qwen(prefix[7:].strip() + "]")
    if not candidates:
        raise ValueError("no complete Qwen boxes to recover")
    return candidates


def select_boxes(candidates, threshold, nms_iou):
    kept = []
    for item in sorted(candidates, key=lambda x: x["score"], reverse=True):
        if item["score"] >= threshold and all(iou(item["box"], other) <= nms_iou for other in kept):
            kept.append(item["box"])
    return kept


def load_model(model_key, revision, device):
    import torch
    from huggingface_hub import HfApi, snapshot_download
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Qwen3VLForConditionalGeneration

    model_id = MODEL_IDS[model_key]
    revision = revision or MODEL_REVISIONS[model_key]
    cache = ROOT / ".cache" / "huggingface"
    cache.mkdir(parents=True, exist_ok=True)
    repo_cache = cache / ("models--" + model_id.replace("/", "--"))
    snapshot = repo_cache / "snapshots" / revision
    if not all((snapshot / name).is_file() for name in ("config.json", "model.safetensors", "preprocessor_config.json")):
        info = HfApi(token=False).model_info(model_id, revision=revision, files_metadata=True)
        revision = info.sha
        weight_bytes = sum(f.size or 0 for f in info.siblings if f.rfilename.endswith(".safetensors"))
        # Existing blobs count as reusable space; the download itself never deletes user files.
        cached_bytes = sum(p.stat().st_size for p in (repo_cache / "blobs").glob("*") if p.is_file())
        required = max(0, weight_bytes - cached_bytes) + 2 * 1024**3
        if shutil.disk_usage(cache).free < required:
            raise RuntimeError(f"insufficient disk space: need {required/1024**3:.2f} GiB including 2 GiB reserve")
        snapshot = snapshot_download(model_id, revision=revision, cache_dir=cache, token=False,
                                     allow_patterns=["*.json", "*.txt", "*.safetensors", "*.jinja"],
                                     max_workers=1)
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, use_fast=False)
    if model_key == "qwen":
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16 if device == "mps" else torch.float32,
            attn_implementation="sdpa")
    else:
        model = AutoModelForZeroShotObjectDetection.from_pretrained(snapshot, local_files_only=True)
    model.eval().to(device)
    return model, processor, revision, str(snapshot)


def infer_image(model_key, model, processor, image, device, prompt, max_new_tokens):
    import torch
    if model_key == "qwen":
        messages = [dict(role="user", content=[dict(type="image", image=image), dict(type="text", text=prompt)])]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                              return_dict=True, return_tensors="pt")
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(device)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        text = processor.batch_decode(output[:, inputs.input_ids.shape[-1]:],
                                      skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return None, text
    queries = prompt if model_key == "dino" else [prompt.split("|")]
    inputs = processor(images=image, text=queries, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    if model_key == "dino":
        processed = processor.post_process_grounded_object_detection(
            outputs, inputs.input_ids, threshold=0.05, text_threshold=0.2,
            target_sizes=[(image.height, image.width)])[0]
    else:
        processed = processor.post_process_object_detection(
            outputs=outputs, threshold=0.05, target_sizes=[(image.height, image.width)])[0]
    candidates = [dict(box=xyxy_box(coords, image.width, image.height), score=float(score))
                  for coords, score in zip(processed["boxes"].detach().cpu().tolist(),
                                           processed["scores"].detach().cpu().tolist())]
    return candidates, None


def run(args):
    if args.ipv4:
        # This machine's IPv6 route fails TLS; keep certificate verification intact.
        original_getaddrinfo = socket.getaddrinfo
        socket.getaddrinfo = lambda host, port, family=0, type=0, proto=0, flags=0: original_getaddrinfo(
            host, port, socket.AF_INET, type, proto, flags)
    import torch
    from PIL import Image, ImageOps
    manifest = read_jsonl(args.manifest)
    device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
    prompt = args.prompt or PROMPTS[args.model]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}; choose a new run output")
    started = time.perf_counter()
    model, processor, revision, snapshot = load_model(args.model, args.revision, device)
    config = {
        "started_at": datetime.now(timezone.utc).isoformat(), "model": MODEL_IDS[args.model],
        "revision": revision, "snapshot": snapshot, "device": device, "prompt": prompt,
        "threshold": args.threshold, "candidate_threshold": 0.05, "text_threshold": 0.2,
        "nms_iou": args.nms_iou, "max_new_tokens": args.max_new_tokens,
        "do_sample": False, "batch_size": 1, "exif_transpose": True,
        "manifest_sha256": hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        "images_expected": len(manifest), "smoke_limit": args.limit,
        "python": platform.python_version(), "platform": platform.platform(),
        "packages": {p: importlib.metadata.version(p)
                     for p in ("torch", "torchvision", "transformers", "Pillow", "numpy", "scipy")},
        "image_processor": processor.image_processor.to_dict(),
        "model_load_seconds": time.perf_counter() - started,
    }
    config_path = args.output.with_suffix(".run.json")
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    errors_path = args.output.with_suffix(".errors.jsonl")
    with args.output.open("x", encoding="utf-8") as output, errors_path.open("x", encoding="utf-8") as errors:
        for index, (photo_id, item) in enumerate(manifest.items()):
            if args.limit and index >= args.limit:
                break
            timer = time.perf_counter()
            row = dict(photo_id=photo_id, boxes=None, error=None)
            try:
                image_path = args.manifest.parent / item["image_path"]
                with Image.open(image_path) as source:
                    image = ImageOps.exif_transpose(source).convert("RGB")
                candidates, raw_text = infer_image(args.model, model, processor, image, device, prompt, args.max_new_tokens)
                if raw_text is not None:
                    row["raw_text"] = raw_text
                    candidates = parse_qwen(raw_text)
                row.update(boxes=select_boxes(candidates, args.threshold, args.nms_iou), candidates=candidates)
                if device == "mps":
                    torch.mps.synchronize()
            except Exception as exc:
                row["boxes"] = None
                row["error"] = f"{type(exc).__name__}: {exc}"
                errors.write(json.dumps(dict(photo_id=photo_id, error=row["error"]), ensure_ascii=False) + "\n")
                errors.flush()
            row["seconds"] = time.perf_counter() - timer
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()
            print(f"{index+1}/{len(manifest)} photo_id={photo_id} boxes={len(row['boxes']) if row['boxes'] is not None else 'ERROR'} seconds={row['seconds']:.3f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    infer = sub.add_parser("infer")
    infer.add_argument("--model", choices=MODEL_IDS, required=True)
    infer.add_argument("--manifest", type=Path, required=True)
    infer.add_argument("--output", type=Path, required=True)
    infer.add_argument("--revision", help="defaults to the exact pinned model commit")
    infer.add_argument("--device", choices=("mps", "cpu"))
    infer.add_argument("--prompt")
    infer.add_argument("--threshold", type=float, default=0.25)
    infer.add_argument("--nms-iou", type=float, default=0.5)
    # shortcut: dense scenes can exceed 768 tokens; recalibrate the cap before relying on Qwen.
    infer.add_argument("--max-new-tokens", type=int, default=768)
    infer.add_argument("--limit", type=int, default=0, help="nonzero only for smoke tests")
    infer.add_argument("--ipv4", action="store_true", help="use when this network's IPv6 TLS route fails")
    post = sub.add_parser("postprocess")
    post.add_argument("--predictions", type=Path, required=True)
    post.add_argument("--output", type=Path, required=True)
    post.add_argument("--threshold", type=float, required=True)
    post.add_argument("--nms-iou", type=float, default=0.5)
    export = sub.add_parser("export")
    export.add_argument("--predictions", type=Path, required=True)
    export.add_argument("--manifest", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--expected-count", type=int, default=400)
    export.add_argument("--recover-qwen-prefix", action="store_true",
                        help="recover a recognized truncated response with the pinned Qwen settings")
    args = parser.parse_args()
    if args.command == "infer":
        if not (0.05 <= args.threshold <= 1 and 0 <= args.nms_iou <= 1 and args.limit >= 0):
            parser.error("invalid threshold, NMS IoU, or limit")
        run(args)
        return
    rows = read_jsonl(args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")
    if args.command == "postprocess":
        if not 0.05 <= args.threshold <= 1 or not 0 <= args.nms_iou <= 1:
            parser.error("threshold must be >= saved candidate threshold 0.05; NMS must be in [0,1]")
        for row in rows.values():
            if row.get("error") is None:
                row["boxes"] = select_boxes(row["candidates"], args.threshold, args.nms_iou)
        run_config = dict(source=str(args.predictions), threshold=args.threshold, nms_iou=args.nms_iou,
                          source_sha256=hashlib.sha256(args.predictions.read_bytes()).hexdigest())
        args.output.with_suffix(".run.json").write_text(json.dumps(run_config, indent=2) + "\n")
    else:
        manifest = read_jsonl(args.manifest)
        if len(manifest) != args.expected_count:
            raise ValueError(f"expected {args.expected_count} images, got {len(manifest)}")
        if args.recover_qwen_prefix:
            validate_rows(rows, manifest, allow_errors=True)
            for row in rows.values():
                if (str(row.get("error")).startswith("JSONDecodeError:")
                        and isinstance(row.get("raw_text"), str)):
                    row["boxes"] = select_boxes(recover_qwen_prefix(row["raw_text"]), 0.25, 0.5)
                    row["error"] = None
        validate_rows(rows, manifest)
        rows = {photo_id: dict(photo_id=photo_id, boxes=rows[photo_id]["boxes"], error=None)
                for photo_id in manifest}
        validate_submission(rows, manifest)
    content = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows.values())
    if args.command == "export" and len(content.encode("utf-8")) > 5 * 1024 * 1024:
        raise ValueError("submission exceeds 5 MiB")
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(content)


if __name__ == "__main__":
    main()
