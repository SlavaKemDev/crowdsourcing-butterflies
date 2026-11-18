"""Object-level evaluation and strict Contest JSONL validation (standard library)."""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def read_jsonl(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{path}: duplicate JSON key {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"{path}: invalid non-finite JSON constant {value}")

    rows = {}
    with Path(path).open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                raise ValueError(f"{path}:{line_number}: empty JSONL line")
            row = json.loads(line, object_pairs_hook=unique_object, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: JSONL row must be an object")
            photo_id = row.get("photo_id")
            if type(photo_id) is not int or photo_id <= 0 or photo_id in rows:
                raise ValueError(f"{path}:{line_number}: invalid or duplicate photo_id")
            rows[photo_id] = row
    if not rows:
        raise ValueError(f"{path}: empty dataset")
    return rows


def validate_box(box):
    fields = ("left", "top", "width", "height")
    if not isinstance(box, dict) or not all(k in box for k in fields):
        raise ValueError("box must have left, top, width, height")
    values = [box[k] for k in fields]
    if not all(type(v) in (int, float) and math.isfinite(v) for v in values):
        raise ValueError("box coordinates must be finite numbers")
    left, top, width, height = values
    if not (0 <= left < 1 and 0 <= top < 1 and 0 < width <= 1 and 0 < height <= 1
            and left + width <= 1 + 1e-9 and top + height <= 1 + 1e-9):
        raise ValueError("box must be nonempty and inside [0, 1]")


def validate_rows(rows, expected_ids, allow_errors=False):
    if set(rows) != set(expected_ids):
        raise ValueError(f"photo_id mismatch: missing={sorted(set(expected_ids)-set(rows))}, "
                         f"extra={sorted(set(rows)-set(expected_ids))}")
    for photo_id, row in rows.items():
        if row.get("error") is not None:
            if allow_errors and isinstance(row["error"], str) and row["error"]:
                if row.get("boxes") is not None:
                    raise ValueError(f"{photo_id}: failed row must have boxes=null")
                continue
            raise ValueError(f"{photo_id}: unresolved processing error")
        if not isinstance(row.get("boxes"), list):
            raise ValueError(f"{photo_id}: boxes must be a list")
        for box in row["boxes"]:
            validate_box(box)


def validate_submission(rows, expected_ids):
    """Contest accepts a narrower schema than internal prediction/evaluation records."""
    validate_rows(rows, expected_ids)
    for photo_id, row in rows.items():
        if (type(row.get("photo_id")) is not int or row["photo_id"] != photo_id
                or set(row) - {"photo_id", "photo_url", "boxes", "error"}):
            raise ValueError(f"{photo_id}: invalid submission identity or extra row fields")
        if any(set(box) != {"left", "top", "width", "height"} for box in row["boxes"]):
            raise ValueError(f"{photo_id}: extra submission box fields")


def iou(a, b):
    width = max(0, min(a["left"] + a["width"], b["left"] + b["width"])
                - max(a["left"], b["left"]))
    height = max(0, min(a["top"] + a["height"], b["top"] + b["height"])
                 - max(a["top"], b["top"]))
    intersection = width * height
    union = a["width"] * a["height"] + b["width"] * b["height"] - intersection
    return intersection / union if union else 0.0


def matched_count(truth, prediction, threshold):
    """Maximum cardinality matching; a greedy IoU match can undercount TP."""
    edges = [[j for j, target in enumerate(truth) if iou(box, target) >= threshold]
             for box in prediction]
    owner = {}

    def augment(index, visited):
        for target in edges[index]:
            if target not in visited:
                visited.add(target)
                if target not in owner or augment(owner[target], visited):
                    owner[target] = index
                    return True
        return False

    return sum(augment(index, set()) for index in range(len(prediction)))


def summarize(truth, predictions, ids, threshold):
    tp = fp = fn = negative_fp = negative_images_with_fp = errors = 0
    negative_images = 0
    seconds = []
    for photo_id in ids:
        target = truth[photo_id]["boxes"]
        row = predictions[photo_id]
        failed = row.get("error") is not None
        boxes = [] if failed else row["boxes"]
        paired = matched_count(target, boxes, threshold)
        tp += paired
        fp += len(boxes) - paired
        fn += len(target) - paired
        errors += failed
        if not target:
            negative_images += 1
            negative_fp += len(boxes)
            negative_images_with_fp += bool(boxes)
        duration = row.get("seconds")
        if type(duration) in (int, float) and math.isfinite(duration) and duration >= 0:
            seconds.append(duration)
    return {
        "images": len(ids), "iou_threshold": threshold, "tp": tp, "fp": fp, "fn": fn,
        "precision": tp / (tp + fp) if tp + fp else 0.0,
        "recall": tp / (tp + fn) if tp + fn else 0.0,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0,
        "processing_errors": errors, "negative_images": negative_images,
        "negative_false_positive_boxes": negative_fp,
        "negative_images_with_false_positives": negative_images_with_fp,
        "timed_images": len(seconds), "total_seconds": sum(seconds),
        "mean_seconds": sum(seconds) / len(seconds) if seconds else None,
    }


def evaluate(truth, predictions, manifest=None):
    validate_rows(truth, truth)
    validate_rows(predictions, truth, allow_errors=True)
    groups = defaultdict(list)
    if manifest is not None:
        if set(manifest) != set(truth):
            raise ValueError("manifest and golden set photo_ids differ")
        for photo_id, row in manifest.items():
            groups[row.get("group", "unspecified")].append(photo_id)
    return {
        "all": {str(t): summarize(truth, predictions, list(truth), t) for t in (0.5, 0.75)},
        "groups": {group: {str(t): summarize(truth, predictions, ids, t)
                           for t in (0.5, 0.75)} for group, ids in groups.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    score = sub.add_parser("evaluate")
    score.add_argument("--golden", type=Path, required=True)
    score.add_argument("--predictions", type=Path, required=True)
    score.add_argument("--manifest", type=Path)
    score.add_argument("--output", type=Path, required=True)
    check = sub.add_parser("validate")
    check.add_argument("--predictions", type=Path, required=True)
    check.add_argument("--manifest", type=Path, required=True)
    check.add_argument("--expected-count", type=int, default=400)
    args = parser.parse_args()
    predictions = read_jsonl(args.predictions)
    manifest = read_jsonl(args.manifest) if args.manifest else None
    if args.command == "validate":
        if len(manifest) != args.expected_count:
            raise ValueError(f"expected {args.expected_count} images, got {len(manifest)}")
        if args.predictions.stat().st_size > 5 * 1024 * 1024:
            raise ValueError("submission exceeds 5 MiB")
        validate_submission(predictions, manifest)
        print(f"Valid: {len(predictions)} unique photo_ids, no processing errors")
    else:
        result = evaluate(read_jsonl(args.golden), predictions, manifest)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(result["all"], indent=2))


if __name__ == "__main__":
    main()
