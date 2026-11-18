"""Collect and freeze the team's CC0 evaluation images from iNaturalist."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import re
import subprocess
import tempfile
import time
from urllib.parse import urlencode, urlparse

from PIL import Image, ImageOps


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 20261117
STRATA = [
    ("butterflies", 200, {"taxon_id": 47224, "term_id": 1, "term_value_id": 2}),
    ("moths", 100, {"taxon_id": 47157, "without_taxon_id": 47224, "term_id": 1, "term_value_id": 2}),
    ("beetles", 15, {"taxon_id": 47208}),
    ("flies", 15, {"taxon_id": 47822}),
    ("bees_wasps_ants", 15, {"taxon_id": 47201}),
    ("true_bugs", 15, {"taxon_id": 47744}),
    ("flowering_plants", 20, {"taxon_id": 47126, "term_id": 12, "term_value_id": 13}),
    ("larvae", 10, {"taxon_id": 47157, "term_id": 1, "term_value_id": 6}),
    ("pupae", 10, {"taxon_id": 47157, "term_id": 1, "term_value_id": 4}),
]


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def download(url, max_bytes=15_000_000):
    if urlparse(url).scheme != "https":
        raise ValueError("HTTPS is required")
    # System curl uses the working system TLS stack on this machine.
    # A real output file lets curl truncate partial responses before a retry.
    with tempfile.TemporaryDirectory(prefix="hw5-download-") as directory:
        target = Path(directory) / "response"
        result = subprocess.run([
            "curl", "-4", "--fail", "--silent", "--show-error", "--location",
            "--proto", "=https", "--proto-redir", "=https", "--retry", "3", "--retry-all-errors",
            "--connect-timeout", "15", "--max-time", "60", "--max-filesize", str(max_bytes),
            "--user-agent", "HW5-CC0-evaluation/1.0", "--output", str(target), url,
        ], capture_output=True)
        if result.returncode:
            raise OSError(result.stderr.decode(errors="replace").strip())
        body = target.read_bytes()
        if len(body) > max_bytes:
            raise ValueError("Response exceeds size limit")
        return body


def candidates(observation, group):
    """An observation's license never substitutes for its photos' licenses."""
    for photo in sorted(observation.get("photos", []), key=lambda p: p["id"]):
        if photo.get("license_code") != "cc0":
            continue
        match = re.search(r"/square\.(jpg|jpeg|png|webp)(?:\?|$)", photo.get("url", ""), re.I)
        if not match:
            continue
        ext = match.group(1).lower()
        photo_id = int(photo["id"])
        base = f"https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}"
        taxon = observation.get("taxon") or {}
        user = observation.get("user") or {}
        yield {
            "photo_id": photo_id,
            "observation_id": int(observation["id"]),
            "observation_url": f"https://www.inaturalist.org/observations/{observation['id']}",
            "photo_page_url": f"https://www.inaturalist.org/photos/{photo_id}",
            "image_url": f"{base}/large.{ext}",
            "original_url": f"{base}/original.{ext}",
            "image_path": f"images/{photo_id}.{ext}",
            "license_code": "cc0",
            "attribution": photo.get("attribution", ""),
            "author": photo.get("attribution_name") or user.get("name") or user.get("login"),
            "observer_login": user.get("login"),
            "observer_id": user.get("id"),
            "taxon_id": taxon.get("id"),
            "taxon_name": taxon.get("name"),
            "taxon_ancestor_ids": taxon.get("ancestor_ids", []),
            "annotations": [
                {k: annotation.get(k) for k in ("controlled_attribute_id", "controlled_value_id")}
                for annotation in observation.get("annotations", [])
            ],
            "group": group,
            "source_group_is_ground_truth": False,
        }


def collect_candidates(group, params, pilot=False):
    snapshot = DATA / "sources" / f"{group}.json"
    page_limit = 1 if pilot or group not in {"butterflies", "moths"} else 4
    items, queries = [], []
    if snapshot.exists():
        cached = json.loads(snapshot.read_text(encoding="utf-8"))
        items, queries = cached["candidates"], cached["queries"]
        if len(queries) >= page_limit or cached.get("exhausted"):
            return items
    # Snapshot a broad candidate pool before selecting the fixed sample.
    exhausted = False
    for page in range(len(queries) + 1, page_limit + 1):
        query = {
            "photos": "true", "photo_license": "cc0", "quality_grade": "research",
            "per_page": 200, "page": page, "order_by": "id", "order": "desc",
            "d2": "2026-11-16", **params,
        }
        url = "https://api.inaturalist.org/v1/observations?" + urlencode(query)
        response = json.loads(download(url, max_bytes=30_000_000))
        queries.append({"url": url, "total_results": response["total_results"], "fetched_at": datetime.now(timezone.utc).isoformat()})
        for observation in response["results"]:
            items.extend(candidates(observation, group))
        print(f"Candidates {group}: page {page}, {len(items)} CC0 photos", flush=True)
        if len(response["results"]) < 200:
            exhausted = True
            break
        time.sleep(1.2)
    random.Random(f"{SEED}:{group}").shuffle(items)
    write_json(snapshot, {"fetched_at": datetime.now(timezone.utc).isoformat(), "queries": queries, "candidates": items, "exhausted": exhausted})
    return items


def image_info(body):
    with Image.open(BytesIO(body)) as image:
        if max(image.size) > 1024:
            raise ValueError("Unexpected image dimensions for the large variant")
        image.verify()
    with Image.open(BytesIO(body)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.load()
        if min(image.size) < 160:
            raise ValueError("Image's shorter dimension is below 160 pixels")
        small = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS).tobytes()
        dhash = sum((small[y * 9 + x] > small[y * 9 + x + 1]) << (y * 8 + x) for y in range(8) for x in range(8))
        return {
            "width": image.width,
            "height": image.height,
            "sha256": hashlib.sha256(body).hexdigest(),
            "pixel_sha256": hashlib.sha256(f"{image.size}:".encode() + image.tobytes()).hexdigest(),
            "dhash": f"{dhash:016x}",
            "bytes": len(body),
        }


def fetch_image(row):
    path = DATA / row["image_path"]
    body = path.read_bytes() if path.exists() else download(row["image_url"])
    info = image_info(body)
    return row, info, body


def save_manifest(rows):
    path = DATA / "manifest.jsonl"
    temporary = path.with_suffix(".jsonl.tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    temporary.replace(path)


def is_duplicate(info, previous):
    # shortcut: dHash is approximate; visually audit rejected pairs if it changes coverage.
    return any(
        info["sha256"] == row["sha256"]
        or info["pixel_sha256"] == row["pixel_sha256"]
        or (int(info["dhash"], 16) ^ int(row["dhash"], 16)).bit_count() <= 3
        for row in previous
    )


def collect(pilot=False):
    DATA.mkdir(exist_ok=True)
    (DATA / "images").mkdir(exist_ok=True)
    if (DATA / "dataset.lock.json").exists():
        frozen_manifest = DATA / "manifest.jsonl"
        lock = json.loads((DATA / "dataset.lock.json").read_text(encoding="utf-8"))
        if hashlib.sha256(frozen_manifest.read_bytes()).hexdigest() != lock["manifest_sha256"]:
            raise ValueError("Frozen manifest checksum changed")
        for row in map(json.loads, frozen_manifest.read_text(encoding="utf-8").splitlines()):
            path = DATA / row["image_path"]
            if not path.exists():
                body = download(row["image_url"])
                if hashlib.sha256(body).hexdigest() != row["sha256"]:
                    raise ValueError(f"Downloaded content changed: {row['photo_id']}")
                path.write_bytes(body)
                print(f"Restored frozen photo {row['photo_id']}", flush=True)
        return verify()
    manifest = DATA / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text().splitlines()] if manifest.exists() else []
    seen_observations = {row["observation_id"] for row in rows}
    seen_photos = {row["photo_id"] for row in rows}
    counts = Counter(row["group"] for row in rows)
    authors = Counter((row["group"], row["observer_id"]) for row in rows)
    failures_path = DATA / "collection_failures.json"
    failures = json.loads(failures_path.read_text(encoding="utf-8")) if failures_path.exists() else []
    for group, count, params in STRATA:
        if pilot:
            count = 2
        pool = collect_candidates(group, params, pilot=pilot)
        pending = []
        for row in pool:
            if row["photo_id"] in seen_photos or row["observation_id"] in seen_observations:
                continue
            pending.append(row)
        for start in range(0, len(pending), 8):
            if counts[group] >= count:
                break
            batch = pending[start:start + min(8, count - counts[group])]
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [(row, executor.submit(fetch_image, row)) for row in batch]
                for row, future in futures:
                    try:
                        candidate, info, body = future.result()
                        author_key = (group, candidate["observer_id"])
                        if candidate["observation_id"] in seen_observations or authors[author_key] >= 10:
                            continue
                        if is_duplicate(info, rows):
                            failures.append({"photo_id": candidate["photo_id"], "reason": "duplicate_or_near_duplicate"})
                            continue
                        if sum(item["bytes"] for item in rows) + info["bytes"] > 750_000_000:
                            raise RuntimeError("Dataset exceeds the 750 MB storage budget")
                        image_path = DATA / candidate["image_path"]
                        image_path.write_bytes(body)
                        rows.append({**candidate, **info})
                        counts[group] += 1
                        authors[author_key] += 1
                        seen_photos.add(candidate["photo_id"])
                        seen_observations.add(candidate["observation_id"])
                        save_manifest(rows)
                        print(f"Downloaded {len(rows)}/400: {group} {counts[group]}/{count}, photo {candidate['photo_id']}", flush=True)
                    except (OSError, ValueError) as exc:
                        failures.append({"photo_id": row["photo_id"], "reason": str(exc)})
        write_json(DATA / "collection_failures.json", failures)
        if counts[group] != count:
            raise RuntimeError(f"Only {counts[group]}/{count} usable images in {group}; expand its candidate snapshot before freezing")
    if pilot:
        (DATA / "pilot_manifest.jsonl").write_bytes(manifest.read_bytes())
        print("Pilot images ready; manifest is not frozen until full collection completes.", flush=True)
        return
    write_json(DATA / "dataset.lock.json", {
        "frozen_at": datetime.now(timezone.utc).isoformat(), "seed": SEED,
        "count": len(rows), "groups": dict(counts),
        "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "total_image_bytes": sum(row["bytes"] for row in rows),
        "image_variant": "large", "maximum_dimension": 1024,
        "source_groups_are_ground_truth": False,
    })
    verify()


def verify():
    manifest = DATA / "manifest.jsonl"
    rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    lock = json.loads((DATA / "dataset.lock.json").read_text(encoding="utf-8"))
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != lock["manifest_sha256"]:
        raise ValueError("Frozen manifest checksum changed")
    if len(rows) != 400 or len({r["photo_id"] for r in rows}) != 400 or len({r["observation_id"] for r in rows}) != 400:
        raise ValueError("Expected 400 unique photos and observations")
    if Counter(row["group"] for row in rows) != Counter({name: count for name, count, _ in STRATA}):
        raise ValueError("Unexpected dataset composition")
    previous = []
    for row in rows:
        if row["license_code"] != "cc0" or row["source_group_is_ground_truth"] is not False:
            raise ValueError(f"Invalid provenance for {row['photo_id']}")
        info = image_info((DATA / row["image_path"]).read_bytes())
        if any(info[key] != row[key] for key in info):
            raise ValueError(f"Image metadata/checksum changed: {row['photo_id']}")
        if is_duplicate(info, previous):
            raise ValueError(f"Duplicate image: {row['photo_id']}")
        previous.append(row)
    print(json.dumps({"verified": len(rows), "groups": lock["groups"], "bytes": lock["total_image_bytes"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["pilot", "collect", "verify"])
    arguments = parser.parse_args()
    verify() if arguments.command == "verify" else collect(pilot=arguments.command == "pilot")
