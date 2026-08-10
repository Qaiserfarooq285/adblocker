#!/usr/bin/env python3
"""Assemble data/dataset/{images,labels}/{train,val} from three sources:

  1. data/annotations_real/  — real hand-annotated frames (YOLO-seg export
     from CVAT/Roboflow: either a flat set of <stem>.txt label files, or the
     standard images/ + labels/ export layout). Matched back to the actual
     frame image by filename stem, searched under data/annotations_real/images/
     first and then under data/frames/**/.

  2. data/synthetic/         — output of generate_synthetic.py, which already
     contains its own deterministic train/val split; that split is preserved
     as-is (never re-derived here).

  3. data/negatives/         — flat folder of hard-negative images (frames
     with non-gambling logos). Added with empty label files and split by a
     seeded hash so re-running is deterministic.

Real frames are split into train/val by TIME CHUNK PER SOURCE VIDEO, parsed
from the "<slug>_t<seconds>" filename convention written by
extract_frames.py — never randomly, because adjacent frames sampled at
extract.fps are near-duplicates and a random split would leak near-identical
frames across train/val and fake the validation score. Each chunk is
assigned to train or val as a whole, so it is structurally impossible for
two frames from the same video-time-window to land on opposite sides.

--append merges newly available real annotations, synthetic data, and
negatives into the existing dataset without wiping prior work. A persisted
split manifest (data/dataset/.split_manifest.json) remembers which chunk
went to which split so previously-assembled real frames never move sides
when new videos are appended later. Without --append, the dataset is fully
rebuilt from whatever currently exists on disk.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

from common import IMAGE_EXTS, die, load_config

FRAME_STEM_RE = re.compile(r"^(?P<slug>.+)_t(?P<sec>\d+(?:\.\d+)?)$")
MANIFEST_NAME = ".split_manifest.json"


def stable_fraction(key: str) -> float:
    """Deterministic pseudo-random value in [0, 1) derived from a string key,
    stable across runs/machines (unlike Python's salted hash())."""
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2 ** 64


def parse_chunk_key(stem: str, chunk_seconds: float) -> str:
    m = FRAME_STEM_RE.match(stem)
    if not m:
        # No timestamp info available (e.g. hand-picked still not from
        # extract_frames.py) — treat each such frame as its own chunk so it
        # can't accidentally get grouped with unrelated frames.
        return f"unknown#{stem}"
    slug = m.group("slug")
    sec = float(m.group("sec"))
    chunk_id = int(sec // chunk_seconds)
    return f"{slug}#{chunk_id}"


def load_manifest(dataset_dir: Path) -> dict:
    path = dataset_dir / MANIFEST_NAME
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_manifest(dataset_dir: Path, manifest: dict) -> None:
    (dataset_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def find_real_label_files(annotations_dir: Path) -> list[Path]:
    labels_sub = annotations_dir / "labels"
    search_dir = labels_sub if labels_sub.exists() else annotations_dir
    return sorted(p for p in search_dir.rglob("*.txt") if p.name != "classes.txt")


def find_image_for_stem(stem: str, annotations_dir: Path, frames_dir: Path) -> Path | None:
    images_sub = annotations_dir / "images"
    if images_sub.exists():
        for ext in IMAGE_EXTS:
            candidate = images_sub / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    # fall back to searching data/frames/<slug>/<stem>.<ext>
    m = FRAME_STEM_RE.match(stem)
    search_roots = [frames_dir / m.group("slug")] if m else [frames_dir]
    for root in search_roots:
        if not root.exists():
            continue
        for ext in IMAGE_EXTS:
            candidate = root / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    for ext in IMAGE_EXTS:
        hits = list(frames_dir.rglob(f"{stem}{ext}"))
        if hits:
            return hits[0]
    return None


def clear_split_dirs(dataset_dir: Path, prefix_filter=None) -> None:
    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = dataset_dir / kind / split
            d.mkdir(parents=True, exist_ok=True)
            for f in list(d.iterdir()):
                if prefix_filter is None or f.name.startswith(prefix_filter):
                    f.unlink()


def copy_pair(img_src: Path, lbl_src: Path | None, dataset_dir: Path, split: str, out_stem: str) -> None:
    img_dst = dataset_dir / "images" / split / f"{out_stem}{img_src.suffix.lower()}"
    lbl_dst = dataset_dir / "labels" / split / f"{out_stem}.txt"
    shutil.copy2(img_src, img_dst)
    if lbl_src is not None and lbl_src.exists():
        shutil.copy2(lbl_src, lbl_dst)
    else:
        lbl_dst.write_text("")


def assemble_real(cfg: dict, dataset_dir: Path, append: bool) -> dict:
    p = cfg["paths"]
    val_ratio = cfg["assemble"]["val_ratio"]
    chunk_seconds = cfg["assemble"]["time_chunk_seconds"]
    annotations_dir = p["annotations_real"]
    frames_dir = p["frames"]

    if not annotations_dir.exists() or not any(annotations_dir.rglob("*.txt")):
        print(f"[real] no annotations found under '{annotations_dir}', skipping.")
        return {"train": 0, "val": 0}

    label_files = find_real_label_files(annotations_dir)
    manifest = load_manifest(dataset_dir) if append else {}

    counts = {"train": 0, "val": 0}
    missing_images = 0
    for lbl in label_files:
        stem = lbl.stem
        img = find_image_for_stem(stem, annotations_dir, frames_dir)
        if img is None:
            missing_images += 1
            continue

        chunk_key = parse_chunk_key(stem, chunk_seconds)
        if chunk_key not in manifest:
            manifest[chunk_key] = "val" if stable_fraction(chunk_key) < val_ratio else "train"
        split = manifest[chunk_key]

        copy_pair(img, lbl, dataset_dir, split, stem)
        counts[split] += 1

    save_manifest(dataset_dir, manifest)
    if missing_images:
        print(f"[real] warning: {missing_images} label file(s) had no matching image and were skipped.")
    return counts


def assemble_synthetic(cfg: dict, dataset_dir: Path) -> dict:
    p = cfg["paths"]
    synthetic_dir = p["synthetic"]
    counts = {"train": 0, "val": 0}

    clear_split_dirs(dataset_dir, prefix_filter="synth_")

    for split in ("train", "val"):
        img_dir = synthetic_dir / "images" / split
        lbl_dir = synthetic_dir / "labels" / split
        if not img_dir.exists():
            continue
        for img in sorted(img_dir.iterdir()):
            if img.suffix.lower() not in IMAGE_EXTS:
                continue
            lbl = lbl_dir / f"{img.stem}.txt"
            copy_pair(img, lbl, dataset_dir, split, img.stem)
            counts[split] += 1

    if counts["train"] == 0 and counts["val"] == 0:
        print(f"[synthetic] no data found under '{synthetic_dir}', skipping (run generate_synthetic.py).")
    return counts


def assemble_negatives(cfg: dict, dataset_dir: Path) -> dict:
    p = cfg["paths"]
    val_ratio = cfg["assemble"]["val_ratio"]
    negatives_dir = p["negatives"]
    counts = {"train": 0, "val": 0}

    clear_split_dirs(dataset_dir, prefix_filter="neg_")

    if not negatives_dir.exists() or not any(
        f.suffix.lower() in IMAGE_EXTS for f in negatives_dir.iterdir() if f.is_file()
    ):
        print(f"[negatives] no images found under '{negatives_dir}', skipping.")
        return counts

    for img in sorted(negatives_dir.iterdir()):
        if not img.is_file() or img.suffix.lower() not in IMAGE_EXTS:
            continue
        split = "val" if stable_fraction(f"neg#{img.stem}") < val_ratio else "train"
        out_stem = f"neg_{img.stem}"
        copy_pair(img, None, dataset_dir, split, out_stem)
        counts[split] += 1

    return counts


def main():
    cfg = load_config()
    p = cfg["paths"]
    dataset_dir = p["dataset"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--append", action="store_true",
                     help="Merge new data into the existing dataset instead of rebuilding from scratch")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (dataset_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    if not args.append:
        print("Rebuilding dataset from scratch (pass --append to merge instead of wiping)...")
        clear_split_dirs(dataset_dir)
        manifest_path = dataset_dir / MANIFEST_NAME
        if manifest_path.exists():
            manifest_path.unlink()
    else:
        print("Appending to existing dataset (previously split real chunks keep their side)...")

    real_counts = assemble_real(cfg, dataset_dir, args.append)
    synth_counts = assemble_synthetic(cfg, dataset_dir)
    neg_counts = assemble_negatives(cfg, dataset_dir)

    print("\n--- Summary ---")
    for name, counts in (("real", real_counts), ("synthetic", synth_counts), ("negatives", neg_counts)):
        print(f"{name:10s} train={counts['train']:5d}  val={counts['val']:5d}")
    total_train = sum(c["train"] for c in (real_counts, synth_counts, neg_counts))
    total_val = sum(c["val"] for c in (real_counts, synth_counts, neg_counts))
    print(f"{'total':10s} train={total_train:5d}  val={total_val:5d}")

    if total_train == 0:
        die("Assembled dataset has 0 training images. Add real annotations and/or run generate_synthetic.py first.")


if __name__ == "__main__":
    main()
