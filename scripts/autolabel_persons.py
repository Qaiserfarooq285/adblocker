#!/usr/bin/env python3
"""Auto-label `person` instances across the assembled dataset using a
pretrained (COCO) yolo11x-seg model, appending them as class 0 polygons to
whatever label files already exist (betting_board / betting_overlay from
annotation or synthetic generation).

Run this AFTER assemble_dataset.py has populated data/dataset/images/{train,val}
and BEFORE (or after, it's idempotent-ish) train.py — person boxes are what
process_video.py later protects from being painted over.

The operation is idempotent: before writing a result it removes existing
class-0 lines from that image's label file. This makes resumable automation
safe to run again after new data arrives.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from tqdm import tqdm

from common import IMAGE_EXTS, die, load_config

CLASS_PERSON = 0  # data/data.yaml: 0 -> person


def find_dataset_images(dataset_dir: Path) -> list[Path]:
    images = []
    for split in ("train", "val"):
        img_dir = dataset_dir / "images" / split
        if img_dir.exists():
            images += sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return images


def label_path_for(image_path: Path) -> Path:
    # .../images/train/foo.jpg -> .../labels/train/foo.txt
    parts = list(image_path.parts)
    idx = parts.index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def without_person_labels(text: str) -> str:
    """Preserve hand/synthetic board labels while replacing auto person labels."""
    return "\n".join(line for line in text.splitlines()
                     if line.strip() and line.split(maxsplit=1)[0] != str(CLASS_PERSON))


def main():
    cfg = load_config()
    p = cfg["paths"]
    acfg = cfg["autolabel"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=acfg["model"])
    ap.add_argument("--conf", type=float, default=acfg["conf"])
    ap.add_argument("--batch", type=int, default=acfg["batch"])
    ap.add_argument("--device", type=str, default=str(acfg["device"]))
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        die("ultralytics is not installed. Run: pip install -r requirements.txt")

    dataset_dir = p["dataset"]
    images = find_dataset_images(dataset_dir)
    if not images:
        die(
            f"No images found under '{dataset_dir}/images/{{train,val}}'. "
            "Run scripts/assemble_dataset.py first."
        )

    print(f"Loading {args.model} for person auto-labeling ({len(images)} images)...")
    model = YOLO(args.model)

    labeled = 0
    for i in tqdm(range(0, len(images), args.batch), desc="Auto-labeling persons"):
        batch = images[i:i + args.batch]
        results = model.predict(
            source=[str(x) for x in batch],
            conf=args.conf,
            classes=[0],  # COCO class 0 = person
            device=args.device,
            verbose=False,
            retina_masks=True,
        )
        for img_path, result in zip(batch, results):
            if result.masks is None or len(result.masks) == 0:
                continue
            h, w = result.orig_shape
            lines = []
            for polygon in result.masks.xy:
                if len(polygon) < 3:
                    continue
                norm = []
                for (x, y) in polygon:
                    norm.extend([f"{max(0, min(1, x / w)):.6f}", f"{max(0, min(1, y / h)):.6f}"])
                lines.append(" ".join([str(CLASS_PERSON)] + norm))

            if not lines:
                continue

            lbl_path = label_path_for(img_path)
            lbl_path.parent.mkdir(parents=True, exist_ok=True)
            existing = without_person_labels(lbl_path.read_text()) if lbl_path.exists() else ""
            if existing and not existing.endswith("\n"):
                existing += "\n"
            lbl_path.write_text(existing + "\n".join(lines) + "\n")
            labeled += 1

    print(f"Done. Added person labels to {labeled}/{len(images)} images.")


if __name__ == "__main__":
    main()
