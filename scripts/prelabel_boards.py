#!/usr/bin/env python3
"""Model-assisted pre-labeling: run a model over real video frames and write
DRAFT betting_board / betting_overlay polygon labels for human review in CVAT
or Roboflow, instead of drawing every polygon from scratch.

This is a bootstrapping tool, not a replacement for human review. Draft
labels can be wrong, missing, or imprecise - always correct them before
trusting them as ground truth. The intended loop is:

  1. Get draft labels for real frames (see the two modes below).
  2. Import the output folder into CVAT/Roboflow, fix the draft polygons
     instead of drawing from nothing, export.
  3. Copy/merge the corrected export into data/annotations_real/.
  4. assemble_dataset.py --append, then train.py --weights models/best.pt
     to fine-tune on the newly-real data.

Two modes, because there is a chicken-and-egg problem at the start:

  --zero-shot  Prompt an open-vocabulary segmenter (YOLOE) with plain English
               ("stadium advertising", "led perimeter board"). Needs no
               trained checkpoint at all. Use this for round 1. A model
               trained only on synthetic pasted logos scores real broadcast
               boards at ~0.07 confidence, so it cannot bootstrap itself -
               it produces zero draft labels and the loop never starts.

  (default)    Use our own trained checkpoint. Use this from round 2 on, once
               models/best.pt has seen real hand-corrected data. Each round
               makes the next round's drafts better.

Usage:
    python scripts/prelabel_boards.py --zero-shot --video match   # round 1
    python scripts/prelabel_boards.py --video match               # round 2+
    python scripts/prelabel_boards.py --video match --conf 0.1
    python scripts/prelabel_boards.py               # all frames under data/frames/
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm

from common import IMAGE_EXTS, die, load_config

CLASS_NAMES = {0: "person", 1: "betting_board", 2: "betting_overlay"}
CLASS_IDS = {name: idx for idx, name in CLASS_NAMES.items()}
PRELABEL_CLASSES = (1, 2)  # never pre-label person; autolabel_persons.py handles that post-assembly


def build_zeroshot_model(zcfg: dict):
    """Load YOLOE and prime it with the text prompts from config.yaml.

    Returns (model, prompt_class_ids) where prompt_class_ids[i] is the
    betting_* class id that YOLOE's i-th prompt maps back onto - YOLOE
    numbers its classes by prompt order, and those indices mean nothing to
    our dataset until they're translated here."""
    try:
        from ultralytics import YOLOE
    except ImportError:
        die("This ultralytics build has no YOLOE (open-vocabulary) support. Upgrade: pip install -U ultralytics")

    prompts, prompt_class_ids = [], []
    for class_name, class_prompts in zcfg["prompts"].items():
        if class_name not in CLASS_IDS:
            die(f"Unknown class '{class_name}' in prelabel.zeroshot.prompts - expected one of {sorted(CLASS_IDS)}.")
        for prompt in class_prompts:
            prompts.append(prompt)
            prompt_class_ids.append(CLASS_IDS[class_name])
    if not prompts:
        die("prelabel.zeroshot.prompts is empty - nothing to prompt YOLOE with.")

    model = YOLOE(zcfg["model"])
    model.set_classes(prompts, model.get_text_pe(prompts))
    print(f"Zero-shot prompts: {', '.join(prompts)}")
    return model, prompt_class_ids


def already_annotated_stems(annotations_dir: Path) -> set[str]:
    """Frame stems that already have a real (trusted) label on disk, so we
    don't waste review time on frames that are already done."""
    stems = set()
    for sub in (annotations_dir, annotations_dir / "labels"):
        if sub.exists():
            stems |= {f.stem for f in sub.glob("*.txt") if f.name != "classes.txt"}
    return stems


def main():
    cfg = load_config()
    p = cfg["paths"]
    pcfg = cfg["prelabel"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=str, default=None,
                     help="Limit to frames under data/frames/<video_slug>/ (default: every video)")
    ap.add_argument("--frames-dir", type=str, default=None,
                     help="Explicit frame directory (overrides --video). Use for a "
                          "balanced per-sport subset of one multi-sport recording.")
    ap.add_argument("--label", type=str, default=None,
                     help="Name for the review output folder (default: the dir name)")
    ap.add_argument("--zero-shot", action="store_true",
                     help="Prompt YOLOE by text instead of using a trained checkpoint (use this for round 1)")
    ap.add_argument("--model", type=str, default=None,
                     help="Weights to use (default: models/best.pt, or prelabel.zeroshot.model under --zero-shot)")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--imgsz", type=int, default=None,
                     help="Inference size (default: 640, or prelabel.zeroshot.imgsz under --zero-shot)")
    ap.add_argument("--device", type=str, default=str(pcfg["device"]))
    ap.add_argument("--include-empty", action="store_true",
                     help="Also copy frames with zero candidate detections, for manually catching what the model missed")
    ap.add_argument("--no-skip-annotated", dest="skip_annotated", action="store_false", default=True,
                     help="Don't skip frames that already have a label in data/annotations_real/")
    ap.add_argument("--out", type=str, default=None,
                     help="Output dir (default: data/prelabel_review/<video or 'all'>)")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    zcfg = pcfg.get("zeroshot", {}) if args.zero_shot else {}
    conf = args.conf if args.conf is not None else (zcfg.get("conf", pcfg["conf"]) if args.zero_shot else pcfg["conf"])
    imgsz = args.imgsz if args.imgsz is not None else (zcfg.get("imgsz", 640) if args.zero_shot else 640)

    if args.zero_shot:
        # YOLOE weights are pretrained and downloaded on demand, not a path we own.
        if args.model:
            zcfg = {**zcfg, "model": args.model}
        model_desc = zcfg.get("model", "yoloe-11l-seg.pt")
    else:
        model_path = Path(args.model) if args.model else (p["models"] / "best.pt")
        if not model_path.exists():
            die(
                f"Model weights not found: {model_path}. Pre-labeling needs a model that already knows "
                "betting_board/betting_overlay - train at least once first (even on synthetic-only data "
                "via scripts/train.py), or use --zero-shot to prompt an open-vocabulary model instead "
                "and skip the checkpoint entirely."
            )
        model_desc = str(model_path)

    frames_root = p["frames"]
    if args.frames_dir:
        # An explicit directory, used for a balanced per-sport subset: one long
        # recording can contain several sports under a single video slug, and
        # prelabelling all of it would let the longest sport dominate.
        frame_dir = Path(args.frames_dir)
        if not frame_dir.exists():
            die(f"'{frame_dir}' not found.")
        frame_paths = sorted(f for f in frame_dir.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        label = args.label or frame_dir.name
    elif args.video:
        frame_dir = frames_root / args.video
        if not frame_dir.exists():
            die(f"'{frame_dir}' not found. Run extract_frames.py first, or check the video slug.")
        frame_paths = sorted(f for f in frame_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        label = args.video
    else:
        frame_paths = sorted(f for f in frames_root.rglob("*") if f.suffix.lower() in IMAGE_EXTS)
        label = "all"
    if not frame_paths:
        die(f"No frames found under '{frames_root}'. Run extract_frames.py first.")

    if args.skip_annotated:
        skip_stems = already_annotated_stems(p["annotations_real"])
        before = len(frame_paths)
        frame_paths = [f for f in frame_paths if f.stem not in skip_stems]
        skipped = before - len(frame_paths)
        if skipped:
            print(f"Skipping {skipped} frame(s) that already have a real annotation.")
    if not frame_paths:
        die("Every candidate frame already has a real annotation on disk - nothing left to pre-label.")

    out_dir = Path(args.out) if args.out else (p["prelabel_review"] / label)
    images_out = out_dir / "images"
    labels_out = out_dir / "labels"
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    (out_dir / "classes.txt").write_text("\n".join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES)) + "\n")

    if args.zero_shot:
        model, prompt_class_ids = build_zeroshot_model(zcfg)
        predict_kwargs = {}  # YOLOE's class ids are prompt indices; filtering by ours is meaningless here
        min_area = zcfg.get("min_area_frac", 0.0)
        max_area = zcfg.get("max_area_frac", 1.0)
    else:
        try:
            from ultralytics import YOLO
        except ImportError:
            die("ultralytics is not installed. Run: pip install -r requirements.txt")
        model = YOLO(model_desc)
        prompt_class_ids = None
        predict_kwargs = {"classes": list(PRELABEL_CLASSES)}
        min_area, max_area = 0.0, 1.0

    mode = "zero-shot" if args.zero_shot else "trained-model"
    print(f"Pre-labeling {len(frame_paths)} frame(s) with {model_desc} "
          f"({mode}, conf={conf}, imgsz={imgsz})...")

    # retina_masks upsamples every mask to full source resolution - a real cost
    # on 4K frames, and pointless here because we only ever read masks.xyn
    # (normalized polygons). Keep it off for the high-imgsz zero-shot pass.
    #
    # One predict() call per image, not a single call over the whole file
    # list with stream=True: with YOLOE + a long source list, some internal
    # buffer scales with the list length rather than staying flat per-frame
    # as stream=True is supposed to guarantee, and blows a ~40GB allocation
    # on a 12GB card well before frame 1 even finishes. Per-image calls have
    # a small, bounded footprint (verified on this same model/imgsz over all
    # 2289 frames of this video with no incident).
    written = 0
    empty = 0
    dropped = 0
    try:
        for frame_path in tqdm(frame_paths, desc="Pre-labeling"):
            result = model.predict(
                source=str(frame_path), conf=conf, device=args.device,
                imgsz=imgsz, retina_masks=not args.zero_shot, verbose=False,
                **predict_kwargs,
            )[0]
            lines = []
            if result.masks is not None:
                cls = result.boxes.cls.cpu().numpy().astype(int)
                for i, polygon in enumerate(result.masks.xyn):
                    if len(polygon) < 3:
                        continue
                    class_id = prompt_class_ids[cls[i]] if prompt_class_ids is not None else int(cls[i])

                    xs, ys = polygon[:, 0], polygon[:, 1]
                    area_frac = float((xs.max() - xs.min()) * (ys.max() - ys.min()))
                    if not (min_area <= area_frac <= max_area):
                        dropped += 1
                        continue

                    norm = []
                    for (x, y) in polygon:
                        norm.extend([f"{max(0.0, min(1.0, float(x))):.6f}", f"{max(0.0, min(1.0, float(y))):.6f}"])
                    lines.append(" ".join([str(class_id)] + norm))

            if not lines:
                empty += 1
                if not args.include_empty:
                    continue

            shutil.copy2(frame_path, images_out / frame_path.name)
            (labels_out / f"{frame_path.stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
            written += 1
    except Exception as exc:
        if type(exc).__name__ != "OutOfMemoryError":
            raise
        die(
            f"CUDA ran out of memory at imgsz={imgsz}. Zero-shot mode is resolution-hungry by design "
            "(thin perimeter boards vanish at low res). Either wait for other GPU jobs to finish - "
            "check `nvidia-smi` - or retry with a smaller size, e.g. --imgsz 1600, accepting lower "
            "recall on distant boards."
        )

    print(f"\nDone. Wrote {written} frame(s) with draft labels to {out_dir}")
    if dropped:
        print(f"({dropped} detection(s) dropped by the min/max area filters)")
    if args.zero_shot:
        print(
            "Zero-shot drafts flag ALL pitchside advertising, not just betting ads - the reviewer's job "
            "is to delete the non-betting ones and tighten the polygons."
        )
    if args.include_empty:
        print(f"({empty} frame(s) had no candidate detections and were included empty for manual review)")
    else:
        print(f"({empty} frame(s) had no candidate detections and were skipped - use --include-empty to keep them)")
    print(
        f"\nNext: review/correct these in CVAT or Roboflow (import as a YOLO segmentation set using "
        f"'{out_dir / 'classes.txt'}' for class names), then copy/merge the corrected export into "
        f"'{p['annotations_real']}' before running assemble_dataset.py."
    )


if __name__ == "__main__":
    main()
