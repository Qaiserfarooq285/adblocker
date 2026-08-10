#!/usr/bin/env python3
"""Honest generalization check: run the model on a video that was NEVER
used in training (not in data/raw_videos/, or a segment you deliberately
excluded) and dump annotated preview frames to outputs/holdout_preview/.

This is the check that actually answers "does this generalize to a brand or
broadcast I haven't trained on", as opposed to val metrics from a dataset
built from the same videos the model trained on.

Usage:
    python scripts/eval_holdout.py --video /path/to/never_trained_on.mp4
    python scripts/eval_holdout.py --video clip.mp4 --fps 0.5 --max-frames 40
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from common import die, load_config, slugify


def main():
    cfg = load_config()
    p = cfg["paths"]
    icfg = cfg["inference"]
    ex = cfg["extract"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=str, required=True, help="Path to a held-out video not used in training")
    ap.add_argument("--model", type=str, default=str(icfg["weights"]))
    ap.add_argument("--fps", type=float, default=ex["fps"], help="Sampling rate for preview frames")
    ap.add_argument("--max-frames", type=int, default=60, help="Cap on number of preview frames written")
    ap.add_argument("--conf", type=float, default=icfg["conf"])
    ap.add_argument("--device", type=str, default=str(icfg["device"]))
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        die(f"Video not found: {video_path}")

    model_path = Path(args.model)
    if not model_path.exists():
        die(f"Model weights not found: {model_path}. Train first with scripts/train.py.")

    try:
        from ultralytics import YOLO
    except ImportError:
        die("ultralytics is not installed. Run: pip install -r requirements.txt")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        die(f"Could not open video: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = max(1, round(src_fps / args.fps))

    slug = slugify(video_path.stem)
    out_dir = p["holdout_preview"] / slug
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    idx = 0
    written = 0
    print(f"Sampling '{video_path.name}' every {frame_interval} frames (~{args.fps} fps)...")
    while written < args.max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % frame_interval == 0:
            results = model.predict(source=frame, conf=args.conf, device=args.device,
                                     retina_masks=True, verbose=False)
            annotated = results[0].plot()
            t = idx / src_fps
            out_path = out_dir / f"{slug}_t{t:.2f}.jpg"
            cv2.imwrite(str(out_path), annotated)
            written += 1
        idx += 1
    cap.release()

    print(f"Done. Wrote {written} annotated preview frame(s) to {out_dir}")


if __name__ == "__main__":
    main()
