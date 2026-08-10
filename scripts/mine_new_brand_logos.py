#!/usr/bin/env python3
"""Mine labelled logo instances for brands the supplied dataset does not cover.

The hand-labelled Roboflow export is soccer footage carrying Betway, BK8,
ROLLBIT, DEBET and Betano. The videos to be processed carry Stake, Polymarket,
bet365, FanDuel, theScore Bet and Betty.ca across five North American sports. A
detector trained only on the first set has never seen the second, so those
brands go unhidden.

Rather than hand-draw them, this reads the frames: full-frame OCR returns text
AND the box that text sits in, so a token that matches a brand alias IS a
labelled logo instance at no annotation cost. OCR is used as the seed precisely
because it is high-precision - a region whose pixels spell "polymarket" is a
Polymarket logo, whereas a template or colour match is a guess. Pseudo-labelling
compounds its own errors, so the seed has to be the trustworthy cue.

Output is YOLO detection format, single class 0 = logo, ready to merge into the
existing dataset's train split.

    python scripts/mine_new_brand_logos.py --frames data/frames_by_sport
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from filter_betting_boards import match_brand, ocr_frame

# Brands the supplied dataset already covers well; mining them again adds
# little and risks drowning the new ones out.
ALREADY_COVERED = {"betano", "betway", "bk8", "rollbit", "debet"}


def mine_one(job):
    path, upscale, pad_frac, min_px = job
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    boxes, brands = [], []
    for text, (x0, y0, x1, y1) in ocr_frame(img, upscale=upscale):
        brand, _ = match_brand([text])
        if brand is None or brand == "generic":
            continue                       # need a NAMED brand to be a logo box
        if brand in ALREADY_COVERED:
            continue
        bw, bh = x1 - x0, y1 - y0
        if bw < min_px or bh < min_px * 0.4:
            continue
        # OCR boxes hug the glyphs; a logo box normally includes the mark and a
        # little padding, which is what the reference dataset's boxes look like.
        px, py = pad_frac * bw, pad_frac * bh
        X0 = max(0.0, x0 - px)
        Y0 = max(0.0, y0 - py)
        X1 = min(float(w), x1 + px)
        Y1 = min(float(h), y1 + py)
        boxes.append(((X0 + X1) / 2 / w, (Y0 + Y1) / 2 / h,
                      (X1 - X0) / w, (Y1 - Y0) / h))
        brands.append(brand)
    return {"path": str(path), "boxes": boxes, "brands": brands, "wh": (w, h)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default="data/frames_by_sport",
                    help="directory tree of frames to mine")
    ap.add_argument("--out", default="data/mined_logos")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--upscale", type=float, default=2.0,
                    help="OCR upscale; board text is small at native resolution")
    ap.add_argument("--pad-frac", type=float, default=0.18)
    ap.add_argument("--min-px", type=float, default=14.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--segments", default="data/segments.json")
    args = ap.parse_args()

    frames_root = Path(args.frames)
    paths = sorted(q for q in frames_root.rglob("*")
                   if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if args.limit:
        paths = paths[:args.limit]
    if not paths:
        raise SystemExit(f"no frames under {frames_root}")

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)

    segs = []
    sp = Path(args.segments)
    if sp.exists():
        segs = json.loads(sp.read_text())["segments"]

    def sport_of(stem: str) -> str:
        m = re.search(r"_t(\d+(?:\.\d+)?)", stem)
        if not m:
            return "?"
        t = float(m.group(1))
        for s in segs:
            if s["start"] <= t <= s["end"]:
                return s.get("name", s.get("id", "?"))
        return "?"

    print(f"[mine] OCR over {len(paths)} frames at {args.upscale}x "
          f"({args.workers} workers)...")
    jobs = [(p, args.upscale, args.pad_frac, args.min_px) for p in paths]

    brand_counter, sport_counter = Counter(), Counter()
    n_frames = n_boxes = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(mine_one, jobs, chunksize=4), 1):
            if not r or not r["boxes"]:
                continue
            stem = Path(r["path"]).stem
            sport = sport_of(stem)
            img = cv2.imread(r["path"])
            if img is None:
                continue
            name = f"mined_{sport}_{stem}"
            cv2.imwrite(str(out / "images" / f"{name}.jpg"), img)
            (out / "labels" / f"{name}.txt").write_text(
                "\n".join(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"
                          for cx, cy, bw, bh in r["boxes"]) + "\n")
            n_frames += 1
            n_boxes += len(r["boxes"])
            brand_counter.update(r["brands"])
            sport_counter[sport] += len(r["boxes"])
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)} frames scanned, {n_boxes} logo boxes mined")

    print(f"\n[mine] {n_frames} frames with >=1 new-brand logo, {n_boxes} boxes")
    print(f"\n{'brand':18s} {'boxes':>7s}")
    for b, n in brand_counter.most_common():
        print(f"{b:18s} {n:>7d}")
    print(f"\n{'sport':20s} {'boxes':>7s}")
    for s, n in sport_counter.most_common():
        print(f"{s:20s} {n:>7d}")
    (out / "mine_summary.json").write_text(json.dumps(
        {"frames": n_frames, "boxes": n_boxes,
         "brands": dict(brand_counter), "per_sport": dict(sport_counter)},
        indent=2) + "\n")
    print(f"\n[mine] wrote {out}")


if __name__ == "__main__":
    main()
