#!/usr/bin/env python3
"""Draw polygons on random assembled training images.

The one cheap check that catches a whole class of silent disasters: labels
shifted, normalised the wrong way, or sitting on the wrong object. If this looks
wrong, nothing downstream can be right, and a training run is hours."""
from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from common import load_config

COLORS = {0: (0, 220, 220), 1: (0, 255, 0), 2: (255, 150, 0)}
NAMES = {0: "person", 1: "board", 2: "overlay"}


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--out", default="outputs/sanity_montage.jpg")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = cfg["paths"]["dataset"] / "images" / args.split
    lroot = cfg["paths"]["dataset"] / "labels" / args.split
    imgs = sorted(q for q in root.iterdir() if q.suffix.lower() in (".jpg", ".png"))
    if not imgs:
        raise SystemExit(f"no images in {root}")
    random.Random(args.seed).shuffle(imgs)

    tiles = []
    for p in imgs[:args.n]:
        im = cv2.imread(str(p))
        if im is None:
            continue
        h, w = im.shape[:2]
        lp = lroot / (p.stem + ".txt")
        counts = {}
        if lp.exists():
            for line in lp.read_text().splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                c = int(parts[0])
                v = np.array([float(x) for x in parts[1:]], np.float32)
                if v.size < 6:
                    continue
                pts = (v.reshape(-1, 2) * np.array([w, h], np.float32)).astype(np.int32)
                cv2.polylines(im, [pts], True, COLORS.get(c, (255, 255, 255)), 2)
                counts[c] = counts.get(c, 0) + 1
        tag = " ".join(f"{NAMES.get(k, k)}:{v}" for k, v in sorted(counts.items())) or "EMPTY"
        im = cv2.resize(im, (480, int(480 * h / w)))
        cv2.rectangle(im, (0, 0), (479, 26), (0, 0, 0), -1)
        cv2.putText(im, tag, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        tiles.append(im)

    hh = min(t.shape[0] for t in tiles)
    rows = [np.hstack([t[:hh] for t in tiles[i:i + 4]]) for i in range(0, len(tiles), 4)]
    rows = [r for r in rows if r.shape[1] == rows[0].shape[1]]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), np.vstack(rows))
    print(f"[sanity] wrote {out} ({len(tiles)} tiles from {args.split})")


if __name__ == "__main__":
    main()
