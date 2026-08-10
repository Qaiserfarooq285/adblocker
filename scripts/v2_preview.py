#!/usr/bin/env python3
"""Build a side-by-side original|cleaned preview for one clip.

Judging "does it blink / does it drift / does it cover players" from a single
still is impossible, so this writes a synchronised comparison video plus a
contact sheet of the frames where the renderer changed the most pixels.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=0.0, help="0 = whole clip")
    args = ap.parse_args()

    vids = sorted((ROOT / "data" / "raw_videos").glob("*.mp4"))
    src = vids[args.clip - 1]
    clean = ROOT / "outputs" / "clips" / f"clip{args.clip:02d}_clean.mp4"
    if not clean.exists():
        raise SystemExit(f"missing {clean}")

    out_v = ROOT / "outputs" / f"preview_clip{args.clip:02d}.mp4"
    out_v.parent.mkdir(parents=True, exist_ok=True)

    # Stack originals on top of cleaned, labelled, at half width so the pair
    # fits on screen without the viewer hunting for the difference.
    vf = ("[0:v]scale=960:-2,drawtext=text='ORIGINAL':x=20:y=20:fontsize=32:"
          "fontcolor=white:box=1:boxcolor=black@0.6[a];"
          "[1:v]scale=960:-2,drawtext=text='BETTING ADS REMOVED':x=20:y=20:"
          "fontsize=32:fontcolor=white:box=1:boxcolor=black@0.6[b];"
          "[a][b]vstack=inputs=2")
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-i", str(clean),
           "-filter_complex", vf, "-c:v", "libx264", "-crf", "20",
           "-preset", "medium", "-pix_fmt", "yuv420p"]
    if args.seconds > 0:
        cmd += ["-t", str(args.seconds)]
    cmd += [str(out_v)]
    subprocess.run(cmd, check=False)

    # Contact sheet: the frames the renderer touched hardest.
    cap_a = cv2.VideoCapture(str(src))
    cap_b = cv2.VideoCapture(str(clean))
    scored = []
    idx = 0
    while True:
        ok_a, fa = cap_a.read()
        ok_b, fb = cap_b.read()
        if not (ok_a and ok_b):
            break
        if idx % 15 == 0:
            d = cv2.absdiff(fa, fb).max(axis=2)
            m = cv2.morphologyEx((d > 25).astype(np.uint8), cv2.MORPH_OPEN,
                                 np.ones((5, 5), np.uint8))
            scored.append((int(m.sum()), idx, fa.copy(), fb.copy()))
        idx += 1
    cap_a.release()
    cap_b.release()

    scored.sort(key=lambda t: -t[0])
    tiles = []
    for changed, i, fa, fb in scored[:4]:
        ys, xs = np.where(cv2.absdiff(fa, fb).max(axis=2) > 25)
        if len(ys) == 0:
            continue
        y0 = max(0, int(np.percentile(ys, 1)) - 60)
        y1 = min(fa.shape[0], int(np.percentile(ys, 99)) + 60)
        pair = np.vstack([fa[y0:y1], fb[y0:y1]])
        s = 1100.0 / pair.shape[1]
        tiles.append(cv2.resize(pair, (1100, max(20, int(pair.shape[0] * s)))))
        print(f"  frame {i}: {changed} px changed")
    if tiles:
        sheet = ROOT / "outputs" / f"preview_clip{args.clip:02d}.jpg"
        cv2.imwrite(str(sheet), np.vstack(tiles))
        print(f"[preview] {sheet}")
    print(f"[preview] {out_v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
