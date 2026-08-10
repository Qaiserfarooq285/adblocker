#!/usr/bin/env python3
"""Measure the finished video against the things that are supposed to be true.

Everything here is recovered from the OUTPUT video by differencing it against
its source, so it checks what was actually rendered rather than what the
pipeline believed it rendered.

Reported per run:

  panel thickness      perpendicular, so a slanted board is judged fairly
  holes in the panel   unpainted components inside the panel's own footprint,
                       split by whether they sit at the TOP edge (the crowd
                       -notch bug) or reach the bottom (a real occluder)
  hole stability       frame-to-frame change in occluder area, which is what
                       a blinking referee looks like numerically

    python scripts/verify_output.py --source data/raw_videos/match.mp4 \
                                    --output outputs/match_blocked.mp4
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def panel_regions(src: np.ndarray, out: np.ndarray, min_area: int = 4000):
    """(footprint, painted) masks for each rendered panel in this frame.

    The output video is re-encoded, so sharp broadcast graphics - the scorebug,
    the lower third - ring with codec noise that clears a naive difference
    threshold and reads as a 'painted region' sitting nowhere near a board. A
    real panel is a FLAT frozen colour, so it is separated from that noise by
    requiring the changed pixels to be nearly uniform in the output."""
    diff = cv2.absdiff(src, out).max(axis=2)
    painted = (diff > 30).astype(np.uint8)
    painted = cv2.morphologyEx(painted, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    closed = cv2.morphologyEx(painted, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out_regions = []
    for c in cnts:
        if cv2.contourArea(c) < min_area:
            continue
        foot = np.zeros(painted.shape, np.uint8)
        cv2.fillConvexPoly(foot, cv2.convexHull(c), 1)
        inside = painted & foot
        px = out[inside > 0]
        if len(px) < 500:
            continue
        if float(px.reshape(-1, 3).std(axis=0).mean()) > 18.0:
            continue          # not a flat fill: codec ringing, or a fade frame
        out_regions.append((foot, inside))
    return out_regions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=100000)
    args = ap.parse_args()

    s = cv2.VideoCapture(args.source)
    o = cv2.VideoCapture(args.output)
    if not s.isOpened() or not o.isOpened():
        raise SystemExit("could not open source and/or output")

    n = 0
    frames_with_panel = 0
    top_holes = 0            # crowd notches: hole hanging off the panel's TOP edge
    bottom_holes = 0         # genuine occluders reaching the bottom
    occluder_series: list[float] = []
    thicknesses: list[float] = []

    while n < args.max_frames:
        ok1, src = s.read()
        ok2, out = o.read()
        if not (ok1 and ok2):
            break
        n += 1
        if (n - 1) % args.stride:
            continue

        frame_occ = 0.0
        for foot, painted in panel_regions(src, out):
            frames_with_panel += 1
            ys, xs = np.where(foot > 0)
            # thickness of a slanted strip = area / horizontal span
            span = max(1, xs.max() - xs.min())
            thicknesses.append(float(foot.sum()) / span)

            holes = (foot & (1 - painted)).astype(np.uint8)
            holes = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
            nh, lab, st, _ = cv2.connectedComponentsWithStats(holes, 8)
            for i in range(1, nh):
                if st[i, cv2.CC_STAT_AREA] < 400:
                    continue
                hx = st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] // 2
                col = np.where(foot[:, hx] > 0)[0]
                if len(col) == 0:
                    continue
                ftop, fbot = col.min(), col.max()
                htop = st[i, cv2.CC_STAT_TOP]
                hbot = htop + st[i, cv2.CC_STAT_HEIGHT]
                band = max(1, fbot - ftop)
                if hbot < fbot - 0.25 * band:      # never reaches the pitch side
                    top_holes += 1
                else:
                    bottom_holes += 1
                    frame_occ += st[i, cv2.CC_STAT_AREA]
        occluder_series.append(frame_occ)

    s.release()
    o.release()

    print(f"frames compared            : {n}")
    print(f"frames with a rendered panel: {frames_with_panel}")
    if thicknesses:
        t = np.array(thicknesses)
        print(f"panel thickness (px)       : median={np.median(t):.0f} "
              f"p95={np.percentile(t, 95):.0f} max={t.max():.0f}")
    print(f"TOP-edge holes (crowd notches, should be ~0): {top_holes}")
    print(f"bottom-reaching holes (real occluders)      : {bottom_holes}")

    occ = np.array(occluder_series)
    live = occ > 0
    if live.any():
        # a blink is an occluder present, gone, then back - count sign flips
        flips = int((np.diff(live.astype(int)) != 0).sum())
        runs = []
        cur = 0
        for v in live:
            if v:
                cur += 1
            elif cur:
                runs.append(cur)
                cur = 0
        if cur:
            runs.append(cur)
        print(f"frames with an occluder    : {int(live.sum())} / {len(occ)}")
        print(f"appear/disappear transitions: {flips}  "
              f"(longest continuous occluder run: {max(runs) if runs else 0} frames)")


if __name__ == "__main__":
    main()
