#!/usr/bin/env python3
"""Did anyone standing in front of a board get painted over?

This answers the referee complaint directly, and it deliberately does NOT trust
the pipeline's own bookkeeping. It re-detects people in the SOURCE frame, finds
which of them overlap the region the output actually painted, and then measures
whether those people's pixels survived into the output.

    preserved   the person's pixels are unchanged  -> they render in front
    painted     the panel was drawn over them      -> the bug

Per person-track it reports the run of frames where they overlap a panel and how
many of those frames painted over them, so a blink shows up as a nonzero count
rather than as an impression.

    python scripts/verify_occluders.py --source clip.mp4 --output out.mp4
"""
from __future__ import annotations

import argparse
import collections

import cv2
import numpy as np

from common import load_config


def painted_mask(src: np.ndarray, out: np.ndarray) -> np.ndarray:
    """Flat-fill pixels the output changed. The uniformity test matters: the
    output is re-encoded, so sharp broadcast graphics ring with codec noise that
    clears a plain difference threshold and is not a panel at all."""
    diff = cv2.absdiff(src, out).max(axis=2)
    m = (diff > 30).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    keep = np.zeros_like(m)
    closed = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((41, 41), np.uint8))
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in cnts:
        if cv2.contourArea(c) < 4000:
            continue
        foot = np.zeros_like(m)
        cv2.fillConvexPoly(foot, cv2.convexHull(c), 1)
        px = out[(m & foot) > 0]
        if len(px) < 500 or float(px.reshape(-1, 3).std(axis=0).mean()) > 18.0:
            continue
        keep |= foot
    return keep


def main():
    cfg = load_config()
    icfg = cfg["inference"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--dark-value", type=float, default=100.0,
                    help="median V below this counts as a dark kit")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(icfg["person_model"])

    s = cv2.VideoCapture(args.source)
    o = cv2.VideoCapture(args.output)
    if not s.isOpened() or not o.isOpened():
        raise SystemExit("could not open source and/or output")

    per_track = collections.defaultdict(lambda: {"frames": [], "painted": [], "dark": 0})
    n = 0
    while True:
        ok1, src = s.read()
        ok2, out = o.read()
        if not (ok1 and ok2):
            break
        n += 1
        if (n - 1) % args.stride:
            continue
        panel = painted_mask(src, out)
        if panel.sum() < 3000:
            continue

        res = model.track(src, persist=True, conf=args.conf, imgsz=icfg["imgsz"],
                          device=str(icfg["device"]), classes=[0],
                          retina_masks=True, verbose=False)[0]
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            continue
        h, w = src.shape[:2]
        masks = res.masks.data.cpu().numpy()
        if masks.shape[1:] != (h, w):
            masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                              for m in masks])
        ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None
               else np.arange(len(masks)))

        for i, tid in enumerate(ids):
            person = (masks[i] > 0.5)
            overlap = int((person & (panel > 0)).sum())
            if overlap < 600:
                continue          # not standing in front of a board
            # Was the person painted over? Compare only the overlapping pixels.
            sel = person & (panel > 0)
            d = cv2.absdiff(src, out).max(axis=2)[sel]
            painted_frac = float((d > 30).mean())
            rec = per_track[int(tid)]
            rec["frames"].append(n)
            rec["painted"].append(painted_frac)
            px = src[person]
            hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
            if float(np.median(hsv[:, 2])) < args.dark_value:
                rec["dark"] += 1

    s.release()
    o.release()

    print(f"frames compared: {n} (stride {args.stride})")
    print(f"people found overlapping a panel: {len(per_track)}\n")
    print(f"{'id':>6} {'frames':>7} {'dark':>5} {'blinks':>7} {'worst':>6}  verdict")
    total_blinks = 0
    for tid, rec in sorted(per_track.items(), key=lambda kv: -len(kv[1]["frames"])):
        f = rec["frames"]
        p = np.array(rec["painted"])
        blinks = int((p > 0.25).sum())      # >25% of their overlap repainted
        total_blinks += blinks
        verdict = "PRESERVED" if blinks == 0 else f"PAINTED OVER on {blinks} frame(s)"
        tag = "yes" if rec["dark"] > len(f) / 2 else "-"
        print(f"{tid:>6} {len(f):>7} {tag:>5} {blinks:>7} {p.max():>6.2f}  {verdict}")
    print(f"\ntotal person-frames painted over: {total_blinks}")


if __name__ == "__main__":
    main()
