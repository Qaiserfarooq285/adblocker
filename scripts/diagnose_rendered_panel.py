#!/usr/bin/env python3
"""Compare the RENDERED panel in an output video against the detection quad
that produced it, on the same frame.

The height diagnostic showed the detector's quad hugging the board, yet the
output shows a thick slab over the goal. Something between detection and
composite is inflating it, and this isolates which: it recovers the painted
region from the output frame by matching the frozen fill colour, then reports
its geometry next to the detection's.

    python scripts/diagnose_rendered_panel.py --frame 631
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import load_config
from fill import canonical_quad, expand_quad, quad_from_mask

CLASS_BOARD, CLASS_OVERLAY = 1, 2


def strip_thickness(q: np.ndarray) -> float:
    """Perpendicular thickness of a (tl,tr,br,bl) strip - the number that says
    whether it hugs a thin LED board. A slanted strip has a large bounding-box
    height while still being thin, so bbox height alone cannot answer this."""
    tl, tr, br, bl = q
    top_mid, bot_mid = (tl + tr) / 2.0, (bl + br) / 2.0
    return float(np.linalg.norm(bot_mid - top_mid))


def describe(name: str, q: np.ndarray, h: int) -> None:
    bh = q[:, 1].max() - q[:, 1].min()
    print(f"  {name:22s} bbox_h={bh:6.1f}px ({bh / h:5.1%})  "
          f"thickness={strip_thickness(q):6.1f}px  "
          f"corners={np.round(q).astype(int).tolist()}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="data/raw_videos/match.mp4")
    ap.add_argument("--rendered", default="outputs/match_blocked.mp4")
    ap.add_argument("--frame", type=int, default=631)
    ap.add_argument("--out", default="outputs/rendered_vs_detected.jpg")
    args = ap.parse_args()

    cfg = load_config()
    icfg = dict(cfg["inference"])

    def grab(path, idx):
        cap = cv2.VideoCapture(path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, f = cap.read()
        cap.release()
        return f if ok else None

    raw = grab(args.video, args.frame)
    rendered = grab(args.rendered, args.frame)
    if raw is None or rendered is None:
        print("could not read frames")
        return
    h, w = raw.shape[:2]
    print(f"frame {args.frame}  {w}x{h}\n")

    # --- what the detector produced ---
    from ultralytics import YOLO
    model = YOLO(str(icfg["weights"]))
    res = model.predict(raw, conf=icfg["conf"], iou=icfg["iou"], imgsz=icfg["imgsz"],
                        device=str(icfg["device"]), classes=[CLASS_BOARD, CLASS_OVERLAY],
                        retina_masks=True, verbose=False)[0]
    det_quads = []
    if res.masks is not None:
        masks = res.masks.data.cpu().numpy()
        if masks.shape[1:] != (h, w):
            masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks])
        for i in range(len(masks)):
            q = quad_from_mask((masks[i] > 0.5).astype(np.uint8))
            if q is not None:
                det_quads.append(q)

    print("DETECTED (this frame, no tracking/EMA):")
    for i, q in enumerate(det_quads):
        describe(f"det[{i}]", q, h)
    if det_quads:
        pts = np.concatenate(det_quads).astype(np.float32)
        merged = canonical_quad(cv2.boxPoints(cv2.minAreaRect(pts)))
        describe("merged+margin", expand_quad(merged, icfg["panel_margin_frac"]), h)

    # --- what actually got painted: the flat region that differs from source ---
    diff = cv2.absdiff(raw, rendered).max(axis=2)
    painted = (diff > 18).astype(np.uint8)
    painted = cv2.morphologyEx(painted, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    painted = cv2.morphologyEx(painted, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    cnts, _ = cv2.findContours(painted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = [c for c in cnts if cv2.contourArea(c) > 2000]
    print(f"\nRENDERED (recovered from output, {len(cnts)} region(s)):")
    rq = []
    for i, c in enumerate(sorted(cnts, key=cv2.contourArea, reverse=True)):
        q = canonical_quad(cv2.boxPoints(cv2.minAreaRect(c)))
        rq.append(q)
        describe(f"painted[{i}]", q, h)
        print(f"  {'':22s} area={cv2.contourArea(c):.0f}px^2")

    if det_quads and rq:
        dt = strip_thickness(expand_quad(canonical_quad(
            cv2.boxPoints(cv2.minAreaRect(np.concatenate(det_quads).astype(np.float32)))),
            icfg["panel_margin_frac"]))
        pt = strip_thickness(rq[0])
        print(f"\n>>> painted strip is {pt / max(dt, 1e-6):.2f}x the thickness of "
              f"the merged detection ({pt:.0f}px vs {dt:.0f}px)")

    vis = raw.copy()
    for q in det_quads:
        cv2.polylines(vis, [np.round(q).astype(np.int32)], True, (0, 0, 255), 3)
    for q in rq:
        cv2.polylines(vis, [np.round(q).astype(np.int32)], True, (0, 255, 255), 3)
    cv2.putText(vis, "red = detection    yellow = actually painted", (20, 44),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 255), 3)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, vis)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
