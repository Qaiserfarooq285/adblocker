#!/usr/bin/env python3
"""Measure how reliably the person model finds figures at the board, and what
the ROI second pass is actually doing to resolution.

The referee blink is a detection-stability problem, so the useful number is
not "does it detect him" on one frame but "on what fraction of consecutive
frames", at each candidate setting. This runs the full-frame pass and the ROI
pass over a frame range and reports per-frame person coverage inside the board
strip, plus the effective scale factor each pass applies to the crop.

    python scripts/diagnose_person_roi.py --input data/raw_videos/match.mp4 --start-frame 600 --n 60
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np

from common import load_config
from fill import expand_quad, quad_from_mask

CLASS_BOARD, CLASS_OVERLAY = 1, 2


def board_quad(model, raw, icfg):
    h, w = raw.shape[:2]
    res = model.predict(raw, conf=icfg["conf"], iou=icfg["iou"], imgsz=icfg["imgsz"],
                        device=str(icfg["device"]), classes=[CLASS_BOARD, CLASS_OVERLAY],
                        retina_masks=True, verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        return None
    m = res.masks.data.cpu().numpy()
    if m.shape[1:] != (h, w):
        m = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in m])
    quads = [q for q in (quad_from_mask((x > 0.5).astype(np.uint8)) for x in m) if q is not None]
    if not quads:
        return None
    pts = np.concatenate(quads).astype(np.float32)
    from fill import canonical_quad
    return expand_quad(canonical_quad(cv2.boxPoints(cv2.minAreaRect(pts))), icfg["panel_margin_frac"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw_videos/match.mp4")
    ap.add_argument("--start-frame", type=int, default=600)
    ap.add_argument("--n", type=int, default=60)
    args = ap.parse_args()

    cfg = load_config()
    icfg = dict(cfg["inference"])
    from ultralytics import YOLO
    board_model = YOLO(str(icfg["weights"]))
    person_model = YOLO(icfg["person_model"])

    cap = cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    settings = [
        ("full conf=0.15 imgsz=1280", dict(mode="full", conf=0.15, imgsz=1280)),
        ("full conf=0.10 imgsz=1280", dict(mode="full", conf=0.10, imgsz=1280)),
        ("roi  conf=0.15 imgsz=640 (current)", dict(mode="roi", conf=0.15, imgsz=640)),
        ("roi  conf=0.10 upscale=2.0", dict(mode="roi", conf=0.10, upscale=2.0)),
    ]
    hits = {name: 0 for name, _ in settings}
    areas = {name: [] for name, _ in settings}
    scale_note = {}
    n_eval = 0

    for i in range(args.n):
        ok, raw = cap.read()
        if not ok:
            break
        h, w = raw.shape[:2]
        q = board_quad(board_model, raw, icfg)
        if q is None:
            continue
        n_eval += 1
        # the zone we care about: the strip plus a band below it (pitch side)
        x0, y0 = np.floor(q.min(axis=0)).astype(int)
        x1, y1 = np.ceil(q.max(axis=0)).astype(int)
        strip_h = max(1, y1 - y0)
        zx0, zy0 = max(0, x0), max(0, y0 - 8)
        zx1, zy1 = min(w, x1), min(h, y1 + int(1.2 * strip_h))
        zone = np.zeros((h, w), np.uint8)
        zone[zy0:zy1, zx0:zx1] = 1

        for name, s in settings:
            found = np.zeros((h, w), np.uint8)
            if s["mode"] == "full":
                r = person_model.predict(raw, conf=s["conf"], imgsz=s["imgsz"],
                                         device=str(icfg["device"]), classes=[0],
                                         retina_masks=True, verbose=False)[0]
                if r.masks is not None and len(r.masks) > 0:
                    m = r.masks.data.cpu().numpy()
                    if m.shape[1:] != (h, w):
                        m = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in m])
                    found = (m > 0.5).any(axis=0).astype(np.uint8)
                scale_note[name] = f"{s['imgsz'] / max(w, h):.2f}x native"
            else:
                pad = int(0.6 * strip_h)
                cx0, cy0 = max(0, x0 - pad), max(0, y0 - pad)
                cx1, cy1 = min(w, x1 + pad), min(h, y1 + pad)
                crop = raw[cy0:cy1, cx0:cx1]
                ch, cw = crop.shape[:2]
                if "upscale" in s:
                    inp = cv2.resize(crop, None, fx=s["upscale"], fy=s["upscale"],
                                     interpolation=cv2.INTER_CUBIC)
                    imgsz = int(np.ceil(max(inp.shape[:2]) / 32) * 32)
                    scale_note[name] = f"{imgsz / max(ch, cw):.2f}x native crop"
                else:
                    inp, imgsz = crop, s["imgsz"]
                    scale_note[name] = f"{imgsz / max(ch, cw):.2f}x native crop"
                r = person_model.predict(inp, conf=s["conf"], imgsz=imgsz,
                                         device=str(icfg["device"]), classes=[0],
                                         retina_masks=True, verbose=False)[0]
                if r.masks is not None and len(r.masks) > 0:
                    m = r.masks.data.cpu().numpy()
                    mm = (m > 0.5).any(axis=0).astype(np.uint8)
                    mm = cv2.resize(mm, (cw, ch), interpolation=cv2.INTER_NEAREST)
                    found[cy0:cy1, cx0:cx1] = mm
            a = int((found & zone).sum())
            areas[name].append(a)
            if a > 500:
                hits[name] += 1

    cap.release()
    print(f"\nframes with a board detection: {n_eval}\n")
    print(f"{'setting':38s} {'frames w/ person in zone':>26s} {'median px':>10s}  effective scale")
    for name, _ in settings:
        med = int(np.median(areas[name])) if areas[name] else 0
        print(f"{name:38s} {hits[name]:>10d} / {n_eval:<13d} {med:>10d}  {scale_note.get(name, '')}")


if __name__ == "__main__":
    main()
