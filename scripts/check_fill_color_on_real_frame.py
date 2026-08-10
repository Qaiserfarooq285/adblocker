#!/usr/bin/env python3
"""Measure the panel fill colour on REAL footage and render a comparison
swatch: the old whole-region per-channel median vs the new board-sourced,
outlier-rejected, muted colour.

Synthetic unit tests can't settle this - a median is robust to tidy synthetic
arrangements, and the actual complaint ("bright orange-brown block") comes
from real board pixels: white slogan text, specular LED glare, and dugout
showing through the detection's margin. So this runs the real detector on a
real frame and prints/writes what each strategy would paint.

Usage:
    .venv/bin/python scripts/check_fill_color_on_real_frame.py \
        --frame data/frames/spain_vs_argentina_19_07_2026_4k/..._t1970.00.jpg
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import die, load_config
from fill import (board_fill_color, canonical_quad, expand_quad, quad_from_mask,
                  sample_band_quad, scene_average_color)


def main():
    cfg = load_config()
    icfg = cfg["inference"]
    mcfg = dict(icfg.get("mute", {}))

    ap = argparse.ArgumentParser()
    ap.add_argument("--frame", type=str, required=True)
    ap.add_argument("--out", type=str, default="outputs/fill_color_check.jpg")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    frame = cv2.imread(args.frame)
    if frame is None:
        die(f"Could not read frame: {args.frame}")

    from ultralytics import YOLO
    model = YOLO(str(icfg["weights"]))
    person_model = YOLO(icfg["person_model"])

    res = model.predict(frame, conf=icfg["panel_create_conf"], imgsz=icfg["imgsz"],
                        device=str(icfg["device"]), classes=[1, 2],
                        retina_masks=True, verbose=False)[0]
    if res.masks is None or len(res.masks) == 0:
        die("No betting detection on this frame - pick a frame where the board is visible.")

    h, w = frame.shape[:2]
    pres = person_model.predict(frame, conf=icfg["person_conf"], imgsz=icfg["imgsz"],
                                device=str(icfg["device"]), classes=[0],
                                retina_masks=True, verbose=False)[0]
    person = np.zeros((h, w), np.uint8)
    if pres.masks is not None and len(pres.masks) > 0:
        pm = pres.masks.data.cpu().numpy()
        if pm.shape[1:] != (h, w):
            pm = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in pm])
        person = (pm > 0.5).any(axis=0).astype(np.uint8)
    px = int(icfg["person_dilate_px"])
    person_d = cv2.dilate(person, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1,) * 2))

    masks = res.masks.data.cpu().numpy()
    if masks.shape[1:] != (h, w):
        masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks])
    conf = res.boxes.conf.cpu().numpy()
    best = int(np.argmax(conf))
    quad = quad_from_mask((masks[best] > 0.5).astype(np.uint8))
    if quad is None:
        die("Detection mask too small to fit a quad.")
    quad = expand_quad(canonical_quad(quad), icfg["panel_margin_frac"])

    scene_avg = scene_average_color(frame, person)

    # OLD: per-channel median over the whole region, people excluded
    m_all = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(m_all, np.round(quad).astype(np.int32), 1)
    m_all[person_d > 0] = 0
    old = np.median(frame[m_all > 0], axis=0).astype(np.float32)

    new = board_fill_color(frame, quad, person_d, scene_avg, mcfg)
    if new is None:
        die("board_fill_color could not sample the band on this frame.")

    def hsv(c):
        return cv2.cvtColor(np.uint8([[np.clip(c, 0, 255)]]), cv2.COLOR_BGR2HSV)[0, 0]

    print(f"detection conf     : {conf[best]:.3f}")
    print(f"scene average  BGR : {scene_avg.round(1).tolist()}  HSV {hsv(scene_avg).tolist()}")
    print(f"OLD whole-region   : {old.round(1).tolist()}  HSV {hsv(old).tolist()}")
    print(f"NEW board+muted    : {new.round(1).tolist()}  HSV {hsv(new).tolist()}")
    print(f"brightness vs scene: old {hsv(old)[2]} / new {hsv(new)[2]} / scene {hsv(scene_avg)[2]} "
          f"(new must be <= scene)")

    # visual: board crop, then both fills applied to it
    x0, y0 = np.floor(quad.min(axis=0)).astype(int)
    x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
    pad = int(0.8 * (y1 - y0))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    crop = frame[y0:y1, x0:x1].copy()

    def painted(color):
        out = crop.copy()
        q = (np.round(quad) - np.array([x0, y0])).astype(np.int32)
        cv2.fillConvexPoly(out, q, tuple(float(v) for v in color))
        return out

    strip = np.vstack([crop, painted(old), painted(new)])
    for i, label in enumerate(("ORIGINAL", "OLD whole-region median", "NEW board-sourced + muted")):
        cv2.putText(strip, label, (12, 30 + i * crop.shape[0]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.out, strip)
    print(f"\nWrote comparison to {args.out}")


if __name__ == "__main__":
    main()
