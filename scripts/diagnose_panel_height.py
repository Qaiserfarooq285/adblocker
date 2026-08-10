#!/usr/bin/env python3
"""Measure what the panel geometry ACTUALLY does, frame by frame.

Written to answer one question before changing any code: is the existing
over-capture clamp running and failing, or is it running and doing nothing?
Those need opposite fixes, and the difference is invisible from the output
video alone.

For every frame it prints the raw detection heights, the merged strip height,
the clamp bound height, and whether the clamp changed anything - all in pixels
and as a fraction of frame height. It also dumps the worst offenders as JPEGs
so the tall-panel frames can be looked at directly.

    python scripts/diagnose_panel_height.py --input data/raw_videos/match.mp4
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from common import die, load_config
from fill import canonical_quad, expand_quad, quad_from_mask

CLASS_BOARD, CLASS_OVERLAY = 1, 2


def quad_height(q: np.ndarray) -> float:
    """Vertical extent of the quad's bounding box - the number that decides
    whether the panel reaches the crowd, regardless of the quad's rotation."""
    return float(q[:, 1].max() - q[:, 1].min())


def main():
    cfg = load_config()
    icfg = dict(cfg["inference"])

    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--start", type=float, default=None)
    ap.add_argument("--end", type=float, default=None)
    ap.add_argument("--max-frames", type=int, default=400)
    ap.add_argument("--dump-top", type=int, default=6, help="save this many tallest-panel frames")
    ap.add_argument("--outdir", default="outputs/height_diag")
    args = ap.parse_args()

    from ultralytics import YOLO
    from common import LOCAL_CONFIG_PATH

    print(f"[config] config.yaml"
          + (f" + {LOCAL_CONFIG_PATH.name} (overlay active)" if LOCAL_CONFIG_PATH.exists() else ""))
    # The point of this line: if the key is absent, no height cap exists at all
    # and every "the clamp should have caught it" theory is moot.
    print(f"[config] inference.max_board_height_frac = "
          f"{icfg.get('max_board_height_frac', '<< ABSENT - NO HEIGHT CAP CONFIGURED >>')}")
    print(f"[config] inference.panel_margin_frac     = {icfg.get('panel_margin_frac')}")
    print(f"[config] inference.merge_gap_height_ratio= {icfg.get('merge_gap_height_ratio')}")
    print(f"[config] inference.occlusion_zone_down_frac = "
          f"{icfg.get('occlusion_zone_down_frac', '<< ABSENT - NO OCCLUSION ZONE >>')}")

    model = YOLO(str(icfg["weights"]))
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        die(f"cannot open {args.input}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if args.start:
        cap.set(cv2.CAP_PROP_POS_MSEC, args.start * 1000.0)
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[video] {src_w}x{src_h} @ {fps:.2f}fps\n")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    rows = []          # (merged_height_frac, frame_idx, frame, quads)
    n = 0
    clamp_changed = 0
    clamp_total = 0
    while n < args.max_frames:
        ok, raw = cap.read()
        if not ok:
            break
        n += 1
        if args.end and (args.start or 0) + n / fps > args.end:
            break

        res = model.track(raw, tracker=str(icfg["tracker"]), persist=True,
                          conf=icfg["conf"], iou=icfg["iou"], imgsz=icfg["imgsz"],
                          device=str(icfg["device"]), classes=[CLASS_BOARD, CLASS_OVERLAY],
                          retina_masks=True, verbose=False)[0]
        if res.boxes is None or len(res.boxes) == 0 or res.masks is None:
            continue

        masks = res.masks.data.cpu().numpy()
        if masks.shape[1:] != (src_h, src_w):
            masks = np.stack([cv2.resize(m, (src_w, src_h), interpolation=cv2.INTER_NEAREST)
                              for m in masks])
        confs = res.boxes.conf.cpu().numpy()

        raw_quads = []
        for i in range(len(masks)):
            q = quad_from_mask((masks[i] > 0.5).astype(np.uint8))
            if q is not None:
                raw_quads.append((float(confs[i]), q))
        if not raw_quads:
            continue

        # replicate process_video's merge exactly: same-line regions fuse
        members = sorted(raw_quads, key=lambda t: t[1][:, 0].min())
        groups: list[list] = []
        for conf, q in members:
            placed = False
            for g in groups:
                gq = g[-1][1]
                gy0, gy1 = gq[:, 1].min(), gq[:, 1].max()
                my0, my1 = q[:, 1].min(), q[:, 1].max()
                h_a, h_b = gy1 - gy0, my1 - my0
                if (min(gy1, my1) - max(gy0, my0) > 0.5 * min(h_a, h_b)
                        and q[:, 0].min() - gq[:, 0].max()
                        < icfg["merge_gap_height_ratio"] * max(h_a, h_b)):
                    g.append((conf, q))
                    placed = True
                    break
            if not placed:
                groups.append([(conf, q)])

        for g in groups:
            pts = np.concatenate([q for _, q in g]).astype(np.float32)
            merged = canonical_quad(cv2.boxPoints(cv2.minAreaRect(pts)))
            bound = expand_quad(merged, icfg["panel_margin_frac"])
            hm, hb = quad_height(merged), quad_height(bound)
            # bound is merged+margin, so the clamp can only ever bite if the
            # rendered quad drifts OUTSIDE it - never on the detection itself
            clamp_total += 1
            if hb < hm - 0.5:
                clamp_changed += 1
            raw_hs = [quad_height(q) for _, q in g]
            frac = hm / src_h
            print(f"f{n:4d} parts={len(g)} raw_h={[f'{h:.0f}' for h in raw_hs]} "
                  f"merged_h={hm:6.1f}px ({frac:5.1%} of frame) "
                  f"bound_h={hb:6.1f}px  clamp_bit={'YES' if hb < hm - 0.5 else 'no'}")
            rows.append((frac, n, raw.copy(), merged, bound))

    cap.release()

    if not rows:
        print("\nNo detections in the sampled range.")
        return

    fracs = np.array([r[0] for r in rows])
    print(f"\n===== SUMMARY over {len(rows)} panel-instances, {n} frames =====")
    print(f"panel height as fraction of frame height:")
    print(f"   min={fracs.min():.1%}  median={np.median(fracs):.1%}  "
          f"p90={np.percentile(fracs, 90):.1%}  max={fracs.max():.1%}")
    print(f"instances taller than 9% of frame : {(fracs > 0.09).sum()} / {len(fracs)}"
          f"  ({(fracs > 0.09).mean():.1%})")
    print(f"instances taller than 15% of frame: {(fracs > 0.15).sum()} / {len(fracs)}")
    print(f"clamp (panel_margin_frac) actually reduced height: {clamp_changed} / {clamp_total}")

    rows.sort(key=lambda r: -r[0])
    for i, (frac, idx, img, merged, bound) in enumerate(rows[:args.dump_top]):
        vis = img.copy()
        cv2.polylines(vis, [np.round(merged).astype(np.int32)], True, (0, 0, 255), 3)
        cv2.polylines(vis, [np.round(bound).astype(np.int32)], True, (255, 255, 0), 2)
        cap_h = 0.09 * src_h
        y_bot = merged[:, 1].max()
        cv2.line(vis, (0, int(y_bot - cap_h)), (src_w, int(y_bot - cap_h)), (0, 255, 0), 2)
        cv2.putText(vis, f"f{idx} h={frac:.1%} (green = 9% cap from bottom edge)",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        path = outdir / f"tall_{i:02d}_f{idx}_{frac:.3f}.jpg"
        cv2.imwrite(str(path), vis)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
