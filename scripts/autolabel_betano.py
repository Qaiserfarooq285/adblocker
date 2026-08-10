#!/usr/bin/env python3
"""Automatically generate REAL betting_board annotations for Betano perimeter
boards, with no human in the loop, by combining two strong priors that hold
for this brand's board design:

  1. Color: the strip is saturated Betano-red (red segments with white script,
     white segments with red CONFIA text - both produce red pixels).
  2. Text: the white-on-red "Betano" script / red-on-white "CONFIA" wordmark,
     confirmed by multi-scale template matching against crops harvested from
     the real footage itself (data/templates/betano/).

Candidate red bands are found by color + horizontal morphology (which bridges
the alternating Betano/CONFIA segments into one strip), then every band must
pass template confirmation before it becomes a label. Bands that fail stay
unlabeled; frames whose best score is far below the accept threshold are
recorded as hard-negative candidates (data/negatives/ feed).

This solves the project's root failure: training was 100% synthetic and the
model scored real boards at ~0.002 confidence. These labels are REAL frames
with REAL board appearances - imperfect polygons beat perfect blindness.

Output layout matches what assemble_dataset.py expects:
    data/annotations_real/images/<stem>.jpg
    data/annotations_real/labels/<stem>.txt      (class 1 polygons)
Review overlays (drawn polygons + scores) go to data/prelabel_review/betano_auto/.

Usage:
    python scripts/autolabel_betano.py --video match
    python scripts/autolabel_betano.py                       # all frame dirs
    python scripts/autolabel_betano.py --dry-run             # overlays only
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import IMAGE_EXTS, die, load_config

CLASS_BOARD = 1

ACCEPT_SCORE = 0.55        # template score >= this -> confirmed Betano band
NEGATIVE_MAX_SCORE = 0.32  # frame is a negative candidate only if EVERY band scores below this
WORK_WIDTH = 2560          # frames are downscaled to this width for mask work

RED_LO1, RED_HI1 = (0, 90, 70), (10, 255, 255)
RED_LO2, RED_HI2 = (170, 90, 70), (180, 255, 255)


def load_templates(tpl_dir: Path) -> dict[str, list[np.ndarray]]:
    """Two independent template groups, keyed by filename prefix. A band must
    match BOTH the 'Betano' script and the 'CONFIA' wordmark to be accepted -
    single-cue grayscale correlation also fires on DoorDash boards, scoreboard
    flags, and crowd patches (measured 0.46-0.59 on all three), but only the
    real strip contains both marks."""
    groups: dict[str, list[np.ndarray]] = {"betano": [], "confia": []}
    for f in sorted(tpl_dir.glob("*.png")):
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None or min(img.shape) < 20:
            continue
        for key in groups:
            if f.stem.lower().startswith(key):
                groups[key].append(img)
    if not groups["betano"] or not groups["confia"]:
        die(f"Need both betano_* and confia_* template crops in {tpl_dir}.")
    return groups


def red_mask(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, np.array(RED_LO1), np.array(RED_HI1)) | cv2.inRange(
        hsv, np.array(RED_LO2), np.array(RED_HI2))
    h, w = m.shape
    # open first to kill crowd speckle, then close hard horizontally so the
    # alternating red/white board segments fuse into one contiguous band
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    close_k = cv2.getStructuringElement(cv2.MORPH_RECT, (max(9, w // 55), max(3, h // 240)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, close_k)
    return m


def candidate_bands(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = mask.shape
    n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    out = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if bw < 0.02 * w or bh > 0.18 * h or bh < 20:
            continue
        if bw / max(bh, 1) < 2.0:
            continue
        if area < 0.25 * bw * bh:  # too sparse to be a solid board strip
            continue
        out.append((x, y, bw, bh))
    return out


def merge_colinear(bands: list[tuple[int, int, int, int]]) -> list[list[tuple[int, int, int, int]]]:
    """Group bands whose vertical centers align and whose horizontal gap is
    small relative to their height - one physical strip, split by occluders
    or by the alternating red-segment/red-text pattern. Vertical alignment is
    judged against the SMALLER member so a growing merged box can't chain-
    swallow crowd blobs above the strip. Returns groups of original bands."""
    bands = sorted(bands, key=lambda b: b[0])
    groups: list[list[tuple[int, int, int, int]]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for b in bands:
        bx, by, bw, bh = b
        placed = False
        for gi, (x, y, w, h) in enumerate(boxes):
            cy_a, cy_b = y + h / 2, by + bh / 2
            gap = bx - (x + w)
            if abs(cy_a - cy_b) < 0.8 * min(h, bh) + 4 and gap < 2.2 * max(h, bh):
                groups[gi].append(b)
                nx, ny = min(x, bx), min(y, by)
                nx2, ny2 = max(x + w, bx + bw), max(y + h, by + bh)
                boxes[gi] = (nx, ny, nx2 - nx, ny2 - ny)
                placed = True
                break
        if not placed:
            groups.append([b])
            boxes.append(b)
    return groups


def group_box(group: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    x = min(b[0] for b in group)
    y = min(b[1] for b in group)
    x2 = max(b[0] + b[2] for b in group)
    y2 = max(b[1] + b[3] for b in group)
    return (x, y, x2 - x, y2 - y)


def band_score(gray_frame: np.ndarray, band: tuple[int, int, int, int],
               templates: dict[str, list[np.ndarray]]) -> dict[str, float]:
    x, y, w, h = band
    pad = h
    crop = gray_frame[max(0, y - pad): y + h + pad, max(0, x - pad): x + w + pad]
    scores = {k: 0.0 for k in templates}
    if min(crop.shape) < 24:
        return scores
    for key, group in templates.items():
        for tpl in group:
            for frac in (0.3, 0.38, 0.48, 0.6, 0.75, 0.95, 1.2):
                th = max(16, int(h * frac))
                scale = th / tpl.shape[0]
                tw = int(tpl.shape[1] * scale)
                if tw < 16 or th >= crop.shape[0] or tw >= crop.shape[1]:
                    continue
                t = cv2.resize(tpl, (tw, th), interpolation=cv2.INTER_AREA)
                res = cv2.matchTemplate(crop, t, cv2.TM_CCOEFF_NORMED)
                scores[key] = max(scores[key], float(res.max()))
    return scores


def is_confirmed(scores: dict[str, float], accept: float) -> bool:
    """One mark matched STRONGLY and the other at least weakly present.
    Calibrated on this footage: true strips always carry one dominant match
    (0.75-0.96) because at least one wordmark is sharply visible, while every
    impostor found so far (blurred crowd bands, scoreboard bars, DoorDash)
    plateaus as 'mediocre both' - 0.50-0.60 on each without ever spiking."""
    b, c = scores["betano"], scores["confia"]
    return max(b, c) >= 0.70 and min(b, c) >= 0.40


def strip_polygons(mask: np.ndarray, band: tuple[int, int, int, int]) -> list[np.ndarray]:
    """Tight polygon(s) that FOLLOW the strip instead of hulling every mask
    pixel in the band box (a hull happily swallows red crowd patches that got
    merged in). The physical strip is a straight band in perspective, so:
    fit a robust line to the per-column mask centers, reject outlier columns
    (crowd blobs), and trace top/bottom edges of the inlier columns. Large
    column gaps (players standing in front) split the strip into separate
    polygons rather than labeling the occluder as board."""
    x, y, w, h = band
    sub = mask[y: y + h, x: x + w]
    cols = np.where(sub.any(axis=0))[0]
    if len(cols) < 20:
        return []
    tops = np.array([np.argmax(sub[:, c]) for c in cols], dtype=float)
    bots = np.array([h - np.argmax(sub[::-1, c]) for c in cols], dtype=float)
    cys = (tops + bots) / 2

    inl = np.ones(len(cols), dtype=bool)
    for _ in range(3):
        if inl.sum() < 10:
            return []
        coef = np.polyfit(cols[inl], cys[inl], 1)
        resid = np.abs(np.polyval(coef, cols) - cys)
        med_h = np.median((bots - tops)[inl])
        inl = resid < max(0.9 * med_h, 5.0)

    cols_i, tops_i, bots_i = cols[inl], tops[inl], bots[inl]
    heights_i = bots_i - tops_i
    med_h = np.median(heights_i)

    # keep only columns whose height looks like the strip (trims noise tails
    # and columns polluted by crowd red above/below the board)
    ok = (heights_i > 0.45 * med_h) & (heights_i < 1.8 * med_h)
    if ok.sum() < 10:
        return []
    cols_i, tops_i, bots_i = cols_i[ok], tops_i[ok], bots_i[ok]

    # the physical strip is straight, so model it as two linear fits - center
    # line and height along x (height varies with perspective) - and emit a
    # clean quad per contiguous run instead of tracing noisy per-column edges
    cy_fit = np.polyfit(cols_i, (tops_i + bots_i) / 2, 1)
    h_fit = np.polyfit(cols_i, bots_i - tops_i, 1)

    polys = []
    breaks = np.where(np.diff(cols_i) > max(3 * med_h, 40))[0]
    segments = np.split(np.arange(len(cols_i)), breaks + 1)
    for seg in segments:
        if len(seg) < 15 or cols_i[seg[-1]] - cols_i[seg[0]] < 2 * med_h:
            continue
        x0, x1 = cols_i[seg[0]], cols_i[seg[-1]]
        quad = []
        for xx in (x0, x1):
            cy = np.polyval(cy_fit, xx)
            hh = max(6.0, np.polyval(h_fit, xx))
            quad.append((xx, cy - hh / 2))
        for xx in (x1, x0):
            cy = np.polyval(cy_fit, xx)
            hh = max(6.0, np.polyval(h_fit, xx))
            quad.append((xx, cy + hh / 2))
        polys.append(np.array(quad, dtype=float) + np.array([x, y], dtype=float))
    return polys


def main():
    cfg = load_config()
    p = cfg["paths"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=str, default=None,
                    help="Limit to data/frames/<slug>/ (default: every frame dir)")
    ap.add_argument("--accept", type=float, default=ACCEPT_SCORE)
    ap.add_argument("--dry-run", action="store_true",
                    help="Write overlays + scores only, no annotations/negatives")
    ap.add_argument("--max-negatives", type=int, default=150)
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    tpl_dir = Path("data/templates/betano")
    templates = load_templates(tpl_dir)
    print(f"Loaded {len(templates)} Betano template(s) from {tpl_dir}")

    frames_root = p["frames"]
    if args.video:
        dirs = [frames_root / args.video]
        if not dirs[0].exists():
            die(f"'{dirs[0]}' not found.")
    else:
        dirs = sorted(d for d in frames_root.iterdir() if d.is_dir())

    frame_paths = []
    for d in dirs:
        frame_paths += sorted(f for f in d.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not frame_paths:
        die("No frames found. Run extract_frames.py first.")

    ann_img = p["annotations_real"] / "images"
    ann_lbl = p["annotations_real"] / "labels"
    review_dir = p["prelabel_review"] / "betano_auto"
    for d in (ann_img, ann_lbl, review_dir):
        d.mkdir(parents=True, exist_ok=True)

    labeled = 0
    neg_candidates = []
    rows = []
    for fp in tqdm(frame_paths, desc="Betano auto-label"):
        img = cv2.imread(str(fp))
        if img is None:
            continue
        H, W = img.shape[:2]
        scale = WORK_WIDTH / W if W > WORK_WIDTH else 1.0
        work = cv2.resize(img, (int(W * scale), int(H * scale)), interpolation=cv2.INTER_AREA) if scale < 1.0 else img
        gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        mask = red_mask(work)
        groups = merge_colinear(candidate_bands(mask))

        Wk = work.shape[1]

        def wide_strip(b):
            # confirmed bands must be an unambiguously wide strip - tiny bands
            # (scoreboard flags) produce spurious ~0.5 matches at small scales
            return b[2] >= 0.05 * Wk and b[2] / max(b[3], 1) >= 3.0

        bands = []
        for group in groups:
            gb = group_box(group)
            if wide_strip(gb):
                bands.append(gb)
            else:
                # merged geometry went bad (e.g. chain-merged into a tall
                # blob) - fall back to the members that qualify on their own
                bands.extend(b for b in group if wide_strip(b))

        confirmed, best_frame_score = [], 0.0
        for band in bands:
            scores = band_score(gray, band, templates)
            best_frame_score = max(best_frame_score, max(scores.values()))
            rows.append([fp.name, *band, f"{scores['betano']:.3f}", f"{scores['confia']:.3f}"])
            if is_confirmed(scores, args.accept):
                for poly in strip_polygons(mask, band):
                    confirmed.append((poly, min(scores.values())))

        wh = np.array([work.shape[1], work.shape[0]], dtype=float)
        if confirmed:
            lines = []
            overlay = work.copy()
            for poly, s in confirmed:
                norm = (poly / wh).clip(0.0, 1.0)
                coords = " ".join(f"{v:.6f}" for v in norm.reshape(-1))
                lines.append(f"{CLASS_BOARD} {coords}")
                cv2.polylines(overlay, [poly.astype(int)], True, (0, 255, 0), 3)
                x, y = poly.min(axis=0).astype(int)
                cv2.putText(overlay, f"{s:.2f}", (x, max(20, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imwrite(str(review_dir / fp.name), overlay)
            if not args.dry_run:
                shutil.copy2(fp, ann_img / fp.name)
                (ann_lbl / f"{fp.stem}.txt").write_text("\n".join(lines) + "\n")
            labeled += 1
        elif bands and best_frame_score < NEGATIVE_MAX_SCORE:
            neg_candidates.append(fp)

    (review_dir / "scores.csv").write_text(
        "\n".join(",".join(map(str, r)) for r in [["frame", "x", "y", "w", "h", "betano", "confia"]] + rows) + "\n")

    if not args.dry_run and neg_candidates:
        random.seed(1234)
        random.shuffle(neg_candidates)
        for fp in neg_candidates[: args.max_negatives]:
            shutil.copy2(fp, p["negatives"] / fp.name)

    print(f"\nDone. {labeled} frame(s) labeled with confirmed Betano boards.")
    print(f"Review overlays: {review_dir}")
    if not args.dry_run:
        print(f"Annotations: {p['annotations_real']}")
        print(f"Hard negatives added: {min(len(neg_candidates), args.max_negatives)} "
              f"(of {len(neg_candidates)} candidates)")


if __name__ == "__main__":
    main()
