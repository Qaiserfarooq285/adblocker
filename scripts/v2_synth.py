#!/usr/bin/env python3
"""Targeted synthetic training data for the two measured weak spots.

Held-out recall broke down as:

    Betano 100%   PredictStreet 89%   Kalshi 77%
    small 86%     medium 92%          large/zoomed 60%

So the detector is not weak in general -- it is weak on Kalshi boards and on
boards that fill a lot of the frame.  Generic augmentation spends most of its
effort on cases already at 90-100%, so instead we synthesise those two cases
specifically.

Two generators, both built from real pixels so the composites stay plausible:

  replace  a real board in a real frame is repainted with another brand's
           artwork, keeping the true perspective, lighting and occlusion.
           Sampling is biased towards Kalshi to correct its under-representation.

  zoom     a board crop is laid across a frame at close-up scale, which is the
           view the detector currently misses four times in ten.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from v2_brands import classify, norm  # noqa: E402


def brand_of(poly, ocr_recs) -> str:
    P = np.asarray(poly, np.float32)
    for r in ocr_recs:
        if classify(r["text"]) != "bet":
            continue
        c = np.asarray(r["quad"], np.float32).mean(axis=0)
        if cv2.pointPolygonTest(P, (float(c[0]), float(c[1])), False) >= 0:
            t = norm(r["text"])
            if "alsh" in t:
                return "kalshi"
            if "dictstr" in t or "redict" in t:
                return "predict"
            if "etano" in t or "betan" in t:
                return "betano"
    return "other"


def rectify(img, quad, out_w=None, out_h=None):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    # order: longest edge first
    lens = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
    q = np.roll(q, -int(np.argmax(lens)), axis=0)
    if q[0][0] > q[1][0]:
        q = q[[1, 0, 3, 2]]
    if q[0][1] > q[3][1]:
        q = q[[3, 2, 1, 0]]
    w = int(out_w or round(np.linalg.norm(q[1] - q[0])))
    h = int(out_h or round(np.linalg.norm(q[3] - q[0])))
    if w < 12 or h < 6:
        return None
    M = cv2.getPerspectiveTransform(q, np.float32([[0, 0], [w, 0], [w, h], [0, h]]))
    return cv2.warpPerspective(img, M, (w, h))


def pitch_top_edge(img, min_run=60):
    """Per-column y of the top of the playing surface, or None if no pitch.

    Perimeter boards line the far edge of the grass, so this is where a
    synthetic board has to sit for the composite to be worth training on.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    grass = ((h_ >= 38) & (h_ <= 62) & (s_ > 70) & (v_ > 60)).astype(np.uint8)
    grass = cv2.morphologyEx(grass, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    H, W = grass.shape
    if grass.mean() < 0.15:
        return None
    edge = np.full(W, np.nan, np.float32)
    for x in range(W):
        col = np.nonzero(grass[:, x])[0]
        if len(col) < min_run:
            continue
        edge[x] = col.min()
    if np.isnan(edge).mean() > 0.5:
        return None
    idx = np.arange(W)
    good = ~np.isnan(edge)
    edge = np.interp(idx, idx[good], edge[good])
    return cv2.GaussianBlur(edge.reshape(1, -1), (51, 1), 0).ravel()


def paste(dst, crop, quad, feather=3):
    q = np.asarray(quad, np.float32).reshape(4, 2)
    h, w = crop.shape[:2]
    M = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [w, 0], [w, h], [0, h]]), q)
    warped = cv2.warpPerspective(crop, M, (dst.shape[1], dst.shape[0]))
    m = np.zeros(dst.shape[:2], np.uint8)
    cv2.fillConvexPoly(m, q.astype(np.int32), 255)
    if feather > 1:
        m = cv2.GaussianBlur(m, (feather | 1, feather | 1), 0)
    a = (m.astype(np.float32) / 255.0)[:, :, None]
    return (dst * (1 - a) + warped * a).astype(np.uint8)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=str(ROOT / "data" / "auto_labels" / "manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "synth"))
    ap.add_argument("--n-replace", type=int, default=420)
    ap.add_argument("--n-zoom", type=int, default=320)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    from ultralytics import YOLO
    person_model = YOLO("yolo11m-seg.pt")
    _pcache: dict[str, np.ndarray] = {}

    def people_mask(path: str, img) -> np.ndarray:
        """Cached person mask, so a board can be composited behind players.

        Pasting a board over the players who stand in front of it puts human
        pixels inside the board label, which is the opposite of the occlusion
        the renderer has to reproduce.
        """
        if path in _pcache:
            return _pcache[path]
        m = np.zeros(img.shape[:2], np.uint8)
        r = person_model.predict(img, conf=0.2, classes=[0], imgsz=960,
                                 verbose=False, retina_masks=True)[0]
        if r.masks is not None:
            for k in r.masks.data.cpu().numpy():
                mm = (k > 0.5).astype(np.uint8)
                if mm.shape != m.shape:
                    mm = cv2.resize(mm, (m.shape[1], m.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                m |= mm
        if len(_pcache) < 400:
            _pcache[path] = m
        return m
    man = json.loads(Path(args.manifest).read_text())
    pos = [m for m in man if m["kind"] == "pos"]
    neg = [m for m in man if m["kind"] == "neg"]

    # --- crop library, tagged by brand -------------------------------------
    lib: dict[str, list[np.ndarray]] = defaultdict(list)
    for m in pos:
        ocr_path = m["image"].replace("data/frames", "data/ocr").replace(".jpg", ".json")
        try:
            recs = json.loads(Path(ocr_path).read_text())
        except Exception:
            continue
        img = cv2.imread(m["image"])
        if img is None:
            continue
        for poly in m["label"]:
            b = brand_of(poly, recs)
            if b == "other":
                continue
            c = rectify(img, poly)
            if c is not None and c.shape[0] >= 12 and c.shape[1] >= 60:
                lib[b].append(c)
    counts = {k: len(v) for k, v in lib.items()}
    print(f"[synth] crop library: {counts}")
    if not lib:
        print("[synth] no crops; nothing to do")
        return 1

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    manifest = []

    # Bias towards the under-performing brand rather than sampling uniformly.
    weights = {"kalshi": 0.55, "predict": 0.30, "betano": 0.15}
    brands = [b for b in weights if lib.get(b)]

    def pick_crop():
        b = rng.choices(brands, [weights[b] for b in brands])[0]
        return rng.choice(lib[b]), b

    made = 0
    # --- generator 1: repaint a real board ---------------------------------
    for i in range(args.n_replace):
        m = rng.choice(pos)
        img = cv2.imread(m["image"])
        if img is None or not m["label"]:
            continue
        img = img.copy()
        polys = [np.asarray(p, np.float32) for p in m["label"]]
        target = rng.choice(polys)
        crop, _ = pick_crop()
        q = np.asarray(target, np.float32).reshape(4, 2)
        tw = max(24, int(np.linalg.norm(q[1] - q[0])))
        th = max(8, int(np.linalg.norm(q[3] - q[0])))
        tile = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_LINEAR)
        orig = img.copy()
        img = paste(img, tile, q)
        pmask = people_mask(m["image"], orig)
        img[pmask > 0] = orig[pmask > 0]
        name = f"rep_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        manifest.append({"image": str(out / "images" / f"{name}.jpg"),
                         "label": [p.round(2).tolist() for p in polys], "kind": "pos"})
        made += 1

    # --- generator 2: close-up boards --------------------------------------
    pool = pos + neg
    for i in range(args.n_zoom):
        m = rng.choice(pool)
        img = cv2.imread(m["image"])
        if img is None:
            continue
        img = img.copy()
        H, W = img.shape[:2]
        crop, _ = pick_crop()
        # Anchor to the real pitch boundary.  A board pasted at a random height
        # floats in the middle of the grass, and training on that teaches the
        # detector to fire on open pitch -- worse than the gap being fixed.
        edge = pitch_top_edge(img)
        if edge is None:
            continue
        cx = rng.uniform(0.2, 0.8) * W
        base = float(np.interp(cx, np.arange(W), edge))
        if not (0.10 * H < base < 0.80 * H):
            continue
        length = rng.uniform(0.55, 1.25) * W
        thick = rng.uniform(0.10, 0.24) * H
        cy = base - thick * rng.uniform(0.35, 0.55)   # board sits above the grass
        # Follow the pitch edge rather than tilting at random: a real rail runs
        # parallel to the touchline, so its angle is set by the perspective.
        x0 = int(np.clip(cx - length / 2, 0, W - 1))
        x1 = int(np.clip(cx + length / 2, 0, W - 1))
        if x1 - x0 > 20:
            ang = float(np.arctan2(edge[x1] - edge[x0], x1 - x0))
            ang = float(np.clip(ang, np.deg2rad(-14), np.deg2rad(14)))
        else:
            ang = np.deg2rad(rng.uniform(-6, 6))
        d = np.array([np.cos(ang), np.sin(ang)], np.float32)
        n = np.array([-d[1], d[0]], np.float32)
        # Slight trapezoid so it does not look pasted flat.
        k = rng.uniform(0.85, 1.15)
        c = np.array([cx, cy], np.float32)
        q = np.float32([
            c - d * length / 2 - n * thick / 2,
            c + d * length / 2 - n * thick * k / 2,
            c + d * length / 2 + n * thick * k / 2,
            c - d * length / 2 + n * thick / 2])
        if q[:, 0].min() < -length or q[:, 0].max() > W + length:
            continue
        # Reject placements that are not board-like before spending the paste.
        # A quad lying over the pitch is grass, and one mostly hidden behind
        # players leaves a label with no board in it -- both teach the detector
        # that open pitch and people are advertising boards.
        qm = np.zeros((H, W), np.uint8)
        cv2.fillConvexPoly(qm, q.astype(np.int32), 1)
        qarea = int(qm.sum())
        if qarea < 4000:
            continue
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        grass = ((hsv[:, :, 0] >= 38) & (hsv[:, :, 0] <= 62) &
                 (hsv[:, :, 1] > 70) & (hsv[:, :, 2] > 60)).astype(np.uint8)
        if float((grass & qm).sum()) / qarea > 0.40:
            continue
        pmask = people_mask(m["image"], img)
        if float((pmask & qm).sum()) / qarea > 0.25:
            continue

        tile = cv2.resize(crop, (max(32, int(length)), max(12, int(thick))),
                          interpolation=cv2.INTER_LINEAR)
        orig = img.copy()
        img = paste(img, tile, q)
        img[pmask > 0] = orig[pmask > 0]
        name = f"zoom_{i:05d}"
        cv2.imwrite(str(out / "images" / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
        # Only the pasted board is labelled; it covers whatever was beneath it.
        keep = [q.round(2).tolist()]
        for p in m.get("label", []):
            pp = np.asarray(p, np.float32)
            inside = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(inside, q.astype(np.int32), 1)
            om = np.zeros(img.shape[:2], np.uint8)
            cv2.fillConvexPoly(om, pp.astype(np.int32), 1)
            if om.sum() and (om & inside).sum() / om.sum() < 0.25:
                keep.append(pp.round(2).tolist())
        manifest.append({"image": str(out / "images" / f"{name}.jpg"),
                         "label": keep, "kind": "pos"})
        made += 1

    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"[synth] wrote {made} images -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
