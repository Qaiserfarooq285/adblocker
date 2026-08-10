#!/usr/bin/env python3
"""Second OCR pass over overlapping tiles, merged into the first pass.

RapidOCR's detector downsamples internally, so a wordmark that is small or
low-contrast in a 1920px frame (the green "Kalshi" rails are the worst case)
is simply never proposed.  Feeding it tiles raises the effective resolution
without changing the model.

Results are unioned into data/ocr/<clip>/<frame>.json.  Boxes that duplicate
an existing detection are dropped, so re-running is idempotent.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def quad_box(q) -> tuple[float, float, float, float]:
    a = np.asarray(q, np.float32).reshape(-1, 2)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def iou(a, b) -> float:
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


_ENGINE = None
_ARGS = None


def _init(cfg) -> None:
    """One OCR engine per worker process; onnxruntime is not fork-safe.

    Each engine holds three ONNX sessions, and each session defaults to a
    thread pool the width of the machine.  Left alone, N workers oversubscribe
    the CPU by a factor of 3N and spend their time context-switching -- pinning
    to a single thread measured 2.5x faster even for one process.
    """
    global _ENGINE, _ARGS
    from rapidocr_onnxruntime import RapidOCR
    _ENGINE = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    _ARGS = cfg


def _work(fp_str: str) -> int:
    return process_frame(Path(fp_str), _ENGINE, _ARGS)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--ocr", default=str(ROOT / "data" / "ocr"))
    ap.add_argument("--cols", type=int, default=3)
    ap.add_argument("--rows", type=int, default=2)
    ap.add_argument("--overlap", type=float, default=0.18)
    ap.add_argument("--upscale", type=float, default=1.5)
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    frames = sorted(Path(args.frames).rglob("*.jpg"))
    todo = [str(p) for p in frames
            if not (Path(args.ocr) / p.relative_to(args.frames).with_suffix(".tiles")).exists()]
    print(f"[tiles] {len(todo)} frames pending of {len(frames)}, workers={args.workers}",
          flush=True)

    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        added_total = done = 0
        with ctx.Pool(args.workers, initializer=_init, initargs=(args,)) as pool:
            for added in pool.imap_unordered(_work, todo, chunksize=4):
                added_total += added
                done += 1
                if done % 100 == 0:
                    print(f"[tiles] {done}/{len(todo)} new_boxes={added_total}", flush=True)
        print(f"[tiles] done {done} frames, {added_total} new text boxes")
        return 0

    _init(args)
    added_total = sum(process_frame(Path(p), _ENGINE, args) for p in todo)
    print(f"[tiles] done {len(todo)} frames, {added_total} new text boxes")
    return 0


def process_frame(fp: Path, engine, args) -> int:
    """OCR one frame in tiles; merge new boxes into its record.  Returns count."""
    rel = fp.relative_to(args.frames).with_suffix(".json")
    jf = Path(args.ocr) / rel
    marker = jf.with_suffix(".tiles")
    if marker.exists():
        return 0
    try:
        recs = json.loads(jf.read_text()) if jf.exists() else []
    except Exception:
        recs = []

    img = cv2.imread(str(fp))
    if img is None:
        return 0
    H, W = img.shape[:2]
    boxes = [quad_box(r["quad"]) for r in recs]
    tw, th = W / args.cols, H / args.rows
    ox, oy = tw * args.overlap, th * args.overlap
    added = 0

    for r_i in range(args.rows):
        for c_i in range(args.cols):
            x0 = int(max(0, c_i * tw - ox))
            y0 = int(max(0, r_i * th - oy))
            x1 = int(min(W, (c_i + 1) * tw + ox))
            y1 = int(min(H, (r_i + 1) * th + oy))
            tile = img[y0:y1, x0:x1]
            if tile.size == 0:
                continue
            if args.upscale != 1.0:
                tile = cv2.resize(tile, None, fx=args.upscale, fy=args.upscale,
                                  interpolation=cv2.INTER_CUBIC)
            try:
                res, _ = engine(tile)
            except Exception:
                continue
            for item in (res or []):
                quad, text, conf = item[0], str(item[1]), float(item[2])
                if conf < args.min_conf:
                    continue
                q = np.asarray(quad, np.float32) / args.upscale
                q[:, 0] += x0
                q[:, 1] += y0
                bb = quad_box(q)
                if any(iou(bb, e) > 0.45 for e in boxes):
                    continue
                boxes.append(bb)
                recs.append({"text": text, "conf": round(conf, 3),
                             "quad": q.round(1).tolist(), "src": "tile"})
                added += 1

    jf.parent.mkdir(parents=True, exist_ok=True)
    jf.write_text(json.dumps(recs))
    # Sidecar marker, so "already tiled" never depends on having found
    # something and no placeholder record leaks into the annotator.
    marker.write_text(str(added))
    return added


if __name__ == "__main__":
    raise SystemExit(main())
