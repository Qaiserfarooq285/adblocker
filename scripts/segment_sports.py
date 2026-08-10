#!/usr/bin/env python3
"""Split one long multi-sport video into its constituent sports (Step A2).

The input is a single recording that runs through several different sports in
sequence. Each sport is a different venue, so it has a distinct colour
signature - a green pitch, a wooden court, a blue mat - which is enough to find
the boundaries without knowing what the sports are.

Why change-point detection and not scene-cut detection: a broadcast cuts shots
every few seconds, so cut detection returns hundreds of boundaries and none of
them tells you the SPORT changed. What changes at a sport boundary is the
sustained colour distribution, so frames are binned into blocks of
`bin_seconds`, and an exact dynamic-programming segmentation finds the K-1
boundaries that minimise total within-segment variance. That returns exactly K
segments by construction, which is what "split into 5 sports" asks for.

Segments are named `sport_1`..`sport_K` because guessing the sport's NAME from
pixels is unreliable; a thumbnail sheet is written so the names can be corrected
by hand in data/segments.json.

    python scripts/segment_sports.py --video "15 min 5 sports.mp4" --sports 5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

from common import die, load_config, slugify


def frame_time(p: Path) -> float:
    """Seconds encoded in the filename by extract_frames.py (<slug>_t<sec>.jpg)."""
    m = re.search(r"_t(\d+(?:\.\d+)?)", p.stem)
    return float(m.group(1)) if m else -1.0


def features(paths: list[Path], size: int = 96) -> np.ndarray:
    """One colour-signature vector per frame: a coarse HS histogram plus a
    coarse spatial value layout. The spatial part matters - a basketball court
    and a football pitch can share a hue while looking nothing alike."""
    out = []
    for p in paths:
        im = cv2.imread(str(p))
        if im is None:
            out.append(None)
            continue
        small = cv2.resize(im, (size, size), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
        cv2.normalize(hist, hist)
        grid = cv2.resize(hsv[..., 2], (6, 6), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        out.append(np.concatenate([hist.ravel(), grid.ravel() * 0.5]))
    dim = next(v.shape[0] for v in out if v is not None)
    return np.stack([v if v is not None else np.zeros(dim, np.float32) for v in out])


def segment(x: np.ndarray, k: int) -> list[int]:
    """Exact K-segmentation of a sequence by dynamic programming.

    cost[i, j] is the within-segment sum of squared deviations for bins i..j-1,
    computed from prefix sums so the whole table is O(N^2) rather than O(N^3).
    Returns the K-1 interior boundary indices."""
    n = x.shape[0]
    ps = np.zeros((n + 1, x.shape[1]), np.float64)
    ps2 = np.zeros(n + 1, np.float64)
    ps[1:] = np.cumsum(x, axis=0)
    ps2[1:] = np.cumsum((x ** 2).sum(axis=1))

    def cost(i: int, j: int) -> float:
        m = j - i
        if m <= 0:
            return 0.0
        s = ps[j] - ps[i]
        return float(ps2[j] - ps2[i] - (s @ s) / m)

    INF = float("inf")
    dp = np.full((k + 1, n + 1), INF)
    back = np.zeros((k + 1, n + 1), int)
    dp[0, 0] = 0.0
    for seg in range(1, k + 1):
        for end in range(seg, n + 1):
            best, arg = INF, seg - 1
            for start in range(seg - 1, end):
                if dp[seg - 1, start] == INF:
                    continue
                c = dp[seg - 1, start] + cost(start, end)
                if c < best:
                    best, arg = c, start
            dp[seg, end], back[seg, end] = best, arg

    bounds, end = [], n
    for seg in range(k, 0, -1):
        start = back[seg, end]
        bounds.append(start)
        end = start
    return sorted(b for b in bounds if 0 < b < n)


def main():
    cfg = load_config()
    p = cfg["paths"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="filename inside input/videos/")
    ap.add_argument("--sports", type=int, default=5)
    ap.add_argument("--bin-seconds", type=float, default=5.0)
    ap.add_argument("--out", default="data/segments.json")
    args = ap.parse_args()

    slug = slugify(Path(args.video).stem)
    frame_dir = p["frames"] / slug
    if not frame_dir.exists():
        die(f"No extracted frames at {frame_dir}. Run extract_frames.py first.")

    paths = sorted((q for q in frame_dir.iterdir() if q.suffix.lower() in (".jpg", ".png")),
                   key=frame_time)
    paths = [q for q in paths if frame_time(q) >= 0]
    if len(paths) < args.sports * 4:
        die(f"Only {len(paths)} usable frames - not enough to find {args.sports} segments.")
    times = np.array([frame_time(q) for q in paths])
    print(f"[seg] {len(paths)} frames spanning {times.min():.0f}-{times.max():.0f}s")

    print("[seg] computing colour signatures...")
    feats = features(paths)

    # bin to keep the DP tractable and to stop single odd shots becoming segments
    bin_idx = np.floor(times / args.bin_seconds).astype(int)
    uniq = np.unique(bin_idx)
    binned = np.stack([feats[bin_idx == b].mean(axis=0) for b in uniq])
    bin_time = np.array([times[bin_idx == b].mean() for b in uniq])
    print(f"[seg] {len(uniq)} bins of {args.bin_seconds:.0f}s -> "
          f"searching for {args.sports} segments")

    bounds = segment(binned, args.sports)
    edges = [0] + list(bounds) + [len(uniq)]

    segments = []
    for i in range(len(edges) - 1) :
        b0, b1 = edges[i], edges[i + 1]
        t0 = float(bin_time[b0] - args.bin_seconds / 2)
        t1 = float(bin_time[b1 - 1] + args.bin_seconds / 2)
        mask = (times >= t0) & (times <= t1)
        segments.append({
            "name": f"sport_{i + 1}",
            "start": round(max(0.0, t0), 2),
            "end": round(t1, 2),
            "duration": round(t1 - max(0.0, t0), 2),
            "n_frames": int(mask.sum()),
        })

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"video": args.video, "slug": slug,
                                    "segments": segments}, indent=2) + "\n")

    print(f"\n{'name':10s} {'start':>9s} {'end':>9s} {'dur':>8s} {'frames':>7s}")
    for s in segments:
        print(f"{s['name']:10s} {s['start']:9.1f} {s['end']:9.1f} "
              f"{s['duration']:8.1f} {s['n_frames']:7d}")
    print(f"\n[seg] wrote {out_path}")

    # thumbnail sheet so the segments can be named by hand
    sheet_dir = Path("data/segments")
    sheet_dir.mkdir(parents=True, exist_ok=True)
    tiles = []
    for s in segments:
        mid = (s["start"] + s["end"]) / 2.0
        pick = paths[int(np.argmin(np.abs(times - mid)))]
        im = cv2.imread(str(pick))
        if im is None:
            continue
        im = cv2.resize(im, (480, 270))
        cv2.rectangle(im, (0, 0), (479, 34), (0, 0, 0), -1)
        cv2.putText(im, f"{s['name']}  {s['start']:.0f}-{s['end']:.0f}s",
                    (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(sheet_dir / f"{s['name']}.jpg"), im)
        tiles.append(im)
    if tiles:
        rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(str(sheet_dir / "segments_sheet.jpg"), np.vstack(rows))
        print(f"[seg] wrote {sheet_dir / 'segments_sheet.jpg'} - rename segments in "
              f"{out_path} if you want real sport names")


if __name__ == "__main__":
    main()
