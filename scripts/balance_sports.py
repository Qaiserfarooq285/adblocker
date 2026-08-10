#!/usr/bin/env python3
"""Build an evenly-balanced per-sport frame subset from one multi-sport video.

Without this, training is dominated by whichever sport happens to run longest,
and the model learns that sport's board style best while barely seeing the
others. Betting boards differ enormously by sport - a hockey dasher board, a
courtside LED, branding printed on an MMA canvas, an outfield wall - so an
unbalanced sample is not a small problem.

Frames are symlinked, not copied, so this costs nothing on disk and the original
`<slug>_t<seconds>.jpg` filename survives - assemble_dataset.py needs that
timestamp to split train/val by time chunk rather than at random.

    python scripts/balance_sports.py --per-sport 320
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from common import die, load_config


def frame_time(p: Path) -> float:
    m = re.search(r"_t(\d+(?:\.\d+)?)", p.stem)
    return float(m.group(1)) if m else -1.0


def main():
    cfg = load_config()
    p = cfg["paths"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--segments", default="data/segments.json")
    ap.add_argument("--per-sport", type=int, default=320,
                    help="frames sampled per sport (evenly spread through it)")
    ap.add_argument("--out", default="data/frames_by_sport")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    seg_path = Path(args.segments)
    if not seg_path.exists():
        die(f"{seg_path} not found. Run segment_sports.py first.")
    meta = json.loads(seg_path.read_text())
    frame_dir = p["frames"] / meta["slug"]
    if not frame_dir.exists():
        die(f"{frame_dir} not found. Run extract_frames.py first.")

    paths = sorted((q for q in frame_dir.iterdir()
                    if q.suffix.lower() in (".jpg", ".jpeg", ".png")), key=frame_time)
    times = np.array([frame_time(q) for q in paths])

    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    print(f"{'sport':20s} {'available':>10s} {'sampled':>8s}")
    total = 0
    summary = []
    for s in meta["segments"]:
        name = s.get("name", s.get("id"))
        dest = out_root / name
        if dest.exists() and not args.force:
            n = len(list(dest.glob("*")))
            print(f"{name:20s} {'-':>10s} {n:>8d}  (exists, use --force to redo)")
            summary.append({"sport": name, "sampled": n})
            total += n
            continue
        if dest.exists():
            for q in dest.iterdir():
                q.unlink()
        dest.mkdir(parents=True, exist_ok=True)

        sel = np.where((times >= s["start"]) & (times <= s["end"]))[0]
        if len(sel) == 0:
            print(f"{name:20s} {0:>10d} {0:>8d}")
            continue
        take = min(args.per_sport, len(sel))
        # evenly spread rather than random, so the sample covers the whole
        # segment instead of clustering in one passage of play
        idx = sel[np.linspace(0, len(sel) - 1, take).round().astype(int)]
        for i in sorted(set(idx.tolist())):
            src = paths[i].resolve()
            link = dest / paths[i].name
            if not link.exists():
                link.symlink_to(src)
        n = len(list(dest.glob("*")))
        print(f"{name:20s} {len(sel):>10d} {n:>8d}")
        summary.append({"sport": name, "sampled": n})
        total += n

    (out_root / "balance.json").write_text(json.dumps(
        {"per_sport_requested": args.per_sport, "total": total,
         "sports": summary}, indent=2) + "\n")
    print(f"\n[balance] {total} frames across {len(summary)} sports -> {out_root}")


if __name__ == "__main__":
    main()
