#!/usr/bin/env python3
"""Merge the supplied Roboflow logo dataset with the logos mined from this
footage, into one single-class detection set.

Why merge rather than pick one. The Roboflow set is hand-labelled and precise
but it is Premier League soccer, and most of its boxes sit on players' shirt
sponsors - it teaches "what a betting wordmark looks like" but nothing about
hockey dasher boards, courtside LEDs, an MMA canvas or an outfield wall. The
mined set is exactly the reverse: it is this footage, in all its sports, but
its boxes come from OCR so they are only as good as the text was legible.
Together they cover both axes.

The split is by SOURCE FRAME, not at random. Mined frames are sampled 2/second
from one recording, so neighbouring frames are near-duplicates; splitting those
at random puts a frame in train and its twin in val, and the val score becomes
a memorisation check rather than a measurement.

    python scripts/merge_logo_datasets.py --out data/logo_merged
"""
from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter
from pathlib import Path


def frame_time(stem: str) -> float | None:
    m = re.search(r"_t(\d+(?:\.\d+)?)", stem)
    return float(m.group(1)) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roboflow", default="data/ext_dataset")
    ap.add_argument("--mined", default="data/mined_logos")
    ap.add_argument("--out", default="data/logo_merged")
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--chunk-seconds", type=float, default=20.0,
                    help="mined frames are grouped into time chunks before the "
                         "split so near-duplicates never straddle it")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for s in ("train", "val"):
        (out / s / "images").mkdir(parents=True, exist_ok=True)
        (out / s / "labels").mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stats = Counter()

    # ---- Roboflow: keep its own train/valid/test split intent ----
    rb = Path(args.roboflow)
    for split_in, split_out in (("train", "train"), ("valid", "val"), ("test", "val")):
        idir, ldir = rb / split_in / "images", rb / split_in / "labels"
        if not idir.exists():
            continue
        for img in sorted(idir.iterdir()):
            lbl = ldir / (img.stem + ".txt")
            if not lbl.exists():
                continue
            shutil.copy2(img, out / split_out / "images" / f"rb_{img.name}")
            shutil.copy2(lbl, out / split_out / "labels" / f"rb_{img.stem}.txt")
            stats[f"roboflow_{split_out}"] += 1

    # ---- mined: split by time chunk ----
    mi = Path(args.mined)
    idir, ldir = mi / "images", mi / "labels"
    if idir.exists():
        chunks: dict[int, list[Path]] = {}
        for img in sorted(idir.iterdir()):
            t = frame_time(img.stem)
            key = int(t // args.chunk_seconds) if t is not None else rng.randint(0, 10 ** 6)
            chunks.setdefault(key, []).append(img)
        keys = sorted(chunks)
        rng.shuffle(keys)
        n_val = max(1, int(len(keys) * args.val_frac))
        val_keys = set(keys[:n_val])
        for k, imgs in chunks.items():
            split = "val" if k in val_keys else "train"
            for img in imgs:
                lbl = ldir / (img.stem + ".txt")
                if not lbl.exists():
                    continue
                shutil.copy2(img, out / split / "images" / f"mn_{img.name}")
                shutil.copy2(lbl, out / split / "labels" / f"mn_{img.stem}.txt")
                stats[f"mined_{split}"] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: train/images\nval: val/images\n\n"
        "nc: 1\nnames: ['logo']\n")

    n_train = stats["roboflow_train"] + stats["mined_train"]
    n_val = stats["roboflow_val"] + stats["mined_val"]
    print(f"{'source':16s} {'train':>7s} {'val':>7s}")
    print(f"{'roboflow':16s} {stats['roboflow_train']:>7d} {stats['roboflow_val']:>7d}")
    print(f"{'mined':16s} {stats['mined_train']:>7d} {stats['mined_val']:>7d}")
    print(f"{'TOTAL':16s} {n_train:>7d} {n_val:>7d}")
    print(f"\n[merge] wrote {out / 'data.yaml'}")
    (out / "merge_summary.json").write_text(json.dumps(dict(stats), indent=2) + "\n")


if __name__ == "__main__":
    main()
