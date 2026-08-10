#!/usr/bin/env python3
"""Turn whatever art was dropped into input/logos/ into assets synthetic
compositing can actually use, and derive each brand's bar colour.

Why this exists: generate_synthetic.py gives a 3-channel logo a fully opaque
alpha, so an opaque asset gets pasted as a RECTANGLE and the whole block is
labelled `betting_board`. Train on that and the model learns "flat rectangle
with text", which is how this project previously ended up scoring ~0.002
confidence on real boards. Nine of the ten assets supplied here are opaque.

Each file is sorted into one of three kinds, because they need opposite handling:

  bar    a flat brand-coloured tile with a wordmark on it (bet365 green,
         Polymarket blue, theScore navy...). This is ALREADY close to what one
         segment of a real LED board looks like, so it is kept intact as a
         strip texture - removing its background here would throw away the
         brand colour, which is the most useful part.
  mark   a wordmark on a neutral white/black backdrop, or one that already has
         alpha. The backdrop is not the brand's colour, so it is keyed out and
         the mark is kept with transparency to be composited onto a generated
         bar.
  photo  not artwork at all - a stock photograph of a stadium, watermark and
         crowd included. Pasting this would composite an entire scene into the
         training image. Excluded from compositing; still usable as a
         match template.

For every `bar` a colour.json is written (unless one already exists, since a
hand-measured value from real footage beats one inferred from press art), which
is what gives each hidden panel its brand-matched fill.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import load_config

EXTS = {".png", ".webp", ".jpg", ".jpeg", ".bmp"}


def load_rgba(path: Path) -> np.ndarray | None:
    im = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if im is None:
        return None
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    return im


def border_stats(bgr: np.ndarray, ring: int = 4):
    """Mean and spread of the outer ring - a flat backdrop has a tight ring."""
    h, w = bgr.shape[:2]
    ring = max(1, min(ring, h // 4, w // 4))
    px = np.concatenate([
        bgr[:ring].reshape(-1, 3), bgr[-ring:].reshape(-1, 3),
        bgr[:, :ring].reshape(-1, 3), bgr[:, -ring:].reshape(-1, 3),
    ]).astype(np.float32)
    return px.mean(axis=0), float(px.std(axis=0).mean())


def edge_density(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return float((cv2.Canny(g, 80, 200) > 0).mean())


def key_out_background(rgba: np.ndarray, tol: float = 42.0) -> np.ndarray:
    """Flood the backdrop colour inward from the border and make it transparent.

    Flood fill from the edges (rather than thresholding on colour globally) so a
    white letter enclosed by the mark keeps its fill - a plain "white is
    background" rule eats the inside of an O."""
    bgr = rgba[..., :3]
    h, w = bgr.shape[:2]
    bg, _ = border_stats(bgr)
    dist = np.linalg.norm(bgr.astype(np.float32) - bg[None, None, :], axis=2)
    close = (dist <= tol).astype(np.uint8)

    # keep only backdrop-coloured pixels REACHABLE from the border
    ff = np.zeros((h + 2, w + 2), np.uint8)
    ff[1:-1, 1:-1] = 1 - close          # obstacles are non-backdrop pixels
    reach = np.zeros_like(ff)
    for seed in [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]:
        y, x = seed
        if close[y, x]:
            m = ff.copy()
            cv2.floodFill(m, None, (x + 1, y + 1), 2)
            reach |= (m == 2).astype(np.uint8)
    bg_mask = reach[1:-1, 1:-1].astype(bool)

    out = rgba.copy()
    alpha = out[..., 3].astype(np.float32)
    alpha[bg_mask] = 0.0
    # soften the 1px key edge so composites do not show a hard cut
    alpha = cv2.GaussianBlur(alpha, (3, 3), 0)
    out[..., 3] = np.clip(alpha, 0, 255).astype(np.uint8)
    return out


def dominant_bar_color(rgba: np.ndarray) -> np.ndarray:
    """The tile's background colour = the brand's bar colour. Taken as the
    modal colour of the border ring, which is the backdrop by construction."""
    bgr = rgba[..., :3].reshape(-1, 3)
    q = (bgr // 16 * 16).astype(np.int32)
    keys, counts = np.unique(q, axis=0, return_counts=True)
    return keys[int(np.argmax(counts))].astype(np.float32)


def classify(rgba: np.ndarray) -> tuple[str, dict]:
    bgr = rgba[..., :3]
    a = rgba[..., 3]
    transparent = float((a < 250).mean())
    bg, spread = border_stats(bgr)
    dens = edge_density(bgr)
    info = {"transparent_frac": round(transparent, 3),
            "border_spread": round(spread, 1),
            "edge_density": round(dens, 3),
            "border_bgr": [int(v) for v in np.round(bg)]}

    if transparent > 0.02:
        return "mark", info                      # already has usable alpha
    # Edge density, not border spread, separates artwork from photography.
    # Measured on these assets: flat brand tiles score 0.002-0.022 while the
    # stock stadium photographs score 0.094-0.099. Border spread misjudged it -
    # a tile with a thin edge stripe (bet365 27.1, stake 44.2) looked "busy"
    # and both flat tiles were wrongly thrown away as photos.
    if dens > 0.06:
        return "photo", info
    # flat backdrop: brand colour, or a neutral white/black studio backdrop?
    v = float(cv2.cvtColor(np.uint8([[bg]]), cv2.COLOR_BGR2HSV)[0, 0, 1])
    lum = float(bg.mean())
    if v < 40 and (lum > 200 or lum < 40):
        return "mark", info                      # neutral backdrop -> key it out
    return "bar", info


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--logos", default=None, help="default: paths.logos")
    ap.add_argument("--out", default="data/logos_clean")
    ap.add_argument("--write-color-json", action="store_true", default=True)
    args = ap.parse_args()

    src = Path(args.logos) if args.logos else cfg["paths"]["logos"]
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, dict] = {}
    print(f"{'brand':14s} {'file':40s} {'kind':6s} {'transp':>7s} {'spread':>7s} {'edges':>6s}")
    for brand_dir in sorted(d for d in src.iterdir() if d.is_dir()):
        brand = brand_dir.name
        entries = []
        # Per-brand manual override, because some calls cannot be made from
        # pixels. A tight crop of a real board and a stock photo OF a stadium
        # score the same on every cue tried here (dominant-hue fraction 0.41 vs
        # 0.56 - the stadium scores HIGHER, thanks to a red board plus red
        # shirts), so a human look is the only reliable arbiter.
        overrides = {}
        kj = brand_dir / "kinds.json"
        if kj.exists():
            try:
                overrides = json.loads(kj.read_text())
            except ValueError as exc:
                print(f"{brand:14s} ignoring kinds.json: {exc}")
        for f in sorted(brand_dir.iterdir()):
            if f.suffix.lower() not in EXTS:
                continue
            rgba = load_rgba(f)
            if rgba is None:
                print(f"{brand:14s} {f.name[:40]:40s} UNREADABLE")
                continue
            kind, info = classify(rgba)
            if f.name in overrides:
                kind = overrides[f.name]
                info["override"] = True
            dest_dir = out_root / brand
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / (f.stem + ".png")

            if kind == "mark" and info["transparent_frac"] <= 0.02:
                rgba = key_out_background(rgba)
                info["transparent_frac"] = round(float((rgba[..., 3] < 250).mean()), 3)
            if kind != "photo":
                cv2.imwrite(str(dest), rgba)

            # Every `bar` also yields a wordmark-only variant. A flat brand tile
            # is useless as a match template - it is nearly uniform, so it
            # correlates ~0.95 with any flat board region and separates nothing
            # (measured: non-betting regions scored HIGHER than betting ones).
            # Keying the backdrop off leaves the lettering, which is the part
            # that actually identifies the brand.
            mark_path = None
            if kind == "bar":
                keyed = key_out_background(rgba)
                frac = float((keyed[..., 3] < 250).mean())
                if 0.05 < frac < 0.97:
                    mark_path = dest.with_name(dest.stem + "_mark.png")
                    cv2.imwrite(str(mark_path), keyed)
                    info["mark_transparent_frac"] = round(frac, 3)

            entries.append({"source": str(f), "kind": kind,
                            "clean": str(dest) if kind != "photo" else None,
                            "mark": str(mark_path) if mark_path else None, **info})
            print(f"{brand:14s} {f.name[:40]:40s} {kind:6s} "
                  f"{info['transparent_frac']:7.2f} {info['border_spread']:7.1f} "
                  f"{info['edge_density']:6.3f}")

        manifest[brand] = {"files": entries}

        # brand bar colour -> colour.json (never overwrite a measured one)
        bars = [e for e in entries if e["kind"] == "bar"]
        cj = brand_dir / "color.json"
        if args.write_color_json and bars and not cj.exists():
            bgr = bars[0]["border_bgr"]
            cj.write_text(json.dumps(
                {"bgr": bgr,
                 "_note": "auto-derived from the brand tile's background by "
                          "scripts/prepare_logos.py; replace with a value measured "
                          "off real footage if the panel reads wrong"}, indent=2) + "\n")
            print(f"{'':14s} -> wrote {cj} bgr={bgr}")

    (out_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    kinds = [e["kind"] for v in manifest.values() for e in v["files"]]
    print(f"\n[logos] bars={kinds.count('bar')} marks={kinds.count('mark')} "
          f"photos_excluded={kinds.count('photo')}")
    print(f"[logos] wrote {out_root / 'manifest.json'}")
    usable = {b: sum(1 for e in v["files"] if e["kind"] != "photo") for b, v in manifest.items()}
    empty = [b for b, n in usable.items() if n == 0]
    if empty:
        print(f"[logos] WARNING no usable compositing asset for: {', '.join(empty)}")


if __name__ == "__main__":
    main()
