#!/usr/bin/env python3
"""Generate synthetic training frames by pasting brand logos onto real
background frames.

Brands are auto-discovered from data/logos/<brand>/*.png at runtime — there
is never a hardcoded brand list. Adding a new brand later means creating
data/logos/<newbrand>/, dropping transparent PNGs in it, and re-running this
script; no code changes are needed. Backgrounds are auto-discovered from
every frame under data/frames/**.

Each pasted logo is randomly assigned one of two "modes" that mimic the two
real-world placements the model needs to generalize across:

  * board   -> rotated + perspective-warped, placed in the lower half of the
               frame (sideline/pitch-level boards viewed at an angle),
               labeled class `betting_board`. Occasionally partially cropped
               to mimic a player standing in front of it.
  * overlay -> axis-aligned (small/no rotation, no perspective), high
               opacity, placed near a corner or edge (broadcast graphics
               burned into the feed), labeled class `betting_overlay`.

Output goes to data/synthetic/{images,labels}/{train,val}/ with a
deterministic seeded split, ready for assemble_dataset.py to merge into
data/dataset/.
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from common import IMAGE_EXTS, die, load_config

CLASS_BOARD = 1     # data/data.yaml: 1 -> betting_board
CLASS_OVERLAY = 2   # data/data.yaml: 2 -> betting_overlay

# Generic slogan words for composed strips (Fix 1: real boards like
# "Betano <> CONFIA <> Betano" are ONE physical strip with a slogan segment
# between repeated logos, not two separate ads - training must match the
# whole-strip labeling convention in README.md #2, not per-logo fragments).
STRIP_SLOGANS = ["BET NOW", "PLAY SAFE", "WIN BIG", "JOIN NOW", "LIVE ODDS", "18+"]


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def discover_brands(logos_dir: Path) -> dict[str, list[Path]]:
    brands = {}
    for sub in sorted(p for p in logos_dir.iterdir() if p.is_dir()):
        files = sorted(f for f in sub.iterdir() if f.suffix.lower() in IMAGE_EXTS)
        if files:
            brands[sub.name] = files
    return brands


def discover_backgrounds(frames_dir: Path) -> list[Path]:
    return sorted(p for p in frames_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS)


# ---------------------------------------------------------------------------
# Logo loading / transform
# ---------------------------------------------------------------------------

def load_logo_rgba(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Could not read logo: {path}")
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA)
    elif img.shape[2] == 3:
        alpha = np.full(img.shape[:2], 255, dtype=np.uint8)
        img = np.dstack([img, alpha])
    return img


def resize_logo(logo: np.ndarray, target_w: int) -> np.ndarray:
    h, w = logo.shape[:2]
    target_w = max(4, target_w)
    scale = target_w / w
    target_h = max(4, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    return cv2.resize(logo, (target_w, target_h), interpolation=interp)


def rotate_and_warp(logo: np.ndarray, rotation_deg: float, perspective_strength: float,
                     rng: random.Random) -> np.ndarray:
    """Apply rotation + optional perspective warp via a single homography,
    returning a new (tight-cropped) RGBA canvas containing the result."""
    h, w = logo.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

    angle = rng.uniform(-rotation_deg, rotation_deg)
    theta = np.radians(angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = np.array([rot @ (pt - [cx, cy]) + [cx, cy] for pt in src])

    if perspective_strength > 0:
        diag = np.hypot(w, h)
        jitter = diag * perspective_strength
        offsets = np.array([[rng.uniform(-jitter, jitter), rng.uniform(-jitter, jitter)] for _ in range(4)])
        dst = rotated + offsets
    else:
        dst = rotated

    min_xy = dst.min(axis=0)
    dst -= min_xy
    tw, th = int(np.ceil(dst[:, 0].max())), int(np.ceil(dst[:, 1].max()))
    tw, th = max(tw, 4), max(th, 4)

    H = cv2.getPerspectiveTransform(src.astype(np.float32), dst.astype(np.float32))
    warped = cv2.warpPerspective(
        logo, H, (tw, th), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0),
    )
    return warped


def build_strip_patch(logo: np.ndarray, rng: random.Random) -> np.ndarray:
    """Compose [logo | slogan text | logo] on one solid colored bar, so a
    fraction of board examples teach the model to see the WHOLE contiguous
    strip as a single region - matching the "one polygon per strip, not per
    logo" labeling convention - instead of only ever seeing isolated logos.
    Real boards commonly repeat a brand's mark either side of its slogan
    (e.g. "Betano <> CONFIA <> Betano"); this mimics that layout generically,
    without hardcoding any specific brand's wordmark."""
    lh, lw = logo.shape[:2]
    gap = max(2, int(lw * rng.uniform(0.15, 0.35)))
    slogan_w = max(8, int(lw * rng.uniform(0.7, 1.1)))
    total_w = lw + gap + slogan_w + gap + lw

    opaque = logo[..., 3] > 128
    bar_color = (logo[..., :3][opaque].mean(axis=0) if opaque.any()
                else np.array([40, 40, 200], dtype=np.float32))
    alt_color = (np.array([250, 250, 250], dtype=np.float32) if bar_color.mean() < 128
                else np.array([25, 25, 25], dtype=np.float32))

    strip = np.zeros((lh, total_w, 4), dtype=np.uint8)
    strip[..., :3] = bar_color.astype(np.uint8)
    strip[..., 3] = 255

    x = 0
    strip[:, x:x + lw] = logo
    x += lw + gap
    strip[:, x:x + slogan_w, :3] = alt_color.astype(np.uint8)
    strip[:, x:x + slogan_w, 3] = 255
    text_color = tuple(int(c) for c in bar_color) if alt_color.mean() > 128 else (255, 255, 255)
    font_scale = max(0.3, lh / 55.0)
    cv2.putText(strip, rng.choice(STRIP_SLOGANS), (int(slogan_w * 0.06), int(lh * 0.65)),
                cv2.FONT_HERSHEY_DUPLEX, font_scale, text_color, max(1, lh // 30), cv2.LINE_AA)
    x += slogan_w + gap
    strip[:, x:x + lw] = logo
    return strip


def apply_partial_occlusion(logo: np.ndarray, max_frac: float, rng: random.Random) -> np.ndarray:
    """Zero out the alpha channel in a random strip to mimic a player
    cropping part of the logo out of view."""
    h, w = logo.shape[:2]
    logo = logo.copy()
    side = rng.choice(["left", "right", "top", "bottom"])
    frac = rng.uniform(0.1, max_frac)
    if side == "left":
        logo[:, : int(w * frac), 3] = 0
    elif side == "right":
        logo[:, w - int(w * frac):, 3] = 0
    elif side == "top":
        logo[: int(h * frac), :, 3] = 0
    else:
        logo[h - int(h * frac):, :, 3] = 0
    return logo


def jitter_brightness_contrast(logo: np.ndarray, brightness_range: float, contrast_range: float,
                                rng: random.Random) -> np.ndarray:
    logo = logo.copy()
    contrast = 1.0 + rng.uniform(-contrast_range, contrast_range)
    brightness = rng.uniform(-brightness_range, brightness_range) * 127
    rgb = logo[..., :3].astype(np.float32)
    rgb = rgb * contrast + brightness
    logo[..., :3] = np.clip(rgb, 0, 255).astype(np.uint8)
    return logo


def maybe_blur(logo: np.ndarray, prob: float, max_kernel: int, rng: random.Random) -> np.ndarray:
    if rng.random() > prob or max_kernel < 3:
        return logo
    k = rng.choice(range(3, max_kernel + 1, 2))
    logo = logo.copy()
    logo[..., :3] = cv2.GaussianBlur(logo[..., :3], (k, k), 0)
    return logo


def scale_opacity(logo: np.ndarray, factor: float) -> np.ndarray:
    logo = logo.copy()
    logo[..., 3] = np.clip(logo[..., 3].astype(np.float32) * factor, 0, 255).astype(np.uint8)
    return logo


# ---------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------

def sample_board_position(w: int, h: int, tw: int, th: int, rng: random.Random) -> tuple[int, int]:
    y_hi = h - th  # highest valid top-left y so the patch still fits in the frame
    y_lo = min(int(h * 0.5), y_hi)  # prefer the lower half, but never past y_hi
    x_hi = max(0, w - tw)
    x0 = rng.randint(0, x_hi)
    y0 = rng.randint(y_lo, y_hi)
    return x0, y0


def sample_overlay_position(w: int, h: int, tw: int, th: int, margin_frac: float,
                             rng: random.Random) -> tuple[int, int]:
    margin_x = max(tw, int(w * margin_frac))
    margin_y = max(th, int(h * margin_frac))
    horiz = rng.choice(["left", "right"])
    vert = rng.choice(["top", "bottom"])
    x_hi = max(0, min(margin_x, w) - tw)
    y_hi = max(0, min(margin_y, h) - th)
    x0 = rng.randint(0, x_hi) if horiz == "left" else max(0, w - tw - rng.randint(0, x_hi))
    y0 = rng.randint(0, y_hi) if vert == "top" else max(0, h - th - rng.randint(0, y_hi))
    return x0, y0


# ---------------------------------------------------------------------------
# Compositing + label extraction
# ---------------------------------------------------------------------------

def extract_polygon(alpha: np.ndarray, min_visible_area_frac: float,
                     original_area: float) -> list[tuple[float, float]] | None:
    mask = (alpha > 20).astype(np.uint8) * 255
    visible_frac = (mask > 0).sum() / max(original_area, 1)
    if visible_frac < min_visible_area_frac:
        return None
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < 20:
        return None
    epsilon = 0.01 * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    if len(approx) < 3:
        return None
    return [(float(pt[0][0]), float(pt[0][1])) for pt in approx]


def composite_logo(bg: np.ndarray, patch: np.ndarray, x0: int, y0: int) -> None:
    h, w = patch.shape[:2]
    roi = bg[y0:y0 + h, x0:x0 + w]
    alpha = (patch[..., 3:4].astype(np.float32)) / 255.0
    blended = patch[..., :3].astype(np.float32) * alpha + roi.astype(np.float32) * (1 - alpha)
    bg[y0:y0 + h, x0:x0 + w] = blended.astype(np.uint8)


def simulate_jpeg_artifacts(img: np.ndarray, quality: int) -> np.ndarray:
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        return img
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


# ---------------------------------------------------------------------------
# Main generation loop
# ---------------------------------------------------------------------------

def generate_one(idx: int, bg_path: Path, brands: dict[str, list[Path]], scfg: dict,
                  rng: random.Random) -> tuple[np.ndarray, list[str]] | None:
    bg = cv2.imread(str(bg_path), cv2.IMREAD_COLOR)
    if bg is None:
        return None
    h, w = bg.shape[:2]
    labels: list[str] = []

    n_logos = rng.randint(scfg["logos_per_image_min"], scfg["logos_per_image_max"])
    brand_names = list(brands.keys())

    for _ in range(n_logos):
        brand = rng.choice(brand_names)
        logo_path = rng.choice(brands[brand])
        try:
            logo = load_logo_rgba(logo_path)
        except ValueError:
            continue

        is_board = rng.random() < scfg["board_prob"]

        target_w = int(w * rng.uniform(scfg["scale_min"], scfg["scale_max"]))
        logo = resize_logo(logo, target_w)

        if is_board and rng.random() < scfg.get("strip_prob", 0.0):
            logo = build_strip_patch(logo, rng)
        original_area = float(logo.shape[0] * logo.shape[1])

        if is_board:
            rotation = scfg["rotation_deg"]
            perspective = scfg["perspective_warp_strength"]
        else:
            rotation = min(2.0, scfg["rotation_deg"])
            perspective = 0.0

        patch = rotate_and_warp(logo, rotation, perspective, rng)
        patch = jitter_brightness_contrast(
            patch, scfg["brightness_jitter"], scfg["contrast_jitter"], rng
        )
        patch = maybe_blur(patch, scfg["blur_prob"], scfg["blur_kernel_max"], rng)
        patch = scale_opacity(patch, rng.uniform(0.55, 1.0) if is_board else rng.uniform(0.85, 1.0))

        if rng.random() < scfg["partial_occlusion_prob"]:
            patch = apply_partial_occlusion(patch, scfg["partial_occlusion_max_frac"], rng)

        th, tw = patch.shape[:2]
        if tw >= w or th >= h:
            continue

        if is_board:
            x0, y0 = sample_board_position(w, h, tw, th, rng)
        else:
            x0, y0 = sample_overlay_position(w, h, tw, th, scfg["overlay_zone_margin_frac"], rng)

        polygon = extract_polygon(patch[..., 3], scfg["min_visible_area_frac"], original_area)
        if polygon is None:
            continue

        composite_logo(bg, patch, x0, y0)

        class_id = CLASS_BOARD if is_board else CLASS_OVERLAY
        norm = []
        for (px, py) in polygon:
            nx = (x0 + px) / w
            ny = (y0 + py) / h
            norm.extend([f"{np.clip(nx, 0, 1):.6f}", f"{np.clip(ny, 0, 1):.6f}"])
        labels.append(" ".join([str(class_id)] + norm))

    if scfg["jpeg_artifact_prob"] > 0 and rng.random() < scfg["jpeg_artifact_prob"]:
        q = rng.randint(scfg["jpeg_quality_min"], scfg["jpeg_quality_max"])
        bg = simulate_jpeg_artifacts(bg, q)

    return bg, labels


def main():
    cfg = load_config()
    p = cfg["paths"]
    scfg = dict(cfg["synthetic"])

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-images", type=int, default=scfg["num_images"])
    ap.add_argument("--val-ratio", type=float, default=scfg["val_ratio"])
    ap.add_argument("--seed", type=int, default=scfg["seed"])
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    logos_dir = p["logos"]
    frames_dir = p["frames"]
    out_dir = p["synthetic"]

    brands = discover_brands(logos_dir)
    if not brands:
        die(
            f"No brand logo folders with images found under '{logos_dir}'. "
            "Create data/logos/<brand>/ and drop transparent-bg PNGs inside."
        )
    backgrounds = discover_backgrounds(frames_dir)
    if not backgrounds:
        die(
            f"No background frames found under '{frames_dir}'. "
            "Run scripts/extract_frames.py first."
        )

    print(f"Discovered {len(brands)} brand(s): {', '.join(brands)}")
    print(f"Discovered {len(backgrounds)} background frame(s)")

    # Every run must be a clean, self-contained regeneration: the RNG draw
    # sequence (and therefore which background/placement each index gets)
    # depends on --num-images, so leftover files from a previous run with a
    # different count can end up with the SAME filename but DIFFERENT
    # content in train vs val - real train/val leakage. Clearing first keeps
    # each run deterministic and self-consistent instead of layering on
    # whatever happened to be on disk already.
    for split in ("train", "val"):
        img_dir = out_dir / "images" / split
        lbl_dir = out_dir / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lbl_dir.exists():
            shutil.rmtree(lbl_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    indices = list(range(args.num_images))
    rng.shuffle(indices)
    n_val = int(args.num_images * args.val_ratio)
    val_set = set(indices[:n_val])

    written = 0
    for i in tqdm(range(args.num_images), desc="Generating synthetic frames"):
        bg_path = backgrounds[rng.randrange(len(backgrounds))]
        result = generate_one(i, bg_path, brands, scfg, rng)
        if result is None:
            continue
        img, labels = result

        split = "val" if i in val_set else "train"
        stem = f"synth_{i:06d}"
        img_path = out_dir / "images" / split / f"{stem}.jpg"
        lbl_path = out_dir / "labels" / split / f"{stem}.txt"

        cv2.imwrite(str(img_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
        lbl_path.write_text("\n".join(labels) + ("\n" if labels else ""))
        written += 1

    print(f"Done. Wrote {written} synthetic images (+labels) to {out_dir}")
    print("Re-run this script any time new brand folders or background frames appear.")


if __name__ == "__main__":
    main()
