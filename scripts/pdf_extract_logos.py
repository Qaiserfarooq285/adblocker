#!/usr/bin/env python3
"""Extract brand logo images from the gambling-sites PDF.

The PDF is a register of licensed operators: each row carries a corporate name,
a logo image, and a set of product tags.  We want the logos, each filed under a
brand slug.  The corporate name ("Hillside (International Sports) ENC") is
rarely the brand ("bet365"), so the slug comes from OCR of the wordmark itself
and only falls back to the nearest page text when OCR reads nothing.

Writes:
  input/logos/<brand>/logo_XX.png   RGBA, background keyed out where flat
  data/pdf_logos/index.json         per-asset provenance + OCR text
  data/pdf_logos/brand_aliases.json {slug: [alias, ...]} for the OCR matcher
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import fitz
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

MIN_SIDE = 24
MIN_AREA = 900

# Words that appear on the register page but are not brands.
STOP = {
    "casino", "sports", "betting", "sport", "bet", "limited", "ltd", "inc",
    "corp", "gaming", "games", "group", "international", "holdings", "canada",
    "ontario", "enc", "bv", "nv", "plc", "llc", "com", "ca", "online", "live",
    "the", "and", "of", "operator", "licensed", "logo",
}


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "", text.lower())
    return s


def brand_from_text(text: str) -> str:
    """Reduce an OCR string to a plausible brand slug."""
    if not text:
        return ""
    text = text.lower()
    text = re.sub(r"https?://|www\.", " ", text)
    text = re.sub(r"\.(com|ca|net|org|io|bet)\b", " ", text)
    # Keep the longest run of letters/digits that is not a stopword.
    parts = [p for p in re.split(r"[^a-z0-9]+", text) if p]
    parts = [p for p in parts if p not in STOP and len(p) >= 3]
    if not parts:
        return ""
    # Brands are frequently written as two touching tokens (bet + 365).
    joined = "".join(parts[:2]) if len(parts) > 1 and len(parts[0]) <= 6 else parts[0]
    return slugify(joined)[:24]


def key_out_background(bgr: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """Return RGBA with a flat surrounding backdrop made transparent."""
    h, w = bgr.shape[:2]
    if alpha is None:
        alpha = np.full((h, w), 255, np.uint8)
    border = np.concatenate([
        bgr[0, :].reshape(-1, 3), bgr[-1, :].reshape(-1, 3),
        bgr[:, 0].reshape(-1, 3), bgr[:, -1].reshape(-1, 3),
    ])
    bg = np.median(border, axis=0)
    spread = float(np.median(np.abs(border - bg)))
    if spread < 12.0:  # backdrop really is flat
        dist = np.linalg.norm(bgr.astype(np.float32) - bg, axis=2)
        fg = (dist > max(28.0, spread * 4 + 20)).astype(np.uint8) * 255
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        # Only trust the key-out if it keeps a sane amount of the tile.
        frac = float(fg.mean()) / 255.0
        if 0.02 < frac < 0.85:
            alpha = cv2.min(alpha, fg)
    return np.dstack([bgr, alpha])


def trim(rgba: np.ndarray) -> np.ndarray:
    ys, xs = np.where(rgba[:, :, 3] > 8)
    if ys.size == 0:
        return rgba
    return rgba[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def load_ocr():
    try:
        from rapidocr_onnxruntime import RapidOCR
        return RapidOCR()
    except Exception as exc:  # pragma: no cover - optional dependency
        print(f"[warn] OCR unavailable ({exc}); falling back to page text")
        return None


def ocr_text(engine, rgba: np.ndarray) -> tuple[str, float]:
    if engine is None:
        return "", 0.0
    img = rgba[:, :, :3].copy()
    a = rgba[:, :, 3]
    img[a < 8] = 255
    scale = max(1.0, 160.0 / max(1, img.shape[0]))
    if scale > 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    best_txt, best_conf = "", 0.0
    for variant in (img, 255 - img):
        try:
            res, _ = engine(variant)
        except Exception:
            continue
        if not res:
            continue
        txt = " ".join(str(r[1]) for r in res)
        conf = float(np.mean([float(r[2]) for r in res]))
        if conf > best_conf:
            best_txt, best_conf = txt, conf
    return best_txt.strip(), best_conf


def page_text_near(page, bbox) -> str:
    """Text whose block overlaps the image band vertically."""
    y0, y1 = bbox[1], bbox[3]
    hits = []
    for blk in page.get_text("blocks"):
        bx0, by0, bx1, by1, txt = blk[0], blk[1], blk[2], blk[3], blk[4]
        if by1 < y0 - 30 or by0 > y1 + 30:
            continue
        hits.append((abs((by0 + by1) / 2 - (y0 + y1) / 2), txt.strip()))
    hits.sort()
    return " ".join(t for _, t in hits[:2])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=None)
    ap.add_argument("--out", default=str(ROOT / "input" / "logos"))
    ap.add_argument("--meta", default=str(ROOT / "data" / "pdf_logos"))
    args = ap.parse_args()

    pdf = args.pdf
    if pdf is None:
        cands = sorted(ROOT.glob("input/**/*.pdf"))
        if not cands:
            print("no PDF found under input/")
            return 1
        pdf = str(cands[0])
    print(f"[pdf] {pdf}")

    out_root = Path(args.out)
    meta_root = Path(args.meta)
    raw_root = meta_root / "raw"
    out_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)

    engine = load_ocr()
    doc = fitz.open(pdf)
    index = []
    per_brand: dict[str, int] = defaultdict(int)
    aliases: dict[str, set[str]] = defaultdict(set)
    unnamed = 0

    for pno, page in enumerate(doc):
        for imeta in page.get_images(full=True):
            xref = imeta[0]
            try:
                info = doc.extract_image(xref)
            except Exception:
                continue
            buf = np.frombuffer(info["image"], np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            if img.ndim == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            alpha = None
            if img.shape[2] == 4:
                alpha = img[:, :, 3]
                img = img[:, :, :3]
            h, w = img.shape[:2]
            if min(h, w) < MIN_SIDE or h * w < MIN_AREA:
                continue

            rgba = trim(key_out_background(img, alpha))
            if min(rgba.shape[:2]) < MIN_SIDE:
                continue

            txt, conf = ocr_text(engine, rgba)
            brand = brand_from_text(txt)
            source = "ocr"
            if not brand:
                try:
                    rects = page.get_image_rects(xref)
                    near = page_text_near(page, rects[0]) if rects else ""
                except Exception:
                    near = ""
                brand = brand_from_text(near)
                source = "pagetext"
            if not brand:
                unnamed += 1
                brand = f"unknown_{unnamed:03d}"
                source = "none"

            per_brand[brand] += 1
            bdir = out_root / brand
            bdir.mkdir(parents=True, exist_ok=True)
            name = f"logo_{per_brand[brand]:02d}.png"
            cv2.imwrite(str(bdir / name), rgba)
            cv2.imwrite(str(raw_root / f"p{pno:03d}_x{xref}.png"), img)

            if txt:
                cleaned = re.sub(r"\s+", " ", txt.lower()).strip()
                if 2 <= len(cleaned) <= 40:
                    aliases[brand].add(cleaned)
                    aliases[brand].add(slugify(cleaned))
            aliases[brand].add(brand)

            index.append({
                "page": pno, "xref": xref, "brand": brand, "file": str(bdir / name),
                "ocr_text": txt, "ocr_conf": round(conf, 3), "name_source": source,
                "w": int(rgba.shape[1]), "h": int(rgba.shape[0]),
            })

    meta_root.mkdir(parents=True, exist_ok=True)
    (meta_root / "index.json").write_text(json.dumps(index, indent=2))
    (meta_root / "brand_aliases.json").write_text(
        json.dumps({k: sorted(v) for k, v in aliases.items()}, indent=2))

    named = sum(1 for r in index if r["name_source"] != "none")
    print(f"[pdf] extracted {len(index)} logos across {len(per_brand)} brands "
          f"({named} named, {unnamed} unnamed)")
    for b, n in sorted(per_brand.items(), key=lambda kv: -kv[1])[:40]:
        print(f"    {b:<24} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
