#!/usr/bin/env python3
"""Turn generic "advertising board" drafts into BETTING-board labels by reading
the wordmark.

The zero-shot pass flags every ad board it can see, because an open-vocabulary
detector has no concept of gambling - on this footage that is ~8,100 polygons
across five sports, most of them Coca-Cola, Scotiabank, Bud Light and the rest.
Something has to decide which ones are betting ads.

Cues that were considered and rejected:

  brand colour   each brand does have a distinctive bar colour, but so do
                 plenty of innocent sponsors - Coca-Cola red sits on top of
                 Betano red, Bud Light blue on top of FanDuel blue. Colour
                 alone cannot separate them.
  template match the supplied art is clean frontal press-kit rendering while
                 the boards are small, perspective-warped and motion-blurred.
                 This project already learned that lesson the other way round:
                 the Betano bootstrap that worked used templates harvested from
                 the footage itself, not press logos.

What does separate them is the text on the board, so each drafted region is
OCR'd and matched against brand aliases plus generic gambling vocabulary. A
board whose text says "bet365" is a betting board no matter what sport it is in
or how the board is shaped.

    python scripts/filter_betting_boards.py --review data/prelabel_review/sports5
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from difflib import SequenceMatcher
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

from common import load_config

CLASS_BOARD, CLASS_OVERLAY = 1, 2

# Brand aliases. Keys must match the folder names under input/logos/ where they
# correspond; extra entries cost nothing and catch brands with no logo supplied,
# which is the whole point of keeping the model's classes brand-agnostic.
BRANDS: dict[str, tuple[str, ...]] = {
    "bet365": ("bet365", "bet 365", "bet36s", "bet36"),
    "betano": ("betano", "confia", "betan"),
    "fanduel": ("fanduel", "fan duel", "fandue"),
    "polymarket": ("polymarket", "poly market", "polymarke"),
    "stake": ("stake.com", "stake com", "stake"),
    # NOT bare "thescore": theScore is a sports broadcaster whose name sits in
    # the scorebug, while theScore BET is the betting brand. Requiring "bet"
    # keeps the board and rejects the graphic.
    "the score bet": ("thescore bet", "thescorebet", "score bet"),
    "draftkings": ("draftkings", "draft kings", "draftking"),
    "betmgm": ("betmgm", "bet mgm"),
    "caesars": ("caesars sportsbook",),
    "pointsbet": ("pointsbet", "points bet"),
    "bet99": ("bet99", "bet 99"),
    "pinnacle": ("pinnacle",),
    "betway": ("betway", "bet way"),
    "unibet": ("unibet",),
    "bwin": ("bwin",),
    "888sport": ("888sport", "888 sport"),
    "fanatics": ("fanatics sportsbook",),
    # Brands found by reading this footage rather than from the logo folders -
    # keeping the model's classes brand-agnostic is what lets it hide these too.
    "betty": ("bettyca", "betty ca", "betty"),
    "bodog": ("bodog",),
    "sports interaction": ("sports interaction", "sportsinteraction"),
    "proline": ("proline", "pro line"),
    "northstar": ("northstar bets", "northstarbets"),
    "leovegas": ("leovegas", "leo vegas"),
    "casumo": ("casumo",),
    "coolbet": ("coolbet",),
    # "powerplay" REMOVED. PowerPlay is a licensed operator and appears in the
    # PDF register, but "POWER PLAY" is ordinary hockey terminology printed in
    # the scorebug during every penalty. Mining it labelled 392 boxes on
    # broadcast graphics, which taught the model that a scorebug is a betting
    # board - it fires there at conf 0.88, and the "floating box" in the hockey
    # render is that panel resizing with the graphic. A brand whose name
    # collides with the sport's own vocabulary costs far more than it wins.
}


# Registered operators whose names collide with the vocabulary of the sport
# itself. They are genuinely licensed gambling brands - that is why they are in
# the PDF - but on this footage the collision costs far more than the brand wins.
PDF_BRAND_BLOCK = {"powerplay"}
# ...and alias fragments that must never enter the table from the PDF harvest,
# for the same reason. "thescore" alone is the broadcaster; "thescore bet" is
# the sportsbook and stays.
ALIAS_BLOCK = ("power play", "powerplay")


def _load_pdf_aliases(path: str = "data/pdf_logos/brand_aliases.json") -> None:
    """Fold in brands harvested from the operator-list PDF supplied alongside
    the footage.

    That document is a register of licensed gambling operators, so it is the
    authoritative answer to "which brands count as betting" - far better than a
    list written from memory. Its logo artwork was OCR'd to recover the brand
    names, which is how 58 canonical brands and 112 aliases arrive here without
    anybody typing them. Missing a brand costs recall directly: an unrecognised
    board is silently dropped from training as if it were a Coca-Cola ad."""
    p = Path(path)
    if not p.exists():
        return
    try:
        extra = json.loads(p.read_text())
    except ValueError:
        return
    for brand, aliases in extra.items():
        if brand in PDF_BRAND_BLOCK:
            continue
        cur = set(BRANDS.get(brand, ()))
        cur.update(a for a in aliases
                   if len(a) >= 4 and not any(b in a for b in ALIAS_BLOCK)
                   and a.strip() not in ("thescore", "the score"))
        if cur:
            BRANDS[brand] = tuple(sorted(cur))


_load_pdf_aliases()

# Words that are one OCR slip away from a brand but are not one. Hockey is full
# of "skate" and basketball of "state"; both sit within fuzzy range of "stake".
FUZZY_BLOCK = ("skate", "skates", "state", "states", "steak", "stack", "stage",
               "score", "scores", "beer", "best", "bell", "shots", "shoot",
               "stats", "stand", "stars", "store", "stone", "start", "steal",
               "sport", "sports", "point", "points", "power", "party",
               # the broadcaster, not the sportsbook: "thescore" fuzzily scores
               # 0.84 against the alias "thescorebet", so without this the name
               # in the scorebug reads as a betting brand
               "thescore", "thescoreapp")

# Generic gambling vocabulary, for a betting board whose brand is unknown.
GENERIC = ("sportsbook", "sports book", "betting", "bet now", "free bet",
           "odds boost", "parlay", "wager", "gamble", "gambling", "casino",
           "bet responsibly", "gamble responsibly", "19+", "21+")

# Explicit non-betting sponsors seen on this footage. Not strictly required -
# absence of a betting match already rejects them - but recording them makes the
# rejection auditable and gives assemble_dataset a hard-negative list.
NEGATIVES = ("coca-cola", "coca cola", "cocacola", "scotiabank", "sportchek",
             "bud light", "budlight", "monster", "crypto.com", "doordash",
             "kraken", "marriott", "aramco", "globant", "verizon", "gatorade",
             "tangerine", "rogers", "canadian tire", "jackpot city")


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9 ]+", "", s.lower()).strip()


def match_brand(texts: list[str]) -> tuple[str | None, str]:
    """(brand, matched_text) or (None, joined_text).

    Exact substring is not enough on real OCR output. Measured on this footage
    the reader returns 'Stnke' for Stake, 'POINImarket' and 'POyMarket' for
    Polymarket, and splits theScore Bet into three tokens - 'the', 'score',
    'BE' - so a contiguous 'score bet' never matches. Hence: match against the
    joined text, the space-stripped joined text, AND fuzzily against individual
    tokens, with a blocklist for the near-misses that a fuzzy test would
    otherwise turn into false positives ('skate' in hockey is 0.8 similar to
    'stake')."""
    toks = [norm(t) for t in texts if norm(t)]
    joined = " ".join(toks)
    squashed = joined.replace(" ", "")
    if not joined:
        return None, ""

    def fuzzy_hit(alias: str) -> bool:
        an = norm(alias)
        if not an:
            return False
        if an in joined or an.replace(" ", "") in squashed:
            return True
        if len(an) < 5:
            return False                      # too short to fuzz safely
        if " " in an:
            # Multi-word aliases must match exactly. Fuzzily, the single token
            # "thescore" scores 0.80 against "thescore bet" - so the
            # broadcaster's name in the scorebug would pass as the sportsbook,
            # which is the contamination this whole guard exists to stop.
            return False
        for t in toks:
            if len(t) < 5 or t in FUZZY_BLOCK:
                continue
            # 0.80 is one substitution in a five-letter word, which is exactly the
            # error rate seen here ("Stnke" for Stake at 0.80, "POINImarket" for
            # Polymarket at 0.76). Tighter than this and real brands are missed;
            # looser and FUZZY_BLOCK stops carrying the load.
            if SequenceMatcher(None, t, an).ratio() >= 0.80:
                return True
        return False

    for brand, aliases in BRANDS.items():
        for a in aliases:
            if fuzzy_hit(a):
                return brand, a
    for g in GENERIC:
        if fuzzy_hit(g):
            return "generic", g
    return None, joined


def poly_from_line(line: str, w: int, h: int):
    parts = line.split()
    cls = int(parts[0])
    v = np.array([float(x) for x in parts[1:]], np.float32)
    if v.size < 6:
        return cls, None
    pts = v.reshape(-1, 2) * np.array([w, h], np.float32)
    return cls, pts


def crop_for_ocr(img: np.ndarray, pts: np.ndarray, min_h: int = 56,
                 max_w: int = 1600) -> np.ndarray | None:
    """Axis-aligned crop around the polygon, upscaled so the wordmark is legible.
    Board text is often only ~12px tall in frame, which no OCR will read."""
    x0 = int(max(0, np.floor(pts[:, 0].min()) - 3))
    x1 = int(min(img.shape[1], np.ceil(pts[:, 0].max()) + 3))
    y0 = int(max(0, np.floor(pts[:, 1].min()) - 3))
    y1 = int(min(img.shape[0], np.ceil(pts[:, 1].max()) + 3))
    if x1 - x0 < 12 or y1 - y0 < 6:
        return None
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    scale = max(1.0, min_h / h)
    if w * scale > max_w:
        scale = max_w / w
    if scale > 1.01:
        crop = cv2.resize(crop, (max(1, int(w * scale)), max(1, int(h * scale))),
                          interpolation=cv2.INTER_CUBIC)
    return crop


def _grad(gray: np.ndarray) -> np.ndarray:
    """Normalised gradient magnitude - the lettering's strokes, without the
    flat fill or the exposure."""
    g = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    m = cv2.magnitude(gx, gy)
    mx = float(m.max())
    return (m / mx * 255.0).astype(np.uint8) if mx > 1e-6 else m.astype(np.uint8)


_TEMPLATES: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] | None = None


def load_brand_templates(clean_root: Path = Path("data/logos_clean"),
                         canon_w: int = 200):
    """Grayscale templates per brand, from the assets prepare_logos.py kept.

    Both kinds are useful and for different reasons: a `bar` matches a whole
    board segment (brand-coloured bar plus wordmark, which is what a real board
    segment looks like), while a `mark` matches just the wordmark and carries an
    alpha channel that becomes a matchTemplate mask so the backdrop is ignored."""
    out: dict[str, list[tuple[np.ndarray, np.ndarray | None]]] = {}
    manifest = clean_root / "manifest.json"
    if not manifest.exists():
        return out
    data = json.loads(manifest.read_text())
    for brand, info in data.items():
        tpls = []
        for e in info.get("files", []):
            if e.get("kind") == "photo":
                continue
            # WORDMARK only. The flat `bar` tile is deliberately not used as a
            # template: being near-uniform it correlates ~0.95 with any flat
            # board region, and measured on this footage the non-betting regions
            # scored HIGHER than the betting ones. Only the lettering carries
            # brand-identifying structure.
            src = e.get("mark") or (e.get("clean") if e.get("kind") == "mark" else None)
            if not src:
                continue
            im = cv2.imread(src, cv2.IMREAD_UNCHANGED)
            if im is None:
                continue
            alpha = im[..., 3] if (im.ndim == 3 and im.shape[2] == 4) else None
            bgr = im[..., :3] if im.ndim == 3 else cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
            h, w = bgr.shape[:2]
            if w < 8 or h < 8:
                continue
            sc = canon_w / w
            g = _grad(cv2.cvtColor(cv2.resize(bgr, (canon_w, max(8, int(h * sc)))),
                                   cv2.COLOR_BGR2GRAY))
            m = None
            if alpha is not None:
                m = cv2.resize(alpha, (g.shape[1], g.shape[0]))
                m = (m > 40).astype(np.uint8) * 255
                if float((m > 0).mean()) < 0.03:
                    m = None
            tpls.append((g, m))
        if tpls:
            out[brand] = tpls
    return out


def template_scores(crop: np.ndarray) -> dict[str, float]:
    """Best normalised-correlation score per brand over several template scales.

    The wordmark occupies an unknown fraction of a detected board, so the
    template is tried at a range of widths relative to the crop rather than
    assuming it fills it."""
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = load_brand_templates()
    if not _TEMPLATES:
        return {}
    # Match on gradient magnitude, not intensity. Board lighting, LED
    # brightness and exposure vary wildly; the STROKES of the lettering are
    # what stay constant, and gradients keep those while discarding the flat
    # fill that made plain intensity matching saturate.
    gray = _grad(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY))
    ch, cw = gray.shape[:2]
    best: dict[str, float] = {}
    for brand, tpls in _TEMPLATES.items():
        b = 0.0
        for g, m in tpls:
            th, tw = g.shape[:2]
            for frac in (0.35, 0.5, 0.7, 0.9):
                w = max(8, int(cw * frac))
                h = max(8, int(th * (w / tw)))
                if h >= ch or w >= cw:
                    continue
                t = cv2.resize(g, (w, h))
                mm = cv2.resize(m, (w, h)) if m is not None else None
                try:
                    if mm is not None:
                        r = cv2.matchTemplate(gray, t, cv2.TM_CCORR_NORMED, mask=mm)
                    else:
                        r = cv2.matchTemplate(gray, t, cv2.TM_CCOEFF_NORMED)
                except cv2.error:
                    continue
                r = r[np.isfinite(r)]
                if r.size:
                    b = max(b, float(r.max()))
        best[brand] = b
    return best


_OCR = None


def ocr_texts(crop: np.ndarray) -> list[str]:
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    res, _ = _OCR(crop)
    if not res:
        return []
    return [r[1] for r in res if len(r) > 1 and isinstance(r[1], str)]


def ocr_frame(img: np.ndarray, upscale: float = 1.0):
    """Full-frame OCR -> [(text, box_xyxy)].

    One pass over the whole frame beats one pass per candidate region on both
    counts. Recall: the detector sees the wordmark in its natural context and at
    consistent scale, rather than in a tight crop whose borders cut the
    lettering. Cost: one OCR call per frame instead of one per polygon, and
    there are ~5 polygons per frame here."""
    global _OCR
    if _OCR is None:
        from rapidocr_onnxruntime import RapidOCR
        _OCR = RapidOCR()
    src = img
    if upscale > 1.01:
        src = cv2.resize(img, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
    res, _ = _OCR(src)
    out = []
    if not res:
        return out
    for r in res:
        if len(r) < 2 or not isinstance(r[1], str):
            continue
        pts = np.asarray(r[0], np.float32) / max(1.0, upscale)
        out.append((r[1], (float(pts[:, 0].min()), float(pts[:, 1].min()),
                           float(pts[:, 0].max()), float(pts[:, 1].max()))))
    return out


def _box_overlaps(box, pts: np.ndarray, pad: float) -> bool:
    bx0, by0, bx1, by1 = box
    px0, py0 = pts[:, 0].min() - pad, pts[:, 1].min() - pad
    px1, py1 = pts[:, 0].max() + pad, pts[:, 1].max() + pad
    ix = min(bx1, px1) - max(bx0, px0)
    iy = min(by1, py1) - max(by0, py0)
    if ix <= 0 or iy <= 0:
        return False
    barea = max(1.0, (bx1 - bx0) * (by1 - by0))
    return (ix * iy) / barea >= 0.55      # most of the text sits on this board


def process_one(job):
    img_path, lbl_path, min_h, probe = Path(job[0]), Path(job[1]), job[2], job[3]
    img = cv2.imread(str(img_path))
    if img is None:
        return None
    h, w = img.shape[:2]
    lines = [l for l in lbl_path.read_text().splitlines() if l.strip()]
    kept, brands, cues, rejected, samples = [], [], Counter(), 0, []
    for line in lines:
        cls, pts = poly_from_line(line, w, h)
        if pts is None:
            continue
        if cls == CLASS_OVERLAY:
            kept.append(line)          # broadcast graphic, not a brand-readable ad
            continue
        crop = crop_for_ocr(img, pts, min_h=min_h)
        if crop is None:
            rejected += 1
            continue
        texts = ocr_texts(crop)
        brand, joined = match_brand(texts)
        if probe:
            samples.append({"brand": brand, "text": joined[:70]})
        if brand is not None:
            kept.append(line)
            brands.append(brand)
            cues["ocr"] += 1
        else:
            rejected += 1
    return {"image": str(img_path), "label": str(lbl_path),
            "kept": kept, "brands": brands, "cues": dict(cues),
            "rejected": rejected, "n_in": len(lines), "samples": samples}


def main():
    cfg = load_config()
    p = cfg["paths"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True, help="a data/prelabel_review/<name> dir")
    ap.add_argument("--out", default=None, help="default: paths.annotations_real")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--segments", default="data/segments.json")
    ap.add_argument("--ocr-min-height", type=int, default=96,
                    help="upscale each region so its height is at least this "
                         "before OCR; board text is often ~12px tall in frame")
    ap.add_argument("--probe", action="store_true",
                    help="dump per-region OCR text and template scores, for "
                         "calibrating --template-thresh before a full run")
    args = ap.parse_args()

    review = Path(args.review)
    img_dir, lbl_dir = review / "images", review / "labels"
    out_dir = Path(args.out) if args.out else p["annotations_real"]
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

    jobs = []
    for img in sorted(img_dir.iterdir()):
        if img.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        lbl = lbl_dir / (img.stem + ".txt")
        if lbl.exists():
            jobs.append((str(img), str(lbl), args.ocr_min_height, args.probe))
    if args.limit:
        jobs = jobs[:args.limit]
    print(f"[filter] OCR over {len(jobs)} frames ({args.workers} workers)...")

    segs = []
    sp = Path(args.segments)
    if sp.exists():
        segs = json.loads(sp.read_text())["segments"]

    def sport_of(stem: str) -> str:
        m = re.search(r"_t(\d+(?:\.\d+)?)", stem)
        if not m:
            return "?"
        t = float(m.group(1))
        for s in segs:
            if s["start"] <= t <= s["end"]:
                return s.get("name", s.get("id", "?"))
        return "?"

    brand_counter, sport_kept, sport_frames = Counter(), Counter(), Counter()
    cue_counter = Counter()
    probe_rows: list[dict] = []
    n_kept_poly = n_in_poly = n_frames_kept = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for i, r in enumerate(ex.map(process_one, jobs, chunksize=4), 1):
            if r is None:
                continue
            cue_counter.update(r.get("cues", {}))
            probe_rows.extend(r.get("samples", []))
            n_in_poly += r["n_in"]
            stem = Path(r["image"]).stem
            sport = sport_of(stem)
            brand_counter.update(r["brands"])
            board_lines = [l for l in r["kept"] if l.split()[0] == str(CLASS_BOARD)]
            n_kept_poly += len(board_lines)
            sport_kept[sport] += len(board_lines)
            if board_lines:
                n_frames_kept += 1
                sport_frames[sport] += 1
                cv2.imwrite(str(out_dir / "images" / f"{stem}.jpg"), cv2.imread(r["image"]))
                (out_dir / "labels" / f"{stem}.txt").write_text("\n".join(r["kept"]) + "\n")
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} frames, {n_kept_poly} betting polys kept")

    (out_dir / "classes.txt").write_text("person\nbetting_board\nbetting_overlay\n")

    if args.probe and probe_rows:
        print(f"\n[probe] {len(probe_rows)} regions; template-score percentiles by "
              f"whether OCR read a brand:")
        for tag, rows in (("OCR matched   ", [r for r in probe_rows if r["ocr_brand"]]),
                          ("OCR no match  ", [r for r in probe_rows if not r["ocr_brand"]])):
            if not rows:
                continue
            sc = np.array([r["tpl_score"] for r in rows])
            print(f"  {tag} n={len(rows):5d}  p50={np.percentile(sc, 50):.3f} "
                  f"p90={np.percentile(sc, 90):.3f} p99={np.percentile(sc, 99):.3f} "
                  f"max={sc.max():.3f}")
        print("  sample OCR hits:")
        for r in [r for r in probe_rows if r["ocr_brand"]][:10]:
            print(f"    {r['ocr_brand']:10s} tpl={r['tpl_score']:.2f} "
                  f"text={r['text'][:44]!r}")
        Path("data/_probe_scores.json").write_text(json.dumps(probe_rows[:4000], indent=1))
        print("  wrote data/_probe_scores.json")

    print(f"\n[filter] board polygons: {n_in_poly} drafted -> {n_kept_poly} betting "
          f"({n_kept_poly / max(1, n_in_poly):.1%} kept)")
    print(f"[filter] accepted by cue: {dict(cue_counter)}")
    print(f"[filter] frames with >=1 betting board: {n_frames_kept}")
    print(f"\n{'brand':16s} {'polys':>7s}")
    for b, n in brand_counter.most_common():
        print(f"{b:16s} {n:>7d}")
    print(f"\n{'sport':20s} {'frames':>7s} {'polys':>7s}")
    for s in sorted(set(list(sport_frames) + list(sport_kept))):
        print(f"{s:20s} {sport_frames[s]:>7d} {sport_kept[s]:>7d}")
    print(f"\n[filter] wrote real annotations to {out_dir}")

    summary = {"drafted": n_in_poly, "kept": n_kept_poly,
               "frames_kept": n_frames_kept,
               "brands": dict(brand_counter),
               "per_sport_frames": dict(sport_frames),
               "per_sport_polys": dict(sport_kept)}
    (out_dir / "filter_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
