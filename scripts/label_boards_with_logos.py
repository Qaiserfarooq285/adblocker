#!/usr/bin/env python3
"""Keep the board drafts that actually contain a betting wordmark.

Two imperfect sources are combined into one good label here:

  zero-shot drafts   strip-shaped polygons covering EVERY advertising board,
                     because an open-vocabulary detector has no concept of
                     gambling. Right shape, no idea which ones matter.
  logo detector      trained on the supplied Roboflow set plus the logos mined
                     from this footage. Knows a betting wordmark when it sees
                     one, but outputs small logo boxes - the wrong thing to
                     hide, since covering only the wordmark leaves the rest of
                     the advert on screen.

A board polygon is kept when a logo box lands inside it. The polygon supplies
the geometry, the logo supplies the verdict. Logo boxes that fall outside every
drafted board become boards in their own right, expanded along the strip, so a
board the zero-shot pass missed entirely is not lost.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from common import die, load_config

CLASS_BOARD, CLASS_OVERLAY = 1, 2


def poly_from_line(line: str, w: int, h: int):
    parts = line.split()
    cls = int(parts[0])
    v = np.array([float(x) for x in parts[1:]], np.float32)
    if v.size < 6:
        return cls, None
    return cls, v.reshape(-1, 2) * np.array([w, h], np.float32)


def box_in_poly(box, pts: np.ndarray, min_frac: float = 0.6) -> bool:
    """Is most of the logo box inside this board's bounding region?"""
    bx0, by0, bx1, by1 = box
    px0, py0 = pts[:, 0].min(), pts[:, 1].min()
    px1, py1 = pts[:, 0].max(), pts[:, 1].max()
    ix = min(bx1, px1) - max(bx0, px0)
    iy = min(by1, py1) - max(by0, py0)
    if ix <= 0 or iy <= 0:
        return False
    return (ix * iy) / max(1.0, (bx1 - bx0) * (by1 - by0)) >= min_frac


def strip_from_box(box, w: int, h: int, grow_x: float = 2.2, grow_y: float = 1.35):
    """A fallback board for a logo the draft pass missed: a strip centred on the
    wordmark, wider than tall, since an advert is a bar and not a square."""
    bx0, by0, bx1, by1 = box
    cx, cy = (bx0 + bx1) / 2.0, (by0 + by1) / 2.0
    bw, bh = (bx1 - bx0) * grow_x, (by1 - by0) * grow_y
    x0, x1 = max(0.0, cx - bw / 2), min(float(w), cx + bw / 2)
    y0, y1 = max(0.0, cy - bh / 2), min(float(h), cy + bh / 2)
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], np.float32)


def main():
    cfg = load_config()
    p = cfg["paths"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--review", required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--segments", default="data/segments.json")
    ap.add_argument("--device", default="0")
    args = ap.parse_args()

    review = Path(args.review)
    img_dir, lbl_dir = review / "images", review / "labels"
    if not img_dir.exists():
        die(f"{img_dir} not found")
    out_dir = Path(args.out) if args.out else p["annotations_real"]
    (out_dir / "images").mkdir(parents=True, exist_ok=True)
    (out_dir / "labels").mkdir(parents=True, exist_ok=True)

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

    from ultralytics import YOLO
    det = YOLO(args.detector)

    imgs = sorted(q for q in img_dir.iterdir()
                  if q.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"[label] {len(imgs)} frames, detector={args.detector}")

    per_sport_frames, per_sport_polys = Counter(), Counter()
    n_kept = n_drafted = n_added = n_frames = 0

    for i, img_path in enumerate(imgs, 1):
        lbl = lbl_dir / (img_path.stem + ".txt")
        if not lbl.exists():
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]

        r = det.predict(img, conf=args.conf, imgsz=args.imgsz, device=args.device,
                        verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes):
            boxes = [tuple(float(v) for v in b) for b in r.boxes.xyxy.cpu().numpy()]

        lines_out = []
        drafts = [l for l in lbl.read_text().splitlines() if l.strip()]
        n_drafted += sum(1 for l in drafts if l.split()[0] == str(CLASS_BOARD))
        claimed = [False] * len(boxes)

        for line in drafts:
            cls, pts = poly_from_line(line, w, h)
            if pts is None:
                continue
            if cls == CLASS_OVERLAY:
                continue                      # broadcast graphics are not the target
            hit = False
            for bi, b in enumerate(boxes):
                if box_in_poly(b, pts):
                    claimed[bi] = True
                    hit = True
            if hit:
                lines_out.append(line)
                n_kept += 1

        # logos with no drafted board around them: synthesise a strip
        for bi, b in enumerate(boxes):
            if claimed[bi]:
                continue
            q = strip_from_box(b, w, h)
            norm = (q / np.array([w, h], np.float32)).clip(0, 1).reshape(-1)
            lines_out.append(f"{CLASS_BOARD} " + " ".join(f"{v:.6f}" for v in norm))
            n_added += 1

        if lines_out:
            sport = sport_of(img_path.stem)
            per_sport_frames[sport] += 1
            per_sport_polys[sport] += len(lines_out)
            n_frames += 1
            cv2.imwrite(str(out_dir / "images" / f"{img_path.stem}.jpg"), img)
            (out_dir / "labels" / f"{img_path.stem}.txt").write_text(
                "\n".join(lines_out) + "\n")
        if i % 200 == 0:
            print(f"  {i}/{len(imgs)} frames, {n_kept} kept + {n_added} added")

    (out_dir / "classes.txt").write_text("person\nbetting_board\nbetting_overlay\n")
    print(f"\n[label] drafts {n_drafted} -> kept {n_kept} "
          f"({n_kept / max(1, n_drafted):.1%}); +{n_added} from unmatched logos")
    print(f"[label] {n_frames} frames written to {out_dir}")
    print(f"\n{'sport':20s} {'frames':>7s} {'polys':>7s}")
    for s in sorted(set(list(per_sport_frames) + list(per_sport_polys))):
        print(f"{s:20s} {per_sport_frames[s]:>7d} {per_sport_polys[s]:>7d}")

    (out_dir / "label_summary.json").write_text(json.dumps({
        "drafted": n_drafted, "kept": n_kept, "added_from_logos": n_added,
        "frames": n_frames, "per_sport_frames": dict(per_sport_frames),
        "per_sport_polys": dict(per_sport_polys)}, indent=2) + "\n")


if __name__ == "__main__":
    main()
