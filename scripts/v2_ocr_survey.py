#!/usr/bin/env python3
"""OCR every extracted frame and dump raw text boxes.

This is a survey, not a labeller: it makes no decision about what is a betting
brand.  It writes one JSON record per frame so the annotation step can be
re-run with different brand rules without paying the OCR cost again.

Output: data/ocr/<clip>/<frame>.json  ->  [{"text","conf","quad"}, ...]
        data/ocr/vocab.json           ->  {token: count} across the corpus
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=str(ROOT / "data" / "frames"))
    ap.add_argument("--out", default=str(ROOT / "data" / "ocr"))
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--min-conf", type=float, default=0.45)
    args = ap.parse_args()

    from rapidocr_onnxruntime import RapidOCR
    engine = RapidOCR()

    frames = sorted(Path(args.frames).rglob("*.jpg"))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    vocab: Counter[str] = Counter()
    done = 0

    for fp in frames:
        rel = fp.relative_to(args.frames)
        dst = out_root / rel.with_suffix(".json")
        if dst.exists():
            try:
                recs = json.loads(dst.read_text())
                for r in recs:
                    for t in re.split(r"[^A-Za-z0-9]+", r["text"].lower()):
                        if len(t) >= 3:
                            vocab[t] += 1
                done += 1
                continue
            except Exception:
                pass
        img = cv2.imread(str(fp))
        if img is None:
            continue
        h, w = img.shape[:2]
        scale = args.width / float(w)
        if abs(scale - 1.0) > 0.02:
            img = cv2.resize(img, (args.width, int(round(h * scale))))
        else:
            scale = 1.0
        try:
            res, _ = engine(img)
        except Exception as exc:
            print(f"[warn] {rel}: {exc}")
            res = None
        recs = []
        for item in (res or []):
            quad, text, conf = item[0], str(item[1]), float(item[2])
            if conf < args.min_conf:
                continue
            q = (np.asarray(quad, np.float32) / scale).round(1).tolist()
            recs.append({"text": text, "conf": round(conf, 3), "quad": q})
            for t in re.split(r"[^A-Za-z0-9]+", text.lower()):
                if len(t) >= 3:
                    vocab[t] += 1
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(json.dumps(recs))
        done += 1
        if done % 100 == 0:
            print(f"[ocr] {done}/{len(frames)}", flush=True)

    (out_root / "vocab.json").write_text(json.dumps(dict(vocab.most_common()), indent=1))
    print(f"[ocr] done {done} frames, {len(vocab)} distinct tokens")
    for t, n in vocab.most_common(80):
        print(f"    {t:<22} {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
