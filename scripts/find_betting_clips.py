#!/usr/bin/env python3
"""Find, per sport, the time ranges where a betting board is actually on screen.

A demo clip is only worth cutting where there is something to hide. The trained
model is run over a sampled sweep of the video; frames with a confident betting
detection are grouped into runs, short gaps are bridged (a board does not cease
to exist because one frame was motion-blurred), and runs shorter than
`--min-frames` are dropped so a single false positive cannot define a clip.

Writes data/betting_clips.json: every clip per sport, plus `best_clip`, the one
run_all.py cuts the demo from."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from common import die, load_config

CLASS_BOARD, CLASS_OVERLAY = 1, 2


def main():
    cfg = load_config()
    p = cfg["paths"]
    icfg = cfg["inference"]

    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--segments", default="data/segments.json")
    ap.add_argument("--out", default="data/betting_clips.json")
    ap.add_argument("--model", default=None)
    ap.add_argument("--sample-fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-frames", type=int, default=6)
    ap.add_argument("--bridge-seconds", type=float, default=2.5)
    args = ap.parse_args()

    vp = Path(args.video)
    if not vp.exists():
        vp = p["raw_videos"] / args.video
    if not vp.exists():
        die(f"video not found: {args.video}")

    segs = json.loads(Path(args.segments).read_text())["segments"]
    weights = args.model or icfg["weights"]
    from ultralytics import YOLO
    model = YOLO(str(weights))

    cap = cv2.VideoCapture(str(vp))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, int(round(fps / args.sample_fps)))
    print(f"[clips] {vp.name} {total} frames @{fps:.1f}fps, sampling every {stride}")

    hits = []          # (seconds, best_conf)
    idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            r = model.predict(frame, conf=args.conf, imgsz=icfg["imgsz"],
                              device=str(icfg["device"]),
                              classes=[CLASS_BOARD, CLASS_OVERLAY],
                              verbose=False)[0]
            best = 0.0
            if r.boxes is not None and len(r.boxes):
                cls = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                sel = cf[cls == CLASS_BOARD]
                best = float(sel.max()) if sel.size else 0.0
            if best > 0:
                hits.append((idx / fps, best))
            if idx % (stride * 200) == 0:
                print(f"  {idx}/{total} frames, {len(hits)} betting hits")
        idx += 1
    cap.release()

    times = np.array([h[0] for h in hits])
    confs = np.array([h[1] for h in hits])
    out = {"video": str(vp), "sample_fps": args.sample_fps, "conf": args.conf,
           "per_sport": {}}

    for s in segs:
        name = s.get("name", s.get("id"))
        m = (times >= s["start"]) & (times <= s["end"])
        st, sc = times[m], confs[m]
        clips = []
        if st.size:
            start = prev = st[0]
            acc = [sc[0]]
            for t, c in zip(st[1:], sc[1:]):
                if t - prev <= args.bridge_seconds:
                    prev = t
                    acc.append(c)
                    continue
                clips.append((start, prev, len(acc), float(np.mean(acc))))
                start = prev = t
                acc = [c]
            clips.append((start, prev, len(acc), float(np.mean(acc))))
        clips = [c for c in clips if c[2] >= args.min_frames]
        clips.sort(key=lambda c: (c[1] - c[0]) * c[3], reverse=True)
        out["per_sport"][name] = {
            "n_hit_frames": int(st.size),
            "clips": [{"start": round(a, 2), "end": round(b, 2),
                       "frames": n, "mean_conf": round(cf, 3)}
                      for a, b, n, cf in clips],
            "best_clip": ({"start": round(clips[0][0], 2), "end": round(clips[0][1], 2),
                           "frames": clips[0][2], "mean_conf": round(clips[0][3], 3)}
                          if clips else None),
        }

    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"\n{'sport':20s} {'hits':>6s} {'clips':>6s}  best clip")
    for name, info in out["per_sport"].items():
        b = info["best_clip"]
        txt = f"{b['start']:.0f}-{b['end']:.0f}s conf={b['mean_conf']:.2f}" if b else "NONE"
        print(f"{name:20s} {info['n_hit_frames']:>6d} {len(info['clips']):>6d}  {txt}")
    print(f"\n[clips] wrote {args.out}")


if __name__ == "__main__":
    main()
