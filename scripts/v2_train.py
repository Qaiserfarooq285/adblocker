#!/usr/bin/env python3
"""Train the single-class betting-ad segmentation model.

Resumable: re-running picks up `runs/v2/<name>/weights/last.pt` when present,
so an interrupted run (power cut, session end) continues instead of restarting.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "dataset" / "data.yaml"))
    ap.add_argument("--weights", default="yolo11s-seg.pt")
    ap.add_argument("--epochs", type=int, default=45)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--name", default="ad_seg3")
    args = ap.parse_args()

    from ultralytics import YOLO

    run_dir = ROOT / "runs" / "v2" / args.name
    last = run_dir / "weights" / "last.pt"
    best = run_dir / "weights" / "best.pt"
    results = None

    # Already finished?  Publish the weights and return.  Without this the
    # orchestrator's train step would call resume() on a completed run every
    # time the watchdog restarts it, and never reach the render stage.
    done_epochs = 0
    csv_path = run_dir / "results.csv"
    if csv_path.exists():
        try:
            done_epochs = sum(1 for _ in csv_path.read_text().splitlines()) - 1
        except Exception:
            done_epochs = 0
    if best.exists() and done_epochs >= args.epochs:
        print(f"[train] already complete ({done_epochs}/{args.epochs} epochs)")
    elif last.exists():
        print(f"[train] resuming from {last} at epoch {done_epochs}")
        try:
            model = YOLO(str(last))
            results = model.train(resume=True)
        except Exception as exc:
            # Early stopping ends a run below the requested epoch count, so the
            # check above will not catch it and resume() then refuses.  With
            # usable weights on disk that is success, not failure -- treating it
            # as an error would stall the unattended run short of rendering.
            if not best.exists():
                raise
            print(f"[train] resume declined ({exc}); using existing best.pt")
    else:
        model = YOLO(args.weights)
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            project=str(ROOT / "runs" / "v2"),
            name=args.name,
            exist_ok=True,
            patience=15,
            # Rails are upright and thin: never flip vertically or rotate.
            degrees=0.0, flipud=0.0, fliplr=0.5,
            # Close-up boards were the worst case at 60% recall, so let the
            # sampler show the model a much wider range of board sizes.
            translate=0.10, scale=0.80, shear=0.0, perspective=0.0,
            # Stadiums differ in lighting far more than in geometry.
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            mosaic=1.0, close_mosaic=15, copy_paste=0.0,
            cache=False, workers=8, seed=0, verbose=True,
        )

    best = run_dir / "weights" / "best.pt"
    if best.exists():
        dst = ROOT / "models" / "ad_seg_best.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best, dst)
        print(f"[train] best -> {dst}")
    try:
        print("[train] final metrics:", results.results_dict)
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
