#!/usr/bin/env python3
"""Export models/best.pt to ONNX and int8 TFLite for the Android APK.

The .tflite file produced here (models/best.tflite) is what actually ships
inside the Android app; the .onnx export is kept as an intermediate/
debugging artifact.

Usage:
    python scripts/export_tflite.py
    python scripts/export_tflite.py --weights models/best.pt --imgsz 640
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from common import die, load_config, resolved_data_yaml


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f}{unit}"
        size /= 1024
    return f"{size:.1f}TB"


def main():
    cfg = load_config()
    p = cfg["paths"]
    ecfg = cfg["export"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=str, default=ecfg["weights"])
    ap.add_argument("--imgsz", type=int, default=ecfg["imgsz"])
    ap.add_argument("--int8", type=bool, default=ecfg["int8"])
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        die(f"Weights not found: {weights_path}. Train first with scripts/train.py.")

    try:
        from ultralytics import YOLO
    except ImportError:
        die("ultralytics is not installed. Run: pip install -r requirements.txt")

    model = YOLO(str(weights_path))
    models_dir = p["models"]
    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Exporting {weights_path} -> ONNX (imgsz={args.imgsz})...")
    onnx_path = Path(model.export(format="onnx", imgsz=args.imgsz))
    onnx_dest = models_dir / "best.onnx"
    if onnx_path.resolve() != onnx_dest.resolve():
        shutil.copy2(onnx_path, onnx_dest)

    print(f"Exporting {weights_path} -> TFLite int8={args.int8} (imgsz={args.imgsz})...")
    export_kwargs = dict(format="tflite", int8=args.int8, imgsz=args.imgsz)
    if args.int8:
        # INT8 quantization calibrates against real images; reuse the
        # training dataset for representative data.
        export_kwargs["data"] = str(resolved_data_yaml(cfg))
    tflite_path = Path(model.export(**export_kwargs))
    tflite_dest = models_dir / "best.tflite"
    if tflite_path.resolve() != tflite_dest.resolve():
        shutil.copy2(tflite_path, tflite_dest)

    print("\n--- Export summary ---")
    print(f"ONNX:   {onnx_dest}  ({human_size(onnx_dest.stat().st_size)})")
    print(f"TFLite: {tflite_dest}  ({human_size(tflite_dest.stat().st_size)})  <- ships in the Android APK")


if __name__ == "__main__":
    main()
