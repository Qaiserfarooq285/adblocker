#!/usr/bin/env python3
"""Train (or fine-tune) the YOLO11-seg model on data/dataset via data/data.yaml.

Fresh training (from COCO pretrain):
    python scripts/train.py

Larger run to measure an accuracy ceiling (still ships yolo11n to the APK):
    python scripts/train.py --model yolo11s-seg.pt

Adaptive fine-tuning after adding new videos/brands (run assemble_dataset.py
--append first):
    python scripts/train.py --weights models/best.pt --epochs 40 --lr0 0.001

--weights starts from the current trained model instead of the COCO
checkpoint named by --model, and defaults epochs/lr0 to the finetune_* values
in config.yaml (lower LR so fine-tuning nudges the model rather than
overwriting what it already learned).
"""
from __future__ import annotations

import argparse
import shutil

from common import die, load_config, resolved_data_yaml


def main():
    cfg = load_config()
    p = cfg["paths"]
    tcfg = cfg["train"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=str, default=tcfg["model"], help="Base COCO-pretrained checkpoint to start from (ignored if --weights is given)")
    ap.add_argument("--weights", type=str, default=None, help="Path to an existing trained .pt to fine-tune from (e.g. models/best.pt)")
    ap.add_argument("--data", type=str, default=None, help="Path to data.yaml (default: data/data.yaml)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=tcfg["imgsz"])
    def batch_arg(v):
        """int, or float for 'fraction of GPU memory'.

        Without an explicit type argparse hands the value through as a STRING,
        so --batch worked when it came from config.yaml (already an int) and
        blew up the moment anyone passed it on the command line:
        "'batch=-1' is of invalid type str"."""
        s = str(v).strip()
        try:
            return int(s)
        except ValueError:
            return float(s)

    ap.add_argument("--batch", type=batch_arg, default=tcfg["batch"],
                    help="Batch size, -1 to auto-fit GPU memory, or a fraction like 0.7")
    ap.add_argument("--device", type=str, default=str(tcfg["device"]))
    ap.add_argument("--amp", type=bool, default=tcfg["amp"])
    ap.add_argument("--workers", type=int, default=tcfg["workers"])
    ap.add_argument("--cache", type=bool, default=tcfg["cache"])
    ap.add_argument("--patience", type=int, default=tcfg["patience"])
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--project", type=str, default=tcfg["project"])
    ap.add_argument("--name", type=str, default=tcfg["name"])
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    try:
        from ultralytics import YOLO
    except ImportError:
        die("ultralytics is not installed. Run: pip install -r requirements.txt")

    if not p["data_yaml"].exists() and args.data is None:
        die(f"'{p['data_yaml']}' not found. It ships with the repo — did you move/delete it?")
    data_yaml = args.data or str(resolved_data_yaml(cfg))

    fine_tuning = args.weights is not None
    start_from = args.weights if fine_tuning else args.model
    epochs = args.epochs if args.epochs is not None else (tcfg["finetune_epochs"] if fine_tuning else tcfg["epochs"])
    lr0 = args.lr0 if args.lr0 is not None else (tcfg["finetune_lr0"] if fine_tuning else None)

    mode = "Fine-tuning from" if fine_tuning else "Training fresh from"
    print(f"{mode} '{start_from}' | epochs={epochs} imgsz={args.imgsz} batch={args.batch} device={args.device}")

    model = YOLO(start_from)
    train_kwargs = dict(
        data=data_yaml,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        amp=args.amp,
        workers=args.workers,
        cache=args.cache,
        patience=args.patience,
        project=args.project,
        name=args.name,
    )
    if lr0 is not None:
        train_kwargs["lr0"] = lr0

    results = model.train(**train_kwargs)

    run_dir = results.save_dir
    best_ckpt = run_dir / "weights" / "best.pt"
    if not best_ckpt.exists():
        die(f"Training finished but no checkpoint found at {best_ckpt}")

    p["models"].mkdir(parents=True, exist_ok=True)
    dest = p["models"] / "best.pt"
    shutil.copy2(best_ckpt, dest)
    print(f"Copied best checkpoint -> {dest}")


if __name__ == "__main__":
    main()
