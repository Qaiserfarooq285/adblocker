#!/usr/bin/env python3
"""Fully automatic, resumable betting-ad removal for one video.

Default workflow (no manual review): extract frames -> zero-shot board proposals
-> PDF-brand OCR filtering -> dataset/person segmentation -> fine-tune ->
verified video render.  Every stage is recorded in a JSON state file, so run
the same command again after an interruption and it continues safely.

    .venv/bin/python scripts/run_automatic.py --input data/raw_videos/match.mp4
    .venv/bin/python scripts/run_automatic.py --input match.mp4 --render-only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from common import die, load_config, slugify

REPO = Path(__file__).resolve().parent.parent
PY = REPO / ".venv" / "bin" / "python"


def run_stage(name: str, cmd: list[str], state_path: Path, state: dict) -> None:
    if state.get(name, {}).get("status") == "done":
        print(f"[auto] {name}: already complete")
        return
    print(f"[auto] {name}: {' '.join(cmd)}", flush=True)
    state[name] = {"status": "running", "at": datetime.now().isoformat(timespec="seconds")}
    state_path.write_text(json.dumps(state, indent=2) + "\n")
    result = subprocess.run(cmd, cwd=REPO)
    if result.returncode:
        state[name] = {"status": "failed", "exit_code": result.returncode,
                       "at": datetime.now().isoformat(timespec="seconds")}
        state_path.write_text(json.dumps(state, indent=2) + "\n")
        raise SystemExit(result.returncode)
    state[name] = {"status": "done", "at": datetime.now().isoformat(timespec="seconds")}
    state_path.write_text(json.dumps(state, indent=2) + "\n")


def main() -> None:
    cfg = load_config()
    p = cfg["paths"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Video path, or a filename under data/raw_videos")
    ap.add_argument("--output", default=None, help="Default: outputs/<video>_verified_blocked.mp4")
    ap.add_argument("--render-only", action="store_true", help="Reuse current models; skip data generation and fine-tuning")
    ap.add_argument("--device", default=str(cfg["inference"]["device"]))
    args = ap.parse_args()

    src = Path(args.input)
    if not src.exists():
        src = p["raw_videos"] / args.input
    if not src.exists():
        die(f"Input video not found: {args.input}")
    if not PY.exists():
        die(f"Virtualenv Python not found: {PY}")

    slug = slugify(src.stem)
    output = Path(args.output) if args.output else p["outputs"] / f"{slug}_verified_blocked.mp4"
    review = p["prelabel_review"] / f"auto_{slug}"
    state_path = REPO / "logs" / f"automatic_{slug}.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {}

    if not args.render_only:
        # extract_frames deliberately accepts filenames relative to raw_videos;
        # keeping training inputs there also makes frame/label provenance stable.
        if src.resolve().parent != p["raw_videos"].resolve():
            die("Automatic training requires the input video under data/raw_videos/. "
                "Move/copy it there, or use --render-only for an external path.")
        run_stage("01_extract", [str(PY), "scripts/extract_frames.py", "--video", src.name, "--fps", "1"], state_path, state)
        run_stage("02_propose_boards", [str(PY), "scripts/prelabel_boards.py", "--zero-shot", "--video", slug,
                                         "--out", str(review), "--no-skip-annotated", "--device", args.device], state_path, state)
        # OCR accepts only named brands in the PDF-derived registry. Generic
        # advertising-board proposals never become training labels.
        run_stage("03_filter_pdf_brands", [str(PY), "scripts/filter_betting_boards.py", "--review", str(review),
                                            "--out", str(p["annotations_real"]), "--workers", "6"], state_path, state)
        run_stage("04_assemble", [str(PY), "scripts/assemble_dataset.py", "--append"], state_path, state)
        run_stage("05_people", [str(PY), "scripts/autolabel_persons.py", "--device", args.device], state_path, state)
        run_stage("06_finetune_segmentation", [str(PY), "scripts/train.py", "--weights", "models/best.pt",
                                                 "--device", args.device], state_path, state)

    run_stage("07_render_verified_video", [str(PY), "scripts/process_video.py", "--input", str(src),
                                             "--output", str(output), "--device", args.device], state_path, state)
    print(f"[auto] COMPLETE: {output}")
    print(f"[auto] state: {state_path}")


if __name__ == "__main__":
    main()
