#!/usr/bin/env python3
"""Run the entire remaining pipeline unattended, from logo detector to the
per-sport demo videos.

Written to survive the operator being away and the controlling session dying:
launch it detached and it carries on. Every step writes a stamp when it
finishes, so re-running skips completed work and resumes at the first unfinished
step - a crash or a kill costs that one step, not the whole run.

    setsid nohup .venv/bin/python scripts/run_all.py > logs/run_all.log 2>&1 &

Progress is appended to logs/run_all.log and a machine-readable state file is
kept at logs/run_all_state.json, so `tail -f` or reading that file both answer
"where is it now".
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
PY = str(REPO / ".venv" / "bin" / "python")
LOGS = REPO / "logs"
STAMPS = REPO / "data" / ".stamps" / "run_all"
STATE = LOGS / "run_all_state.json"

DEMO_SECONDS = 60          # per-sport demo clip length
SEG_EPOCHS = 100
SEG_IMGSZ = 1280           # NOT 640: thin boards vanish; see config.yaml

# False: cut ONE betting-visible clip per sport and hide the ads in it. That is
# the cheap way to find out whether hiding actually works on each sport - a full
# pass is 115,170 frames and 8-16 hours, which is a lot to spend before knowing
# the panel behaves on a hockey board or an MMA canvas.
# True: render every sport end to end, in resumable chunks. Flip this once the
# clips look right.
FULL_SPORT_VIDEOS = False
# ...but render it in chunks and concatenate. The whole video is ~64 minutes and
# the renderer runs two models plus optical flow per frame, so this is a
# multi-hour job; at one file per sport a power cut two hours into a segment
# loses two hours. At this granularity it loses minutes, and each finished chunk
# is watchable immediately.
CHUNK_SECONDS = 300


def log(msg: str) -> None:
    line = f"[{datetime.now():%H:%M:%S}] {msg}"
    print(line, flush=True)


def set_state(step: str, status: str, extra: dict | None = None) -> None:
    LOGS.mkdir(parents=True, exist_ok=True)
    state = {}
    if STATE.exists():
        try:
            state = json.loads(STATE.read_text())
        except ValueError:
            state = {}
    state[step] = {"status": status, "at": datetime.now().isoformat(timespec="seconds")}
    if extra:
        state[step].update(extra)
    state["_current"] = step if status == "running" else state.get("_current", "")
    STATE.write_text(json.dumps(state, indent=2) + "\n")


def done(step: str) -> bool:
    return (STAMPS / f"{step}.done").exists()


def mark(step: str) -> None:
    STAMPS.mkdir(parents=True, exist_ok=True)
    (STAMPS / f"{step}.done").write_text(datetime.now().isoformat() + "\n")


def run(cmd: list[str], step: str) -> None:
    log(f"$ {' '.join(str(c) for c in cmd[:6])}{' ...' if len(cmd) > 6 else ''}")
    r = subprocess.run([str(c) for c in cmd], cwd=REPO)
    if r.returncode != 0:
        raise RuntimeError(f"{step}: exit {r.returncode}")


def step(name: str, required: bool = True):
    """Skip when already stamped, record status, keep the traceback.

    `required=False` marks a step the run can finish without. Synthetic data,
    person labels and the sanity montage all improve the result but none of
    them is load-bearing: real board labels plus a trained model still produce
    demo videos. Aborting the whole overnight run because an optional step
    tripped would trade a slightly worse model for no model at all."""
    def deco(fn):
        def wrapper(*a, **kw):
            if done(name):
                log(f"== {name}: already done, skipping")
                return True
            log(f"== {name}: START{'' if required else '  (optional)'}")
            set_state(name, "running")
            t0 = time.time()
            try:
                fn(*a, **kw)
            except Exception as exc:                       # noqa: BLE001
                set_state(name, "failed", {"error": str(exc), "required": required})
                log(f"== {name}: FAILED after {time.time() - t0:.0f}s -> {exc}")
                traceback.print_exc()
                if required:
                    return False
                log(f"== {name}: optional, continuing without it")
                return True
            mark(name)
            set_state(name, "done", {"seconds": round(time.time() - t0)})
            log(f"== {name}: DONE in {time.time() - t0:.0f}s")
            return True
        wrapper.required = required
        wrapper.step_name = name
        return wrapper
    return deco


# ---------------------------------------------------------------------------

@step("01_logo_detector")
def train_logo_detector():
    """Detector that says 'this region contains a betting wordmark'. Not the
    hider - its job is to decide which board strips are BETTING boards."""
    from ultralytics import YOLO
    m = YOLO("yolo11s.pt")
    m.train(data=str(REPO / "data/logo_merged/data.yaml"), epochs=45, imgsz=960,
            batch=-1, device=0, amp=True, workers=8, patience=15,
            project="runs/logo", name="merged", exist_ok=True, plots=True)


def logo_weights() -> str:
    """Where Ultralytics actually put the detector.

    It prepends the task directory to `project`, so a project of 'runs/logo'
    lands at 'runs/detect/runs/logo/...'. Hard-coding the path we asked for is
    what broke the first unattended run after the training itself had already
    succeeded - 2.4 hours of work stranded behind a filename."""
    for c in (REPO / "runs/detect/runs/logo/merged/weights/best.pt",
              REPO / "runs/logo/merged/weights/best.pt",
              REPO / "runs/detect/merged/weights/best.pt"):
        if c.exists():
            return str(c)
    found = sorted(REPO.glob("runs/**/merged/weights/best.pt"))
    if found:
        return str(found[-1])
    raise FileNotFoundError("logo detector best.pt not found under runs/")


@step("02_board_labels")
def build_board_labels():
    """Zero-shot board drafts, kept only where the logo detector finds a betting
    wordmark inside them. This is what turns 'every advertising board' into
    'betting boards', and it produces STRIP-level polygons - the thing the
    renderer has to hide - rather than the logo boxes themselves."""
    run([PY, "scripts/label_boards_with_logos.py",
         "--review", "data/prelabel_review/sports5",
         "--detector", logo_weights(),
         "--out", "data/annotations_real",
         "--segments", "data/segments.json"], "02_board_labels")


@step("03_synthetic", required=False)
def synthetic():
    run([PY, "scripts/generate_synthetic.py", "--num-images", "3000"], "03_synthetic")


@step("04_assemble")
def assemble():
    run([PY, "scripts/assemble_dataset.py"], "04_assemble")


@step("05_persons", required=False)
def persons():
    # Person polygons are written into data/dataset, so assembly must happen
    # first. The previous order silently left the training set without them.
    run([PY, "scripts/autolabel_persons.py"], "05_persons")


@step("06_sanity_montage", required=False)
def sanity():
    run([PY, "scripts/sanity_montage.py", "--out", "outputs/sanity_montage.jpg"],
        "06_sanity_montage")


@step("07_train_seg")
def train_seg():
    """The hider. imgsz 1280 because a perimeter board is a thin strip and at
    640 it shrinks to ~20px tall and recall collapses - measured on this repo."""
    run([PY, "scripts/train.py", "--epochs", str(SEG_EPOCHS),
         "--imgsz", str(SEG_IMGSZ), "--batch", "-1"], "07_train_seg")


@step("08_betting_clips")
def betting_clips():
    run([PY, "scripts/find_betting_clips.py",
         "--video", "15 min 5 sports.mp4",
         "--segments", "data/segments.json",
         "--out", "data/betting_clips.json"], "08_betting_clips")


@step("09_demos")
def demos():
    """Hide the ads and write one video per sport.

    Two modes, see FULL_SPORT_VIDEOS. Either way the work is done in chunks that
    are skipped when their file already exists, so a killed run resumes at the
    chunk boundary rather than restarting."""
    import subprocess as sp

    segs = json.loads((REPO / "data/segments.json").read_text())["segments"]
    outdir = REPO / "outputs"
    chunkdir = outdir / "chunks"
    outdir.mkdir(exist_ok=True)
    chunkdir.mkdir(exist_ok=True)
    ffmpeg = os.environ.get("FFMPEG", "ffmpeg")

    # --- decide the time range to render per sport ---
    ranges: list[tuple[str, float, float]] = []
    if FULL_SPORT_VIDEOS:
        for s_ in sorted(segs, key=lambda x: float(x["end"]) - float(x["start"])):
            ranges.append((s_.get("name", s_.get("id")),
                           float(s_["start"]), float(s_["end"])))
        log(f"   FULL mode: {len(ranges)} sports end-to-end, {CHUNK_SECONDS}s chunks")
    else:
        cp = REPO / "data/betting_clips.json"
        clips = json.loads(cp.read_text()) if cp.exists() else {}
        per = clips.get("per_sport", {})
        for s_ in segs:
            name = s_.get("name", s_.get("id"))
            best = (per.get(name) or {}).get("best_clip")
            if best:
                a = float(best["start"])
                b = min(float(best["end"]), a + DEMO_SECONDS)
                if b - a < 5:                       # too short to judge anything
                    b = min(a + DEMO_SECONDS, float(s_["end"]))
                ranges.append((name, a, b))
            else:
                # No betting detected in this sport. Still render from the middle
                # of it: a clip showing nothing hidden is evidence about the
                # model, whereas a missing file is just a gap.
                mid = (float(s_["start"]) + float(s_["end"])) / 2.0
                ranges.append((name, mid, min(mid + DEMO_SECONDS, float(s_["end"]))))
                log(f"   {name}: no betting clip found - using segment midpoint")
        log(f"   CLIP mode: {len(ranges)} sports x <={DEMO_SECONDS}s")

    made = []
    for name, t0, t1 in ranges:
        final = outdir / (f"sport_{name}_blocked.mp4" if FULL_SPORT_VIDEOS
                          else f"demo_{name}.mp4")
        if final.exists():
            log(f"   {name}: {final.name} exists, skipping")
            made.append(str(final))
            continue
        log(f"   {name}: {t0:.0f}-{t1:.0f}s ({(t1 - t0) / 60:.1f} min)")

        parts, cur, idx = [], t0, 0
        while cur < t1:
            nxt = min(cur + CHUNK_SECONDS, t1)
            part = chunkdir / f"{name}_{idx:03d}.mp4"
            if not part.exists():
                try:
                    run([PY, "scripts/process_video.py",
                         "--input", "15 min 5 sports.mp4",
                         "--start", f"{cur:.2f}", "--end", f"{nxt:.2f}",
                         "--output", str(part)], f"09_demos:{name}:{idx}")
                except RuntimeError as exc:
                    log(f"   {name} chunk {idx}: FAILED ({exc}) - continuing")
            if part.exists():
                parts.append(part)
            cur, idx = nxt, idx + 1

        if not parts:
            log(f"   {name}: nothing rendered, skipping")
            continue
        if len(parts) == 1:
            parts[0].replace(final)
        else:
            listing = chunkdir / f"{name}_parts.txt"
            listing.write_text("".join(f"file '{q.resolve()}'\n" for q in parts))
            r = sp.run([ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
                        "-f", "concat", "-safe", "0", "-i", str(listing),
                        "-c", "copy", str(final)], cwd=REPO)
            if r.returncode != 0:
                log(f"   {name}: concat FAILED - chunks are in {chunkdir}")
                continue
        log(f"   {name}: wrote {final.name}")
        made.append(str(final))

    (LOGS / "demos.json").write_text(json.dumps(made, indent=2) + "\n")


def main():
    LOGS.mkdir(parents=True, exist_ok=True)
    log("=" * 70)
    log("run_all: full pipeline, unattended")
    log(f"repo={REPO}")
    log("=" * 70)
    t0 = time.time()
    steps = [train_logo_detector, build_board_labels, synthetic, assemble,
             persons, sanity, train_seg, betting_clips, demos]
    for fn in steps:
        if not fn():
            log(f"ABORTING: required step {getattr(fn, 'step_name', '?')} failed. "
                "Fix it and re-run run_all.py - completed steps are stamped "
                "and will be skipped.")
            set_state("_run", "failed")
            return 1
    set_state("_run", "done", {"total_seconds": round(time.time() - t0)})
    log("=" * 70)
    log(f"ALL DONE in {(time.time() - t0) / 60:.1f} min")
    for p in sorted((REPO / "outputs").glob("demo_*.mp4")):
        log(f"  {p}")
    log("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
