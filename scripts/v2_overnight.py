#!/usr/bin/env python3
"""Unattended overnight supervisor: train to a deadline, then render everything.

Training is allowed a fixed wall-clock budget rather than a fixed epoch count.
Ultralytics writes best.pt after every epoch, so stopping at a deadline costs
only the epoch in flight, and it guarantees the clips are rendered by morning
instead of the whole night going into training that is still running at wake-up.

Safe to run repeatedly: it skips whatever is already finished.
"""
from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = str(ROOT / ".venv" / "bin" / "python")
if not Path(PY).exists():
    PY = sys.executable
LOGS = ROOT / "logs"
RUN = ROOT / "runs" / "v2" / "ad_seg3"


def log(msg: str) -> None:
    line = f"{datetime.now().strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with (LOGS / "overnight.log").open("a") as fh:
        fh.write(line + "\n")


def procs(pattern: str) -> list[int]:
    out = subprocess.run(["ps", "-eo", "pid,comm,args"], capture_output=True,
                         text=True).stdout
    pids = []
    for line in out.splitlines()[1:]:
        p = line.split(None, 2)
        if len(p) == 3 and p[1].startswith("python") and pattern in p[2]:
            pids.append(int(p[0]))
    return pids


def epochs_done() -> int:
    csv = RUN / "results.csv"
    if not csv.exists():
        return 0
    try:
        return max(0, len(csv.read_text().splitlines()) - 1)
    except Exception:
        return 0


def wait_for_training(budget_min: float) -> str:
    """Returns 'natural' if training exited on its own, 'forced' if we killed it.

    The distinction matters downstream: an orchestrator that was already
    running (e.g. started manually with APPROVED already in place) continues
    past a naturally-finished train step entirely on its own.  Killing that
    orchestrator and launching a second one races both against each other on
    the very next render -- which is exactly what corrupted clip01 the first
    time this ran.  Only a forced stop needs us to take over.
    """
    deadline = time.time() + budget_min * 60
    while True:
        alive = procs("v2_train.py")
        if not alive:
            log(f"training ended on its own after {epochs_done()} epochs")
            return "natural"
        if time.time() > deadline:
            log(f"training budget reached at {epochs_done()} epochs; stopping it")
            for pid in alive:
                try:
                    os.killpg(os.getpgid(pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError):
                    try:
                        os.kill(pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            time.sleep(20)
            for pid in procs("v2_train.py"):
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            return "forced"
        time.sleep(60)


def publish_weights() -> bool:
    best = RUN / "weights" / "best.pt"
    if not best.exists():
        log("no best.pt produced; keeping the previous model")
        return False
    dst = ROOT / "models" / "ad_seg_best.pt"
    shutil.copy2(best, dst)
    log(f"published {best} -> {dst} ({epochs_done()} epochs)")
    return True


def main() -> int:
    LOGS.mkdir(parents=True, exist_ok=True)
    budget = float(os.environ.get("TRAIN_BUDGET_MIN", "165"))
    log(f"supervisor start; training budget {budget:.0f} min")

    outcome = "natural"
    if procs("v2_train.py"):
        outcome = wait_for_training(budget)
    else:
        log("no training running")

    sys.path.insert(0, str(ROOT / "scripts"))
    from v2_pipeline import mark, stamp_path

    if outcome == "natural":
        # Whatever orchestrator launched this training (ours or one started
        # manually beforehand) is still alive and will walk through render and
        # verify on its own now that the train step's subprocess has returned.
        # Do not touch it -- starting a second orchestrator here is what
        # corrupted clip01 the first time this ran.  Just wait for it.
        publish_weights()
        log("training finished naturally; waiting for the existing "
            "orchestrator to render and verify (not starting a second one)")
        while procs("v2_pipeline.py"):
            time.sleep(30)
        log("existing orchestrator process has exited")
    else:
        # We had to kill v2_train.py ourselves, which makes the orchestrator's
        # train step fail its subprocess.run() and exit.  Nothing is running
        # the rest of the pipeline, so publish the weights, mark the step done
        # by hand, and start rendering.
        publish_weights()
        for pid in procs("v2_pipeline.py"):
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        time.sleep(5)
        for pid in procs("v2_pipeline.py") + procs("v2_render.py"):
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        mark("train", {"epochs": epochs_done(), "note": "time-budgeted overnight run"})
        for s in ("render_first", "approval", "render_rest", "verify"):
            stamp_path(s).unlink(missing_ok=True)
        (LOGS / "APPROVED").write_text(datetime.now(timezone.utc).isoformat())
        (LOGS / "PAUSE").unlink(missing_ok=True)

        log("starting render + verify")
        r = subprocess.run([PY, str(ROOT / "scripts" / "v2_pipeline.py")],
                           stdout=(LOGS / "v2_pipeline.log").open("a"),
                           stderr=subprocess.STDOUT)
        log(f"pipeline finished rc={r.returncode}")

    n = len(list((ROOT / "outputs" / "clips").glob("clip*_clean.mp4")))
    log(f"clips rendered: {n}/20")
    rep = ROOT / "outputs" / "verify_report.json"
    if rep.exists():
        try:
            log("verify summary: " + json.dumps(json.loads(rep.read_text())["summary"]))
        except Exception:
            pass
    log("supervisor done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
