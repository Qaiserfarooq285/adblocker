#!/usr/bin/env bash
# Keep the football pipeline moving without supervision.
#
# Cron runs this every 10 minutes and once at boot.  It starts the orchestrator
# only when nothing is already running, so a power cut resumes automatically
# from the last completed stage and a live run is never disturbed.
#
# Create logs/PAUSE to stop it taking any action.
set -uo pipefail

ROOT="/home/qaiserfarooq/Downloads/Ad Blocker"
cd "$ROOT" || exit 0

mkdir -p logs
[ -f logs/PAUSE ] && exit 0

# cron runs with a minimal PATH, so a bare `python` is not the project's
# interpreter -- and often is not on PATH at all.  Always use the venv by path.
PYBIN="$ROOT/.venv/bin/python"
[ -x "$PYBIN" ] || PYBIN="$(command -v python3)"
[ -x "$PYBIN" ] || exit 0

# Nothing left to do?
if [ -f data/.stamps/v2/verify.done ]; then
    exit 0
fi

# The pipeline itself holds an flock, so a concurrent start is refused rather
# than racing; this process check just avoids the noise of a doomed launch.
if pgrep -af "v2_pipeline.py|v2_train.py|v2_render.py|v2_ocr_tiles.py|v2_ocr_survey.py" \
        | grep -qv watchdog; then
    exit 0
fi

echo "$(date -Is) starting pipeline" >> logs/v2_watchdog.log
setsid nohup "$PYBIN" scripts/v2_pipeline.py >> logs/v2_pipeline.log 2>&1 < /dev/null &
disown
