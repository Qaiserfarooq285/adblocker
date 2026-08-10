#!/usr/bin/env bash
# Keep run_all.py alive across crashes, reboots and power cuts.
#
# run_all.py already stamps every finished step, so relaunching it is always
# safe: it skips completed work and resumes at the first unfinished step. This
# just makes sure something relaunches it when nobody is at the keyboard.
#
# Install (idempotent):
#   crontab -l 2>/dev/null | grep -v 'Ad Blocker/scripts/watchdog.sh' > /tmp/ct
#   { cat /tmp/ct; echo "@reboot sleep 60 && '/home/.../scripts/watchdog.sh'";
#     echo "*/10 * * * * '/home/.../scripts/watchdog.sh'"; } | crontab -
set -u
REPO="/home/qaiserfarooq/Downloads/Ad Blocker"
cd "$REPO" || exit 0
mkdir -p logs

# Paused by hand? stay out of the way. Needed because otherwise the watchdog
# fights anyone trying to stop the pipeline to change a setting: kill it and it
# is back within ten minutes, running the old config.
if [ -f logs/PAUSE ]; then
  exit 0
fi

# Finished? then never start again.
if [ -f data/.stamps/run_all/09_demos.done ]; then
  exit 0
fi

# Already running? leave it alone. Match the interpreter+script, not the word
# "run_all", so a grep of the crontab or an editor buffer cannot look like it.
if pgrep -f "[p]ython .*scripts/run_all\.py" >/dev/null 2>&1; then
  exit 0
fi

echo "[$(date '+%F %T')] watchdog: run_all not running, (re)starting" >> logs/watchdog.log
setsid nohup "$REPO/.venv/bin/python" scripts/run_all.py >> logs/run_all.log 2>&1 < /dev/null &
exit 0
