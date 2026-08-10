#!/usr/bin/env bash
# Wait for the in-flight run to end, drop the demos rendered with the old
# football-tuned config, then let the watchdog re-render them.
#
# The three deleted here were produced before max_board_height_frac 0.09 ->
# 0.75 and the field_of_play/grass/pan drops were disabled, so they show the
# blinking and the trimmed panel. baseball and american_football start after
# the change and are kept.
set -u
cd "/home/qaiserfarooq/Downloads/Ad Blocker" || exit 1
while pgrep -f "scripts/run_all\.py" >/dev/null 2>&1; do sleep 20; done
echo "[$(date '+%F %T')] requeue: run_all exited, clearing stale demos" >> logs/watchdog.log
for s in ice_hockey basketball mma; do
  rm -f "outputs/demo_${s}.mp4" outputs/chunks/${s}_*.mp4 outputs/chunks/${s}_parts.txt
done
rm -f outputs/*.video_only.mp4 outputs/*.trimmed_input.mp4
rm -f data/.stamps/run_all/09_demos.done
rm -f logs/PAUSE
echo "[$(date '+%F %T')] requeue: cleared, watchdog will re-render" >> logs/watchdog.log
