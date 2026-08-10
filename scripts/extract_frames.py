#!/usr/bin/env python3
"""Extract frames from every video in data/raw_videos/ (or a single --video).

Output: data/frames/<video_slug>/<video_slug>_t<seconds>.jpg
Both the source video and the timestamp are recoverable from the filename,
which assemble_dataset.py relies on to do a time-based train/val split.

Two extraction modes (--mode):
  interval  (default) - fixed fps sampling, e.g. one frame every 1/fps seconds.
                          For sparse sampling (interval >= 2s, e.g. --fps 0.25
                          on a long match) this seeks directly to each target
                          timestamp in parallel instead of decoding the whole
                          video sequentially - for a multi-hour 4K video this
                          is the difference between minutes and hours, since a
                          sequential fps= filter still has to decode every
                          input frame even when only keeping a fraction of them.
  scene                - ffmpeg scene-change detection: only frames where the
                          frame-to-frame scene score exceeds --scene-threshold
                          are kept. Produces fewer near-duplicate frames than
                          interval sampling over static shots, and is more
                          likely to catch the moment a rotating LED sideline
                          board switches to a different ad, which a fixed
                          interval can straddle or miss entirely. Always a
                          sequential full decode - can be slow on long videos.

Examples:
    python scripts/extract_frames.py                       # all videos, default fps
    python scripts/extract_frames.py --video match1.mp4    # just one video
    python scripts/extract_frames.py --fps 2                # denser sampling
    python scripts/extract_frames.py --video match1.mp4 --start 600 --end 720 --fps 5
        # dense re-extraction over a 2-minute ad-heavy segment
    python scripts/extract_frames.py --mode scene --scene-threshold 0.1
        # diverse, de-duplicated frames instead of fixed-interval sampling
    python scripts/extract_frames.py --fps 0.25 --workers 12
        # sparse sampling of a long video, seeking in parallel
    python scripts/extract_frames.py --force                # re-extract even if done
"""
from __future__ import annotations

import argparse
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm

from common import find_videos, load_config, require_ffmpeg, require_ffprobe, slugify, die

# Below this many seconds between sampled frames, sequential fps= decode is
# faster than paying a seek cost per frame; at/above it, parallel seeking wins
# (and wins by a lot on long videos, since sequential decode cost scales with
# video duration regardless of how sparse the output is).
SEEK_MODE_MIN_INTERVAL_S = 2.0


def probe_duration(video: Path, ffprobe: str) -> float:
    cmd = [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(video)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return float(result.stdout.strip())


def _jpg_qscale(quality_0_100: int) -> int:
    """ffmpeg's -qscale:v for mjpeg is 2 (best) .. 31 (worst); map from a
    0-100 'quality' knob so config.yaml stays intuitive."""
    q = max(1, min(100, quality_0_100))
    return round(2 + (31 - 2) * (1 - q / 100))


def _rename_tmp_files(out_dir: Path, slug: str, ext: str, timestamps: list[float]) -> int:
    tmp_files = sorted(out_dir.glob(f"_tmp_{slug}_*.{ext}"))
    count = 0
    for i, f in enumerate(tmp_files):
        t = timestamps[i] if i < len(timestamps) else (timestamps[-1] + (i - len(timestamps) + 1) if timestamps else float(i))
        new_name = out_dir / f"{slug}_t{t:.2f}.{ext}"
        f.rename(new_name)
        count += 1
    return count


def extract_interval(video: Path, out_dir: Path, fps: float, start: float | None,
                      end: float | None, ext: str, quality: int, ffmpeg: str,
                      ffprobe: str, workers: int) -> int:
    interval = 1.0 / fps
    if interval >= SEEK_MODE_MIN_INTERVAL_S:
        return extract_interval_seek(video, out_dir, fps, start, end, ext, quality, ffmpeg, ffprobe, workers)
    return extract_interval_sequential(video, out_dir, fps, start, end, ext, quality, ffmpeg)


def extract_interval_sequential(video: Path, out_dir: Path, fps: float, start: float | None,
                                 end: float | None, ext: str, quality: int, ffmpeg: str) -> int:
    """Dense sampling: decode straight through with ffmpeg's fps= filter.
    Efficient when the gap between kept frames is small relative to seek
    overhead (short clips, or fps close to the source frame rate)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(video.stem)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if end is not None:
        duration = end - (start or 0)
        if duration <= 0:
            die(f"--end ({end}) must be greater than --start ({start or 0})")
        cmd += ["-t", str(duration)]

    vf = f"fps={fps}"
    cmd += ["-vf", vf, "-qscale:v", str(_jpg_qscale(quality))]

    # %d gets the running frame index; we rename to real timestamps after.
    tmp_pattern = str(out_dir / f"_tmp_{slug}_%06d.{ext}")
    cmd += [tmp_pattern]

    subprocess.run(cmd, check=True)

    base_t = start or 0.0
    n = len(list(out_dir.glob(f"_tmp_{slug}_*.{ext}")))
    timestamps = [base_t + i / fps for i in range(n)]
    return _rename_tmp_files(out_dir, slug, ext, timestamps)


def extract_interval_seek(video: Path, out_dir: Path, fps: float, start: float | None,
                           end: float | None, ext: str, quality: int, ffmpeg: str,
                           ffprobe: str, workers: int) -> int:
    """Sparse sampling: seek directly to each target timestamp instead of
    decoding the whole video. A per-frame seek+grab costs roughly a constant
    couple of seconds regardless of where in the video it lands (nearest
    keyframe + a short decode), so this scales with the NUMBER of frames
    wanted rather than the video's duration - the opposite of the sequential
    fps= filter, which must decode every source frame no matter how sparse
    the output is. Runs `workers` grabs concurrently since each is an
    independent ffmpeg process."""
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(video.stem)

    duration = probe_duration(video, ffprobe)
    base_t = start or 0.0
    end_t = end if end is not None else duration
    if end_t <= base_t:
        die(f"Nothing to extract: start ({base_t}) is at or past the video/segment end ({end_t}).")

    interval = 1.0 / fps
    timestamps = []
    t = base_t
    while t < end_t:
        timestamps.append(t)
        t += interval

    def grab(t: float) -> bool:
        out_path = out_dir / f"{slug}_t{t:.2f}.{ext}"
        cmd = [
            ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-ss", str(t), "-i", str(video), "-frames:v", "1",
            "-qscale:v", str(_jpg_qscale(quality)), str(out_path),
        ]
        result = subprocess.run(cmd, capture_output=True)
        return result.returncode == 0 and out_path.exists()

    count = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for ok in tqdm(executor.map(grab, timestamps), total=len(timestamps), desc=f"Seeking ({workers} parallel)"):
            if ok:
                count += 1
    return count


def extract_scene(video: Path, out_dir: Path, threshold: float, start: float | None,
                   end: float | None, ext: str, quality: int, ffmpeg: str) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    slug = slugify(video.stem)

    cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "info"]
    if start is not None:
        cmd += ["-ss", str(start)]
    cmd += ["-i", str(video)]
    if end is not None:
        duration = end - (start or 0)
        if duration <= 0:
            die(f"--end ({end}) must be greater than --start ({start or 0})")
        cmd += ["-t", str(duration)]

    # select only keeps frames whose scene-change score exceeds `threshold`;
    # showinfo prints each kept frame's presentation timestamp to stderr so
    # we can recover real timestamps (scene cuts land at irregular times,
    # unlike interval mode where they're derived from a fixed fps).
    vf = f"select='gt(scene,{threshold})',showinfo"
    tmp_pattern = str(out_dir / f"_tmp_{slug}_%06d.{ext}")
    cmd += ["-vf", vf, "-vsync", "vfr", "-qscale:v", str(_jpg_qscale(quality)), tmp_pattern]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        die(f"ffmpeg scene-detection extraction failed for {video.name}:\n{result.stderr[-2000:]}")

    pts_times = [float(m) for m in re.findall(r"pts_time:(\d+\.?\d*)", result.stderr)]
    base_t = start or 0.0
    timestamps = [base_t + t for t in pts_times]
    return _rename_tmp_files(out_dir, slug, ext, timestamps)


def already_extracted(out_dir: Path) -> bool:
    return out_dir.exists() and any(out_dir.glob("*_t*.jpg")) or any(
        out_dir.glob("*_t*.png") if out_dir.exists() else []
    )


def main():
    cfg = load_config()
    p = cfg["paths"]
    ex = cfg["extract"]

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", type=str, default=None, help="Limit extraction to a single video filename (in data/raw_videos/)")
    ap.add_argument("--mode", choices=["interval", "scene"], default=ex["mode"],
                     help=f"'interval' = fixed fps sampling, 'scene' = ffmpeg scene-change detection (default: {ex['mode']})")
    ap.add_argument("--fps", type=float, default=ex["fps"], help=f"Frames per second to sample, mode=interval (default: {ex['fps']})")
    ap.add_argument("--scene-threshold", type=float, default=ex["scene_threshold"],
                     help=f"ffmpeg scene-score cutoff, mode=scene (default: {ex['scene_threshold']})")
    ap.add_argument("--start", type=float, default=None, help="Start time in seconds (for dense re-extraction of a segment)")
    ap.add_argument("--end", type=float, default=None, help="End time in seconds")
    ap.add_argument("--workers", type=int, default=ex["parallel_workers"],
                     help=f"Parallel ffmpeg processes for sparse seek-based interval extraction (default: {ex['parallel_workers']})")
    ap.add_argument("--force", action="store_true", help="Re-extract even if frames already exist for this video")
    ap.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    args = ap.parse_args()

    ffmpeg = require_ffmpeg()
    ffprobe = require_ffprobe()

    raw_videos_dir = p["raw_videos"]
    if not raw_videos_dir.exists():
        die(f"'{raw_videos_dir}' does not exist. Create it and drop match videos inside.")

    if args.video:
        video_path = raw_videos_dir / args.video
        if not video_path.exists():
            die(f"Video not found: {video_path}")
        videos = [video_path]
    else:
        videos = find_videos(raw_videos_dir)
        if not videos:
            die(
                f"No videos found in '{raw_videos_dir}'. Supported extensions: "
                "mp4, mkv, ts, mov, avi, webm. Drop match videos there and re-run."
            )

    frames_root = p["frames"]
    total = 0
    for video in videos:
        slug = slugify(video.stem)
        out_dir = frames_root / slug

        if not args.force and (args.start is None) and already_extracted(out_dir):
            print(f"[skip] {video.name} — frames already extracted at {out_dir} (use --force to redo)")
            continue

        if args.mode == "scene":
            print(f"[extract] {video.name} -> {out_dir} (mode=scene, threshold={args.scene_threshold})")
            count = extract_scene(
                video=video, out_dir=out_dir, threshold=args.scene_threshold,
                start=args.start, end=args.end, ext=ex["image_ext"],
                quality=ex["jpg_quality"], ffmpeg=ffmpeg,
            )
        else:
            print(f"[extract] {video.name} -> {out_dir} (mode=interval, fps={args.fps})")
            count = extract_interval(
                video=video, out_dir=out_dir, fps=args.fps,
                start=args.start, end=args.end, ext=ex["image_ext"],
                quality=ex["jpg_quality"], ffmpeg=ffmpeg, ffprobe=ffprobe,
                workers=args.workers,
            )
        print(f"  -> wrote {count} frames")
        total += count

    print(f"Done. {total} frames extracted across {len(videos)} video(s).")


if __name__ == "__main__":
    main()
