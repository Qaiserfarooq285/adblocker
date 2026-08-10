"""Shared helpers for betting-ad-blocker scripts: config loading, path
resolution, and small utilities used across the pipeline. Every script in
this folder imports from here instead of re-implementing config parsing.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = REPO_ROOT / "config.yaml"
LOCAL_CONFIG_PATH = REPO_ROOT / "config.local.yaml"


def deep_merge(base: dict, overlay: dict) -> dict:
    """Recursively overlay one dict onto another, returning a new dict."""
    out = dict(base)
    for key, val in (overlay or {}).items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], val)
        else:
            out[key] = val
    return out


def load_config(config_path: str | Path | None = None) -> dict:
    """Load config.yaml, then overlay config.local.yaml if it exists.

    The overlay is how notebooks/pipeline.ipynb sets values from its own
    config cell: config.yaml keeps the documentation and the defaults (and
    its comments, which a programmatic yaml round-trip would destroy), while
    the generated overlay holds only the handful of keys the notebook is
    currently overriding. Delete config.local.yaml to fall back to defaults.

    Returns a plain dict; paths inside `paths:` are resolved to absolute Path
    objects relative to the repo root so scripts can run from any cwd."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        die(f"Config file not found: {path}")
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if config_path is None and LOCAL_CONFIG_PATH.exists():
        with open(LOCAL_CONFIG_PATH) as f:
            cfg = deep_merge(cfg, yaml.safe_load(f) or {})

    resolved = {}
    for key, val in cfg.get("paths", {}).items():
        p = Path(val)
        resolved[key] = p if p.is_absolute() else (REPO_ROOT / p)
    cfg["paths"] = resolved
    return cfg


def die(message: str, code: int = 1) -> None:
    """Print a friendly error to stderr and exit. Use for user-facing
    problems (missing ffmpeg, empty folders, bad paths) — not for bugs."""
    print(f"[error] {message}", file=sys.stderr)
    sys.exit(code)


def resolved_data_yaml(cfg: dict) -> Path:
    """Ultralytics resolves a relative `path:` in a data.yaml against its own
    global DATASETS_DIR setting (or cwd), not against the yaml file's own
    location. To keep data/data.yaml portable and cwd-independent, write a
    sibling copy with `path:` replaced by the dataset dir's absolute path,
    and hand that to Ultralytics instead."""
    src = cfg["paths"]["data_yaml"]
    with open(src) as f:
        data = yaml.safe_load(f)
    data["path"] = str(cfg["paths"]["dataset"].resolve())
    out = src.parent / ".data.resolved.yaml"
    with open(out, "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)
    return out


def require_ffmpeg() -> str:
    """Return the path to the ffmpeg binary or exit with a clear message."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        die(
            "ffmpeg was not found on PATH. Install it first, e.g.:\n"
            "  Ubuntu/Debian:  sudo apt install ffmpeg\n"
            "  macOS:          brew install ffmpeg\n"
            "  Windows:        https://ffmpeg.org/download.html"
        )
    return ffmpeg


def require_ffprobe() -> str:
    """Return the path to the ffprobe binary or exit with a clear message.
    ffprobe ships alongside ffmpeg in virtually every distribution."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe is None:
        die("ffprobe was not found on PATH (it normally ships alongside ffmpeg - reinstall ffmpeg).")
    return ffprobe


def require_nonempty_dir(path: Path, hint: str) -> list[Path]:
    """Return sorted entries in `path`, or exit with a friendly message if
    the directory is missing or empty. `hint` tells the user what to do."""
    if not path.exists() or not any(path.iterdir()):
        die(f"'{path}' is empty or missing. {hint}")
    return sorted(path.iterdir())


VIDEO_EXTS = {".mp4", ".mkv", ".ts", ".mov", ".avi", ".webm"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_videos(raw_videos_dir: Path) -> list[Path]:
    return sorted(
        p for p in raw_videos_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS
    )


def slugify(name: str) -> str:
    """Turn a filename stem into a filesystem/label-safe slug."""
    out = []
    for ch in name.strip().lower():
        if ch.isalnum():
            out.append(ch)
        elif ch in (" ", "-", "_", "."):
            out.append("_")
    slug = "".join(out)
    while "__" in slug:
        slug = slug.replace("__", "_")
    return slug.strip("_") or "video"
