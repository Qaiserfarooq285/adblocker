#!/usr/bin/env python3
"""Offline demo pipeline: video in -> video out with betting ads covered by a
flat, frozen, board-coloured panel, while never painting over people.

The three collaborating pieces:

  fill.py     the ONLY fill path. A panel is a flat quad of colour sampled
              from the board's own face and then muted (see that module for
              why sampling the whole region produced a distracting orange
              block). cv2.inpaint and every blur fallback are gone.

  persons.py  person masks with tracked ids, held across detection gaps and
              advanced by the camera affine, plus a second pass inside the
              panel ROIs. People always win over the panel.

  this file   camera-motion estimation, scene-cut detection, whole-strip
              merging, panel lifecycle (create/smooth/propagate/fade), and
              the per-frame order of operations:

                detect -> merge strips -> smooth/propagate quads -> clamp ->
                rasterize panel mask -> dilate person masks -> subtract ->
                feather edge -> composite fill

Inference runs at a deliberately low conf to FEED the tracker; creating a
panel still needs `panel_create_frames` strong detections, while sustaining
one needs almost nothing (trackers/betting_bytetrack.yaml). On frames with no
detection a live panel is carried by the global camera affine rather than
frozen or dropped, so it stays glued to the board through motion-blurred pans.

--debug writes a side-by-side diagnostic video and per-frame state logs.

Usage:
    python scripts/process_video.py --input data/raw_videos/match1.mp4
    python scripts/process_video.py --input match1.mp4 --start 1955 --end 1985
    python scripts/process_video.py --input match1.mp4 --debug
"""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np

from common import die, load_config, require_ffmpeg, slugify
from fill import (PanelTrack, canonical_quad, clamp_strip_thickness, composite_panels,
                  expand_quad, goal_structure_mask, hue_distance, load_brand_colors,
                  looks_like_grass, measure_bottom_edge_offset, persons_in_front_mask,
                  pick_fill_color, quad_from_mask, quad_overlap, sample_band_color,
                  sample_band_quad, scene_average_color, shift_quad_edge,
                  strip_thickness)
from persons import PersonMaskTracker
from brand_evidence import BettingBrandVerifier

CLASS_PERSON = 0
CLASS_BOARD = 1
CLASS_OVERLAY = 2
CLASS_NAMES = {0: "person", 1: "betting_board", 2: "betting_overlay"}


def renderable_panel(quad: np.ndarray, cls: int, frame_w: int, frame_h: int,
                     safety: dict) -> tuple[bool, str]:
    """Reject implausibly large detections before sticky tracking paints them."""
    if not safety.get("enabled", True):
        return True, ""
    area_frac = abs(float(cv2.contourArea(quad.astype(np.float32)))) / max(1.0, frame_w * frame_h)
    x0, y0 = quad.min(axis=0)
    x1, y1 = quad.max(axis=0)
    bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
    if cls == CLASS_BOARD:
        if area_frac > float(safety.get("max_board_area_frac", 0.12)):
            return False, f"area {area_frac:.3f}"
        thickness = strip_thickness(quad)
        if thickness > float(safety.get("max_board_thickness_frac", 0.30)) * frame_h:
            return False, f"thickness {thickness:.0f}px"
        aspect = max(bw, bh) / min(bw, bh)
        if aspect < float(safety.get("min_board_aspect_ratio", 1.6)):
            return False, f"aspect {aspect:.2f}"
    elif area_frac > float(safety.get("max_overlay_area_frac", 0.08)):
        return False, f"overlay area {area_frac:.3f}"
    return True, ""


# ---------------------------------------------------------------------------
# Camera motion + scene cuts
# ---------------------------------------------------------------------------

class CameraMotion:
    """Global affine between consecutive frames from sparse LK flow.

    The mask used to blank rows 30-95% and columns 25-75% - a huge central
    block - on the theory that the pitch centre is full of players moving
    against the camera. That is a football assumption twice over: it presumes
    the interesting background is around the edges, and that the centre is
    textureless enough to skip. On a hockey broadcast the rink fills the frame
    and the ice is nearly featureless, so blanking the middle left almost
    nothing to track and the affine came out wrong. Measured on the hockey
    clip, panel motion residual against this affine was LARGER than simply
    assuming the panel had not moved (7.8px vs 5.2px) - i.e. the estimate was
    worse than useless, and a panel locked to it inherited the error.

    RANSAC already rejects the moving players, so the mask is now a light touch
    and is dropped entirely whenever it starves the tracker of features."""

    def __init__(self, w: int, h: int, exclude_frac: float = 0.0):
        self.prev_gray: np.ndarray | None = None
        self.w, self.h = w, h
        self.feature_mask = None
        if exclude_frac > 0:
            m = np.full((h, w), 255, dtype=np.uint8)
            y0 = int((0.5 - exclude_frac / 2) * h)
            y1 = int((0.5 + exclude_frac / 2) * h)
            x0 = int((0.5 - exclude_frac / 2) * w)
            x1 = int((0.5 + exclude_frac / 2) * w)
            m[y0:y1, x0:x1] = 0
            self.feature_mask = m
        self.n_features = 0
        self.n_fallback = 0

    def _detect(self, gray, mask):
        return cv2.goodFeaturesToTrack(gray, maxCorners=600, qualityLevel=0.01,
                                       minDistance=10, mask=mask, blockSize=7)

    def update(self, gray: np.ndarray) -> np.ndarray | None:
        affine = None
        if self.prev_gray is not None:
            p0 = self._detect(self.prev_gray, self.feature_mask)
            # Starved by the mask? try again on the whole frame rather than
            # return a bad affine or none at all.
            if (p0 is None or len(p0) < 60) and self.feature_mask is not None:
                p0 = self._detect(self.prev_gray, None)
                self.n_fallback += 1
            self.n_features = 0 if p0 is None else len(p0)
            if p0 is not None and len(p0) >= 12:
                p1, st, _ = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, p0, None,
                                                     winSize=(21, 21), maxLevel=3)
                if p1 is not None:
                    good = st.ravel() == 1
                    if good.sum() >= 12:
                        affine, _ = cv2.estimateAffinePartial2D(
                            p0[good], p1[good], method=cv2.RANSAC,
                            ransacReprojThreshold=2.5, maxIters=3000,
                            confidence=0.995)
        self.prev_gray = gray
        return affine


def ghost_reason(tr: PanelTrack, frame: np.ndarray, person_mask_dilated: np.ndarray,
                 affine: np.ndarray | None, pcfg: dict, band_frac: float,
                 frame_w: int, frame_h: int) -> str | None:
    """Why this propagated panel should be dropped, or None to keep coasting.

    A panel with no detection is running on inertia. Inertia is fine for a
    dropped frame or two; it is NOT fine as a way to keep a panel alive after
    the board has left the shot, which is how a ghost ends up sliding across
    the centre circle. Every check here answers the same question: is there
    still real evidence a board is under this quad?"""
    if tr.missed > int(pcfg.get("max_propagation_frames", 6)):
        return "propagation cap"

    if affine is None:
        return "no camera motion estimate"

    # A fast pan with no detections means the board is leaving frame - let it
    # go rather than drag it along with the camera.
    pan = float(np.hypot(affine[0, 2], affine[1, 2]))
    if pan > float(pcfg.get("pan_drop_px", 25)):
        return f"pan {pan:.0f}px with no detection"

    # A perimeter board is never in the middle of the PITCH - but that is a
    # statement about football, not about sport. A basketball courtside LED, a
    # hockey dasher board and the branding printed on an MMA canvas all sit low
    # and central in frame, exactly where this box says a board cannot be.
    # Measured on the five-sport render it fired 69 times out of 153 drops,
    # killing live panels and making them blink. Set field_of_play to null for
    # anything that is not football.
    fop = pcfg.get("field_of_play")
    if fop:
        fx0, fy0, fx1, fy1 = fop
        cx, cy = tr.quad.mean(axis=0)
        if (fx0 * frame_w <= cx <= fx1 * frame_w) and (fy0 * frame_h <= cy <= fy1 * frame_h):
            return "drifted into the field of play"

    if pcfg.get("verify_content", True) and tr.ref_color is not None:
        under = sample_band_color(frame, tr.quad, person_mask_dilated, band_frac)
        if under is not None:
            # Grass is likewise football-only: there is none on ice, hardwood or
            # a canvas, so this can only produce false drops there.
            if pcfg.get("drop_on_grass", True) and looks_like_grass(under):
                return "now over grass"
            # An LED board CYCLES ITS ADVERT. Holding it to the hue it had when
            # the track was born drops it the moment the ad changes, which is
            # the board doing its job rather than the panel drifting off it.
            if hue_distance(under, tr.ref_color) > float(pcfg.get("content_hue_tol", 30)):
                return "content no longer matches the board"
    return None


def enforce_thin_strip(tr: PanelTrack, frame: np.ndarray, icfg: dict, ema: float) -> tuple:
    """Final geometry guard, applied after merging/smoothing/propagation and
    before anything is drawn. Returns (thickness_px, cap_px, bit, snap_dy).

    Two independent guarantees:

      thickness cap  the panel may never be thicker than
                     `max_board_height_frac` of frame height, measured
                     PERPENDICULAR to the strip. Over-thick detections always
                     overshoot upward into the crowd, so the trim is anchored
                     at the bottom edge (see clamp_strip_thickness).

      bottom anchor  the lower edge is pinned to the board/grass seam found in
                     the image. Measured on this footage the detector's bottom
                     edge is already good to 3-5px, but 3-5px of shortfall is a
                     visible red sliver of un-hidden advert along the bottom of
                     the panel. The correction is EMA'd onto the track rather
                     than applied raw, because a per-frame re-snap is precisely
                     the kind of thing that makes a panel vibrate."""
    frame_h = frame.shape[0]
    cap = float(icfg.get("max_board_height_frac", 0.09)) * frame_h
    before = strip_thickness(tr.quad)
    tr.quad = clamp_strip_thickness(tr.quad, cap)
    bit = before > cap + 0.5

    snap_dy = 0.0
    scfg = icfg.get("bottom_snap", {}) or {}
    if scfg.get("enabled", True):
        max_shift = float(scfg.get("max_shift_frac", 0.35)) * max(1.0, strip_thickness(tr.quad))
        measured = measure_bottom_edge_offset(frame, tr.quad, max_shift)
        if measured is not None:
            tr.bottom_offset = ema * measured + (1.0 - ema) * tr.bottom_offset
        # decay toward zero when the seam cannot be seen (over the goalmouth),
        # so a stale correction never outlives the evidence for it
        else:
            tr.bottom_offset *= (1.0 - ema)
        snap_dy = float(np.clip(tr.bottom_offset, -max_shift, max_shift))
        if abs(snap_dy) >= 0.5:
            tr.quad = canonical_quad(shift_quad_edge(tr.quad, snap_dy, "bottom"))
    return strip_thickness(tr.quad), cap, bit, snap_dy


def hsv_hist(frame_small: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame_small, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist


# ---------------------------------------------------------------------------
# Debug rendering
# ---------------------------------------------------------------------------

def draw_debug_left(raw: np.ndarray, det_info, person_mask: np.ndarray,
                    person_tracks, tracks: dict[int, PanelTrack],
                    ocfg: dict | None = None, dilate_kernel=None,
                    gcfg: dict | None = None, geom=None,
                    occluders: np.ndarray | None = None) -> np.ndarray:
    """Raw per-detection boxes (red, with conf) so fragmentation is visible;
    person mask tinted so a referee miss is obvious; each panel's
    raw-detection-union bound in cyan so over-capture is visible against it;
    the actual rendered panel in green (detected) or orange (camera-motion
    propagated); the colour-sample band in magenta.

    Occlusion accounting is the point of the person colours here: a person who
    actually punches a hole in the panel is drawn SOLID, and one ignored as
    being behind the board is drawn hollow with 'ign'. When a referee goes
    missing, this says immediately whether he was undetected or detected and
    then discarded as crowd."""
    vis = raw.copy()
    tint = vis.copy()
    tint[person_mask > 0] = (0, 255, 255)
    vis = cv2.addWeighted(tint, 0.35, vis, 0.65, 0)

    # who actually occludes: the tracker's LATCHED verdict when available, so
    # the overlay shows what was really subtracted rather than a fresh guess
    if occluders is not None:
        occl = occluders.copy()
    else:
        occl = np.zeros(raw.shape[:2], np.uint8)
        if ocfg is not None:
            for tr in tracks.values():
                if tr.active:
                    occl |= persons_in_front_mask(person_mask, tr.quad, raw.shape[0],
                                                  ocfg, dilate_kernel)
    if occl.any():
        t2 = vis.copy()
        t2[occl > 0] = (0, 0, 255)
        vis = cv2.addWeighted(t2, 0.45, vis, 0.55, 0)

    if gcfg is not None and gcfg.get("enabled", True):
        for tr in tracks.values():
            if not tr.active:
                continue
            g = goal_structure_mask(raw, tr.quad, gcfg)
            if g is not None:
                vis[g > 0] = (255, 255, 255)

    for tr in person_tracks:
        x, y = tr.origin
        ph, pw = tr.patch.shape[:2]
        patch_area = int(tr.patch.sum())
        # a held patch can hang off the frame edge, so clip both sides to the
        # overlapping window before ANDing them
        fh, fw = occl.shape[:2]
        cx0, cy0 = max(0, x), max(0, y)
        cx1, cy1 = min(fw, x + pw), min(fh, y + ph)
        hit = 0
        if occl.any() and cx0 < cx1 and cy0 < cy1:
            hit = int((occl[cy0:cy1, cx0:cx1]
                       & tr.patch[cy0 - y:cy1 - y, cx0 - x:cx1 - x]).sum())
        occluding = patch_area > 0 and hit > 0.2 * patch_area
        col = (0, 140, 255) if tr.held else (0, 220, 220)
        if getattr(tr, "provisional", False):
            col = (255, 0, 255)                      # motion-fallback mask
        cv2.rectangle(vis, (x, y), (x + pw, y + ph), col, 2 if occluding else 1)
        flags = ("H" if tr.held else "") + ("L" if getattr(tr, "in_front", False) else "") \
                + ("M" if getattr(tr, "provisional", False) else "")
        label = f"p{tr.tid}{flags}{'' if occluding else ' ign'}"
        cv2.putText(vis, label, (x, max(12, y - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)

    geom_by_id = {g[0]: g for g in (geom or [])}

    for tid, conf, quad in det_info:
        cv2.polylines(vis, [np.round(quad).astype(np.int32)], True, (0, 0, 255), 1)
        x, y = quad.min(axis=0).astype(int)
        cv2.putText(vis, f"{conf:.2f}", (x, max(14, y - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    for tr in tracks.values():
        if not tr.active:
            continue
        if tr.bound_quad is not None:
            cv2.polylines(vis, [np.round(tr.bound_quad).astype(np.int32)], True, (255, 255, 0), 2)
        cv2.polylines(vis, [np.round(sample_band_quad(tr.quad, 0.6)).astype(np.int32)],
                      True, (255, 0, 255), 1)
        color = (0, 255, 0) if tr.detected_this_frame else (255, 160, 0)
        tag = "D" if tr.detected_this_frame else f"P{tr.missed}"
        cv2.polylines(vis, [np.round(tr.quad).astype(np.int32)], True, color, 2)
        x, y = tr.quad.min(axis=0).astype(int)
        g = geom_by_id.get(tr.tid)
        extra = ""
        if g is not None:
            _, thick, cap, bit, snap = g
            extra = f" t{thick:.0f}/{cap:.0f}{'!' if bit else ''} dy{snap:+.0f}"
        cv2.putText(vis, f"id{tr.tid} {tag} fade{tr.fade:.1f}{extra}",
                    (x, min(vis.shape[0] - 6, y + 24)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return vis


# ---------------------------------------------------------------------------
# Audio remux
# ---------------------------------------------------------------------------

def remux_audio(video_only_path: Path, source_with_audio: Path, final_path: Path, ffmpeg: str) -> None:
    cmd = [
        ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(video_only_path), "-i", str(source_with_audio),
        "-c:v", "copy", "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest", str(final_path),
    ]
    if subprocess.run(cmd).returncode != 0:
        print("[warn] audio remux failed, keeping video-only output.")
        video_only_path.replace(final_path)
    else:
        video_only_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    cfg = load_config()
    p = cfg["paths"]
    icfg = dict(cfg["inference"])
    mcfg = dict(icfg.get("mute", {}))
    pcfg = dict(icfg.get("propagation", {}))
    ocfg = dict(icfg.get("occlusion", {}))
    gcfg = dict(icfg.get("protect_goal", {}))

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=str, required=True, help="Path to an input video (any resolution/fps)")
    ap.add_argument("--output", type=str, default=None, help="Output path (default: outputs/<name>_blocked.mp4)")
    ap.add_argument("--start", type=float, default=None, help="Start time in seconds")
    ap.add_argument("--end", type=float, default=None, help="End time in seconds")
    ap.add_argument("--model", type=str, default=None, help="Weights (default: inference.weights)")
    ap.add_argument("--conf", type=float, default=icfg["conf"],
                    help="Tracker feed confidence - keep LOW; panel creation is gated separately")
    ap.add_argument("--iou", type=float, default=icfg["iou"])
    ap.add_argument("--imgsz", type=int, default=icfg["imgsz"])
    ap.add_argument("--device", type=str, default=str(icfg["device"]))
    ap.add_argument("--debug", action="store_true",
                    help="Also write <output>_debug.mp4 side-by-side and log per-frame state")
    ap.add_argument("--capture-review", action="store_true",
                    help="Save one frame + draft labels per newly-tracked instance for later review")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()

    ffmpeg = require_ffmpeg()

    input_path = Path(args.input)
    if not input_path.exists():
        candidate = p["raw_videos"] / args.input
        if candidate.exists():
            input_path = candidate
        else:
            die(f"Input video not found: {args.input}")

    model_path = Path(args.model) if args.model else Path(icfg["weights"])
    if not model_path.exists():
        die(f"Model weights not found: {model_path}. Train first with scripts/train.py.")

    tracker_path = Path(icfg["tracker"])
    if not tracker_path.exists():
        die(f"Tracker config not found: {tracker_path}")

    p["outputs"].mkdir(parents=True, exist_ok=True)
    output_path = Path(args.output) if args.output else p["outputs"] / f"{input_path.stem}_blocked.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_video_path = output_path.with_suffix(".video_only.mp4")

    trimmed_path = None
    source_path = input_path
    if args.start is not None or args.end is not None:
        if args.end is not None and args.end <= (args.start or 0):
            die(f"--end ({args.end}) must be greater than --start ({args.start or 0})")
        trimmed_path = output_path.with_suffix(".trimmed_input.mp4")
        trim_cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if args.start is not None:
            trim_cmd += ["-ss", str(args.start)]
        trim_cmd += ["-i", str(input_path)]
        if args.end is not None:
            trim_cmd += ["-t", str(args.end - (args.start or 0))]
        trim_cmd += ["-c", "copy", str(trimmed_path)]
        print(f"Trimming input to [{args.start or 0}, {args.end if args.end is not None else 'end'}]s...")
        subprocess.run(trim_cmd, check=True)
        source_path = trimmed_path

    try:
        from ultralytics import YOLO
    except ImportError:
        die("ultralytics is not installed. Run: pip install -r requirements.txt")

    betting_model = YOLO(str(model_path))
    person_model = YOLO(icfg["person_model"])   # stock COCO weights, persons only
    try:
        brand_verifier = BettingBrandVerifier(icfg.get("brand_verification", {}), args.device)
    except FileNotFoundError as exc:
        die(str(exc))

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        die(f"Could not open video: {source_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp_video_path), fourcc, src_fps, (src_w, src_h))
    if not writer.isOpened():
        die(f"Could not open VideoWriter for {tmp_video_path}")

    debug_writer = None
    if args.debug:
        debug_path = output_path.with_name(output_path.stem + "_debug.mp4")
        debug_writer = cv2.VideoWriter(str(debug_path), fourcc, src_fps, (src_w * 2, src_h))
        if not debug_writer.isOpened():
            die(f"Could not open debug VideoWriter for {debug_path}")
        print(f"[debug] writing side-by-side diagnostics to {debug_path}")

    # Renderer init log - this line existing is the regression guard: if you
    # ever see anything but 'frozen-board-median-muted' here, a smear path
    # crept back in.
    # Surface the config source: a stale config.local.yaml silently overriding
    # config.yaml is invisible otherwise, and produced exactly one wrong-colour
    # run before this line existed.
    from common import LOCAL_CONFIG_PATH
    print(f"[config] config.yaml"
          + (f" + {LOCAL_CONFIG_PATH.name} (overlay active)" if LOCAL_CONFIG_PATH.exists() else ""))

    brand_colors = load_brand_colors(p["logos"], icfg.get("brand_color_file", "color.json"))
    if brand_colors:
        pinned = ", ".join(f"{k}=BGR{np.round(v).astype(int).tolist()}" for k, v in brand_colors.items())
        print(f"[fill] pinned brand colour(s): {pinned}")
    else:
        print(f"[fill] no pinned brand colours (drop a {icfg.get('brand_color_file', 'color.json')} "
              f"into input/logos/<brand>/ to pin one); sampling the board instead")

    print(f"[fill] renderer=frozen-colour-flat-panel feather={icfg['panel_feather_px']}px "
          f"band={mcfg.get('sample_band_frac', 0.6)} sat x{mcfg.get('saturation_mult', 0.75)} "
          f"val x{mcfg.get('value_mult', 0.8)} scene_blend={mcfg.get('scene_blend', 0.0)} "
          f"(colour frozen once per track; cv2.inpaint removed; no blur fallback exists)")
    print(f"[track] tracker={tracker_path} feed_conf={args.conf} "
          f"create>={icfg['panel_create_conf']}x{icfg['panel_create_frames']}f "
          f"ema={icfg['ema_alpha']}")
    vcfg = icfg.get("brand_verification", {})
    print(f"[verify] named PDF-brand OCR + logo detector required="
          f"{vcfg.get('enabled', True)} (logo={vcfg.get('logo_weights', 'disabled')})")
    print(f"[ghost] propagate<={pcfg.get('max_propagation_frames', 6)}f "
          f"pan_drop>{pcfg.get('pan_drop_px', 25)}px verify_content={pcfg.get('verify_content', True)} "
          f"field_of_play={pcfg.get('field_of_play')}")
    print(f"[person] model={icfg['person_model']} conf={icfg['person_conf']} "
          f"hold={icfg['person_hold_frames']}f dilate={icfg['person_dilate_px']}px "
          f"roi_pass={icfg['roi_person_pass']} roi_upscale={icfg.get('roi_upscale', 2.0)}x "
          f"retina_masks=True")
    print(f"[geom] max_board_height_frac={icfg.get('max_board_height_frac', 0.09)} "
          f"(={icfg.get('max_board_height_frac', 0.09) * src_h:.0f}px, measured PERPENDICULAR "
          f"to the strip - a slanted board has a much larger bbox height) "
          f"bottom_snap={(icfg.get('bottom_snap', {}) or {}).get('enabled', True)}")
    print(f"[occl] zone=board..+{ocfg.get('occlusion_zone_down_frac', 0.25)}xH down, "
          f"{ocfg.get('occlusion_zone_up_px', 6)}px up; only people whose FEET reach the "
          f"board's bottom edge occlude (crowd behind the board is ignored); "
          f"verdict LATCHED after {ocfg.get('front_latch_frames', 2)} frame(s) and never "
          f"re-decided (a per-frame verdict is what made the referee blink)")
    print(f"[occl] motion_person_fallback={icfg.get('motion_person_fallback', True)} "
          f"(grass-anchored moving blobs the detector missed)")
    print(f"[goal] protect_goal={gcfg.get('enabled', True)} "
          f"vertical_len={gcfg.get('vertical_len_frac', 1.2)}x strip thickness")
    print(f"Processing '{source_path.name}' ({src_w}x{src_h} @ {src_fps:.2f}fps, ~{total_frames} frames)")

    video_slug = slugify(input_path.stem)
    capture_dir = None
    captured_track_ids: set[int] = set()
    n_captured = 0
    if args.capture_review:
        capture_dir = p["prelabel_review"] / f"{video_slug}_live"
        (capture_dir / "images").mkdir(parents=True, exist_ok=True)
        (capture_dir / "labels").mkdir(parents=True, exist_ok=True)
        (capture_dir / "classes.txt").write_text("\n".join(CLASS_NAMES[i] for i in sorted(CLASS_NAMES)) + "\n")
        print(f"Capturing new detections for review to {capture_dir}")

    # motion runs on half-res grays; its mask must match that size
    motion = CameraMotion(src_w // 2, src_h // 2,
                          float(icfg.get("motion_exclude_frac", 0.0)))
    people = PersonMaskTracker(person_model, icfg, args.imgsz)
    tracks: dict[int, PanelTrack] = {}
    prev_hist = None
    ema = float(icfg["ema_alpha"])
    dilate_px = int(icfg["person_dilate_px"])
    dilate_kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate_px + 1,) * 2)
                     if dilate_px > 0 else None)
    n_frames = 0
    n_cuts = 0
    n_clamped = 0
    n_rejected_geometry = 0
    thickness_log: list[float] = []
    start_time = time.time()

    while True:
        ok, raw = cap.read()
        if not ok:
            break
        n_frames += 1
        frame = raw.copy()
        gray = cv2.cvtColor(cv2.resize(raw, (src_w // 2, src_h // 2)), cv2.COLOR_BGR2GRAY)

        # --- scene-cut check (HSV hist correlation; robust to fast pans) ---
        hist = hsv_hist(cv2.resize(raw, (320, 180)))
        if prev_hist is not None:
            correl = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            if correl < icfg["scene_cut_min_correl"]:
                n_cuts += 1
                print(f"[cut] frame {n_frames}: hist correlation {correl:.3f} < "
                      f"{icfg['scene_cut_min_correl']} - clearing {len(tracks)} track(s)")
                tracks.clear()
                people.tracks.clear()
                motion.prev_gray = None
        prev_hist = hist

        # --- global camera motion (half-res grays; scale affine back up) ---
        affine_half = motion.update(gray)
        affine = None
        if affine_half is not None:
            affine = affine_half.copy()
            affine[:, 2] *= 2.0   # translation back to full-res pixels

        # --- 1. detect (low conf feeds the sticky tracker) ---
        result = betting_model.track(
            raw, tracker=str(tracker_path), persist=True,
            conf=args.conf, iou=args.iou, imgsz=args.imgsz,
            device=args.device, classes=[CLASS_BOARD, CLASS_OVERLAY],
            retina_masks=True, verbose=False,
        )[0]

        # --- persons: tracked, held across gaps, plus a pass inside the panel
        #     ROIs (last frame's panels - they barely move between frames) ---
        roi_quads = [t.bound_quad if t.bound_quad is not None else t.quad
                     for t in tracks.values() if t.active]
        person_mask, n_person_det, n_person_held = people.update(raw, affine, roi_quads)
        person_mask_dilated = (cv2.dilate(person_mask, dilate_kernel)
                               if dilate_kernel is not None else person_mask)
        # People in FRONT of the board, decided per track and latched (see
        # PersonTrack.in_front). Dilated here so the subtraction keeps a margin
        # around them, and used on every frame - detected, held or propagated.
        occluders = people.last_occluders
        occluders_dilated = (cv2.dilate(occluders, dilate_kernel)
                             if (occluders is not None and dilate_kernel is not None)
                             else occluders)
        scene_avg = scene_average_color(raw, person_mask)

        # --- 2. merge strips ACROSS detections: same-line betting regions
        #        whose horizontal gap is small are one physical ad strip
        #        ("Betano | CONFIA | Betano" - CONFIA is Betano's slogan, not
        #        a neighbour). The gap cap is what stops the merge from ever
        #        bridging across to Kraken/Marriott: those are not betting
        #        detections, and a betting region on their far side is farther
        #        than merge_gap_height_ratio x height away. ---
        det_quads: dict[int, tuple[float, np.ndarray, np.ndarray]] = {}
        det_info = []
        if result.boxes is not None and len(result.boxes) > 0 and result.boxes.id is not None:
            cls_arr = result.boxes.cls.cpu().numpy().astype(int)
            conf_arr = result.boxes.conf.cpu().numpy()
            id_arr = result.boxes.id.cpu().numpy().astype(int)
            mask_arr = None
            if result.masks is not None:
                mask_arr = result.masks.data.cpu().numpy()
                if mask_arr.shape[1:] != (src_h, src_w):
                    mask_arr = np.stack([cv2.resize(m, (src_w, src_h),
                                                    interpolation=cv2.INTER_NEAREST) for m in mask_arr])
            members = []
            if mask_arr is not None:
                for i, tid in enumerate(id_arr):
                    quad = quad_from_mask((mask_arr[i] > 0.5).astype(np.uint8))
                    if quad is None:
                        continue
                    xs, ys = quad[:, 0], quad[:, 1]
                    members.append((int(tid), float(conf_arr[i]), int(cls_arr[i]),
                                    xs.min(), xs.max(), ys.min(), ys.max(), quad))
                    det_info.append((int(tid), float(conf_arr[i]), quad))

            members.sort(key=lambda t: t[3])
            groups: list[list] = []
            for mem in members:
                placed = False
                for g in groups:
                    _, _, _, gx0, gx1, gy0, gy1, _ = g[-1]
                    _, _, _, mx0, mx1, my0, my1, _ = mem
                    h_a, h_b = gy1 - gy0, my1 - my0
                    v_overlap = min(gy1, my1) - max(gy0, my0)
                    gap = mx0 - gx1
                    if (v_overlap > 0.5 * min(h_a, h_b)
                            and gap < icfg["merge_gap_height_ratio"] * max(h_a, h_b)):
                        g.append(mem)
                        placed = True
                        break
                if not placed:
                    groups.append([mem])

            for g in groups:
                gid = min(m[0] for m in g)
                conf = max(m[1] for m in g)
                cls_g = g[0][2]
                pts = np.concatenate([m[7] for m in g]).astype(np.float32)
                quad = expand_quad(canonical_quad(cv2.boxPoints(cv2.minAreaRect(pts))),
                                   icfg["panel_margin_frac"])
                okay, why = renderable_panel(quad, cls_g, src_w, src_h,
                                              icfg.get("render_safety", {}))
                if not okay:
                    n_rejected_geometry += 1
                    if n_rejected_geometry <= 10:
                        print(f"[safety] frame {n_frames}: rejected {CLASS_NAMES[cls_g]} {why}")
                    continue
                det_quads[gid] = (conf, quad, quad.copy())
                if gid not in tracks:
                    # a re-identified strip must not restart the K-frame gate
                    # (that would blink); inherit an overlapping panel's state
                    # Associate by GEOMETRY, not by tracker id. Measured on
                    # the hockey clip the tracker produced 304 panel births in
                    # 60s (~5/second): every re-id starts a fresh panel in a
                    # slightly different place, which is what reads as the block
                    # jumping around. Matching against the affine-PREDICTED quad
                    # rather than the stale one is what makes this survive a
                    # fast pan, and 0.35 is loose enough to re-attach after a
                    # few missed frames.
                    thr = float(icfg.get("panel_match_iou", 0.35))
                    heir, best_iou = None, thr
                    for tr_ in tracks.values():
                        if tr_.cls != cls_g:
                            continue
                        ref = tr_.quad
                        if affine is not None:
                            ref = canonical_quad(cv2.transform(
                                ref.reshape(-1, 1, 2), affine).reshape(-1, 2))
                        ov = quad_overlap(ref, quad)
                        if ov > best_iou:
                            heir, best_iou = tr_, ov
                    if heir is not None:
                        del tracks[heir.tid]
                        heir.tid = gid
                        heir.dying = False
                        tracks[gid] = heir
                    else:
                        tracks[gid] = PanelTrack(tid=gid, cls=cls_g, quad=quad.copy(),
                                                 bound_quad=quad.copy(), fade=0.0)

        if capture_dir is not None and det_quads:
            new_tids = [tid for tid in det_quads if tid not in captured_track_ids]
            if new_tids:
                lines = []
                for tid in new_tids:
                    norm = (det_quads[tid][1] / np.array([src_w, src_h], dtype=float)).clip(0, 1)
                    lines.append(f"{tracks[tid].cls} " + " ".join(f"{v:.6f}" for v in norm.reshape(-1)))
                    captured_track_ids.add(tid)
                stem = f"{video_slug}_t{n_frames / src_fps:.2f}"
                cv2.imwrite(str(capture_dir / "images" / f"{stem}.jpg"), raw)
                (capture_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")
                n_captured += 1

        # --- 3. smooth / propagate quads, then clamp ---
        fade_step = 1.0 / max(1, int(icfg["fade_frames"]))
        dropped = []
        for tid, tr in tracks.items():
            det = det_quads.get(tid)
            tr.detected_this_frame = det is not None
            if det is not None:
                conf, quad, bound = det
                tr.dying = False                 # fresh evidence resurrects a fading panel
                tr.missed = 0
                # The clamp bound is EMA-smoothed like the quad itself. Taking
                # it raw from each frame's detection let detection jitter reach
                # the rendered edge directly, bypassing the smoothing - which
                # is what made the panel vibrate.
                tr.bound_quad = (bound if tr.bound_quad is None else
                                 canonical_quad(ema * bound + (1.0 - ema) * tr.bound_quad))
                if not tr.active:
                    tr.hits = tr.hits + 1 if conf >= icfg["panel_create_conf"] else 0
                    tr.quad = quad
                    if tr.hits >= icfg["panel_create_frames"] and n_frames >= tr.verify_after_frame:
                        verified, evidence = brand_verifier.verify(raw, quad)
                        if not verified:
                            # Keep the segment candidate alive, but require fresh
                            # detections before checking it again. A generic board
                            # must never become paint merely through persistence.
                            tr.hits = 0
                            tr.verify_after_frame = n_frames + int(vcfg.get("retry_frames", 90))
                            print(f"[verify] frame {n_frames}: panel {tid} rejected ({evidence}); "
                                  f"retry after frame {tr.verify_after_frame}")
                            continue
                        color, ref, src = pick_fill_color(raw, quad, person_mask_dilated,
                                                          scene_avg, mcfg, brand_colors,
                                                          float(icfg.get("brand_color_snap_hue", 30)))
                        if color is not None:
                            # Frozen HERE, once, for the life of the track. There
                            # is deliberately no other assignment to tr.color in
                            # this file - a per-frame recompute is a vibrating panel.
                            tr.color, tr.ref_color, tr.color_source = color, ref, src
                            tr.active = True
                            tr.fade = 0.0        # fade in from nothing
                            print(f"[verify] track {tid}: confirmed named brand '{evidence}'")
                            print(f"[fill] track {tid}: colour frozen from {src} "
                                  f"BGR={np.round(color).astype(int).tolist()}")
                else:
                    # RIGID LOCK. A confirmed panel is stuck to a physical
                    # board, so it should ride the camera and nothing else.
                    # Steering it by the detection every frame is what made the
                    # block wobble: the detector jitters a few px per frame and
                    # the panel faithfully copied that jitter (median 6px, p90
                    # 19px of movement with no real motion behind it).
                    #
                    # So: predict with the camera affine, then correct toward
                    # the detection only very slowly - enough to shed affine
                    # drift over a shot, far too slow to be visible as motion.
                    # If the detection disagrees wholesale, the board has moved
                    # or we locked onto the wrong thing, so re-sync outright.
                    prev = tr.quad
                    if affine is not None:
                        prev = canonical_quad(cv2.transform(
                            prev.reshape(-1, 1, 2), affine).reshape(-1, 2))
                    if bool(icfg.get("lock_geometry", True)):
                        if quad_overlap(prev, quad) < float(icfg.get("resync_iou", 0.45)):
                            blended = quad                      # wholesale disagreement
                        else:
                            a = float(icfg.get("lock_correct_alpha", 0.03))
                            blended = canonical_quad(a * quad + (1.0 - a) * prev)
                    else:
                        blended = canonical_quad(ema * quad + (1.0 - ema) * prev)
                    # A panel is stuck to a physical board, so once the camera
                    # motion is accounted for it can only move a little between
                    # frames. Without this the panel teleported up to 892px on
                    # a 1920-wide frame when a detection landed on a different
                    # board. Anything beyond the step limit is not this board.
                    step = float(icfg.get("max_corner_step_px", 36))
                    delta = blended - prev
                    dist = np.linalg.norm(delta, axis=1)
                    if dist.max() > step:
                        blended = canonical_quad(prev + delta * (step / dist.max()))
                    tr.quad = blended
                    tr.fade = min(1.0, tr.fade + fade_step)
            else:
                if not tr.active:
                    dropped.append(tid)          # never-born track lost its feed
                    continue
                tr.missed += 1
                if affine is not None:
                    tr.quad = canonical_quad(
                        cv2.transform(tr.quad.reshape(-1, 1, 2), affine).reshape(-1, 2))
                    if tr.bound_quad is not None:
                        # the clamp bound moves WITH the camera, so propagation
                        # keeps the panel glued without ever growing it
                        tr.bound_quad = canonical_quad(
                            cv2.transform(tr.bound_quad.reshape(-1, 1, 2), affine).reshape(-1, 2))

                reason = ghost_reason(tr, raw, person_mask_dilated, affine, pcfg,
                                      mcfg.get("sample_band_frac", 0.6), src_w, src_h)
                if reason is not None:
                    # Cut it immediately, no fade: a fade would keep drawing the
                    # ghost for several more frames exactly where it shouldn't be.
                    if not tr.dying:
                        print(f"[ghost] frame {n_frames}: dropping panel {tid} - {reason}")
                    dropped.append(tid)
                    continue
                tr.fade = min(1.0, tr.fade + fade_step)
        for tid in dropped:
            del tracks[tid]

        # --- 3b. thin-strip enforcement + bottom-edge anchoring ---
        geom = []
        for tr in tracks.values():
            if not tr.active:
                continue
            thick, thick_cap, bit, snap = enforce_thin_strip(tr, raw, icfg, ema)
            geom.append((tr.tid, thick, thick_cap, bit, snap))
            if bit:
                n_clamped += 1
                print(f"[geom] frame {n_frames}: panel {tr.tid} trimmed to the "
                      f"{thick_cap:.0f}px cap")

        # --- 4-8. rasterize -> subtract occluders -> feather -> composite ---
        frame = composite_panels(frame, list(tracks.values()), person_mask,
                                 int(icfg["panel_feather_px"]),
                                 ocfg=ocfg, dilate_kernel=dilate_kernel, gcfg=gcfg,
                                 occluder_mask=occluders_dilated)

        writer.write(frame)

        thickness_log.extend(g[1] for g in geom)

        if debug_writer is not None:
            left = draw_debug_left(raw, det_info, person_mask, list(people.tracks.values()),
                                   tracks, ocfg, dilate_kernel, gcfg, geom,
                                   occluders=occluders_dilated)
            debug_writer.write(np.hstack([left, frame]))
            states = [f"id{t.tid}:{'D' if t.detected_this_frame else 'P' + str(t.missed)}"
                      f"{'*' if t.active else ''}" for t in tracks.values()]
            gtxt = " ".join(f"id{t}:{th:.0f}px/{c:.0f}{'!' if b else ''}dy{s:+.0f}"
                            for t, th, c, b, s in geom)
            print(f"[debug] f{n_frames}: tracks=[{' '.join(states)}] "
                  f"persons={n_person_det}det+{n_person_held}held geom=[{gtxt}]")

    cap.release()
    writer.release()
    if debug_writer is not None:
        debug_writer.release()

    elapsed = time.time() - start_time
    avg_fps = n_frames / elapsed if elapsed > 0 else 0.0
    print(f"Processed {n_frames} frames in {elapsed:.1f}s ({avg_fps:.2f} fps), {n_cuts} scene cut(s)")
    if thickness_log:
        t = np.array(thickness_log)
        thick_cap = float(icfg.get("max_board_height_frac", 0.09)) * src_h
        print(f"[geom] panel thickness over {len(t)} rendered panel-frames: "
              f"min={t.min():.0f}px median={np.median(t):.0f}px max={t.max():.0f}px "
              f"(cap {thick_cap:.0f}px); trimmed by the cap on {n_clamped} panel-frame(s); "
              f"over cap after clamping: {(t > thick_cap + 0.5).sum()}")
    if n_rejected_geometry:
        print(f"[safety] rejected {n_rejected_geometry} implausible detection group(s) before rendering")
    if capture_dir is not None:
        print(f"Captured {n_captured} frame(s) for review to {capture_dir}")

    print("Re-muxing original audio...")
    remux_audio(tmp_video_path, source_path, output_path, ffmpeg)
    if trimmed_path is not None:
        trimmed_path.unlink(missing_ok=True)
    print(f"Done. Output: {output_path}")


if __name__ == "__main__":
    main()
