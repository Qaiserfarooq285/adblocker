#!/usr/bin/env python3
"""Panel geometry, board-sourced fill colour, and the flat-panel renderer.

This is the ONLY fill path in the pipeline. `cv2.inpaint` and every
blur-based fallback were deliberately removed: they produced a shimmering
grey smear that moved with the content underneath. A hidden ad is a flat,
static, frozen-colour panel or it is nothing.

Colour sourcing (the part that decides whether the panel is invisible or
distracting):

  Sampling the median over the WHOLE detection is wrong. A perimeter board
  is usually shot against the dugout - dark bench, orange-vested stewards,
  deep shadow - and those pixels dominate the median, yielding a bright
  orange-brown block that catches the eye harder than the advert did. So we
  sample a thin band along the board's own face instead (middle 60%
  horizontally, central vertical third), excluding people, and then mute the
  result: drain saturation, take off the shine, pull the tone toward the
  scene average, and cap brightness so the panel can never be a highlight.

  Nothing here is brand- or colour-specific. The colour comes from whatever
  the board actually looks like, so a blue/green/yellow board in a future
  video adapts on its own with no code change.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Panel tracks
# ---------------------------------------------------------------------------

@dataclass
class PanelTrack:
    tid: int
    cls: int
    quad: np.ndarray                  # 4x2 float32, canonical corner order (EMA-smoothed)
    bottom_offset: float = 0.0        # EMA'd correction that pins the panel's
                                      # lower edge to the board/grass seam; kept
                                      # on the track (not recomputed raw each
                                      # frame) because a per-frame snap vibrates
    bound_quad: np.ndarray | None = None   # hard clamp: union of contributing
                                           # detections + margin - the rendered
                                           # panel may NEVER exceed this
    color: np.ndarray | None = None   # frozen muted BGR - computed ONCE at
                                      # confirmation and never recomputed; a
                                      # colour that changes per frame is a
                                      # panel that visibly vibrates
    ref_color: np.ndarray | None = None    # frozen RAW board colour, kept so a
                                           # propagated panel can be checked
                                           # against what it is supposedly over
    color_source: str = ""            # "pinned:<brand>" or "sampled" (debug/log)
    hits: int = 0                     # consecutive strong detections pre-birth
    active: bool = False              # panel is being rendered
    missed: int = 0                   # consecutive propagated (no-det) frames
    fade: float = 1.0                 # render alpha: ramps 0->1 on creation,
                                      # 1->0 on removal (no popping)
    dying: bool = False               # fading out; dropped when fade hits 0
    detected_this_frame: bool = field(default=False, repr=False)
    verify_after_frame: int = 0       # failed logo/OCR checks are rate-limited


# ---------------------------------------------------------------------------
# Quad helpers
# ---------------------------------------------------------------------------

def canonical_quad(pts: np.ndarray) -> np.ndarray:
    """Order 4 corners consistently (tl, tr, br, bl) so EMA between two quads
    never averages mismatched corners into a bowtie."""
    pts = np.asarray(pts, dtype=np.float32).reshape(4, 2)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.stack([
        pts[np.argmin(s)],   # top-left: smallest x+y
        pts[np.argmin(d)],   # top-right: smallest y-x
        pts[np.argmax(s)],   # bottom-right
        pts[np.argmax(d)],   # bottom-left
    ])


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    pts = cv2.findNonZero(mask)
    if pts is None or len(pts) < 20:
        return None
    return canonical_quad(cv2.boxPoints(cv2.minAreaRect(pts)))


def expand_quad(quad: np.ndarray, frac: float) -> np.ndarray:
    """Grow a quad about its centroid by frac per side (the config margin)."""
    c = quad.mean(axis=0, keepdims=True)
    return ((quad - c) * (1.0 + 2.0 * frac) + c).astype(np.float32)


def quad_overlap(a: np.ndarray, b: np.ndarray) -> float:
    """Bbox intersection over the smaller bbox area - cheap overlap measure
    used to let a re-identified strip inherit its predecessor's panel state
    instead of restarting the K-frame creation gate (which would blink)."""
    ax0, ay0 = a.min(axis=0)
    ax1, ay1 = a.max(axis=0)
    bx0, by0 = b.min(axis=0)
    bx1, by1 = b.max(axis=0)
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    smaller = min((ax1 - ax0) * (ay1 - ay0), (bx1 - bx0) * (by1 - by0))
    return float(iw * ih / max(smaller, 1e-6))


def quad_point(quad: np.ndarray, u: float, v: float) -> np.ndarray:
    """Bilinear point inside a (tl, tr, br, bl) quad. u runs left->right along
    the board, v runs top->bottom, both in [0, 1]."""
    tl, tr, br, bl = quad
    top = tl + u * (tr - tl)
    bottom = bl + u * (br - bl)
    return top + v * (bottom - top)


def strip_thickness(quad: np.ndarray) -> float:
    """Distance between a strip's top and bottom edges, measured PERPENDICULAR
    to the strip axis.

    This is the number that says whether a panel hugs a thin LED board, and it
    is not the same as the quad's bounding-box height. Measured on this
    footage, a perimeter board shot down the touchline has a perpendicular
    thickness of ~85px while its bounding box spans ~270px, because the strip
    runs diagonally across the frame. Capping bounding-box height would
    amputate the far end of a perfectly good strip; capping this does not."""
    tl, tr, br, bl = quad
    return float(np.linalg.norm((bl + br) / 2.0 - (tl + tr) / 2.0))


def clamp_strip_thickness(quad: np.ndarray, max_px: float) -> np.ndarray:
    """Trim an over-thick strip to `max_px`, anchored at its BOTTOM edge.

    An LED board sits low, right where the hoardings meet the pitch, and a
    detection that comes out too thick has always overshot upward into the
    crowd rather than downward into the grass. So the bottom edge - the edge we
    can actually verify against the board/grass seam - stays put and the top
    edge is pulled down to meet the cap."""
    t = strip_thickness(quad)
    if t <= max_px or t <= 1e-6:
        return quad
    keep = max_px / t
    tl, tr, br, bl = quad
    return np.stack([bl + (tl - bl) * keep,     # new top-left
                     br + (tr - br) * keep,     # new top-right
                     br, bl]).astype(np.float32)


def quad_edge_y_at_x(quad: np.ndarray, x: float, edge: str = "bottom") -> float:
    """Y of the strip's top or bottom edge at a given x, by linear interpolation
    along that edge. Used to ask 'is this person's feet below the board?'."""
    tl, tr, br, bl = quad
    a, b = (bl, br) if edge == "bottom" else (tl, tr)
    dx = float(b[0] - a[0])
    t = 0.0 if abs(dx) < 1e-6 else float(np.clip((x - a[0]) / dx, 0.0, 1.0))
    return float(a[1] + t * (b[1] - a[1]))


def shift_quad_edge(quad: np.ndarray, dy: float, edge: str = "bottom") -> np.ndarray:
    q = quad.copy().astype(np.float32)
    idx = (3, 2) if edge == "bottom" else (0, 1)
    q[idx[0], 1] += dy
    q[idx[1], 1] += dy
    return q


def measure_bottom_edge_offset(frame: np.ndarray, quad: np.ndarray,
                               max_shift_px: float, samples: int = 12) -> float | None:
    """How far the panel's lower edge sits above the real board/grass seam.

    The detector's bottom edge is good to a few pixels, but a few pixels of
    shortfall is a visible red sliver of un-hidden advert along the bottom of
    the panel. Rather than trust the mask, look for the seam directly: walk
    down each sample column from inside the board and find where grass starts.

    Returns a signed pixel offset (positive = panel must move DOWN), or None
    when too few columns give a clean answer - over the goalmouth there is net
    and crowd below the board instead of grass, and guessing there is worse
    than leaving the edge alone."""
    h, w = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue, sat = hsv[..., 0].astype(np.int16), hsv[..., 1].astype(np.int16)
    grass = ((hue >= 32) & (hue <= 95) & (sat > 45))

    offsets = []
    x0, x1 = float(quad[:, 0].min()), float(quad[:, 0].max())
    for i in range(samples):
        x = int(round(x0 + (x1 - x0) * (i + 0.5) / samples))
        if not (0 <= x < w):
            continue
        yb = quad_edge_y_at_x(quad, x, "bottom")
        lo = int(max(0, yb - max_shift_px))
        hi = int(min(h - 1, yb + max_shift_px))
        if hi - lo < 4:
            continue
        col = grass[lo:hi + 1, x]
        # first row that starts a run of solid grass - a single green pixel is
        # noise, a sustained run is the pitch
        run, start = 0, None
        for j, g in enumerate(col):
            if g:
                if run == 0:
                    start = j
                run += 1
                if run >= 6:
                    break
            else:
                run, start = 0, None
        if run >= 6 and start is not None:
            offsets.append((lo + start) - yb)
    if len(offsets) < 4:
        return None
    return float(np.clip(np.median(offsets), -max_shift_px, max_shift_px))


def goal_structure_mask(frame: np.ndarray, quad: np.ndarray, gcfg: dict) -> np.ndarray | None:
    """Pixels of goal frame and netting inside the panel's footprint, which
    must never be painted over: the goal is IN FRONT of the perimeter board,
    so covering it puts a flat slab across the net.

    Two cues were tried and rejected against real frames, both worth recording
    because both look obviously right on paper:

      whiteness  half the board is white - the CONFIA wordmark sits on white
                 blocks - so this punches holes through the advert.
      thinness   a white top-hat protects THREE TIMES more area in a
                 board-only stretch of the panel than in the goalmouth:
                 letter strokes are exactly as thin as net cord.

    What does separate them is that a goal post is TALLER than the board is
    thick, and a letter cannot be - it has to fit inside the strip. So the
    mask is a vertical morphological opening: bright pixels that survive being
    eroded by a vertical line longer than the strip is thick. A post (~270px
    here) survives, every letter stroke (<=0.6x thickness) is erased.

    Netting is deliberately NOT chased. Its cord is far fainter than the
    lettering, so no threshold catches the mesh without also catching the
    wordmarks, and painting over faint netting is a much smaller error than
    exposing the advert. The posts and crossbar are what read as a solid
    structure being cut in half."""
    h, w = frame.shape[:2]
    thick = max(4.0, strip_thickness(quad))

    pad = int(float(gcfg.get("context_up_frac", 1.5)) * thick)
    x0 = int(max(0, np.floor(quad[:, 0].min()) - 4))
    x1 = int(min(w, np.ceil(quad[:, 0].max()) + 4))
    y0 = int(max(0, np.floor(quad[:, 1].min()) - pad))
    y1 = int(min(h, np.ceil(quad[:, 1].max()) + pad))
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    sat, val = hsv[..., 1], hsv[..., 2]
    # Calibrated on the left-goal frame: at val>=150 the posts are too dim to
    # register at all (they sit in the stand's shadow), which is what made an
    # earlier version of this mask silently return nothing.
    bright = ((val >= int(gcfg.get("min_value", 130)))
              & (sat <= int(gcfg.get("max_saturation", 100)))).astype(np.uint8)

    # The whole discriminator, in one line: survive erosion by a vertical line
    # longer than the board is thick.
    vlen = int(max(int(gcfg.get("min_vertical_px", 15)),
                   float(gcfg.get("vertical_len_frac", 1.2)) * thick)) | 1
    vlen = min(vlen, max(3, (y1 - y0) - 1))
    keep = cv2.morphologyEx(bright, cv2.MORPH_OPEN,
                            cv2.getStructuringElement(cv2.MORPH_RECT, (1, vlen)))
    if not keep.any():
        return None

    n, lab, stats, _ = cv2.connectedComponentsWithStats(keep, 8)
    filtered = np.zeros_like(keep)
    min_area = int(gcfg.get("min_component_px", 80))
    max_w = int(float(gcfg.get("max_width_frac", 0.30)) * thick)
    # Solid frame members only. Net CORD is 2-3px wide and passes every other
    # test here, but protecting the mesh uncovers the advert behind it - the
    # first run that did so left a legible "Bet" showing at the goalmouth.
    # Netting is semi-transparent, so painting over it costs a faint texture;
    # painting over a post cuts a solid white bar in half. Only the second is
    # worth exposing an advert to avoid.
    min_w = int(gcfg.get("min_width_px", 5))
    # ...and it must stand IN FRONT of the board, by the same test used for
    # people: a goal post is planted on the pitch, so it continues BELOW the
    # board's bottom edge onto the grass. A dugout upright, a stanchion or a
    # crowd barrier sits behind the hoardings, which hide its base. Without
    # this, the dugout's uprights were protected on a shot with no goal in it
    # at all, drawing thin vertical stripes that flickered across the panel.
    below = float(gcfg.get("min_below_board_frac", 0.25)) * thick
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        if not (min_w <= stats[i, cv2.CC_STAT_WIDTH] <= max_w):
            continue
        cx = x0 + stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH] / 2.0
        foot = y0 + stats[i, cv2.CC_STAT_TOP] + stats[i, cv2.CC_STAT_HEIGHT]
        if foot < quad_edge_y_at_x(quad, cx, "bottom") + below:
            continue
        filtered[lab == i] = 1
    if not filtered.any():
        return None

    grow = int(gcfg.get("dilate_px", 3))
    if grow > 0:
        filtered = cv2.dilate(filtered, cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * grow + 1,) * 2))
    out = np.zeros((h, w), np.uint8)
    out[y0:y1, x0:x1] = filtered
    return out


def in_front_evidence(top_y: float, feet: float, xc: float,
                      quad: np.ndarray, ocfg: dict) -> bool:
    """Does THIS frame's blob look like someone standing in front of the board?

    Deliberately named 'evidence' and not 'verdict'. It is computed from one
    frame's mask extent, and a mask extent is noisy: when the detector finds
    only a referee's torso and misses his legs, `feet` jumps upward and the
    same person who was in front a frame ago now tests as behind. Callers must
    LATCH this over a track rather than re-deciding every frame - see
    PersonTrack.in_front. Using it directly per frame is what made the
    black-kit referee blink."""
    thick = strip_thickness(quad)
    tol = float(ocfg.get("feet_tolerance_frac", 0.25)) * thick
    protrude = float(ocfg.get("min_protrusion_frac", 0.5)) * thick
    board_top = quad_edge_y_at_x(quad, xc, "top")
    board_bottom = quad_edge_y_at_x(quad, xc, "bottom")

    if feet < board_bottom - tol:
        return False              # mask ends at the board's top: behind it

    # A person is far taller than the strip, so a blob that fits ENTIRELY
    # inside the board band is not one - it is a low-confidence false positive
    # on the advert itself (the black camera pole and the dark patch at the
    # goalmouth both produce these at conf 0.10).
    if (board_top - top_y) < protrude and (feet - board_bottom) < protrude:
        return False
    return True


def persons_in_front_mask(person_mask: np.ndarray, quad: np.ndarray, frame_h: int,
                          ocfg: dict, dilate_kernel) -> np.ndarray:
    """The subset of the person mask that is actually IN FRONT of this board.

    A perimeter board has people on both sides of it, and only one side is an
    occluder. Stewards, substitutes and the front row of the crowd stand
    BEHIND the hoardings: the board hides them from the shins down, so their
    visible mask ENDS at the board's top edge. A player or referee in front of
    it is standing on the grass, so their mask continues well below the board's
    bottom edge. Feet position separates the two reliably; mask overlap alone
    does not, which is why the panel used to be full of crowd-shaped holes
    along its top edge.

    Components are found on the UNdilated mask so two people standing close
    together are not fused into one verdict, and only the survivors get
    dilated."""
    h, w = person_mask.shape[:2]
    down = int(float(ocfg.get("occlusion_zone_down_frac", 0.25)) * frame_h)
    up = int(ocfg.get("occlusion_zone_up_px", 6))

    x0 = int(max(0, np.floor(quad[:, 0].min()) - 8))
    x1 = int(min(w, np.ceil(quad[:, 0].max()) + 8))
    y0 = int(max(0, np.floor(quad[:, 1].min()) - up))
    y1 = int(min(h, np.ceil(quad[:, 1].max()) + down))
    out = np.zeros((h, w), np.uint8)
    if x1 - x0 < 4 or y1 - y0 < 4:
        return out

    sub = person_mask[y0:y1, x0:x1]
    if not sub.any():
        return out

    n, lab, stats, cent = cv2.connectedComponentsWithStats(sub, 8)
    keep = np.zeros_like(sub)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < int(ocfg.get("min_person_px", 120)):
            continue
        top_y = y0 + stats[i, cv2.CC_STAT_TOP]
        if in_front_evidence(top_y, top_y + stats[i, cv2.CC_STAT_HEIGHT],
                             x0 + cent[i][0], quad, ocfg):
            keep[lab == i] = 1
    if not keep.any():
        return out

    out[y0:y1, x0:x1] = keep
    return cv2.dilate(out, dilate_kernel) if dilate_kernel is not None else out


def sample_band_quad(quad: np.ndarray, band_frac: float) -> np.ndarray:
    """The strip of the board we actually trust for colour: horizontally the
    middle `band_frac`, vertically the central third. Staying off the ends and
    the top/bottom rims keeps the sample on the LED face itself rather than on
    its frame, the grass below, or whatever sits behind its edges."""
    band_frac = float(np.clip(band_frac, 0.05, 1.0))
    u0, u1 = 0.5 - band_frac / 2, 0.5 + band_frac / 2
    v0, v1 = 1.0 / 3.0, 2.0 / 3.0
    return np.stack([
        quad_point(quad, u0, v0), quad_point(quad, u1, v0),
        quad_point(quad, u1, v1), quad_point(quad, u0, v1),
    ]).astype(np.float32)


# ---------------------------------------------------------------------------
# Colour
# ---------------------------------------------------------------------------

def _bgr_to_hsv(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(np.asarray(bgr, np.float32).reshape(1, 1, 3).astype(np.uint8),
                        cv2.COLOR_BGR2HSV).reshape(3).astype(np.float32)


def _hsv_to_bgr(hsv: np.ndarray) -> np.ndarray:
    hsv = np.clip(hsv, [0, 0, 0], [179, 255, 255]).astype(np.uint8).reshape(1, 1, 3)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR).reshape(3).astype(np.float32)


def scene_average_color(frame: np.ndarray, person_mask: np.ndarray | None = None) -> np.ndarray:
    """Mean BGR of the whole scene (grass + stands + crowd), people excluded.
    Computed on a thumbnail - this is a broad tone reference, not a detail
    measurement, and it must be cheap enough to run every frame."""
    small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
    if person_mask is not None and person_mask.any():
        pm = cv2.resize(person_mask, (160, 90), interpolation=cv2.INTER_NEAREST)
        px = small[pm == 0]
        if len(px) >= 200:
            return px.mean(axis=0).astype(np.float32)
    return small.reshape(-1, 3).mean(axis=0).astype(np.float32)


def sample_band_color(frame: np.ndarray, quad: np.ndarray, person_mask_dilated: np.ndarray,
                      band_frac: float = 0.6) -> np.ndarray | None:
    """RAW (unmuted) colour of the LED strip's own face, or None when too much
    of the band is occluded to trust. This is also the probe used to check
    whether a propagated panel is still over a board at all."""
    band = sample_band_quad(quad, band_frac)
    m = np.zeros(frame.shape[:2], dtype=np.uint8)
    cv2.fillConvexPoly(m, np.round(band).astype(np.int32), 1)
    m[person_mask_dilated > 0] = 0
    px = frame[m > 0]
    if len(px) < 30:
        return None

    hsv = cv2.cvtColor(px.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3).astype(np.float32)

    # Reject outliers before averaging: stray people the mask missed, specular
    # glints off the LED panel, and shadow edges all sit far from the bulk in
    # saturation/value even when their hue looks plausible.
    med_s, med_v = np.median(hsv[:, 1]), np.median(hsv[:, 2])
    dist = np.abs(hsv[:, 1] - med_s) + np.abs(hsv[:, 2] - med_v)
    keep = dist <= np.percentile(dist, 80)
    if keep.sum() >= 20:
        hsv = hsv[keep]

    # Hue is circular and a red board sits exactly on the 0/179 wrap point,
    # where a plain median lands on the wrong side. Average hue as unit
    # vectors; saturation/value stay medians.
    ang = hsv[:, 0].astype(np.float64) * (2.0 * np.pi / 180.0)
    hue = (np.degrees(np.arctan2(np.sin(ang).mean(), np.cos(ang).mean())) / 2.0) % 180.0
    return _hsv_to_bgr(np.array([hue, np.median(hsv[:, 1]), np.median(hsv[:, 2])], np.float32))


def hue_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Circular hue distance between two BGR colours, in degrees (0-90)."""
    ha, hb = _bgr_to_hsv(a)[0], _bgr_to_hsv(b)[0]
    d = abs(float(ha) - float(hb)) % 180.0
    return min(d, 180.0 - d)


def looks_like_grass(bgr: np.ndarray) -> bool:
    h, s, _ = _bgr_to_hsv(bgr)
    return 30.0 <= float(h) <= 90.0 and float(s) > 50.0


def mute_color(bgr: np.ndarray, scene_avg: np.ndarray, mcfg: dict) -> np.ndarray:
    """Take the glare off a colour without changing what colour it IS.

    Saturation and value come down; hue is left alone. `scene_blend` defaults
    to 0 for exactly that reason - blending hue toward the scene average is
    what previously turned a red board into olive-brown."""
    hsv = _bgr_to_hsv(bgr)
    hsv[1] *= float(mcfg.get("saturation_mult", 0.75))
    hsv[2] *= float(mcfg.get("value_mult", 0.8))
    color = _hsv_to_bgr(hsv)

    blend = float(mcfg.get("scene_blend", 0.0))
    if blend > 0:
        color = (1.0 - blend) * color + blend * scene_avg.astype(np.float32)

    cap = float(mcfg.get("max_relative_brightness", 1.0))
    scene_v = _bgr_to_hsv(scene_avg)[2]
    chsv = _bgr_to_hsv(color)
    if chsv[2] > scene_v * cap and chsv[2] > 1e-3:
        chsv[2] = scene_v * cap          # never a highlight
        color = _hsv_to_bgr(chsv)
    return np.clip(color, 0, 255).astype(np.float32)


def load_brand_colors(logos_dir, filename: str = "color.json") -> dict[str, np.ndarray]:
    """Read pinned brand colours from input/logos/<brand>/color.json (or
    color.txt holding a hex string). A pinned colour is exact and cannot be
    corrupted by one bad frame, which is why it outranks sampling."""
    import json
    from pathlib import Path

    out: dict[str, np.ndarray] = {}
    logos_dir = Path(logos_dir)
    if not logos_dir.exists():
        return out

    def from_hex(text: str):
        t = text.strip().lstrip("#")
        if len(t) != 6:
            return None
        r, g, b = int(t[0:2], 16), int(t[2:4], 16), int(t[4:6], 16)
        return np.array([b, g, r], np.float32)

    for sub in sorted(d for d in logos_dir.iterdir() if d.is_dir()):
        color = None
        jf = sub / filename
        if jf.exists():
            try:
                data = json.loads(jf.read_text())
                if "rgb" in data:
                    r, g, b = data["rgb"][:3]
                    color = np.array([b, g, r], np.float32)
                elif "bgr" in data:
                    color = np.array(data["bgr"][:3], np.float32)
                elif "hex" in data:
                    color = from_hex(str(data["hex"]))
            except (ValueError, KeyError, TypeError) as exc:
                print(f"[fill] ignoring {jf}: {exc}")
        else:
            tf = sub / "color.txt"
            if tf.exists():
                color = from_hex(tf.read_text())
        if color is not None:
            out[sub.name] = np.clip(color, 0, 255).astype(np.float32)
    return out


def pick_fill_color(frame: np.ndarray, quad: np.ndarray, person_mask_dilated: np.ndarray,
                    scene_avg: np.ndarray, mcfg: dict, brand_colors: dict[str, np.ndarray],
                    snap_hue: float = 30.0):
    """Choose the colour to freeze onto a new track.

    Returns (muted_color, reference_raw_color, source_label), or
    (None, None, reason) when neither source can produce one this frame - in
    which case the caller should wait for a cleaner frame rather than freeze
    something wrong for the life of the track."""
    sampled = sample_band_color(frame, quad, person_mask_dilated, mcfg.get("sample_band_frac", 0.6))

    if brand_colors:
        # Classes are brand-agnostic by design, so the board doesn't announce
        # which brand it is. Match on colour: snap to the nearest pinned brand
        # when the sample is close to it, and when only one brand is pinned
        # and the sample failed entirely, fall back to that pin.
        if sampled is not None:
            name, pinned = min(brand_colors.items(), key=lambda kv: hue_distance(sampled, kv[1]))
            if hue_distance(sampled, pinned) <= snap_hue:
                return mute_color(pinned, scene_avg, mcfg), pinned.copy(), f"pinned:{name}"
        elif len(brand_colors) == 1:
            name, pinned = next(iter(brand_colors.items()))
            return mute_color(pinned, scene_avg, mcfg), pinned.copy(), f"pinned:{name}(no-sample)"

    if sampled is None:
        return None, None, "band-occluded"
    return mute_color(sampled, scene_avg, mcfg), sampled.copy(), "sampled"


def board_fill_color(frame, quad, person_mask_dilated, scene_avg, mcfg):
    """Back-compat shim: sample the board band and mute it, with no pinned
    brand colour. Kept so the colour-comparison tool and older tests keep
    working; the pipeline itself calls pick_fill_color()."""
    sampled = sample_band_color(frame, quad, person_mask_dilated,
                               mcfg.get("sample_band_frac", 0.6))
    if sampled is None:
        return None
    return mute_color(sampled, scene_avg, mcfg)


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

def composite_panels(frame: np.ndarray, panels: list[PanelTrack],
                     person_mask: np.ndarray, feather_px: int,
                     ocfg: dict | None = None, dilate_kernel=None,
                     gcfg: dict | None = None,
                     occluder_mask: np.ndarray | None = None) -> np.ndarray:
    """Composite every active panel as a flat frozen-colour region with a
    feathered edge, covering neither the people in front of the board nor the
    goal structure that stands in front of it.

    Three subtractions, in order of how badly getting them wrong shows:

      bound_quad   hard over-capture clamp (union of contributing detections +
                   margin) so EMA overshoot or propagation drift can never grow
                   the panel onto the dugout or a neighbouring non-betting board.
      persons      only those on the PITCH side. Prefer passing the tracker's
                   LATCHED occluder mask via `occluder_mask` - deciding
                   in-front/behind from a single frame's mask extent is what
                   made the black-kit referee blink. The `ocfg` path recomputes
                   it per frame and exists for tests and the debug overlay.
      goal         net and posts, which are in front of the board.

    Pure function so the unit test can drive it directly. Modifies and returns
    `frame`."""
    h, w = frame.shape[:2]
    for tr in panels:
        if not tr.active or tr.color is None or tr.fade <= 0.0:
            continue
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillConvexPoly(mask, np.round(tr.quad).astype(np.int32), 255)
        if tr.bound_quad is not None:              # hard over-capture clamp
            bound = np.zeros((h, w), dtype=np.uint8)
            cv2.fillConvexPoly(bound, np.round(tr.bound_quad).astype(np.int32), 255)
            mask &= bound

        if occluder_mask is not None:
            occluders = occluder_mask
        elif ocfg is not None:
            occluders = persons_in_front_mask(person_mask, tr.quad, h, ocfg, dilate_kernel)
        else:
            occluders = person_mask

        if gcfg is not None and gcfg.get("enabled", True):
            goal = goal_structure_mask(frame, tr.quad, gcfg)
            if goal is not None:
                occluders = occluders | goal

        mask[occluders > 0] = 0                    # subtract BEFORE feathering
        if not mask.any():
            continue
        alpha = mask.astype(np.float32) / 255.0
        if feather_px > 0:
            k = 2 * feather_px + 1
            alpha = cv2.GaussianBlur(alpha, (k, k), 0)
            alpha[occluders > 0] = 0.0             # feather must not bleed onto them
        alpha *= float(np.clip(tr.fade, 0.0, 1.0))
        x, y, bw, bh = cv2.boundingRect(cv2.dilate(mask, np.ones((2 * feather_px + 1,) * 2, np.uint8)))
        if bw == 0 or bh == 0:
            continue
        roi = frame[y:y + bh, x:x + bw].astype(np.float32)
        a = alpha[y:y + bh, x:x + bw][..., None]
        frame[y:y + bh, x:x + bw] = (tr.color * a + roi * (1.0 - a)).astype(np.uint8)
    return frame
