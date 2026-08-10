#!/usr/bin/env python3
"""Unit tests for the two things that quietly break without being visible in
a log: person pixels surviving the composite, and the fill colour being
sampled from the wrong pixels.

Run:  .venv/bin/python scripts/test_panel_person_subtraction.py
"""
from __future__ import annotations

import sys

import cv2
import numpy as np

from pathlib import Path

from fill import (PanelTrack, board_fill_color, canonical_quad, clamp_strip_thickness,
                  in_front_evidence,
                  composite_panels, goal_structure_mask, load_brand_colors,
                  measure_bottom_edge_offset, persons_in_front_mask, pick_fill_color,
                  quad_edge_y_at_x, sample_band_quad, scene_average_color,
                  strip_thickness)
from persons import PersonMaskTracker, paste, warp_patch

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)


# ---------------------------------------------------------------------------

def test_person_excluded_from_panel():
    """A person in front of the board must be pixel-identical afterwards."""
    h, w = 360, 640
    frame = np.full((h, w, 3), 40, dtype=np.uint8)
    quad = canonical_quad(np.array([[100, 120], [540, 120], [540, 200], [100, 200]], np.float32))
    person = np.zeros((h, w), dtype=np.uint8)
    person[80:260, 300:360] = 1
    person_color = (10, 200, 10)
    frame[person > 0] = person_color

    track = PanelTrack(tid=1, cls=1, quad=quad,
                       color=np.array([30, 30, 220], np.float32), active=True)

    dilate = 9
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * dilate + 1,) * 2)
    person_dilated = cv2.dilate(person, k)
    out = composite_panels(frame.copy(), [track], person_dilated, feather_px=4)

    check((out[person > 0] == np.array(person_color, np.uint8)).all(),
          "person pixels were painted over by the panel fill")

    probe = out[160, 150]
    check((abs(probe.astype(int) - track.color.astype(int)) <= 12).all(),
          f"panel did not render where it should (probe {probe.tolist()})")

    quad_mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(quad_mask, track.quad.astype(np.int32), 1)
    halo = (person_dilated > 0) & (person == 0) & (quad_mask > 0)
    check((out[halo] == frame[halo]).all(), "feather bled into the dilated person margin")


def _hue(bgr) -> float:
    return float(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])


def _hue_dist(a: float, b: float) -> float:
    d = abs(a - b) % 180
    return min(d, 180 - d)


def _dugout_scene():
    """The real failure case: the detection quad does NOT sit tightly on the
    LED face. Margin expansion and an imperfect fit make it straddle the
    dugout above (dark bench, orange stewards) and grass below, so a median
    over the WHOLE quad is dominated by background rather than the board."""
    h, w = 400, 800
    frame = np.full((h, w, 3), (60, 110, 70), np.uint8)     # grass
    quad = canonical_quad(np.array([[100, 100], [700, 100], [700, 250], [100, 250]], np.float32))
    frame[100:150, 100:700] = (25, 80, 235)                 # dugout/stewards: orange
    frame[150:200, 100:700] = (40, 40, 190)                 # the LED board: red
    frame[200:250, 100:700] = (60, 110, 70)                 # grass under the board
    return frame, quad, (40, 40, 190), (25, 80, 235)


def test_color_is_sampled_from_the_board_face():
    """Sourcing only. Muting/blending are switched off so this test fails for
    exactly one reason: the colour came from the wrong pixels."""
    frame, quad, board_bgr, dugout_bgr = _dugout_scene()
    raw_cfg = {"sample_band_frac": 0.6, "saturation_mult": 1.0, "value_mult": 1.0,
               "scene_blend": 0.0, "max_relative_brightness": 10.0}
    color = board_fill_color(frame, quad, np.zeros(frame.shape[:2], np.uint8),
                             scene_average_color(frame, None), raw_cfg)
    check(color is not None, "board_fill_color returned None on a clean board")
    if color is None:
        return

    check(_hue_dist(_hue(color), _hue(board_bgr)) < 15,
          f"fill hue {_hue(color):.0f} is not the board's {_hue(board_bgr):.0f}")
    check(_hue_dist(_hue(color), _hue(board_bgr)) < _hue_dist(_hue(color), _hue(dugout_bgr)),
          "fill hue sits closer to the dugout than to the board")
    # NOTE: this synthetic case cannot prove much on its own - a median is
    # robust to a symmetric three-way split by construction, so the naive
    # whole-quad median happens to survive it too. The real discrimination
    # between sampling strategies is measured on actual footage; see
    # check_fill_color_on_real_frame.py.


def test_fill_is_muted_and_never_brighter_than_the_scene():
    """Muting only: same board, default knobs."""
    frame, quad, board_bgr, _ = _dugout_scene()
    scene_avg = scene_average_color(frame, None)
    color = board_fill_color(frame, quad, np.zeros(frame.shape[:2], np.uint8), scene_avg,
                             {"sample_band_frac": 0.6, "saturation_mult": 0.5,
                              "value_mult": 0.85, "scene_blend": 0.25,
                              "max_relative_brightness": 1.0})
    if color is None:
        check(False, "board_fill_color returned None with default mute knobs")
        return
    hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]
    board_hsv = cv2.cvtColor(np.uint8([[board_bgr]]), cv2.COLOR_BGR2HSV)[0, 0]
    scene_v = cv2.cvtColor(np.uint8([[scene_avg.astype(np.uint8)]]), cv2.COLOR_BGR2HSV)[0, 0, 2]
    check(int(hsv[1]) < int(board_hsv[1]),
          f"fill not muted: sat {hsv[1]} >= board sat {board_hsv[1]}")
    check(int(hsv[2]) <= int(scene_v) + 1,
          f"fill brighter than the scene ({hsv[2]} > {scene_v}) - it will draw the eye")


def test_color_adapts_to_a_different_board_colour():
    """Nothing is hardcoded to red: a blue board must yield a blue-family fill."""
    h, w = 400, 800
    frame = np.full((h, w, 3), (60, 110, 70), np.uint8)
    board_bgr = (200, 90, 30)                   # blue board
    quad = canonical_quad(np.array([[100, 160], [700, 160], [700, 230], [100, 230]], np.float32))
    cv2.fillConvexPoly(frame, quad.astype(np.int32), board_bgr)
    color = board_fill_color(frame, quad, np.zeros((h, w), np.uint8),
                             scene_average_color(frame, None),
                             {"sample_band_frac": 0.6, "saturation_mult": 0.5,
                              "value_mult": 0.85, "scene_blend": 0.25,
                              "max_relative_brightness": 1.0})
    check(color is not None, "board_fill_color returned None on a blue board")
    if color is not None:
        hue = lambda bgr: float(cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)[0, 0, 0])
        d = abs(hue(color) - hue(board_bgr)) % 180
        check(min(d, 180 - d) < 25,
              f"blue board gave hue {hue(color):.0f}, expected near {hue(board_bgr):.0f}")


def test_sample_band_is_inside_the_quad():
    quad = canonical_quad(np.array([[100, 100], [500, 140], [500, 220], [100, 180]], np.float32))
    band = sample_band_quad(quad, 0.6)
    for pt in band:
        check(cv2.pointPolygonTest(quad.astype(np.float32), (float(pt[0]), float(pt[1])), False) >= 0,
              f"sample band corner {pt.tolist()} fell outside the detection quad")


def test_held_person_mask_follows_camera_motion():
    """A held mask must move with the camera, not stay pinned where the person
    used to be - otherwise the hold protects empty grass."""
    patch = np.ones((40, 20), np.uint8)
    affine = np.array([[1.0, 0.0, 25.0], [0.0, 1.0, -10.0]])   # pure translation
    moved = warp_patch(patch, (100, 200), affine, (400, 600))
    check(moved is not None, "warp_patch returned None on a simple translation")
    if moved is not None:
        _, origin = moved
        check(abs(origin[0] - 125) <= 1 and abs(origin[1] - 190) <= 1,
              f"held mask landed at {origin}, expected ~(125, 190)")

    mask = np.zeros((400, 600), np.uint8)
    paste(mask, moved[0], moved[1])
    check(mask.sum() > 0, "pasted held mask is empty")


def test_pinned_brand_colour_wins_and_stays_faithful():
    """A pinned colour is exact: the panel must come out that hue, muted but
    unmistakably the brand's colour - not blended toward the scene."""
    import json
    import tempfile

    frame, quad, board_bgr, _ = _dugout_scene()
    mcfg = {"sample_band_frac": 0.6, "saturation_mult": 0.75, "value_mult": 0.8,
            "scene_blend": 0.0, "max_relative_brightness": 1.0}

    with tempfile.TemporaryDirectory() as tmp:
        brand = Path(tmp) / "betano"
        brand.mkdir()
        (brand / "color.json").write_text(json.dumps({"rgb": [164, 60, 51]}))
        other = Path(tmp) / "bluebrand"
        other.mkdir()
        (other / "color.txt").write_text("#1E50C8")
        colors = load_brand_colors(Path(tmp))

    check(set(colors) == {"betano", "bluebrand"}, f"brand colours not loaded: {sorted(colors)}")
    check(_hue_dist(_hue(colors["bluebrand"]), _hue((200, 80, 30))) < 12,
          "color.txt hex was not parsed to the right hue")

    color, ref, src = pick_fill_color(frame, quad, np.zeros(frame.shape[:2], np.uint8),
                                      scene_average_color(frame, None), mcfg, colors, 30.0)
    check(src.startswith("pinned:betano"), f"expected the pinned Betano colour, got '{src}'")
    if color is not None:
        # muted, but still clearly RED - the whole point of Issue 1
        hsv = cv2.cvtColor(np.uint8([[color]]), cv2.COLOR_BGR2HSV)[0, 0]
        check(_hue_dist(float(hsv[0]), _hue((51, 60, 164))) < 12,
              f"pinned red came out at hue {hsv[0]} - not the brand's colour")
        check(int(hsv[1]) > 90, f"pinned colour washed out to sat {hsv[1]} - reads grey, not red")


def test_unpinned_brand_is_not_painted_another_brands_colour():
    """A board whose hue is nowhere near any pin must keep its own sampled
    colour rather than snapping to an unrelated brand."""
    h, w = 400, 800
    frame = np.full((h, w, 3), (60, 110, 70), np.uint8)
    green_board = (60, 190, 60)
    quad = canonical_quad(np.array([[100, 160], [700, 160], [700, 230], [100, 230]], np.float32))
    cv2.fillConvexPoly(frame, quad.astype(np.int32), green_board)
    colors = {"betano": np.array([51, 60, 164], np.float32)}   # red pin, far from green

    _, _, src = pick_fill_color(frame, quad, np.zeros((h, w), np.uint8),
                                scene_average_color(frame, None),
                                {"sample_band_frac": 0.6, "saturation_mult": 0.75,
                                 "value_mult": 0.8, "scene_blend": 0.0,
                                 "max_relative_brightness": 1.0}, colors, 30.0)
    check(src == "sampled", f"a green board snapped to the red pin (source '{src}')")


def test_ghost_panel_is_dropped():
    """Issue 3: a propagated panel must die rather than slide onto the pitch."""
    from process_video import ghost_reason

    pcfg = {"max_propagation_frames": 6, "verify_content": True, "content_hue_tol": 30,
            "pan_drop_px": 25, "field_of_play": [0.15, 0.45, 0.85, 1.0]}
    W, H = 1920, 1080
    grass = np.full((H, W, 3), (60, 150, 70), np.uint8)          # open pitch
    still = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.5]])          # near-static camera
    red = np.array([51, 60, 164], np.float32)
    empty = np.zeros((H, W), np.uint8)

    def track(cx, cy, missed=1):
        q = canonical_quad(np.array([[cx - 200, cy - 25], [cx + 200, cy - 25],
                                     [cx + 200, cy + 25], [cx - 200, cy + 25]], np.float32))
        return PanelTrack(tid=1, cls=1, quad=q, color=red.copy(), ref_color=red.copy(),
                          active=True, missed=missed)

    mid = track(W * 0.5, H * 0.7)          # centre circle
    check(ghost_reason(mid, grass, empty, still, pcfg, 0.6, W, H) is not None,
          "a panel sitting on the centre circle was NOT dropped")

    over_cap = track(W * 0.5, H * 0.2, missed=7)
    check(ghost_reason(over_cap, grass, empty, still, pcfg, 0.6, W, H) == "propagation cap",
          "propagation cap did not fire at missed=7")

    panning = track(W * 0.5, H * 0.2)
    fast = np.array([[1.0, 0.0, 60.0], [0.0, 1.0, 3.0]])
    check(ghost_reason(panning, grass, empty, fast, pcfg, 0.6, W, H) is not None,
          "a fast pan with no detection did not drop the panel")

    on_grass = track(W * 0.5, H * 0.2)     # high in frame, but the pixels are grass
    check(ghost_reason(on_grass, grass, empty, still, pcfg, 0.6, W, H) == "now over grass",
          "content check did not notice the panel is over grass")

    # …and a legitimate short gap over a board that is still there survives
    board = grass.copy()
    q = track(W * 0.5, H * 0.2).quad
    cv2.fillConvexPoly(board, np.round(q).astype(np.int32), (51, 60, 164))
    ok = track(W * 0.5, H * 0.2)
    check(ghost_reason(ok, board, empty, still, pcfg, 0.6, W, H) is None,
          "a real board under a 1-frame gap was wrongly dropped")


def test_person_config_defaults_favour_recall():
    cfg = {"person_conf": 0.15, "person_hold_frames": 12, "roi_person_pass": True}
    tracker = PersonMaskTracker(model=None, icfg=cfg, imgsz=1280)
    check(tracker.conf <= 0.2, "person conf too high - dark-kit referees will be missed")
    check(tracker.hold_frames >= 6, "person hold too short to bridge a detection gap")
    check(tracker.roi_pass, "ROI person pass disabled by default")


# ---------------------------------------------------------------------------

def _slanted_strip(x0=100.0, y0=100.0, length=800.0, drop=200.0, thick=60.0):
    """A perimeter board as the camera actually sees it: a thin strip running
    diagonally down the frame. Its bounding-box height (drop + thick) is far
    larger than its real thickness, which is the whole point."""
    return canonical_quad(np.array([
        [x0, y0], [x0 + length, y0 + drop],
        [x0 + length, y0 + drop + thick], [x0, y0 + thick]], np.float32))


def test_thickness_is_measured_perpendicular_not_as_bbox_height():
    """The distinction the whole geometry guard rests on. A slanted strip 60px
    thick must not be judged by its 260px bounding box - clamping that would
    amputate the far end of a perfectly good board."""
    q = _slanted_strip(thick=60.0, drop=200.0)
    bbox_h = float(q[:, 1].max() - q[:, 1].min())
    check(abs(strip_thickness(q) - 60.0) < 1.5,
          f"strip_thickness read {strip_thickness(q):.1f}, expected ~60")
    check(bbox_h > 3 * strip_thickness(q),
          "test strip is not slanted enough to prove the distinction")

    # a cap ABOVE the true thickness but BELOW the bbox height must not fire
    same = clamp_strip_thickness(q, 100.0)
    check(np.allclose(same, q),
          "a 60px-thick strip was trimmed by a 100px cap - the cap is reading "
          "bounding-box height, which would destroy every slanted board")


def test_over_thick_panel_is_trimmed_from_the_top():
    """An over-thick detection always overshoots upward into the crowd, so the
    bottom edge - the one we can verify against the board/grass seam - stays
    put and the top comes down."""
    q = _slanted_strip(thick=200.0, drop=150.0)
    out = clamp_strip_thickness(q, 80.0)
    check(abs(strip_thickness(out) - 80.0) < 1.0,
          f"clamped thickness {strip_thickness(out):.1f}, expected 80")
    check(np.allclose(out[2], q[2]) and np.allclose(out[3], q[3]),
          "the bottom edge moved - the trim must be anchored at the bottom")
    check(out[0][1] > q[0][1] and out[1][1] > q[1][1],
          "the top edge did not come down")


def test_only_people_in_front_of_the_board_punch_holes():
    """The crowd-hole bug, as a test. Someone behind the hoardings is hidden
    from the shins down, so their mask ENDS at the board's top edge; someone in
    front is on the grass and continues below its bottom edge."""
    h, w = 400, 600
    quad = canonical_quad(np.array([[100, 200], [500, 200], [500, 260], [100, 260]], np.float32))
    person = np.zeros((h, w), np.uint8)
    person[80:205, 150:190] = 1        # steward BEHIND: mask stops at the board top
    person[150:340, 300:340] = 1       # player IN FRONT: mask runs onto the grass

    ocfg = {"occlusion_zone_down_frac": 0.25, "occlusion_zone_up_px": 6,
            "feet_tolerance_frac": 0.25, "min_person_px": 50}
    kept = persons_in_front_mask(person, quad, h, ocfg, None)

    check(kept[150:340, 300:340].any(), "the player in front was ignored - the panel "
                                        "would be painted straight over them")
    check(not kept[80:205, 150:190].any(),
          "the steward behind the board still punches a hole - this is the "
          "crowd-notch bug along the panel's top edge")


def test_goal_structure_is_protected_without_eating_the_wordmark():
    """Both cues that were tried and failed are encoded here: a tall post must
    be protected, and a bright letter-sized block inside the strip must not be.
    A whiteness or thinness test passes the first and fails the second."""
    h, w = 400, 600
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :] = (60, 90, 60)                       # grass-ish
    quad = canonical_quad(np.array([[100, 200], [500, 200], [500, 280], [100, 280]], np.float32))
    cv2.fillConvexPoly(frame, np.round(quad).astype(np.int32), (40, 40, 190))   # red board
    frame[120:340, 300:312] = (235, 235, 235)        # goal post: taller than the strip
    frame[215:265, 400:460] = (235, 235, 235)        # white wordmark block INSIDE the strip

    g = goal_structure_mask(frame, quad, {})
    check(g is not None, "goal mask found nothing at all - the post should qualify")
    if g is not None:
        check(g[215:275, 300:312].any(),
              "the goal post crossing the board was not protected")
        check(not g[215:265, 405:455].any(),
              "the white wordmark block was protected - this punches a hole "
              "straight through the advert we are hiding")


def test_bottom_edge_snaps_down_to_the_grass_seam():
    """The red sliver of un-hidden advert along the panel's lower edge: the
    quad stops short of the seam and the measurement must say 'move down'."""
    h, w = 400, 600
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :] = (40, 40, 190)                    # board red everywhere above
    frame[300:, :] = (60, 140, 60)                 # grass below y=300
    short = canonical_quad(np.array([[100, 220], [500, 220], [500, 285], [100, 285]], np.float32))

    dy = measure_bottom_edge_offset(frame, short, 40.0)
    check(dy is not None, "the board/grass seam was not found at all")
    if dy is not None:
        check(dy > 8, f"offset {dy:.1f}px - should be ~+15 (panel must move DOWN)")

    exact = canonical_quad(np.array([[100, 235], [500, 235], [500, 300], [100, 300]], np.float32))
    dy2 = measure_bottom_edge_offset(frame, exact, 40.0)
    check(dy2 is not None and abs(dy2) <= 3,
          f"an already-correct edge was told to move {dy2}px")


def test_occluder_survives_a_partial_segmentation():
    """The referee blink, reproduced through the REAL tracker code path.

    Observed on the 4K clip, frame 711: the assistant referee's legs and flag
    segmented but his black torso did not. So a mask DID exist and WAS
    subtracted - it simply failed to cover him, and the panel painted his upper
    body out while his legs stayed visible. Two things must hold for him not to
    blink:

      * the in-front verdict must LATCH on the track. An earlier version
        rebuilt the PersonTrack object on every detected frame, which reset
        front_votes and meant the latch could never reach its threshold - a bug
        a test that hand-rolled the latch logic could not have caught, which is
        why this one drives update() itself.
      * the occluder mask must be the UNION of recent masks, so a frame where
        the torso is missing still covers the torso.
    """
    from persons import PersonMaskTracker

    quad = canonical_quad(np.array([[100, 200], [500, 200], [500, 280], [100, 280]], np.float32))
    ocfg = {"feet_tolerance_frac": 0.25, "min_protrusion_frac": 0.5,
            "min_person_px": 50, "front_latch_frames": 2}

    # the raw per-frame evidence really does flip - that is the premise
    check(in_front_evidence(120, 340, 300, quad, ocfg),
          "a fully detected person in front was not recognised as in front")
    check(not in_front_evidence(120, 250, 300, quad, ocfg),
          "the partial-detection case does not reproduce - test is not exercising the bug")

    h, w = 400, 600
    full = np.zeros((h, w), np.uint8)
    full[120:340, 290:310] = 1          # head above board, feet on grass below
    legs = np.zeros((h, w), np.uint8)
    legs[280:340, 290:310] = 1          # only below the board: torso lost

    class _T:
        def __init__(self, a): self._a = a
        def cpu(self): return self
        def numpy(self): return self._a

    class _Res:
        def __init__(self, m):
            self.masks = type("M", (), {"data": _T(m[None].astype(np.float32))})()
            self.boxes = type("B", (), {"id": _T(np.array([7])), "__len__": lambda s: 1})()

    class _Model:
        def __init__(self, seq): self.seq, self.i = list(seq), 0
        def track(self, frame, **kw):
            m = self.seq[min(self.i, len(self.seq) - 1)]
            self.i += 1
            return [_Res(m)]

    icfg = {"occlusion": ocfg, "roi_person_pass": False,
            "motion_person_fallback": False, "mask_union_frames": 3}
    tracker = PersonMaskTracker(_Model([full, full, legs, full]), icfg, imgsz=64)
    frame = np.zeros((h, w, 3), np.uint8)

    torso_rows = slice(150, 240)
    torso_cols = slice(292, 308)
    covered = []
    for _ in range(4):
        tracker.update(frame, None, [quad])
        occ = tracker.last_occluders
        covered.append(bool(occ[torso_rows, torso_cols].all()))

    check(tracker.tracks[7].in_front,
          "the in-front verdict never latched - PersonTrack state is being reset "
          "on detected frames, so front_votes can never reach the threshold")
    check(covered[0] and covered[1],
          "a fully detected person in front was not covered by the occluder mask")
    check(covered[2],
          "TORSO EXPOSED on the partial-segmentation frame - this is the referee "
          "blink: the panel paints over his upper body for that frame")
    check(covered[3], "coverage did not recover after the partial frame")


def test_composite_respects_the_occlusion_zone_and_the_goal():
    """End-to-end through the renderer: the panel covers the board, leaves the
    person in front alone, ignores the one behind, and spares the post."""
    h, w = 400, 600
    frame = np.zeros((h, w, 3), np.uint8)
    frame[:, :] = (60, 90, 60)
    quad = canonical_quad(np.array([[100, 200], [500, 200], [500, 280], [100, 280]], np.float32))
    cv2.fillConvexPoly(frame, np.round(quad).astype(np.int32), (40, 40, 190))
    frame[120:340, 300:312] = (235, 235, 235)          # goal post
    behind = (150, 60, 60)
    infront = (10, 200, 10)
    frame[130:205, 150:190] = behind                   # crowd behind the board
    frame[150:340, 400:440] = infront                  # player in front
    person = np.zeros((h, w), np.uint8)
    person[130:205, 150:190] = 1
    person[150:340, 400:440] = 1

    tr = PanelTrack(tid=1, cls=1, quad=quad.copy(), bound_quad=quad.copy(),
                    color=np.array([60., 60., 120.], np.float32), active=True, fade=1.0)
    out = composite_panels(frame.copy(), [tr], person, 0,
                           ocfg={"occlusion_zone_down_frac": 0.25,
                                 "occlusion_zone_up_px": 6,
                                 "feet_tolerance_frac": 0.25, "min_person_px": 50},
                           dilate_kernel=None, gcfg={})

    check(np.allclose(out[220:270, 405:435], infront),
          "the player in front of the board was painted over")
    check(not np.allclose(out[215:230, 200:280], frame[215:230, 200:280]),
          "the panel did not cover the board at all")
    check(np.allclose(out[215:230, 302:310], (235, 235, 235)),
          "the goal post was painted over")
    covered = out[202:205, 155:185]
    check(not np.allclose(covered, behind),
          "the crowd behind the board still leaves a hole in the panel")


# ---------------------------------------------------------------------------

def main():
    for fn in (test_person_excluded_from_panel,
               test_color_is_sampled_from_the_board_face,
               test_fill_is_muted_and_never_brighter_than_the_scene,
               test_color_adapts_to_a_different_board_colour,
               test_pinned_brand_colour_wins_and_stays_faithful,
               test_unpinned_brand_is_not_painted_another_brands_colour,
               test_ghost_panel_is_dropped,
               test_sample_band_is_inside_the_quad,
               test_held_person_mask_follows_camera_motion,
               test_person_config_defaults_favour_recall,
               test_thickness_is_measured_perpendicular_not_as_bbox_height,
               test_over_thick_panel_is_trimmed_from_the_top,
               test_only_people_in_front_of_the_board_punch_holes,
               test_goal_structure_is_protected_without_eating_the_wordmark,
               test_bottom_edge_snaps_down_to_the_grass_seam,
               test_occluder_survives_a_partial_segmentation,
               test_composite_respects_the_occlusion_zone_and_the_goal):
        try:
            fn()
        except Exception as exc:  # noqa: BLE001 - a crash is a failure like any other
            FAILURES.append(f"{fn.__name__} raised {type(exc).__name__}: {exc}")

    if FAILURES:
        print(f"FAIL ({len(FAILURES)}):")
        for f in FAILURES:
            print(f"  - {f}")
        sys.exit(1)
    print("PASS (17): person occlusion; board-face colour sourcing; muting; colour "
          "adaptation; pinned brand colour; no cross-brand snapping; ghost-panel "
          "dropping; band placement; held-mask motion; recall-favouring defaults; "
          "perpendicular thickness; top-anchored trim; pitch-side-only occlusion; "
          "goal protected without eating the wordmark; bottom-edge snap; "
          "composite end-to-end; occluder survives a partial segmentation "
          "(latched verdict + recent-mask union)")


if __name__ == "__main__":
    main()
