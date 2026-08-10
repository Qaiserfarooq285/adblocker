#!/usr/bin/env python3
"""Hide betting ads in football video.

Stage order, deliberately:
  detect -> fit quad -> track/smooth -> build fill -> subtract people -> composite

Design notes that matter, because earlier versions failed on each of them:

* **Quad, not raw mask.**  A per-frame segmentation mask wobbles at the pixel
  level even when the board is rock still.  Fitting a four-corner quad and
  smoothing the corners removes that wobble without decoupling the panel from
  the board.

* **Detection-gated, no optical-flow propagation.**  The panel is only drawn
  where the model currently sees a board (plus a very short hold across
  dropouts).  Nothing is ever carried by camera motion, which is what made the
  old panel drift across the pitch during pans.

* **People are a hard subtraction, computed independently.**  Person masks come
  from a general COCO segmentation model, not from the ad model, and are
  unioned over several frames so a partially-segmented player is still fully
  protected.

* **Fill is built in rail space.**  A perimeter LED board is a physical object
  with nothing behind it, so the fill synthesises blank board surface sampled
  from the same rail rather than pretending to recover a hidden background.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import deque
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


# ------------------------------------------------------------------ geometry

def canonical(quad: np.ndarray) -> np.ndarray:
    """Corners ordered along the board's long axis, as tl, tr, br, bl.

    Ordering by y-coordinate looks equivalent and is not: on a rail receding
    steeply into the distance, both ends of one short edge can outrank a long
    edge's corners, which silently swaps length with thickness.  Everything
    downstream then works across the rail instead of along it -- the fill
    samples its colour from grass and crowd rather than from the rail.  Anchor
    on the longest edge instead, which no amount of perspective can confuse.
    """
    q = np.asarray(quad, np.float32).reshape(4, 2)
    lengths = [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]
    q = np.roll(q, -int(np.argmax(lengths)), axis=0)
    if q[0][0] > q[1][0]:                       # run the long edge left to right
        q = q[[1, 0, 3, 2]]
    if q[0][1] > q[3][1]:                       # make the long edge the top one
        q = q[[3, 2, 1, 0]]
    return q.astype(np.float32)


def quad_shape(quad: np.ndarray) -> tuple[float, float]:
    """(length along the rail, thickness across it)."""
    q = canonical(quad)
    return (float(np.linalg.norm(q[1] - q[0])), float(np.linalg.norm(q[3] - q[0])))


def merge_overlapping(masks: list[np.ndarray], confs: list[float],
                      iou_thr: float = 0.20, contain_thr: float = 0.50):
    """Mask-level NMS over board proposals.

    The model often proposes a rail twice: a confident piece covering the
    betting run, and a lower-confidence one that over-reaches along the same
    rail into the neighbouring sponsor.  Ultralytics' box NMS keeps both --
    their boxes overlap too little to suppress -- and painting both draws the
    offset union, which both thickens the panel and swallows an innocent
    sponsor.

    Fusing them is tempting and wrong: the union is exactly the over-reach.
    Keep the confident proposal and drop what overlaps it.  Leaving part of a
    rail showing on one frame is recoverable -- the tracker carries the board
    and other frames catch it -- whereas painting over a non-betting sponsor is
    the failure this whole project exists to avoid.
    """
    order = sorted(range(len(masks)), key=lambda i: -confs[i])
    kept: list[dict] = []
    for i in order:
        mi = masks[i]
        ai = int(mi.sum())
        if ai == 0:
            continue
        for g in kept:
            inter = int(np.count_nonzero(g["mask"] & mi))
            if inter == 0:
                continue
            union = int(np.count_nonzero(g["mask"] | mi))
            if not (inter / union > iou_thr or inter / ai > contain_thr):
                continue
            # Same rail split into overlapping pieces, or one proposal
            # over-reaching past the end of the betting run?  Fusing collinear
            # pieces recovers the whole board; fusing an over-reach swallows
            # the neighbouring sponsor.  Shape decides: joining collinear
            # pieces leaves the strip just as thin, while joining anything else
            # fattens it.
            fused = g["mask"] | mi
            fq = quad_from_mask(fused)
            if fq is not None:
                fl, ft = quad_shape(fq)
                _, gt = quad_shape(quad_from_mask(g["mask"]))
                _, it = quad_shape(quad_from_mask(mi))
                if ft <= 1.35 * max(gt, it) and fl / max(ft, 1e-6) >= 3.0:
                    g["mask"] = fused
                    g["conf"] = max(g["conf"], confs[i])
            break
        else:
            kept.append({"mask": mi, "conf": confs[i]})
    return kept


def quad_from_mask(mask: np.ndarray) -> np.ndarray | None:
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 40:
        return None
    return canonical(cv2.boxPoints(cv2.minAreaRect(c)))


def fit_strip(mask: np.ndarray, along_pct: float = 0.5,
              across_pct: float = 2.0) -> np.ndarray | None:
    """Fit one parallelogram tightly to a board mask.

    minAreaRect is driven by extreme pixels, so a few stray mask pixels above
    the board make the rectangle taller than the board itself.  Fitting the
    principal axis and taking robust percentiles across it instead gives a
    strip that matches the board's real thickness, and it is a true
    parallelogram, so all four edges are straight.
    """
    ys, xs = np.nonzero(mask)
    if len(xs) < 40:
        return None
    pts = np.column_stack([xs, ys]).astype(np.float32)
    mean = pts.mean(axis=0)
    centred = pts - mean
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    d = vt[0] / (np.linalg.norm(vt[0]) + 1e-6)
    n = np.array([-d[1], d[0]], np.float32)
    t = centred @ d
    p = centred @ n
    t0, t1 = np.percentile(t, [along_pct, 100 - along_pct])
    p0, p1 = np.percentile(p, [across_pct, 100 - across_pct])
    if t1 - t0 < 8 or p1 - p0 < 3:
        return None
    corners = np.float32([
        mean + d * t0 + n * p0, mean + d * t1 + n * p0,
        mean + d * t1 + n * p1, mean + d * t0 + n * p1])
    return canonical(corners)


def strips_from_frame(masks: list[np.ndarray], shape: tuple[int, int],
                      min_fill: float = 0.70, seg_len: float = 240.0,
                      max_seg: int = 10) -> list[np.ndarray]:
    """Turn all board masks in a frame into clean, non-overlapping strips.

    Painting one quad per detection stacks several near-duplicate rectangles of
    slightly different height, and a union of rectangles is not a rectangle --
    that is what produced the stepped, too-tall band.  Merging the masks first
    and fitting one strip per physical rail gives exactly one clean panel per
    board.  A rail that bends round the corner fills its strip poorly and is
    cut into straight pieces.
    """
    h, w = shape
    if not masks:
        return []
    union = np.zeros((h, w), np.uint8)
    for m in masks:
        union |= (m > 0).astype(np.uint8)
    # Bridge the small gaps the detector leaves between pieces of one rail.
    union = cv2.morphologyEx(union, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))

    out: list[np.ndarray] = []
    n_lbl, labels = cv2.connectedComponents(union, 8)
    for i in range(1, n_lbl):
        comp = (labels == i).astype(np.uint8)
        if int(comp.sum()) < 150:
            continue
        q = fit_strip(comp)
        if q is None:
            continue
        area = cv2.contourArea(q.astype(np.float32))
        if area > 0 and float(comp.sum()) / area >= min_fill:
            out.append(q)
            continue
        length, _ = quad_shape(q)
        k = int(np.clip(round(length / seg_len), 2, max_seg))
        top, bot = q[1] - q[0], q[2] - q[3]
        for j in range(k):
            t0, t1 = j / k, (j + 1) / k
            strip = np.zeros((h, w), np.uint8)
            cv2.fillConvexPoly(strip, np.float32([
                q[0] + top * t0, q[0] + top * t1,
                q[3] + bot * t1, q[3] + bot * t0]).astype(np.int32), 1)
            piece = comp & strip
            pq = fit_strip(piece)
            if pq is not None:
                out.append(pq)
    return out


def quads_from_mask(mask: np.ndarray, min_fill: float = 0.72,
                    seg_len: float = 220.0, max_seg: int = 12) -> list[np.ndarray]:
    """One or more tight quads covering a board mask.

    Perimeter boards bend where they wrap round the corner flag, and a single
    minAreaRect over a bent strip is necessarily far taller than the boards --
    that extra height is what reaches up into the crowd.  When the mask fills
    its rectangle poorly, cut it along its own long axis and fit a rectangle
    per slice, giving a chain of tight rectangles that follows the bend while
    every edge stays straight.
    """
    q = quad_from_mask(mask)
    if q is None:
        return []
    area = cv2.contourArea(q.astype(np.float32))
    if area <= 0:
        return []
    if float(mask.sum()) / area >= min_fill:
        return [q]

    length, _ = quad_shape(q)
    n = int(np.clip(round(length / seg_len), 2, max_seg))
    h, w = mask.shape[:2]
    top, bot = q[1] - q[0], q[2] - q[3]
    out = []
    for i in range(n):
        t0, t1 = i / n, (i + 1) / n
        strip = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(strip, np.float32([
            q[0] + top * t0, q[0] + top * t1,
            q[3] + bot * t1, q[3] + bot * t0]).astype(np.int32), 1)
        piece = mask & strip
        if int(piece.sum()) < 80:
            continue
        pq = quad_from_mask(piece)
        if pq is not None:
            out.append(pq)
    return out or [q]


def quad_iou(a: np.ndarray, b: np.ndarray, shape: tuple[int, int]) -> float:
    h, w = shape
    sc = 256.0 / max(h, w)
    ma = np.zeros((int(h * sc) + 2, int(w * sc) + 2), np.uint8)
    mb = ma.copy()
    cv2.fillConvexPoly(ma, (a * sc).astype(np.int32), 1)
    cv2.fillConvexPoly(mb, (b * sc).astype(np.int32), 1)
    inter = int(np.count_nonzero(ma & mb))
    union = int(np.count_nonzero(ma | mb))
    return inter / union if union else 0.0


# -------------------------------------------------------------------- tracks

class Track:
    __slots__ = ("quad", "mask", "hits", "misses", "conf", "alive", "id")
    _next = 0

    def __init__(self, quad: np.ndarray, conf: float, mask: np.ndarray | None = None):
        self.quad = quad.copy()
        self.mask = mask
        self.hits = 1
        self.misses = 0
        self.conf = conf
        self.alive = False
        Track._next += 1
        self.id = Track._next

    def update(self, quad: np.ndarray, conf: float, alpha: float,
               mask: np.ndarray | None = None) -> None:
        self.quad = (alpha * quad + (1.0 - alpha) * self.quad).astype(np.float32)
        if mask is not None:
            self.mask = mask
        self.hits += 1
        self.misses = 0
        self.conf = conf

    def miss(self) -> None:
        self.misses += 1


class Tracker:
    def __init__(self, iou_thr: float, min_hits: int, max_hold: int, alpha: float):
        self.iou_thr = iou_thr
        self.min_hits = min_hits
        self.max_hold = max_hold
        self.alpha = alpha
        self.tracks: list[Track] = []

    def reset(self) -> None:
        """Drop every track.  Used on a shot change.

        Holding a track across a cut paints the previous shot's board onto
        whatever the new shot happens to show, which reads as the panel
        "carrying over" into unrelated frames.  A cut invalidates all geometry,
        so nothing should survive it.
        """
        self.tracks.clear()

    def step(self, dets: list[tuple], shape) -> list[Track]:
        used = set()
        for tr in self.tracks:
            best, bi = self.iou_thr, -1
            for i, d in enumerate(dets):
                if i in used:
                    continue
                v = quad_iou(tr.quad, d[0], shape)
                if v > best:
                    best, bi = v, i
            if bi >= 0:
                used.add(bi)
                d = dets[bi]
                tr.update(d[0], d[1], self.alpha, d[2] if len(d) > 2 else None)
            else:
                tr.miss()
        for i, d in enumerate(dets):
            if i not in used:
                self.tracks.append(Track(d[0], d[1], d[2] if len(d) > 2 else None))
        self.tracks = [t for t in self.tracks if t.misses <= self.max_hold]
        out = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                t.alive = True
            if t.alive and t.misses <= self.max_hold:
                out.append(t)
        return out


# ----------------------------------------------------------------- rail fill

def rail_fill(frame: np.ndarray, quad: np.ndarray, side_frac: float = 0.55,
              darken: float = 0.90) -> np.ndarray:
    """Synthesise blank board surface for one quad.

    Samples the rail immediately left and right of the betting run at the same
    height, takes a robust dark percentile (LED boards read near-black when not
    lit), and lays it down with the rail's own vertical shading so the patch
    does not look like a sticker.
    """
    h, w = frame.shape[:2]
    q = canonical(quad)
    width = float(np.linalg.norm(q[1] - q[0]))
    height = float(np.linalg.norm(q[3] - q[0]))
    if width < 8 or height < 4:
        return np.zeros((0, 0, 3), np.uint8)

    span = max(12.0, width * side_frac)
    dirv = (q[1] - q[0]) / (np.linalg.norm(q[1] - q[0]) + 1e-6)
    candidates = []
    for sgn in (-1.0, 1.0):
        base_a = q[0] if sgn < 0 else q[1]
        base_b = q[3] if sgn < 0 else q[2]
        a = base_a + dirv * span * sgn
        b = base_b + dirv * span * sgn
        poly = np.array([base_a, a, b, base_b], np.float32) if sgn < 0 else \
               np.array([a, base_a, base_b, b], np.float32)
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, poly.astype(np.int32), 255)
        px = frame[m > 0]
        if px.size < 30:
            continue
        px = px.reshape(-1, 3).astype(np.float32)
        med = np.median(px, axis=0)
        # Boards are flat; grass and crowd are not.  Spread tells them apart.
        spread = float(np.mean(np.abs(px - med)))
        candidates.append((spread, med, px))

    # A rail that ends mid-frame has board on one side and pitch on the other.
    # Averaging both gives the muddy colour that belongs to neither, so take
    # the flatter neighbour and ignore the other entirely.
    base = None
    if candidates:
        spread, med, px = min(candidates, key=lambda c: c[0])
        if spread < 26.0:
            base = np.percentile(px, 35, axis=0)
    if base is None:
        # No usable neighbour: fall back to the board's own unlit tone, which
        # is what an LED rail looks like between messages.
        m = np.zeros((h, w), np.uint8)
        cv2.fillConvexPoly(m, q.astype(np.int32), 255)
        px = frame[m > 0]
        base = (np.percentile(px.reshape(-1, 3), 15, axis=0)
                if px.size else np.array([32.0, 32.0, 32.0]))

    base = np.clip(base * darken, 0, 255)
    ph, pw = max(2, int(round(height))), max(2, int(round(width)))
    patch = np.tile(base.astype(np.float32), (ph, pw, 1))
    # Gentle top-to-bottom falloff mimics how these rails catch stadium light.
    ramp = np.linspace(1.06, 0.94, ph, dtype=np.float32).reshape(ph, 1, 1)
    return np.clip(patch * ramp, 0, 255).astype(np.uint8)


def warp_fill(frame: np.ndarray, quad: np.ndarray, patch: np.ndarray) -> np.ndarray:
    """Project a rail-space patch back onto the frame, returning a full-frame BGR."""
    h, w = frame.shape[:2]
    ph, pw = patch.shape[:2]
    src = np.float32([[0, 0], [pw, 0], [pw, ph], [0, ph]])
    M = cv2.getPerspectiveTransform(src, canonical(quad))
    return cv2.warpPerspective(patch, M, (w, h), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_TRANSPARENT,
                               dst=np.zeros((h, w, 3), np.uint8))


def _palette(patch: np.ndarray, cover: float = 0.90) -> np.ndarray:
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)
    q = lab.astype(np.int32) // 16
    codes = ((q[:, :, 0] << 16) | (q[:, :, 1] << 8) | q[:, :, 2]).ravel()
    keys, counts = np.unique(codes, return_counts=True)
    order = np.argsort(-counts)
    keep = counts[order].cumsum() <= cover * codes.size
    keep[0] = True
    return keys[order][keep]


def _strip(frame: np.ndarray, quad: np.ndarray, t0: float, t1: float) -> np.ndarray | None:
    """Sample the rail between two positions along its own axis (0..1 = quad)."""
    q = canonical(quad)
    top, bot = q[1] - q[0], q[2] - q[3]
    a = q[0] + top * t0
    b = q[0] + top * t1
    c = q[3] + bot * t1
    d = q[3] + bot * t0
    poly = np.float32([a, b, c, d])
    h, w = frame.shape[:2]
    # Sample whatever part of the strip is on screen.  Rejecting the whole
    # strip the moment a corner leaves the frame stops extension dead on
    # exactly the rails that run to the edge, which are the long ones we care
    # about.  Require most of it to be visible so the match stays meaningful.
    inside = ((poly[:, 0] > -4) & (poly[:, 0] < w + 4) &
              (poly[:, 1] > -4) & (poly[:, 1] < h + 4)).sum()
    if inside < 3:
        return None
    wd = int(round(np.linalg.norm(b - a)))
    ht = int(round(np.linalg.norm(d - a)))
    if wd < 6 or ht < 4:
        return None
    M = cv2.getPerspectiveTransform(
        poly, np.float32([[0, 0], [wd, 0], [wd, ht], [0, ht]]))
    return cv2.warpPerspective(frame, M, (wd, ht))


def extend_rail(frame: np.ndarray, quad: np.ndarray, mask: np.ndarray,
                step: float = 0.25, max_grow: float = 1.0,
                match_rel: float = 0.62) -> np.ndarray:
    """Follow a detected board along the rail while the surface still matches.

    The model truncates long runs: every proposal on a full-width rail comes
    back about the same length, so a Betano run continuing behind the goal is
    simply never predicted.  The board itself is the evidence -- step outward
    along the rail axis and keep going while the pixels still belong to the
    same palette, which stops at the neighbouring sponsor because its colours
    are different.

    Conservative on purpose: a high match threshold and a hard cap on growth,
    because over-running into an innocent sponsor is worse than stopping early.
    """
    q = canonical(quad)
    core = _strip(frame, q, 0.0, 1.0)
    if core is None or core.size == 0:
        return q
    pal = _palette(core)
    lab0 = cv2.cvtColor(core, cv2.COLOR_BGR2LAB).astype(np.int32) // 16
    c0 = (lab0[:, :, 0] << 16) | (lab0[:, :, 1] << 8) | lab0[:, :, 2]
    # Judge the extension against how well the board matches *itself*, not an
    # absolute number: a thick slanted rail includes net and crowd inside its
    # strip, so the ceiling is well below 1.0 and a fixed threshold never fires.
    match_min = match_rel * float(np.isin(c0, pal).mean())

    def matches(t0: float, t1: float) -> bool:
        s = _strip(frame, q, t0, t1)
        if s is None or s.size == 0:
            return False
        lab = cv2.cvtColor(s, cv2.COLOR_BGR2LAB).astype(np.int32) // 16
        codes = (lab[:, :, 0] << 16) | (lab[:, :, 1] << 8) | lab[:, :, 2]
        return float(np.isin(codes, pal).mean()) >= match_min

    lo, hi = 0.0, 1.0
    grown = 0.0
    while grown < max_grow and matches(hi, hi + step):
        hi += step
        grown += step
    while grown < max_grow and matches(lo - step, lo):
        lo -= step
        grown += step
    if lo == 0.0 and hi == 1.0:
        return q

    top, bot = q[1] - q[0], q[2] - q[3]
    return canonical(np.float32([
        q[0] + top * lo, q[0] + top * hi, q[3] + bot * hi, q[3] + bot * lo]))


def pitch_mask(frame: np.ndarray, dh: int = 6, ds: int = 48, dv: int = 58) -> np.ndarray:
    """Pixels that are almost certainly turf.

    On steeply-slanted rails near the corner flag the segmentation over-reaches
    downward onto the pitch, and a panel lying on the grass is the single most
    obvious artefact a viewer can spot.  An ad board is never turf, so turf is
    always safe to exclude.

    The threshold is deliberately narrow.  Measured over this footage turf
    occupies H 47-50, S 114-146, V 125-142 -- a very tight cluster -- whereas
    the Kalshi wordmark, the one betting brand that is also green, spans H
    35-117 at up to S 246.  Matching turf tightly removes grass without ever
    sparing a Kalshi board.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_, s_, v_ = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    cand = (h_ >= 38) & (h_ <= 62) & (s_ > 70) & (v_ > 60)
    if int(cand.sum()) < 0.02 * frame.shape[0] * frame.shape[1]:
        return np.zeros(frame.shape[:2], np.uint8)      # no pitch in shot
    ref = np.median(hsv[cand], axis=0)
    m = ((np.abs(h_.astype(np.int16) - int(ref[0])) <= dh) &
         (np.abs(s_.astype(np.int16) - int(ref[1])) <= ds) &
         (np.abs(v_.astype(np.int16) - int(ref[2])) <= dv)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))


def crossing_structure(frame: np.ndarray, area: np.ndarray,
                       v_min: int = 168, s_max: int = 74,
                       outside_frac: float = 0.35) -> np.ndarray:
    """White objects that pass *in front of* a board, e.g. goal posts and net.

    A plain "don't paint white" rule cannot work here: the PredictStreet rails
    are white lettering on black, so it would leave the ad perfectly readable.
    The distinction that does hold is geometric -- a goalpost, crossbar or net
    continues well beyond the board, while a glyph is entirely contained by it.
    So keep any white component that is substantially outside the paint area
    and drop the ones that are not.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    white = ((hsv[:, :, 2] >= v_min) & (hsv[:, :, 1] <= s_max)).astype(np.uint8)
    white = cv2.morphologyEx(white, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(white, 8)
    keep = np.zeros_like(white)
    inside = area > 0
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < 60:
            continue
        comp = labels == i
        overlap = int(np.count_nonzero(comp & inside))
        if overlap == 0:
            continue
        total = int(stats[i, cv2.CC_STAT_AREA])
        if (total - overlap) / total >= outside_frac:
            keep |= comp.astype(np.uint8)
    if keep.any():
        keep = cv2.dilate(keep, np.ones((3, 3), np.uint8), iterations=1)
    return keep * 255


def static_overlay_mask(video: Path, samples: int = 40) -> np.ndarray | None:
    """Broadcast graphics burned into fixed screen positions (clock, logo bug).

    Sampled widely across the clip, the camera has moved a great deal, so any
    pixel that never changes is painted on by the broadcaster rather than part
    of the scene.  Cheap to compute once per clip and it needs no hard-coded
    rectangle, which would not survive a different broadcaster.
    """
    cap = cv2.VideoCapture(str(video))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if n < samples * 2:
        cap.release()
        return None
    idxs = {int(round((i + 0.5) * n / samples)) - 1 for i in range(samples)}
    frames, i = [], 0
    while True:
        ok, fr = cap.read()
        if not ok:
            break
        if i in idxs:
            frames.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
        i += 1
    cap.release()
    if len(frames) < 8:
        return None
    std = np.std(np.stack(frames), axis=0)
    # Tighten until the mask is plausibly just graphics.  A clip whose camera
    # moves little leaves much of the scene static at a loose threshold, and
    # rejecting the whole mask then leaves the clock and score unprotected --
    # which is exactly what happened on clip02.  Only a genuinely burned-in
    # graphic survives the strictest pass.
    for thresh in (7.0, 4.0, 2.5, 1.5):
        m = (std < thresh).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        if m.mean() <= 0.12:
            return cv2.dilate(m, np.ones((9, 9), np.uint8), iterations=1) * 255
    return None


# --------------------------------------------------------------------- people

class PersonMasker:
    """Union-of-recent person masks, so a briefly mis-segmented player stays safe."""

    def __init__(self, model, conf: float, union: int, dilate: int, imgsz: int):
        self.model = model
        self.conf = conf
        self.hist: deque[np.ndarray] = deque(maxlen=max(1, union))
        self.kernel = np.ones((dilate, dilate), np.uint8) if dilate > 0 else None
        self.imgsz = imgsz

    def reset(self) -> None:
        """Forget history at a shot change; the previous shot's people are gone."""
        self.hist.clear()

    @staticmethod
    def _in_front(pmask: np.ndarray, quads: list[np.ndarray], slack: float = 6.0) -> bool:
        """Is this person between the camera and the boards, or behind them?

        Only people on the pitch occlude a board.  Crowd, photographers and
        officials standing behind the rail are themselves occluded by it, and
        subtracting them punches notches out of the panel's top edge -- which
        is what stops it reading as a clean rectangle.

        A person in front stands on the pitch, so their feet fall below the
        board's lower edge; someone behind it is cut off by the board and their
        lowest visible pixel sits at or above that edge.
        """
        ys, xs = np.nonzero(pmask)
        if len(ys) == 0:
            return False
        foot_y = float(ys.max())
        cx = float(xs[ys > ys.max() - 5].mean()) if (ys > ys.max() - 5).any() else float(xs.mean())
        for q in quads:
            c = canonical(q)
            # Bottom edge of the board, evaluated at this person's x.
            (x0, y0), (x1, y1) = c[3], c[2]
            if not (min(x0, x1) - 40 <= cx <= max(x0, x1) + 40):
                continue
            t = 0.0 if abs(x1 - x0) < 1e-3 else (cx - x0) / (x1 - x0)
            board_bottom = y0 + (y1 - y0) * float(np.clip(t, 0.0, 1.0))
            if foot_y > board_bottom - slack:
                return True
        return False

    def __call__(self, frame: np.ndarray,
                 quads: list[np.ndarray] | None = None) -> np.ndarray:
        h, w = frame.shape[:2]
        cur = np.zeros((h, w), np.uint8)
        r = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz,
                               classes=[0], verbose=False, retina_masks=True)[0]
        if r.masks is not None:
            for m in r.masks.data.cpu().numpy():
                mm = (m > 0.5).astype(np.uint8)
                if mm.shape != (h, w):
                    mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                if quads and not self._in_front(mm, quads):
                    continue
                cur |= mm
        if self.kernel is not None:
            cur = cv2.dilate(cur, self.kernel, iterations=1)
        self.hist.append(cur)
        out = self.hist[0].copy()
        for m in list(self.hist)[1:]:
            out |= m
        # Return 0/255, not 0/1.  Callers combine this with 0/255 paint masks,
        # and a 0/1 mask silently degrades bitwise_not into a no-op
        # (bitwise_not(1) == 254, so 255 & 254 == 254 stays painted).
        return out * 255


# ----------------------------------------------------------------------- main

def process(video: Path, out_path: Path, cfg: dict, debug: bool = False) -> dict:
    from ultralytics import YOLO

    ad = YOLO(cfg["ad_weights"])
    person = YOLO(cfg["person_weights"])
    pm = PersonMasker(person, cfg["person_conf"], cfg["person_union"],
                      cfg["person_dilate"], cfg["person_imgsz"])
    tracker = Tracker(cfg["track_iou"], cfg["min_hits"], cfg["max_hold"], cfg["ema_alpha"])
    overlay = static_overlay_mask(video) if cfg["protect_overlay"] else None
    if overlay is not None:
        print(f"[render] static broadcast overlay: {100 * overlay.mean() / 255:.2f}% of frame")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    tmp = out_path.with_suffix(".tmp.mp4")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    feather = int(cfg["feather"])
    stats = {"frames": 0, "det_frames": 0, "painted": 0, "dets": 0, "cuts": 0}
    prev_small = None

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        stats["frames"] += 1

        # Shot-change detection.  Highlight packages cut constantly, and a
        # track held across a cut lands on unrelated content.
        small = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
        if prev_small is not None:
            if float(np.abs(small.astype(np.int16) - prev_small).mean()) > cfg["cut_delta"]:
                tracker.reset()
                pm.reset()
                stats["cuts"] += 1
        prev_small = small.astype(np.int16)

        res = ad.predict(frame, conf=cfg["render_conf"], imgsz=cfg["ad_imgsz"],
                         verbose=False, retina_masks=True)[0]
        dets: list[tuple[np.ndarray, float]] = []
        if res.masks is not None:
            # Shape-filter *before* suppression.  A perimeter rail is a long
            # thin strip; squat blobs are the model hedging.  Filtering after
            # NMS lets a high-confidence blob suppress every good rail it
            # overlaps and then be rejected itself, painting nothing at all --
            # which is precisely what happened once the threshold was lowered.
            raw, confs = [], []
            for mk, cf in zip(res.masks.data.cpu().numpy(),
                              res.boxes.conf.cpu().numpy()):
                mm = (mk > 0.5).astype(np.uint8)
                if mm.shape != (h, w):
                    mm = cv2.resize(mm, (w, h), interpolation=cv2.INTER_NEAREST)
                q = quad_from_mask(mm)
                if q is None:
                    continue
                length, thick = quad_shape(q)
                if thick < 3 or length / thick < cfg["min_aspect"]:
                    continue
                if thick > cfg["max_thick_frac"] * h:
                    continue
                raw.append(mm)
                confs.append(float(cf))
            kept = merge_overlapping(raw, confs)
            conf_of = max((g["conf"] for g in kept), default=0.0)
            for q in strips_from_frame([g["mask"] for g in kept], (h, w)):
                length, thick = quad_shape(q)
                if thick < 3 or length / thick < 1.6:
                    continue
                if thick > cfg["max_thick_frac"] * h:
                    continue
                dets.append((q, conf_of, None))
        if dets:
            stats["det_frames"] += 1
            stats["dets"] += len(dets)

        live = tracker.step(dets, (h, w))

        if live:
            fill = np.zeros((h, w, 3), np.uint8)
            area = np.zeros((h, w), np.uint8)
            grow = np.ones((cfg["mask_grow"], cfg["mask_grow"]), np.uint8) \
                if cfg["mask_grow"] > 1 else None
            for tr in live:
                quad = tr.quad
                patch = rail_fill(frame, quad)
                if patch.size == 0:
                    continue
                fill = np.maximum(fill, warp_fill(frame, quad, patch))
                # The panel is exactly the quad: four straight edges, matching
                # the board it was fitted to.
                #
                # Two earlier attempts to be cleverer both made it worse.
                # Intersecting with the raw segmentation left ragged notches
                # wherever the mask was imperfect, and unioning in "pixels whose
                # colour matches the board" painted every crowd pixel that
                # happened to share that colour -- which is how audience, seats
                # and people behind the boards ended up inside the panel.
                # Anything that must not be painted is subtracted afterwards as
                # its own explicit stage, which keeps those edges sharp too.
                cv2.fillConvexPoly(area, canonical(quad).astype(np.int32), 255)

            if area.any() and cfg["avoid_pitch"]:
                area = cv2.bitwise_and(area, cv2.bitwise_not(pitch_mask(frame) * 255))
            if area.any() and overlay is not None:
                area = cv2.bitwise_and(area, cv2.bitwise_not(overlay))
            if area.any() and cfg["protect_goal"]:
                area = cv2.bitwise_and(area, cv2.bitwise_not(crossing_structure(frame, area)))
            if area.any():
                soft = cv2.GaussianBlur(area, (feather | 1, feather | 1), 0) \
                    if feather > 1 else area
                # Feather first, then cut people out with a hard edge.  Blurring
                # after the subtraction smears alpha back across the silhouette
                # and paints a halo onto the player; a crisp edge is also what
                # a viewer expects where a person occludes a board.
                people = pm(frame, [t.quad for t in live])
                soft = cv2.bitwise_and(soft, cv2.bitwise_not(people))
                a = (soft.astype(np.float32) / 255.0)[:, :, None]
                frame = (frame * (1 - a) + fill * a).astype(np.uint8)
                stats["painted"] += 1

        if debug:
            for tr in live:
                cv2.polylines(frame, [tr.quad.astype(np.int32)], True, (0, 255, 255), 2)
        writer.write(frame)

    cap.release()
    writer.release()

    # Re-encode to H.264 so the result plays everywhere, keeping the original audio.
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", str(tmp), "-i", str(video),
           "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "libx264", "-crf", "20",
           "-preset", "medium", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
           str(out_path)]
    if subprocess.run(cmd).returncode == 0:
        tmp.unlink(missing_ok=True)
    else:
        tmp.replace(out_path)
    return stats


DEFAULTS = dict(
    ad_weights=str(ROOT / "models" / "ad_seg_best.pt"),
    person_weights="yolo11m-seg.pt",
    ad_imgsz=1920, person_imgsz=1280,
    render_conf=0.10, person_conf=0.15,
    person_dilate=9,
    track_iou=0.30, min_hits=2, max_hold=3, ema_alpha=0.45,
    feather=5,
    min_aspect=2.5, max_thick_frac=0.18, mask_grow=3, avoid_pitch=1,
    protect_goal=1, protect_overlay=1, extend_rail=0,
    cut_delta=14.0, person_union=5,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--debug", action="store_true")
    for k, v in DEFAULTS.items():
        ap.add_argument(f"--{k.replace('_', '-')}", type=type(v), default=v)
    args = ap.parse_args()

    cfg = {k: getattr(args, k) for k in DEFAULTS}
    src = Path(args.input)
    out = Path(args.output) if args.output else ROOT / "outputs" / f"{src.stem}_clean.mp4"
    stats = process(src, out, cfg, args.debug)
    print(json.dumps({"input": src.name, "output": str(out), **stats}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
