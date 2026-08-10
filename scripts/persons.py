#!/usr/bin/env python3
"""Person masks that survive detection gaps.

A panel only avoids painting over someone if there is a person mask to
subtract that frame. So a single missed detection is enough to smear the fill
across a human for a frame, and repeated misses read as flicker. Referees in
all-black kit are the known worst case: low contrast, small, often against a
dark dugout, and dropped by the detector for a frame at a time.

The board side already solved this with tracking + camera-motion propagation,
so people get the same treatment:

  * detect with a LOW threshold - a spurious person costs a few unhidden
    pixels, a missed one paints over a human. The asymmetry is not close.
  * track ids (ByteTrack) so a person is a continuing thing, not a fresh
    observation each frame.
  * when a tracked person is missed, keep their last mask alive for
    `person_hold_frames`, advanced by the SAME global camera affine the panels
    use, so the held mask rides along with the pan instead of smearing.
  * an extra cheap detection pass inside the panel ROIs, where small
    near-board figures get lost in a full-frame pass.

Masks are stored as cropped patches plus an origin, not full-frame arrays:
warping a handful of small patches per frame is cheap, warping several 4K
masks is not.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from fill import in_front_evidence, quad_edge_y_at_x, strip_thickness


@dataclass
class PersonTrack:
    tid: int
    patch: np.ndarray            # cropped uint8 mask, 0/1
    origin: tuple[int, int]      # (x, y) of the patch's top-left in frame coords
    missed: int = 0
    held: bool = False           # True when this frame's mask is carried over
    in_front: bool = False       # LATCHED verdict: this person stands in front
                                 # of the board. Latched, never re-decided, because
                                 # the per-frame geometric evidence depends on the
                                 # mask's exact extent - when the detector finds a
                                 # referee's torso but misses his legs, the same
                                 # person tests as 'behind' for a frame and the
                                 # panel paints over him. Nobody walks from in
                                 # front of a perimeter board to behind it.
    front_votes: int = 0         # consecutive frames of positive evidence
    provisional: bool = False    # from the motion fallback, not the detector
    history: list = field(default_factory=list)
    # Recent masks for this person, kept because a DETECTED mask is not
    # necessarily a COMPLETE one. Measured on the 4K clip: on frame 711 the
    # assistant referee's legs and flag were segmented but his black torso was
    # not, so the panel painted his upper body out while his legs stayed - a
    # blink that no amount of holding or latching fixes, because a mask did
    # exist and was subtracted; it simply did not cover him. The occluder mask
    # is therefore the UNION of the last `mask_union_frames` masks, each
    # advanced by the camera affine, which makes coverage monotone through a
    # partial segmentation.


def warp_patch(patch: np.ndarray, origin: tuple[int, int], affine: np.ndarray,
               shape: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]] | None:
    """Move a cropped mask by the global camera affine, returning a new
    (patch, origin). None if the result leaves the frame or degenerates."""
    h, w = patch.shape[:2]
    x, y = origin
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], np.float32)
    moved = cv2.transform(corners.reshape(-1, 1, 2), affine).reshape(-1, 2)
    nx0, ny0 = np.floor(moved.min(axis=0)).astype(int)
    nx1, ny1 = np.ceil(moved.max(axis=0)).astype(int)
    nw, nh = int(nx1 - nx0), int(ny1 - ny0)
    fh, fw = shape
    if nw <= 0 or nh <= 0 or nw > 4 * w + 32 or nh > 4 * h + 32:
        return None
    if nx1 <= 0 or ny1 <= 0 or nx0 >= fw or ny0 >= fh:
        return None
    # patch-local -> new-patch-local: apply the affine in frame coords, then
    # re-origin onto the new crop
    m = affine.astype(np.float32).copy()
    m[:, 2] = (affine[:, :2] @ np.array([x, y], np.float64) + affine[:, 2]) - np.array([nx0, ny0])
    out = cv2.warpAffine(patch, m, (nw, nh), flags=cv2.INTER_NEAREST)
    return out, (int(nx0), int(ny0))


def paste(mask: np.ndarray, patch: np.ndarray, origin: tuple[int, int]) -> None:
    """OR a cropped patch into a full-frame mask, clipped to bounds."""
    fh, fw = mask.shape[:2]
    ph, pw = patch.shape[:2]
    x, y = origin
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(fw, x + pw), min(fh, y + ph)
    if x0 >= x1 or y0 >= y1:
        return
    sub = patch[y0 - y: y1 - y, x0 - x: x1 - x]
    mask[y0:y1, x0:x1] |= sub


class PersonMaskTracker:
    """Full-frame person mask with tracked ids and held masks across gaps."""

    def __init__(self, model, icfg: dict, imgsz: int):
        self.model = model
        self.conf = float(icfg.get("person_conf", 0.15))
        self.tracker = str(icfg.get("person_tracker", "bytetrack.yaml"))
        self.hold_frames = int(icfg.get("person_hold_frames", 12))
        self.roi_pass = bool(icfg.get("roi_person_pass", True))
        self.roi_upscale = float(icfg.get("roi_upscale", 2.0))
        self.roi_conf = float(icfg.get("roi_person_conf", self.conf))
        self.ocfg = dict(icfg.get("occlusion", {}) or {})
        self.mcfg = dict(icfg.get("motion_fallback", {}) or {})
        self.motion_fallback = bool(icfg.get("motion_person_fallback", True))
        self.union_frames = max(1, int(icfg.get("mask_union_frames", 3)))
        self.prev_gray: np.ndarray | None = None
        self.last_occluders: np.ndarray | None = None
        self._next_roi_id = -1000   # ROI-only people get their own id space
        self.device = str(icfg.get("device", 0))
        self.imgsz = imgsz
        self.tracks: dict[int, PersonTrack] = {}

    def _detect(self, frame: np.ndarray) -> list[tuple[int, np.ndarray]]:
        res = self.model.track(
            frame, tracker=self.tracker, persist=True, conf=self.conf,
            imgsz=self.imgsz, device=self.device, classes=[0],
            retina_masks=True, verbose=False,
        )[0]
        out = []
        if res.masks is None or res.boxes is None or len(res.boxes) == 0:
            return out
        h, w = frame.shape[:2]
        masks = res.masks.data.cpu().numpy()
        if masks.shape[1:] != (h, w):
            # exact frame dims, nearest-neighbour: a misaligned upscale makes
            # the whole subtraction silently no-op
            masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) for m in masks])
        ids = (res.boxes.id.cpu().numpy().astype(int) if res.boxes.id is not None
               else np.arange(-1, -len(masks) - 1, -1))
        for i, tid in enumerate(ids):
            out.append((int(tid), (masks[i] > 0.5).astype(np.uint8)))
        return out

    def _roi_instances(self, frame: np.ndarray,
                       roi_quads: list[np.ndarray]) -> list[np.ndarray]:
        """Second pass restricted to the panel ROIs, at genuinely HIGHER
        resolution. Small dark figures right at the board - a black-kit referee
        is the worst case - are exactly what a full-frame pass loses.

        The crop is UPSCALED before inference and imgsz is derived from the
        upscaled crop, which is the whole point. The previous version passed a
        fixed imgsz=640 for a crop that is often ~1100px wide, so it silently
        ran the 'high resolution' pass at 0.69x native - a downscale, and
        coarser than the full-frame pass it was supposed to rescue.

        Returns one mask per detected instance (not a merged mask) so callers
        can give each figure a track that survives the next frame's miss."""
        out: list[np.ndarray] = []
        if not roi_quads:
            return out
        h, w = frame.shape[:2]
        for quad in roi_quads:
            x0, y0 = np.floor(quad.min(axis=0)).astype(int)
            x1, y1 = np.ceil(quad.max(axis=0)).astype(int)
            pad = int(0.6 * max(1, y1 - y0))       # include people standing above the strip
            x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
            x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
            if x1 - x0 < 32 or y1 - y0 < 32:
                continue
            crop = frame[y0:y1, x0:x1]
            ch, cw = crop.shape[:2]
            if self.roi_upscale > 1.0:
                crop_in = cv2.resize(crop, None, fx=self.roi_upscale, fy=self.roi_upscale,
                                     interpolation=cv2.INTER_CUBIC)
            else:
                crop_in = crop
            imgsz = int(np.ceil(max(crop_in.shape[:2]) / 32) * 32)
            res = self.model.predict(crop_in, conf=self.roi_conf, imgsz=imgsz,
                                     device=self.device, classes=[0],
                                     retina_masks=True, verbose=False)[0]
            if res.masks is None or len(res.masks) == 0:
                continue
            m = res.masks.data.cpu().numpy()
            # Generous here on purpose. This gate exists only to keep the crowd
            # BEHIND the board out of the track table (they leaked the hold pool
            # to 46 held vs 34 detected), not to make the in-front/behind call -
            # that is latched per track later. A crowd member's mask ends at the
            # board's top edge, i.e. a full thickness above its bottom, so 0.6
            # still excludes them comfortably while letting a partially detected
            # referee through instead of discarding him for a frame.
            tol = float(self.ocfg.get("roi_feet_tolerance_frac", 0.6)) * strip_thickness(quad)
            for inst in m:
                mm = (inst > 0.5).astype(np.uint8)
                if mm.shape != (ch, cw):
                    mm = cv2.resize(mm, (cw, ch), interpolation=cv2.INTER_NEAREST)
                if mm.sum() < 40:
                    continue
                # Only register plausible OCCLUDERS. The crop around a
                # perimeter board is mostly crowd standing behind it, and at
                # 2x upscale the model finds dozens of them. Registering those
                # as tracks made the hold pool balloon (46 held vs 34 detected
                # on one measured frame); they then went stale, drifted with
                # the camera affine, and started punching holes in the panel.
                ys, xs = np.where(mm > 0)
                feet = y0 + int(ys.max())
                xc = x0 + float(xs.mean())
                if feet < quad_edge_y_at_x(quad, xc, "bottom") - tol:
                    continue
                full = np.zeros((h, w), dtype=np.uint8)
                full[y0:y1, x0:x1] = mm
                out.append(full)
        return out

    def _register_roi(self, inst: np.ndarray) -> None:
        """Fold an ROI-pass instance into the track table so it is HELD across
        the next detection gap like any other person.

        Previously these masks were OR'd into the frame's output and thrown
        away, so anyone the full-frame pass could not see - the referee -
        reappeared and vanished with every flicker of the ROI pass. Matching is
        by mask overlap against existing tracks; an unmatched instance gets its
        own id and starts being tracked."""
        ys, xs = np.where(inst > 0)
        if len(xs) == 0:
            return
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        area = float(inst.sum())

        best, best_iou = None, 0.0
        for tid, tr in self.tracks.items():
            ph, pw = tr.patch.shape[:2]
            tx, ty = tr.origin
            ix0, iy0 = max(x0, tx), max(y0, ty)
            ix1, iy1 = min(x1, tx + pw), min(y1, ty + ph)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            inter = float((inst[iy0:iy1, ix0:ix1]
                           & tr.patch[iy0 - ty:iy1 - ty, ix0 - tx:ix1 - tx]).sum())
            union = area + float(tr.patch.sum()) - inter
            iou = inter / max(union, 1.0)
            if iou > best_iou:
                best, best_iou = tid, iou

        if best is not None and best_iou >= 0.25:
            tr = self.tracks[best]
            tr.patch = inst[y0:y1, x0:x1].copy()
            tr.origin = (x0, y0)
            tr.missed = 0
            tr.held = False
            # A real detection promotes a provisional (motion-fallback) track:
            # it is now backed by the model, so it earns the normal hold instead
            # of being dropped on the next miss. The motion path re-sets this
            # immediately after its own call, so only detections clear it.
            tr.provisional = False
            return best

        tid = self._next_roi_id
        self._next_roi_id -= 1
        self.tracks[tid] = PersonTrack(tid=tid, patch=inst[y0:y1, x0:x1].copy(),
                                       origin=(x0, y0), missed=0, held=False)
        return tid

    def _motion_blobs(self, frame: np.ndarray, affine: np.ndarray | None,
                      quads: list[np.ndarray], covered: np.ndarray) -> list[np.ndarray]:
        """Last-resort masks for figures BOTH detector passes missed.

        The frame is compared against the previous one after cancelling camera
        motion with the affine the panels already use, so only things moving
        against the world survive. Two things keep it from punching random
        holes in the advert:

          * a perimeter board is an LED display whose CONTENT ANIMATES, so its
            own face frame-differences constantly. Blobs are therefore required
            to be anchored on the GRASS below the board - a person in front
            stands on the pitch, an advert animation cannot.
          * anything the detector already covers is excluded, so this only ever
            adds, never contradicts.
        """
        out: list[np.ndarray] = []
        if self.prev_gray is None or not quads:
            return out
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev = self.prev_gray
        if affine is not None:
            prev = cv2.warpAffine(prev, affine, (w, h), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_REPLICATE)
        diff = cv2.absdiff(gray, prev)
        moving = (diff > int(self.mcfg.get("motion_diff_thresh", 26))).astype(np.uint8)
        moving = cv2.morphologyEx(moving, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        moving = cv2.morphologyEx(moving, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        moving[covered > 0] = 0

        min_area = int(self.mcfg.get("motion_min_area_px", 900))
        for quad in quads:
            thick = strip_thickness(quad)
            x0 = int(max(0, quad[:, 0].min() - 8))
            x1 = int(min(w, quad[:, 0].max() + 8))
            y0 = int(max(0, quad[:, 1].min() - 0.5 * thick))
            y1 = int(min(h, quad[:, 1].max() + 2.5 * thick))
            if x1 - x0 < 16 or y1 - y0 < 16:
                continue
            sub = moving[y0:y1, x0:x1]
            if not sub.any():
                continue
            n, lab, st, cent = cv2.connectedComponentsWithStats(sub, 8)
            for i in range(1, n):
                if st[i, cv2.CC_STAT_AREA] < min_area:
                    continue
                top_y = y0 + st[i, cv2.CC_STAT_TOP]
                feet = top_y + st[i, cv2.CC_STAT_HEIGHT]
                xc = x0 + cent[i][0]
                # anchored on the grass, not floating on the advert's face
                if feet < quad_edge_y_at_x(quad, xc, "bottom") + 0.25 * thick:
                    continue
                if st[i, cv2.CC_STAT_HEIGHT] < st[i, cv2.CC_STAT_WIDTH]:
                    continue                       # people are taller than wide
                full = np.zeros((h, w), np.uint8)
                full[y0:y1, x0:x1] = (lab == i).astype(np.uint8)
                out.append(full)
        return out

    def update(self, frame: np.ndarray, affine: np.ndarray | None,
               roi_quads: list[np.ndarray] | None = None) -> tuple[np.ndarray, int, int]:
        """Returns (person_mask, n_detected, n_held) for this frame."""
        h, w = frame.shape[:2]

        # Bring every stored mask into THIS frame's coordinates first. History
        # entries were captured on earlier frames, so they must be advanced by
        # the camera affine before a fresh mask is unioned with them - otherwise
        # the union smears the person backwards across the pan.
        if affine is not None:
            for tr in self.tracks.values():
                warped = []
                for hp, ho in tr.history:
                    moved = warp_patch(hp, ho, affine, (h, w))
                    if moved is not None:
                        warped.append(moved)
                tr.history = warped

        detections = self._detect(frame)
        seen = set()

        for tid, m in detections:
            ys, xs = np.where(m > 0)
            if len(xs) == 0:
                continue
            x0, x1 = int(xs.min()), int(xs.max()) + 1
            y0, y1 = int(ys.min()), int(ys.max()) + 1
            patch = m[y0:y1, x0:x1].copy()
            # UPDATE the existing track rather than replacing it. Replacing the
            # object reset in_front/front_votes on every detected frame, so the
            # latch could never reach its threshold and never latched at all.
            tr = self.tracks.get(tid)
            if tr is None:
                tr = PersonTrack(tid=tid, patch=patch, origin=(x0, y0))
                self.tracks[tid] = tr
            else:
                tr.patch, tr.origin = patch, (x0, y0)
                tr.missed, tr.held, tr.provisional = 0, False, False
            tr.history.append((patch, (x0, y0)))
            del tr.history[:-self.union_frames]
            seen.add(tid)

        # ROI pass BEFORE the miss/hold sweep, so a person only the high-res
        # crop can see counts as detected this frame rather than as a gap.
        n_roi = 0
        if self.roi_pass:
            for inst in self._roi_instances(frame, roi_quads or []):
                tid = self._register_roi(inst)
                if tid is not None:
                    seen.add(tid)
                    n_roi += 1

        dropped = []
        for tid, tr in self.tracks.items():
            if tid in seen:
                continue
            tr.missed += 1
            tr.held = True
            # Motion-fallback masks are re-derived from scratch every frame, so
            # holding them would just pile up stale copies of the same figure
            # and let them drift. Drop on the first miss; if the person is still
            # there, the fallback produces a fresh blob next frame.
            if tr.provisional or tr.missed > self.hold_frames:
                dropped.append(tid)
                continue
            if affine is not None:
                moved = warp_patch(tr.patch, tr.origin, affine, (h, w))
                if moved is None:
                    dropped.append(tid)
                    continue
                tr.patch, tr.origin = moved
            # affine unavailable: hold the mask where it is rather than drop it -
            # a stale mask costs a few unhidden pixels, dropping it exposes a person
        for tid in dropped:
            del self.tracks[tid]

        mask = np.zeros((h, w), dtype=np.uint8)
        for tr in self.tracks.values():
            paste(mask, tr.patch, tr.origin)

        quads = roi_quads or []
        if self.motion_fallback and quads:
            for blob in self._motion_blobs(frame, affine, quads, mask):
                tid = self._register_roi(blob)
                if tid is not None:
                    self.tracks[tid].provisional = True
                    seen.add(tid)
                    paste(mask, self.tracks[tid].patch, self.tracks[tid].origin)
        self.prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # --- latch the in-front verdict, then build the occluder mask ---
        # The verdict is a property of the TRACK, not of this frame's mask.
        # Subtract when there is evidence NOW *or* the track has latched, so a
        # person is cut out from his very first frame and stays cut out through
        # any frame where his mask comes back partial.
        occluders = np.zeros((h, w), dtype=np.uint8)
        need = int(self.ocfg.get("front_latch_frames", 2))
        for tr in self.tracks.values():
            ph, pw = tr.patch.shape[:2]
            tx, ty = tr.origin
            ys, xs = np.where(tr.patch > 0)
            if len(xs) == 0:
                continue
            top_y = ty + int(ys.min())
            feet = ty + int(ys.max())
            xc = tx + float(xs.mean())
            evidence = any(in_front_evidence(top_y, feet, xc, q, self.ocfg) for q in quads)
            if evidence:
                tr.front_votes += 1
                if tr.front_votes >= need:
                    tr.in_front = True
            else:
                tr.front_votes = 0
            if evidence or tr.in_front:
                # UNION of recent masks, not just this frame's. A detected mask
                # is not necessarily a complete one: the assistant referee's
                # legs and flag segmented while his black torso did not, and
                # subtracting only that frame's mask painted his upper body out.
                paste(occluders, tr.patch, tr.origin)
                for hp, ho in tr.history:
                    paste(occluders, hp, ho)

        self.last_occluders = occluders
        n_held = sum(1 for tr in self.tracks.values() if tr.held)
        return mask, len(seen), n_held
