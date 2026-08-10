"""Brand evidence required before a segmented board may be painted.

Segmentation answers *where is the board?*  It does not answer *is this a
betting ad?*  This module makes that second decision from the wordmark: OCR
must match a named gambling brand from the PDF-derived alias registry, and a
separate logo detector must also see a logo in the same board crop.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from filter_betting_boards import crop_for_ocr, match_brand, ocr_texts


class BettingBrandVerifier:
    def __init__(self, cfg: dict, device: str):
        self.cfg = cfg
        self.device = device
        self.detector = None
        weights_value = cfg.get("logo_weights")
        weights = Path(str(weights_value)) if weights_value else None
        if cfg.get("enabled", True) and cfg.get("require_logo_detector", True):
            if weights is None or not weights.is_file():
                raise FileNotFoundError(
                    f"Betting logo verifier weights not found: {weights_value!r}. "
                    "Set inference.brand_verification.logo_weights or disable the verifier explicitly.")
            from ultralytics import YOLO
            self.detector = YOLO(str(weights))

    def verify(self, frame: np.ndarray, quad: np.ndarray) -> tuple[bool, str]:
        """Return a named, PDF-registered brand only with corroborating evidence."""
        if not self.cfg.get("enabled", True):
            return True, "verification-disabled"
        crop = crop_for_ocr(frame, quad, min_h=int(self.cfg.get("ocr_min_height", 112)))
        if crop is None:
            return False, "unreadable-board"
        try:
            brand, _ = match_brand(ocr_texts(crop))
        except Exception as exc:  # OCR must fail closed; never paint on a failed verifier.
            return False, f"ocr-error:{type(exc).__name__}"
        # 'generic' indicates words such as sportsbook but no named PDF brand.
        # It is intentionally insufficient: it produced avoidable false panels.
        if not brand or brand == "generic":
            return False, "no-named-brand"

        if self.detector is not None:
            r = self.detector.predict(crop, conf=float(self.cfg.get("logo_conf", 0.35)),
                                      imgsz=int(self.cfg.get("logo_imgsz", 960)),
                                      device=self.device, verbose=False)[0]
            if r.boxes is None or len(r.boxes) == 0:
                return False, f"{brand}:no-logo-box"
        return True, brand
