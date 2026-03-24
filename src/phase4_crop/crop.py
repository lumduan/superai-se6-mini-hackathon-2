"""Phase 4 — Adaptive Vote Column Crop.

Dynamically detect the rightmost column boundary and crop only the vote
count column from a preprocessed scan.

Two crop modes:

- **Mode A (default)**: adaptive crop of the rightmost column using vertical
  line detection — reduces noise, +10–30% OCR accuracy.
- **Mode B (fallback)**: fixed-ratio crop at 0.80 when detection fails —
  preserves context when the adaptive crop loses too much signal.

``all_fallback_crops`` returns crops at multiple fixed ratios so the
ensemble phase (Phase 11) can pick the best candidate.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Fixed crop ratios tried when adaptive detection fails.
# Each ratio is the left-edge position as a fraction of image width.
FALLBACK_CROP_RATIOS: list[float] = [0.70, 0.75, 0.80, 0.85]

# Minimum left-ratio returned by detection to be considered valid.
# Ratios below this (< 50 % from left) likely capture too much non-vote area.
MIN_VALID_RATIO = 0.5


# ── Core detection ────────────────────────────────────────────────────────────

def detect_rightmost_column_boundary(image_path: str | Path) -> float | None:
    """Detect the left edge of the rightmost vote column via vertical lines.

    Reads the image in grayscale, thresholds to isolate dark ink, then uses
    morphological opening with a tall vertical kernel to isolate vertical
    lines (table borders).  The second-to-last detected vertical line is
    returned as a fraction of the image width — this is the left boundary of
    the rightmost column.

    Parameters
    ----------
    image_path:
        Path to the scan image (any OpenCV-readable format).

    Returns
    -------
    Left-edge ratio (0.0–1.0) of the rightmost column, or ``None`` when
    fewer than two vertical lines can be found.
    """
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("detect_rightmost_column_boundary: cannot read %s", path)
        return None

    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, img.shape[0] // 8)
    )
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    col_sums = np.sum(v_lines, axis=0)
    line_positions = np.where(col_sums > img.shape[0] * 0.2)[0]

    if len(line_positions) < 2:
        logger.debug(
            "detect_rightmost_column_boundary: only %d line position(s) found — "
            "need at least 2",
            len(line_positions),
        )
        return None

    unique_cols = np.unique(line_positions)
    last_line = unique_cols[-2] if len(unique_cols) >= 2 else unique_cols[-1]
    ratio = float(last_line / img.shape[1])
    logger.debug(
        "detect_rightmost_column_boundary: boundary at column %d (ratio=%.3f)",
        last_line,
        ratio,
    )
    return ratio


# ── Public crop API ───────────────────────────────────────────────────────────

def crop_vote_column(image_path: str | Path) -> Image.Image:
    """Adaptively crop the vote column from a scan.

    **Mode A** — adaptive: runs vertical-line detection; if a boundary is
    found at ratio ≥ ``MIN_VALID_RATIO``, crops from that x-position to the
    right edge.

    **Mode B** — fallback: when detection fails (or returns a ratio that is
    too small), crops at the conservative default ratio of 0.80.

    Parameters
    ----------
    image_path:
        Path to the preprocessed scan PNG.

    Returns
    -------
    PIL Image containing only the vote column (right portion of the scan).

    Raises
    ------
    FileNotFoundError
        If *image_path* cannot be opened by Pillow.
    """
    path = Path(image_path)

    left_ratio = detect_rightmost_column_boundary(path)

    if left_ratio is not None and left_ratio >= MIN_VALID_RATIO:
        img = Image.open(path)
        w, h = img.size
        left_px = int(w * left_ratio)
        logger.info(
            "crop_vote_column: adaptive crop at x=%d (ratio=%.3f) for %s",
            left_px,
            left_ratio,
            path.name,
        )
        return img.crop((left_px, 0, w, h))

    # Detection failed or ratio is too small — use conservative fallback.
    fallback_ratio = 0.80
    logger.info(
        "crop_vote_column: adaptive crop failed for %s, using fallback ratio %.2f",
        path.name,
        fallback_ratio,
    )
    img = Image.open(path)
    w, h = img.size
    return img.crop((int(w * fallback_ratio), 0, w, h))


def all_fallback_crops(image_path: str | Path) -> list[Image.Image]:
    """Return cropped images for every fallback ratio.

    Produces one crop per entry in ``FALLBACK_CROP_RATIOS``.  These are
    added as separate ensemble candidates in Phase 11 to give the voting
    step diverse views of the vote column when adaptive detection cannot
    pin down the exact boundary.

    Parameters
    ----------
    image_path:
        Path to the preprocessed scan PNG.

    Returns
    -------
    List of PIL Images, one per ratio in ``FALLBACK_CROP_RATIOS``.

    Raises
    ------
    FileNotFoundError
        If *image_path* cannot be opened by Pillow.
    """
    path = Path(image_path)
    img = Image.open(path)
    w, h = img.size
    crops = [img.crop((int(w * r), 0, w, h)) for r in FALLBACK_CROP_RATIOS]
    logger.debug(
        "all_fallback_crops: produced %d crops for %s (ratios=%s)",
        len(crops),
        path.name,
        FALLBACK_CROP_RATIOS,
    )
    return crops
