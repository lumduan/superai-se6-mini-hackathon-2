"""Phase 3 — Image Preprocessing.

Apply a sharpen correction to the full-page image before OCR to clarify
digit edges and improve Typhoon OCR accuracy on scanned documents.

Scope: sharpen only (unsharp mask).  No crop logic here — cropping is
Phase 4 (fallback only).
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


# ── Step 1 — Sharpen ─────────────────────────────────────────────────────────

def sharpen(img: np.ndarray) -> np.ndarray:
    """Unsharp mask to clarify digit edges.

    Blurs the image then subtracts the blur from the original (scaled) to
    amplify high-frequency detail (digit strokes, grid lines).

    Parameters
    ----------
    img:
        Grayscale or BGR uint8 array.

    Returns
    -------
    Sharpened image with the same shape and dtype.
    """
    blurred = cv2.GaussianBlur(img, (0, 0), 3)
    result = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    logger.debug("Sharpen: unsharp mask applied (alpha=1.5, beta=-0.5, sigma=3)")
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def preprocess_image(image_path: str | Path) -> Image.Image:
    """Preprocessing pipeline: sharpen.

    Reads the image from disk, applies the unsharp mask, and returns a PIL
    Image ready for full-page OCR.  No cropping is performed here — that is
    Phase 4 (fallback only).

    Parameters
    ----------
    image_path:
        Path to the PNG (or any OpenCV-readable format) scan file.

    Returns
    -------
    Sharpened PIL Image (same mode as input).

    Raises
    ------
    FileNotFoundError
        If the image cannot be read from *image_path*.
    """
    path = Path(image_path)
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")

    img = sharpen(img)

    logger.info("preprocess_image: pipeline complete for %s", path.name)
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
