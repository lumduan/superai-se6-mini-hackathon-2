"""Phase 2 — Dynamic Table Page Detection.

Scans all available pages for a document and returns only those that contain
a vote count table.  Three complementary signals are combined (union):

  Signal A — OpenCV Line Detection:
      Dense horizontal + vertical line intersections indicate a grid/table.

  Signal B — OCR Keyword Detection:
      Thai table header keywords found by Tesseract (requires pytesseract +
      a Thai language pack).  Falls back gracefully when unavailable.

  Signal C — Row Count Heuristic:
      More than 5 digit-rich lines → likely a data table.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import IMAGES_DIR

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

TABLE_KEYWORDS = ["คะแนน", "รวมคะแนน", "พรรคการเมือง", "หมายเลข"]

# Minimum non-zero pixel counts (after morphological open) that indicate a
# proper table grid.  Tuned on typical สส.6/1 scan dimensions.
MIN_H_PIXELS = 500
MIN_V_PIXELS = 200

# Minimum digit-rich lines to trigger Signal C heuristic.
DIGIT_LINE_THRESHOLD = 5

# Minimum digits per line to count that line as "digit-rich".
DIGITS_PER_LINE = 3


# ── Signal A — OpenCV Line Detection ────────────────────────────────────────

def has_table_structure(image_path: str | Path) -> bool:
    """Return True when the page contains a dense horizontal + vertical grid.

    Uses morphological open with long thin kernels to isolate ruled lines,
    then checks that enough pixels survive (indicating a real table grid).
    """
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        logger.warning("Signal A: could not read image %s", image_path)
        return False

    _, binary = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY_INV)

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (img.shape[1] // 4, 1)
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, img.shape[0] // 8)
    )

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)

    h_count = cv2.countNonZero(h_lines)
    v_count = cv2.countNonZero(v_lines)

    result = h_count > MIN_H_PIXELS and v_count > MIN_V_PIXELS
    logger.debug(
        "Signal A [%s]: h=%d v=%d → %s",
        Path(image_path).name, h_count, v_count, result,
    )
    return result


# ── Signal B — OCR Keyword Detection ────────────────────────────────────────

def _ocr_text(image_path: str | Path) -> str:
    """Return raw OCR text for a page using pytesseract (Thai+English).

    Returns an empty string if pytesseract or its language data is missing,
    so the caller can degrade gracefully.
    """
    try:
        import pytesseract  # optional dependency
        from PIL import Image as _Image
        return pytesseract.image_to_string(
            _Image.open(str(image_path)), lang="tha+eng"
        )
    except ImportError:
        logger.debug("pytesseract not installed — Signal B disabled")
        return ""
    except Exception as exc:  # language pack missing, etc.
        logger.debug("Signal B OCR failed for %s: %s", image_path, exc)
        return ""


def has_table_keywords(image_path: str | Path) -> bool:
    """Return True when a Thai table header keyword is found via OCR (Signal B)."""
    text = _ocr_text(image_path)
    result = any(kw in text for kw in TABLE_KEYWORDS)
    logger.debug("Signal B [%s]: keyword hit=%s", Path(image_path).name, result)
    return result


# ── Signal C — Row Count Heuristic ──────────────────────────────────────────

def has_digit_rich_rows(image_path: str | Path) -> bool:
    """Return True when more than DIGIT_LINE_THRESHOLD digit-rich lines are found.

    Uses the same OCR text as Signal B but looks at line density rather than
    keywords, so it catches tables where keywords are garbled.
    """
    text = _ocr_text(image_path)
    if not text:
        return False

    digit_lines = sum(
        1
        for line in text.splitlines()
        if sum(c.isdigit() for c in line) > DIGITS_PER_LINE
    )
    result = digit_lines > DIGIT_LINE_THRESHOLD
    logger.debug(
        "Signal C [%s]: digit_lines=%d → %s",
        Path(image_path).name, digit_lines, result,
    )
    return result


# ── Public API ───────────────────────────────────────────────────────────────

def is_table_page(image_path: str | Path) -> bool:
    """Return True if any of the three signals fires for this page.

    Signals are evaluated in order A → B → C (short-circuit on first hit
    to avoid unnecessary OCR calls when structure detection already fired).
    """
    path = Path(image_path)
    if not path.exists():
        logger.warning("Page not found: %s", path)
        return False

    if has_table_structure(path):
        logger.info("is_table_page[A]: %s", path.name)
        return True

    if has_table_keywords(path):
        logger.info("is_table_page[B]: %s", path.name)
        return True

    if has_digit_rich_rows(path):
        logger.info("is_table_page[C]: %s", path.name)
        return True

    logger.debug("is_table_page: no signal for %s", path.name)
    return False


def get_table_pages(
    doc_key: str,
    images_dir: str | Path = IMAGES_DIR,
) -> list[Path]:
    """Return sorted list of page paths that contain a vote count table.

    Checks up to four pages (page1, page2, page3, page4) using the union of
    all three detection signals.  ``party_list`` doc keys are normalised to
    their ``constituency`` image key automatically.

    Parameters
    ----------
    doc_key:
        Document key, e.g. ``"constituency_10_1"`` or ``"party_list_10_1"``.
    images_dir:
        Directory containing PNG scans.

    Returns
    -------
    Sorted list of ``Path`` objects for pages that pass detection.
    """
    import re as _re

    base = Path(images_dir)
    # party_list shares constituency scan files
    image_key = _re.sub(r"^party_list_", "constituency_", doc_key)

    suffixes = ["", "_page2", "_page3", "_page4"]
    candidates = [
        base / f"{image_key}{s}.png" for s in suffixes if (base / f"{image_key}{s}.png").exists()
    ]

    table_pages = [p for p in candidates if is_table_page(p)]
    logger.info(
        "get_table_pages(%s): %d/%d pages are table pages",
        doc_key, len(table_pages), len(candidates),
    )
    return sorted(table_pages)
