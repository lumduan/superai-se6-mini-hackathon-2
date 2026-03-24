"""Phase 11 — Multi-pass Fallback OCR & Ensemble Voting.

Run up to 5 OCR passes when Phase 10 confidence is below threshold.
Full-page passes come first (passes 1–2) for structural diversity;
crop-based passes are used only as later fallbacks (passes 3–4).
Pass 5 uses Tesseract as a structurally independent fallback.

Public API
----------
normalize_length(votes, expected)
    Pad or truncate a vote list to exactly *expected* rows.
preprocess_otsu(image_path)
    Apply Otsu threshold only — structurally different from CLAHE.
fallback_tesseract(image_path)
    Extract vote strings using Tesseract (tha+eng).
ensemble_votes(candidates, expected)
    Weighted per-row majority vote across multiple OCR passes.
apply_sanity_checks(votes)
    Remove impossible values and strip leading zeros.
extract_votes_multipass(image_path, expected, api_key)
    Main entry point — runs all passes and returns the ensemble result.
"""

from __future__ import annotations

import logging
from collections import Counter
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from PIL import Image

from src.phase3_preprocess import preprocess_image
from src.phase4_crop import all_fallback_crops, crop_vote_column, FALLBACK_CROP_RATIOS
from src.phase5_ocr import run_typhoon_ocr
from src.phase6_parse import parse_html_table
from src.phase7_thai_crosscheck import cross_check_vote
from src.phase8_normalize import apply_hard_rules, apply_soft_rules, normalize_votes
from src.phase9_postprocess import extract_total_from_html, total_based_correction
from src.phase10_confidence import (
    compute_document_confidence,
    compute_row_confidence,
    needs_fallback,
)

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────


def normalize_length(votes: list[str], expected: int) -> list[str]:
    """Pad with ``"0"`` or truncate so *votes* has exactly *expected* entries.

    Parameters
    ----------
    votes:
        Extracted vote strings from one OCR pass.
    expected:
        The number of candidate rows expected in this document.

    Returns
    -------
    A list of exactly *expected* strings.

    Examples
    --------
    >>> normalize_length(["100", "200"], 4)
    ['100', '200', '0', '0']
    >>> normalize_length(["100", "200", "300", "400", "500"], 3)
    ['100', '200', '300']
    """
    if len(votes) < expected:
        votes = votes + ["0"] * (expected - len(votes))
    return votes[:expected]


def preprocess_otsu(image_path: str | Path) -> Image.Image:
    """Return an Otsu-thresholded version of the image as a PIL Image.

    Applies binary Otsu thresholding (no CLAHE, no sharpen) to create a
    structurally different input for Pass 2 — this reduces correlated errors
    between Pass 1 (CLAHE + sharpen) and Pass 2.

    Parameters
    ----------
    image_path:
        Path to the raw scan PNG.

    Returns
    -------
    Binarized PIL Image.

    Raises
    ------
    FileNotFoundError
        If *image_path* cannot be read by OpenCV.
    """
    path = Path(image_path)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


def fallback_tesseract(image_path: str | Path) -> list[str]:
    """Extract vote strings from *image_path* using Tesseract (tha+eng).

    Filters output lines to those that contain 3–7 digits — the expected
    length range for Thai election vote counts.

    Parameters
    ----------
    image_path:
        Path to the image file.

    Returns
    -------
    List of digit-only strings extracted from likely vote-count lines.
    Empty list when Tesseract is not installed or returns no useful lines.
    """
    try:
        import pytesseract  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("fallback_tesseract: pytesseract not installed — returning []")
        return []

    try:
        # Pass path as string — pytesseract accepts both PIL Image and file path.
        # Avoids eagerly opening the file before tesseract is ready.
        text = pytesseract.image_to_string(str(image_path), lang="tha+eng")
    except Exception as exc:  # noqa: BLE001
        logger.warning("fallback_tesseract: error running Tesseract — %s", exc)
        return []

    results: list[str] = []
    for line in text.splitlines():
        digit_count = sum(c.isdigit() for c in line)
        if 3 <= digit_count <= 7:
            digits = "".join(c for c in line if c.isdigit())
            if digits:
                results.append(digits)
    return results


def apply_sanity_checks(votes: list[str]) -> list[str]:
    """Remove obviously impossible vote strings and strip leading zeros.

    Rules applied in order per value:
    1. Strings longer than 7 digits → replace with ``"0"`` (impossible count).
    2. Strings with leading zeros and length > 1 → strip leading zeros.
       ``"01234"`` → ``"1234"``; pure ``"0"`` is left unchanged.

    Parameters
    ----------
    votes:
        Ensemble result before final sanity filtering.

    Returns
    -------
    Cleaned vote list of the same length.

    Examples
    --------
    >>> apply_sanity_checks(["12345678", "01234", "0", "100"])
    ['0', '1234', '0', '100']
    """
    result: list[str] = []
    for v in votes:
        if len(v) > 7:
            result.append("0")
        elif v.startswith("0") and len(v) > 1:
            result.append(v.lstrip("0") or "0")
        else:
            result.append(v)
    return result


# ── Core ensemble ─────────────────────────────────────────────────────────────


def ensemble_votes(
    candidates: list[tuple[float, list[str]]],
    expected: int,
) -> list[str]:
    """Weighted per-row majority vote across multiple OCR passes.

    Each candidate is a ``(doc_confidence, votes_list)`` pair.  Before
    voting, every votes list is normalised to *expected* rows.  Per row,
    each candidate's weight is:

        weight = doc_confidence × row_confidence × (1 + agree_bonus)

    where *row_confidence* is derived from Phase 10 and *agree_bonus* is
    the fraction of other passes that agree on the same value — this
    suppresses correlated outliers.

    Parameters
    ----------
    candidates:
        List of ``(document_confidence, votes_list)`` tuples from each pass.
        ``document_confidence`` must be in ``[0.0, 1.0]``.
    expected:
        The number of candidate rows expected in the document.

    Returns
    -------
    List of *expected* vote strings chosen by weighted majority vote.

    Examples
    --------
    >>> ensemble_votes([(1.0, ["100", "200"]), (0.5, ["100", "999"])], 2)
    ['100', '200']
    """
    if not candidates:
        return ["0"] * expected

    normalized = [
        (conf, normalize_length(v, expected)) for conf, v in candidates
    ]

    results: list[str] = []
    n_passes = len(normalized)

    for i in range(expected):
        row_values = [votes[i] for _, votes in normalized]
        consensus: Counter[str] = Counter(row_values)

        vote_weights: Counter[str] = Counter()
        for conf, votes in normalized:
            v = votes[i]
            row_conf = compute_row_confidence(v, i, expected)
            agree_bonus = consensus[v] / n_passes
            weight = conf * row_conf * (1.0 + agree_bonus)
            vote_weights[v] += weight

        results.append(vote_weights.most_common(1)[0][0])

    return results


# ── Per-pass helper ────────────────────────────────────────────────────────────


def _run_pass(
    label: str,
    image: Image.Image | str | Path,
    expected: int,
    api_key: Optional[str],
) -> tuple[float, list[str]]:
    """Run one OCR pass and return ``(doc_confidence, votes)``.

    Applies Phase 5 → 6 → 7 → 8 → 9 in sequence on *image*.  The
    document confidence is computed with Phase 10.

    Parameters
    ----------
    label:
        Descriptive name for logging.
    image:
        Image to process (PIL, path string, or Path).
    expected:
        Number of expected candidate rows.
    api_key:
        Typhoon OCR API key (or ``None`` to use env var).

    Returns
    -------
    Tuple of ``(doc_confidence, votes_list)``.  On failure returns
    ``(0.0, [])``.
    """
    try:
        html = run_typhoon_ocr(image, api_key=api_key)
        parsed = parse_html_table(html)
        ocr_total = extract_total_from_html(html)

        votes: list[str] = []
        for _, raw, digits in parsed:
            crossed = cross_check_vote(raw, normalize_votes(digits))
            hard = apply_hard_rules(crossed)
            votes.append(hard)

        votes = total_based_correction(votes, ocr_total)
        conf = compute_document_confidence(votes, expected, ocr_total)
        logger.info("%s: conf=%.3f rows=%d", label, conf, len(votes))
        return conf, votes

    except Exception as exc:  # noqa: BLE001
        logger.warning("%s: pass failed — %s", label, exc)
        return 0.0, []


# ── Main entry point ──────────────────────────────────────────────────────────


def extract_votes_multipass(
    image_path: str | Path,
    expected: int,
    api_key: Optional[str] = None,
) -> list[str]:
    """Run multiple diverse OCR passes and return ensemble-voted votes.

    Pass order:
    1. Full preprocessed page — CLAHE + sharpen (Phase 3 pipeline) → Typhoon
    2. Full page — Otsu threshold only (structurally diverse) → Typhoon
    3. Adaptive crop of the vote column → Typhoon
    4. All fixed-ratio fallback crops (0.70, 0.75, 0.80, 0.85) → Typhoon
    5. Full page → Tesseract (tha+eng) — independent model

    Early exit after Pass 1 or 2 if the result no longer needs a fallback.

    Parameters
    ----------
    image_path:
        Path to the raw scan PNG.
    expected:
        Number of candidate rows expected in the document.
    api_key:
        Typhoon OCR API key.  Falls back to the ``TYPHOON_OCR_API_KEY`` env
        variable when ``None``.

    Returns
    -------
    List of *expected* vote strings after ensemble voting and sanity checks.
    """
    path = Path(image_path)
    candidates: list[tuple[float, list[str]]] = []

    # Pass 1 — full preprocessed page (PRIMARY: CLAHE + sharpen)
    preprocessed = preprocess_image(str(path))
    conf1, votes1 = _run_pass("Pass 1 (typhoon full preprocessed)", preprocessed, expected, api_key)
    candidates.append((conf1, votes1))
    if not needs_fallback(votes1, expected, conf1):
        logger.info("extract_votes_multipass: early exit after Pass 1 (conf=%.3f)", conf1)
        return apply_sanity_checks(normalize_length(votes1, expected))

    # Pass 2 — full page, Otsu threshold only (structural diversity)
    otsu_img = preprocess_otsu(path)
    conf2, votes2 = _run_pass("Pass 2 (typhoon otsu)", otsu_img, expected, api_key)
    candidates.append((conf2, votes2))
    if not needs_fallback(votes2, expected, conf2):
        logger.info("extract_votes_multipass: early exit after Pass 2 (conf=%.3f)", conf2)
        return apply_sanity_checks(normalize_length(votes2, expected))

    # Pass 3 — adaptive vote-column crop
    crop3 = crop_vote_column(path)
    conf3, votes3 = _run_pass("Pass 3 (typhoon adaptive crop)", crop3, expected, api_key)
    candidates.append((conf3, votes3))

    # Pass 4 — diverse fixed-ratio fallback crops
    for idx, fallback_img in enumerate(all_fallback_crops(path)):
        ratio = FALLBACK_CROP_RATIOS[idx] if idx < len(FALLBACK_CROP_RATIOS) else 0.0
        conf_i, votes_i = _run_pass(
            f"Pass 4.{idx + 1} (typhoon crop {ratio:.2f})",
            fallback_img,
            expected,
            api_key,
        )
        candidates.append((conf_i, votes_i))

    # Pass 5 — Tesseract (fully independent model)
    raw_tess = fallback_tesseract(path)
    votes_tess = [normalize_votes(v) for v in raw_tess]
    conf_tess = compute_document_confidence(votes_tess, expected)
    logger.info("Pass 5 (tesseract): conf=%.3f rows=%d", conf_tess, len(votes_tess))
    candidates.append((conf_tess, votes_tess))

    final = ensemble_votes(candidates, expected)
    return apply_sanity_checks(final)
