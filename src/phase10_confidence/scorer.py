"""Phase 10 — Per-row Confidence Scoring.

Compute confidence at two granularities:

Row-level
---------
``compute_row_confidence(vote, position, total_expected)``
    Returns a 0.0–1.0 score for a single extracted vote string.  Uses Phase 8
    soft-rule multipliers as a base, then applies structural penalties
    (too short, non-digit) that are specific to this row's position in the
    document.

Document-level
--------------
``compute_document_confidence(votes, expected, ocr_total)``
    Returns a 0.0–1.0 score for an entire extracted vote list by checking:

    1. Length mismatch vs expected count.
    2. Zero-vote ratio (many zeros → likely OCR failure).
    3. Short-value ratio (very short digit strings → suspicious).
    4. Distribution plausibility (Phase 9 helper).
    5. Column consistency (Phase 6 helper).
    6. OCR-total checksum alignment (bonus for match, penalty for mismatch).

Fallback detection
------------------
``needs_fallback(votes, expected, confidence)``
    Returns ``True`` when the document confidence is below
    ``CONFIDENCE_THRESHOLD`` *or* the row count differs from expected by more
    than ``MISMATCH_TOLERANCE``.

Public API
----------
compute_row_confidence(vote, position, total_expected)
    Per-row 0.0–1.0 confidence score.
compute_document_confidence(votes, expected, ocr_total)
    Document-level 0.0–1.0 confidence score.
needs_fallback(votes, expected, confidence)
    Whether the Phase 11 multi-pass fallback should be triggered.
"""

from __future__ import annotations

import logging
from typing import Optional

from src.config import CONFIDENCE_THRESHOLD
from src.phase6_parse import has_consistent_column
from src.phase8_normalize import apply_soft_rules
from src.phase9_postprocess import is_reasonable_distribution

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# How many extra or missing rows are tolerated before forcing a fallback.
MISMATCH_TOLERANCE = 2

# Minimum digit-string length below which a per-row penalty is applied.
_SHORT_LENGTH_THRESHOLD = 3

# Per-row penalty applied when the vote string is shorter than
# ``_SHORT_LENGTH_THRESHOLD`` digits.
_SHORT_LENGTH_PENALTY = 0.3

# Fractional match tolerance for OCR-total checksum (1 % of total).
_TOTAL_MATCH_TOLERANCE = 0.01

# Score additions / subtractions for document-level checks.
_BONUS_TOTAL_MATCH = 0.2
_PENALTY_TOTAL_MISMATCH = 0.1
_PENALTY_DISTRIBUTION = 0.2
_PENALTY_COLUMN = 0.15
_PENALTY_ZERO_RATIO_SCALE = 0.3
_PENALTY_SHORT_RATIO_SCALE = 0.2
_MAX_LENGTH_MISMATCH_PENALTY = 0.5


# ── Public API ────────────────────────────────────────────────────────────────


def compute_row_confidence(
    vote: str,
    position: int,  # noqa: ARG001 — reserved for future positional heuristics
    total_expected: int,  # noqa: ARG001 — reserved for future ratio heuristics
) -> float:
    """Return a 0.0–1.0 confidence score for a single vote string.

    The score is derived from the Phase 8 soft-rule multiplier, then
    penalised when the vote string is suspiciously short.

    Parameters
    ----------
    vote:
        Normalised digit string (as produced by Phase 8).
    position:
        Zero-based index of this row within the document (reserved for
        future position-aware heuristics).
    total_expected:
        The number of candidates expected in this document (reserved for
        future ratio-aware heuristics).

    Returns
    -------
    A float in ``[0.0, 1.0]``.

    Examples
    --------
    >>> compute_row_confidence("12345", 0, 10)
    1.0
    >>> compute_row_confidence("1", 0, 10)     # too short
    0.2
    >>> compute_row_confidence("abc", 0, 10)   # non-digit
    0.0
    """
    if not vote or not vote.isdigit():
        return 0.0

    score: float = apply_soft_rules(vote)  # 0.0–1.0

    if len(vote) < _SHORT_LENGTH_THRESHOLD:
        score -= _SHORT_LENGTH_PENALTY

    return max(0.0, score)


def compute_document_confidence(
    votes: list[str],
    expected: int,
    ocr_total: Optional[int] = None,
) -> float:
    """Return a 0.0–1.0 confidence score for a full extracted vote list.

    Starts at ``1.0`` and deducts for each anomaly found.  A small bonus is
    awarded when the computed sum matches the OCR-printed grand total.

    Parameters
    ----------
    votes:
        List of normalised digit strings (as produced by Phase 8 / 9).
    expected:
        The number of candidate rows expected for this document.
    ocr_total:
        Grand total extracted from the OCR output (may be ``None``).

    Returns
    -------
    A float clamped to ``[0.0, 1.0]``.

    Examples
    --------
    >>> compute_document_confidence(["1234", "5678"], 2)
    1.0
    >>> compute_document_confidence([], 5)
    0.0
    """
    if not votes:
        return 0.0

    score: float = 1.0

    # 1. Length mismatch penalty — scaled by relative deviation, capped at 0.5.
    if len(votes) != expected:
        ratio = abs(len(votes) - expected) / max(expected, 1)
        score -= min(_MAX_LENGTH_MISMATCH_PENALTY, ratio)
        logger.debug(
            "compute_document_confidence: length mismatch %d vs %d, penalty=%.3f",
            len(votes),
            expected,
            min(_MAX_LENGTH_MISMATCH_PENALTY, ratio),
        )

    # 2. Zero-vote ratio penalty.
    zero_ratio = sum(v == "0" for v in votes) / len(votes)
    score -= zero_ratio * _PENALTY_ZERO_RATIO_SCALE

    # 3. Short-value ratio penalty (< 2 digits).
    short_ratio = sum(len(v) < 2 for v in votes) / len(votes)
    score -= short_ratio * _PENALTY_SHORT_RATIO_SCALE

    # 4. Distribution plausibility check (Phase 9).
    if not is_reasonable_distribution(votes):
        score -= _PENALTY_DISTRIBUTION
        logger.debug("compute_document_confidence: unreasonable distribution penalty")

    # 5. Column consistency check (Phase 6).
    if not has_consistent_column(votes):
        score -= _PENALTY_COLUMN
        logger.debug("compute_document_confidence: inconsistent column penalty")

    # 6. OCR-total checksum.
    if ocr_total is not None:
        numeric_sum = sum(int(v) for v in votes if v.isdigit())
        relative_error = abs(numeric_sum - ocr_total) / max(ocr_total, 1)
        if relative_error < _TOTAL_MATCH_TOLERANCE:
            score += _BONUS_TOTAL_MATCH
            logger.debug(
                "compute_document_confidence: total match bonus +%.2f (sum=%d, ocr=%d)",
                _BONUS_TOTAL_MATCH,
                numeric_sum,
                ocr_total,
            )
        else:
            score -= _PENALTY_TOTAL_MISMATCH
            logger.debug(
                "compute_document_confidence: total mismatch penalty -%.2f (sum=%d, ocr=%d)",
                _PENALTY_TOTAL_MISMATCH,
                numeric_sum,
                ocr_total,
            )

    return max(0.0, min(1.0, score))


def needs_fallback(
    votes: list[str],
    expected: int,
    confidence: float,
) -> bool:
    """Return ``True`` when Phase 11 multi-pass fallback should be triggered.

    Fallback is needed when *either* condition holds:
    - ``confidence`` is below ``CONFIDENCE_THRESHOLD`` (from ``src.config``).
    - The absolute difference between ``len(votes)`` and ``expected`` exceeds
      ``MISMATCH_TOLERANCE``.

    Parameters
    ----------
    votes:
        List of normalised digit strings.
    expected:
        The number of candidate rows expected for this document.
    confidence:
        Document-level confidence score from
        ``compute_document_confidence``.

    Returns
    -------
    ``True`` if fallback is needed, ``False`` otherwise.

    Examples
    --------
    >>> needs_fallback(["1234", "5678"], 2, 0.9)
    False
    >>> needs_fallback(["1234"], 5, 0.9)   # count mismatch > MISMATCH_TOLERANCE
    True
    >>> needs_fallback(["1234", "5678"], 2, 0.4)  # low confidence
    True
    """
    low_confidence = confidence < CONFIDENCE_THRESHOLD
    count_mismatch = abs(len(votes) - expected) > MISMATCH_TOLERANCE

    if low_confidence or count_mismatch:
        logger.debug(
            "needs_fallback: confidence=%.3f threshold=%.3f low=%s, "
            "count=%d expected=%d mismatch=%s",
            confidence,
            CONFIDENCE_THRESHOLD,
            low_confidence,
            len(votes),
            expected,
            count_mismatch,
        )
        return True

    return False
