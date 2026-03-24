"""Phase 9 — Row Structure Validation & Total-based Correction.

Two checks are applied in order:

Check 1 — Distribution check
    Votes whose median falls below 50 are likely row numbers or OCR noise
    rather than real vote counts.  ``is_reasonable_distribution`` returns
    *False* when the distribution is unreasonable.

Check 2 — Total-based correction
    Thai election documents always print a grand-total row
    (``รวมคะแนนทั้งสิ้น``).  When Typhoon OCR captures this row, the
    extracted total is used as a checksum.  If the sum of the extracted
    vote rows differs from the checksum by a small amount, the single most
    suspicious row is adjusted to close the gap.

    A row's *suspicion score* combines:
    - Soft-confidence penalty from Phase 8 (low or impossible values → higher
      suspicion).
    - Normalised absolute deviation from the document median (outlier rows
      attract higher suspicion).

    The correction is **skipped** when:
    - No OCR total is available.
    - The gap is already 0 (sums match perfectly).
    - The gap is larger than 20 % of the largest single vote (too risky to
      guess which row to fix).
    - The adjusted value would be negative or exceed ``MAX_VOTE``.

Public API
----------
is_reasonable_distribution(votes)
    Return *True* when the vote-count distribution looks plausible.

extract_total_from_html(html)
    Extract the printed grand-total from an HTML table string.

extract_total_from_markdown(markdown)
    Extract the printed grand-total from a plain-text / markdown string.

total_based_correction(votes, ocr_total)
    Adjust the most suspicious row so that ``sum(votes) == ocr_total``.

validate_and_correct(votes, ocr_total)
    Convenience wrapper: distribution check → total-based correction.
"""

from __future__ import annotations

import logging
import re
import statistics
from typing import Optional

from bs4 import BeautifulSoup

from src.config import MAX_VOTE
from src.phase8_normalize import apply_soft_rules

logger = logging.getLogger(__name__)

# Thai keywords that identify a grand-total row in the document.
_TOTAL_KEYWORDS: list[str] = [
    "รวมคะแนนทั้งสิ้น",
    "รวมคะแนน",
    "รวมทั้งสิ้น",
]

# Translation table: Thai digits → Arabic digits.
_THAI_DIGIT_MAP: dict[int, int] = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Minimum median vote value for the distribution to be considered reasonable.
# Below this threshold the extracted values are more likely row numbers or OCR
# noise than genuine vote counts.
_DISTRIBUTION_MEDIAN_MIN = 50

# Maximum allowed gap between the computed sum and the OCR total, expressed as
# a fraction of the largest individual vote.  Gaps larger than this are too
# risky to fix automatically.
_MAX_CORRECTION_RATIO = 0.20

# Weights for the suspicion-score blend:
#   soft_penalty_weight   — penalty from Phase 8 plausibility check
#   deviation_weight      — normalised distance from document median
_SOFT_PENALTY_WEIGHT = 0.7
_DEVIATION_WEIGHT = 0.3


# ── Helpers ───────────────────────────────────────────────────────────────────


def _translate_thai_digits(text: str) -> str:
    """Translate Thai digit characters to ASCII digit characters."""
    return text.translate(_THAI_DIGIT_MAP)


def _extract_digits(text: str) -> str:
    """Return only ASCII digit characters from *text* (after Thai translation)."""
    return "".join(c for c in _translate_thai_digits(text) if c.isdigit())


# ── Public API ────────────────────────────────────────────────────────────────


def is_reasonable_distribution(votes: list[str]) -> bool:
    """Return *True* when the vote-count distribution looks plausible.

    A vote list is considered *unreasonable* when:

    - It contains no numeric values.
    - The median of the non-zero numeric values falls below
      ``_DISTRIBUTION_MEDIAN_MIN`` (50).  Low medians indicate that OCR
      captured row numbers, serial numbers, or digit noise rather than real
      vote counts.

    Parameters
    ----------
    votes:
        List of digit-only strings as produced by Phase 8 normalization.

    Returns
    -------
    ``True`` if the distribution looks plausible, ``False`` otherwise.

    Examples
    --------
    >>> is_reasonable_distribution(["1234", "5678", "9012"])
    True
    >>> is_reasonable_distribution(["1", "2", "3"])
    False
    >>> is_reasonable_distribution([])
    False
    """
    numeric = [int(v) for v in votes if v.isdigit() and v != "0"]
    if not numeric:
        return False
    return statistics.median(numeric) >= _DISTRIBUTION_MEDIAN_MIN


def extract_total_from_html(html: str) -> Optional[int]:
    """Extract the printed grand-total from a Typhoon HTML table.

    Searches for table rows whose combined cell text contains one of the
    ``_TOTAL_KEYWORDS``.  From the matching row the rightmost cell that
    contains 3–8 digits is taken as the total.

    Parameters
    ----------
    html:
        Raw HTML string returned by Typhoon OCR.

    Returns
    -------
    The grand total as an ``int``, or ``None`` if no total row is found or
    no numeric value can be extracted.

    Examples
    --------
    >>> html = "<table><tr><td colspan='3'>รวมคะแนนทั้งสิ้น</td><td>77,982</td></tr></table>"
    >>> extract_total_from_html(html)
    77982
    """
    if not html or not html.strip():
        return None

    soup = BeautifulSoup(html, "html.parser")
    for row in soup.find_all("tr"):
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        row_text = " ".join(cells)
        if not any(kw in row_text for kw in _TOTAL_KEYWORDS):
            continue

        # Rightmost cell with 3–8 digits is the total.
        for cell in reversed(cells):
            digits = _extract_digits(cell)
            if 3 <= len(digits) <= 8:
                total = int(digits)
                logger.debug("extract_total_from_html: found total %d in row %r", total, cells)
                return total

    return None


def extract_total_from_markdown(markdown: str) -> Optional[int]:
    """Extract the printed grand-total from a plain-text or markdown string.

    Scans lines for ``_TOTAL_KEYWORDS`` and extracts the first run of 3–8
    Arabic (or Thai) digits found on that line.

    Parameters
    ----------
    markdown:
        Plain text or markdown string (e.g. fallback OCR output).

    Returns
    -------
    The grand total as an ``int``, or ``None`` if not found.

    Examples
    --------
    >>> extract_total_from_markdown("รวมคะแนน | 77,982")
    77982
    """
    if not markdown or not markdown.strip():
        return None

    for line in markdown.splitlines():
        if not any(kw in line for kw in _TOTAL_KEYWORDS):
            continue
        translated = _translate_thai_digits(line)
        # Remove commas between digits so that "77,982" → "77982".
        normalized_line = re.sub(r"(\d),(\d)", r"\1\2", translated)
        # Match runs of exactly 3–8 digits that are not part of a longer
        # digit sequence (use negative lookahead/lookbehind).
        matches = re.findall(r"(?<!\d)\d{3,8}(?!\d)", normalized_line)
        if matches:
            total = int(matches[-1])  # take last (rightmost) match
            logger.debug("extract_total_from_markdown: found total %d on line %r", total, line)
            return total

    return None


def total_based_correction(
    votes: list[str],
    ocr_total: Optional[int],
) -> list[str]:
    """Adjust the most suspicious row so that ``sum(votes) == ocr_total``.

    The correction algorithm:

    1. If no *ocr_total* is provided, return *votes* unchanged.
    2. Compute the current sum of all digit-valid rows.
    3. If the gap is 0, return *votes* unchanged.
    4. If the gap exceeds 20 % of the largest individual vote, return *votes*
       unchanged (gap is too large to guess a single-row fix).
    5. Rank every valid row by a *suspicion score*:

       ``score = soft_penalty × 0.7 + normalised_deviation × 0.3``

       where *soft_penalty* = ``1.0 − apply_soft_rules(vote)`` (0 = plausible,
       1 = suspicious) and *normalised_deviation* = ``|vote − median| / (median + 1)``.
    6. Add the gap to the most suspicious row's value.
    7. If the adjusted value is out of ``[0, MAX_VOTE]``, return *votes*
       unchanged.

    Parameters
    ----------
    votes:
        List of digit-only strings (Phase 8 output).
    ocr_total:
        Grand total extracted from the OCR output, or ``None``.

    Returns
    -------
    A (possibly corrected) copy of *votes*.  The original list is never
    mutated.

    Examples
    --------
    >>> total_based_correction(["1000", "2000", "3000"], 6100)
    ['1000', '2000', '3100']
    """
    if ocr_total is None:
        return list(votes)

    numeric_vals = [(i, int(v)) for i, v in enumerate(votes) if v.isdigit()]
    current_sum = sum(val for _, val in numeric_vals)

    if current_sum == ocr_total:
        logger.debug("total_based_correction: sum already matches ocr_total=%d", ocr_total)
        return list(votes)

    gap = ocr_total - current_sum
    logger.debug(
        "total_based_correction: current_sum=%d ocr_total=%d gap=%d",
        current_sum,
        ocr_total,
        gap,
    )

    if not numeric_vals:
        return list(votes)

    max_val = max(val for _, val in numeric_vals)
    if max_val > 0 and abs(gap) > max_val * _MAX_CORRECTION_RATIO:
        logger.debug(
            "total_based_correction: gap %d > 20%% of max_val %d — skipping",
            gap,
            max_val,
        )
        return list(votes)

    # Compute suspicion scores.
    all_vals = [val for _, val in numeric_vals]
    median_val = statistics.median(all_vals) if all_vals else 0.0

    def _suspicion(i: int, val: int) -> float:
        soft_penalty = 1.0 - apply_soft_rules(str(val))  # 0 = plausible, 1 = suspicious
        deviation = abs(val - median_val) / (median_val + 1)
        return _SOFT_PENALTY_WEIGHT * soft_penalty + _DEVIATION_WEIGHT * deviation

    best_idx, best_val = max(numeric_vals, key=lambda iv: _suspicion(iv[0], iv[1]))

    adjusted = best_val + gap
    if not (0 <= adjusted <= MAX_VOTE):
        logger.debug(
            "total_based_correction: adjusted value %d out of range [0, %d] — skipping",
            adjusted,
            MAX_VOTE,
        )
        return list(votes)

    corrected = list(votes)
    corrected[best_idx] = str(adjusted)
    logger.info(
        "total_based_correction: row %d corrected %s → %s (gap=%+d, ocr_total=%d)",
        best_idx,
        votes[best_idx],
        corrected[best_idx],
        gap,
        ocr_total,
    )
    return corrected


def validate_and_correct(
    votes: list[str],
    ocr_total: Optional[int] = None,
) -> tuple[list[str], bool]:
    """Run distribution check then total-based correction.

    Parameters
    ----------
    votes:
        List of Phase 8 normalized digit strings.
    ocr_total:
        Grand total from OCR output (may be ``None``).

    Returns
    -------
    ``(corrected_votes, distribution_ok)`` where *distribution_ok* is
    ``True`` when :func:`is_reasonable_distribution` passes.

    Examples
    --------
    >>> validate_and_correct(["1200", "3400", "5600"], 10200)
    (['1200', '3400', '5600'], True)
    >>> validate_and_correct(["1", "2", "3"], None)
    (['1', '2', '3'], False)
    """
    distribution_ok = is_reasonable_distribution(votes)
    if not distribution_ok:
        logger.warning(
            "validate_and_correct: distribution check FAILED "
            "(median of non-zero votes < %d) — votes may be row numbers or noise",
            _DISTRIBUTION_MEDIAN_MIN,
        )

    corrected = total_based_correction(votes, ocr_total)
    return corrected, distribution_ok
