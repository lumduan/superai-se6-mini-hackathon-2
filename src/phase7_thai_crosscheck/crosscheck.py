"""Phase 7 — Thai Text Cross-check with Digit-level Diff.

Thai election documents (Form สส.6/1) print vote counts both as Arabic digits
and as Thai number words in parentheses, e.g.::

    34,405 (สามหมื่นสี่พันสี่ร้อยห้า)

A digit-length match alone is insufficient — OCR can produce the same digit
count but wrong digits (e.g. ``34405`` → ``34485``).  This module uses a
**digit-level edit distance** to detect such errors and prefer the Thai text
value when the mismatch is significant.

Public API
----------
extract_thai_number_text(raw_cell)
    Extract the Thai number word string from the parenthesised sub-string.
digit_distance(a, b)
    Count differing digit positions between two digit strings.
cross_check_vote(raw_cell, digit_vote)
    Main entry point — returns the best vote string after cross-checking.

Helper (private)
----------------
_partial_thai_number(thai_text)
    Regex + fuzzy fallback for when pythainlp cannot parse the Thai number.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Number of digit mismatches that triggers a preference for Thai text value.
# A diff of 1-2 is considered OCR noise (keep digit OCR result).
# A diff ≥ 3 is considered reliable enough to trust the Thai spelling instead.
# Threshold raised from 2 → 3: Thai digit numerals (๖๙๔) are more reliably
# OCR'd than Thai number words; a 2-digit mismatch is still within typical
# word-OCR noise and should not override the numeral reading.
DIGIT_DIFF_THRESHOLD = 3

# If lengths differ by exactly this value, trust the Thai text length.
ACCEPTABLE_LENGTH_DIFF = 1

# Thai magnitude words → numeric value (ordered large → small for parsing).
_THAI_MAGNITUDES: list[tuple[str, int]] = [
    ("ล้าน", 1_000_000),
    ("แสน", 100_000),
    ("หมื่น", 10_000),
    ("พัน", 1_000),
    ("ร้อย", 100),
    ("สิบ", 10),
]

# Thai unit words → digit value.
_THAI_DIGITS_WORD: dict[str, int] = {
    "ศูนย์": 0,
    "หนึ่ง": 1,
    "สอง": 2,
    "สาม": 3,
    "สี่": 4,
    "ห้า": 5,
    "หก": 6,
    "เจ็ด": 7,
    "แปด": 8,
    "เก้า": 9,
}

# Minimum fuzzy-matching ratio (0–100) for rapidfuzz.partial_ratio.
_FUZZY_THRESHOLD = 80


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fuzzy_contains(word: str, text: str) -> bool:
    """Return True if *word* appears in *text* with fuzzy tolerance.

    Uses ``rapidfuzz.fuzz.partial_ratio`` when available; falls back to a
    plain substring check when rapidfuzz is not installed.  The fuzzy path
    handles common OCR distortions in the Thai pronunciation text (e.g.
    ``"ห้า"`` → ``"ห่า"``).
    """
    try:
        from rapidfuzz import fuzz  # type: ignore[import]

        window = len(word) + 2  # small lookahead window
        for i in range(max(1, len(text) - len(word) + 1)):
            candidate = text[i : i + window]
            if fuzz.partial_ratio(word, candidate) >= _FUZZY_THRESHOLD:
                return True
        return False
    except ImportError:
        return word in text  # graceful fallback


# ── Public API ────────────────────────────────────────────────────────────────


def extract_thai_number_text(raw_cell: str) -> str | None:
    """Extract the Thai number word string from the parenthesised part of *raw_cell*.

    Typhoon OCR renders vote cells as::

        "34,405 (สามหมื่นสี่พันสี่ร้อยห้า)"

    This function returns the text inside the first ``(…)`` pair, stripped of
    leading/trailing whitespace.  Returns ``None`` when no parentheses are
    found or the captured group is empty.

    Parameters
    ----------
    raw_cell:
        The raw vote-cell string from the HTML table.

    Returns
    -------
    The Thai number word string, or ``None``.
    """
    match = re.search(r"\(([^)]+)\)", raw_cell)
    if match:
        text = match.group(1).strip()
        return text if text else None
    return None


def digit_distance(a: str, b: str) -> int:
    """Count the number of differing digit positions between *a* and *b*.

    When the strings have different lengths, the absolute length difference is
    added to the count of positional mismatches over the shared prefix — this
    penalises length differences as well as wrong digits.

    Parameters
    ----------
    a, b:
        Strings of ASCII digits to compare.

    Returns
    -------
    Non-negative integer; 0 means *a* and *b* are identical.

    Examples
    --------
    >>> digit_distance("34405", "34405")
    0
    >>> digit_distance("34405", "34485")
    1
    >>> digit_distance("34405", "344850")
    2
    """
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b))
    min_len = min(len(a), len(b))
    positional_diff = sum(x != y for x, y in zip(a[:min_len], b[:min_len]))
    return abs(len(a) - len(b)) + positional_diff


def cross_check_vote(raw_cell: str, digit_vote: str) -> str:
    """Cross-check *digit_vote* against the Thai text representation in *raw_cell*.

    Algorithm
    ---------
    1. Extract the Thai number word from the parenthesised sub-string.
    2. Convert that text to an Arabic digit string using pythainlp; fall back
       to the partial regex parser when pythainlp fails.
    3. If lengths match, compute digit-level edit distance:
       - ``diff ≥ DIGIT_DIFF_THRESHOLD`` → multiple mismatches → trust Thai text.
       - ``diff < DIGIT_DIFF_THRESHOLD`` → minor noise → keep OCR digit.
    4. If lengths differ by exactly 1 → trust Thai text length.
    5. Otherwise keep the original OCR digit string.

    Parameters
    ----------
    raw_cell:
        Raw vote-cell string (may contain both digit and Thai number text).
    digit_vote:
        Normalised digit string already extracted from *raw_cell* by Phase 6.

    Returns
    -------
    The best available digit string — either *digit_vote* or the value derived
    from the Thai number text.
    """
    thai_text = extract_thai_number_text(raw_cell)
    if not thai_text:
        logger.debug("No Thai number text in cell: %r", raw_cell)
        return digit_vote

    thai_num = _convert_thai_text(thai_text)
    if thai_num is None:
        logger.debug("Thai number conversion failed for: %r", thai_text)
        return digit_vote

    len_a = len(thai_num)
    len_b = len(digit_vote)

    if len_a == len_b:
        diff = digit_distance(thai_num, digit_vote)
        if diff >= DIGIT_DIFF_THRESHOLD:
            logger.info(
                "Thai cross-check: digit diff=%d ≥ threshold=%d → using Thai value %r "
                "(was %r) for cell %r",
                diff,
                DIGIT_DIFF_THRESHOLD,
                thai_num,
                digit_vote,
                raw_cell,
            )
            return thai_num
        logger.debug(
            "Thai cross-check: digit diff=%d < threshold=%d → keeping OCR value %r",
            diff,
            DIGIT_DIFF_THRESHOLD,
            digit_vote,
        )
        return digit_vote

    # Different lengths
    if abs(len_a - len_b) == ACCEPTABLE_LENGTH_DIFF:
        logger.info(
            "Thai cross-check: length diff=1 → using Thai value %r (was %r) for cell %r",
            thai_num,
            digit_vote,
            raw_cell,
        )
        return thai_num

    logger.debug(
        "Thai cross-check: length diff=%d > 1 → keeping OCR value %r (Thai: %r)",
        abs(len_a - len_b),
        digit_vote,
        thai_num,
    )
    return digit_vote


# ── Internal conversion helpers ───────────────────────────────────────────────


def _convert_thai_text(thai_text: str) -> str | None:
    """Attempt Thai-word → Arabic-digit conversion using pythainlp, then fallback.

    First tries ``pythainlp.util.thai_word_to_num``; if that raises an
    exception (including the common ``KeyError`` for unrecognised tokens),
    falls through to the regex/fuzzy partial parser.

    Returns
    -------
    A digit string (no commas), or ``None`` on complete failure.
    """
    try:
        from pythainlp.util import thai_word_to_num  # type: ignore[import]

        result = thai_word_to_num(thai_text)
        if result is not None:
            s = str(result)
            if s.lstrip("-").isdigit():
                return s
    except Exception as exc:  # noqa: BLE001
        logger.debug("pythainlp.thai_word_to_num failed for %r: %s", thai_text, exc)

    # Fallback to partial regex parser
    return _partial_thai_number(thai_text)


def _partial_thai_number(thai_text: str) -> str | None:
    """Regex + fuzzy fallback Thai number parser.

    Handles common OCR distortions in the Thai pronunciation text (e.g.
    extra characters, wrong tone marks, split words) by using
    ``rapidfuzz.fuzz.partial_ratio`` for substring matching.

    Supports numbers representable with the standard Thai magnitude system
    (up to ล้าน = 1,000,000).  Numbers above that are outside the realistic
    vote-count range and will not be computed correctly — the caller should
    treat the return value as an approximation.

    Parameters
    ----------
    thai_text:
        Thai number word string (possibly OCR-distorted).

    Returns
    -------
    Arabic digit string, or ``None`` when no recognised word is found.
    """
    total = 0
    remaining = thai_text

    for mag_word, mag_val in _THAI_MAGNITUDES:
        if not _fuzzy_contains(mag_word, remaining):
            continue

        # Split at the matched magnitude word — use exact position when possible.
        if mag_word in remaining:
            idx = remaining.index(mag_word)
        else:
            # Approximate split at the midpoint of the remaining string when the
            # fuzzy match cannot give us a precise position.
            idx = max(0, len(remaining) // 2)

        prefix = remaining[:idx]
        remaining = remaining[idx + len(mag_word):]

        # Determine the coefficient (digit before the magnitude word).
        coeff = 1  # "หมื่น" alone means one หมื่น = 10,000
        for digit_word, digit_val in _THAI_DIGITS_WORD.items():
            if _fuzzy_contains(digit_word, prefix):
                coeff = digit_val
                break

        total += coeff * mag_val

    # Add any remaining unit word
    for digit_word, digit_val in _THAI_DIGITS_WORD.items():
        if _fuzzy_contains(digit_word, remaining):
            total += digit_val
            break

    if total > 0:
        return str(total)

    # Last resort: extract raw digits from the Thai text string (OCR may have
    # mixed Arabic numerals into the pronunciation text).
    fallback_digits = "".join(c for c in thai_text if c.isdigit())
    if fallback_digits:
        logger.debug(
            "_partial_thai_number: fell back to raw digit extraction → %r from %r",
            fallback_digits,
            thai_text,
        )
        return fallback_digits

    return None
