"""Phase 8 — Normalization & Hard Rule Overrides.

Turn raw OCR vote strings into clean Arabic digit strings, then apply hard
rules to reject physically impossible values and soft rules to assign
per-value confidence penalties.

Normalization
-------------
1. Handle empty / placeholder strings → ``"0"``.
2. Translate Thai digits (๐–๙) to Arabic (0–9).
3. Apply unambiguous OCR character substitutions (``O``→``0``, ``l``→``1``,
   ``I``→``1``) **only when** the cleaned string already contains real digits,
   avoiding false conversions on OCR garbage.
4. Strip all non-digit characters.
5. Return ``"0"`` when nothing remains.

Hard rules
----------
- Non-digit string → ``fallback`` (default ``"0"``).
- Value > 1 000 000 → ``fallback`` (impossible in any Thai constituency).

Soft confidence modifiers
--------------------------
- Value > 1 000 000 → 0.1  (essentially impossible)
- Value < ``SOFT_LOW_VOTE`` (20) → 0.5  (suspicious but plausible in fringe races)
- Otherwise → 1.0

Public API
----------
normalize_votes(raw)
    Normalize a raw OCR string to a clean digit string.
apply_soft_rules(vote)
    Return a confidence multiplier for a normalized vote string.
apply_hard_rules(vote, fallback)
    Override truly impossible values; leave small-but-legal values unchanged.
normalize_and_validate(raw, fallback)
    Convenience: normalize then hard-rule in one call.
"""

from __future__ import annotations

import logging

from src.config import MAX_VOTE

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Thai digits → Arabic digits translation table.
_THAI_DIGIT_MAP = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# Unambiguous single-character OCR substitutions.
# Only applied when the string already contains a real digit to avoid
# converting pure OCR garbage into spurious numbers.
_OCR_FIXES: dict[str, str] = {
    "O": "0",
    "o": "0",
    "l": "1",
    "I": "1",
}

# Votes below this threshold get a soft confidence penalty.
# Set deliberately low (20) so that legitimate fringe candidates are not
# hard-overridden — only a penalty is applied.
SOFT_LOW_VOTE = 20


# ── Public API ────────────────────────────────────────────────────────────────


def normalize_votes(raw: str) -> str:
    """Normalize a raw OCR vote string to a clean Arabic digit string.

    Steps applied in order:

    1. Treat empty / placeholder values as ``"0"``.
    2. Translate Thai digits (``๐``–``๙``) to Arabic (``0``–``9``).
    3. Apply OCR character fixes (``O``→``0``, ``l``/``I``→``1``) only when
       there is at least one real digit already present.
    4. Remove all non-digit characters.
    5. Return ``"0"`` if nothing remains.

    Parameters
    ----------
    raw:
        The raw string from an OCR vote cell (may contain Thai digits, commas,
        spaces, misc characters).

    Returns
    -------
    A non-empty string of Arabic digit characters (``"0"`` at minimum).

    Examples
    --------
    >>> normalize_votes("34,405")
    '34405'
    >>> normalize_votes("๓๔,๔๐๕")
    '34405'
    >>> normalize_votes("3O,4O5")
    '30405'
    >>> normalize_votes("-")
    '0'
    >>> normalize_votes("")
    '0'
    """
    if not raw or str(raw).strip() in ("", "-", "—", "–"):
        return "0"

    # Step 1: translate Thai digits
    cleaned = str(raw).translate(_THAI_DIGIT_MAP)

    # Step 2: apply OCR fixes only when real digits are present
    if any(c.isdigit() for c in cleaned):
        chars: list[str] = []
        for ch in cleaned:
            if ch.isdigit():
                chars.append(ch)
            elif ch in _OCR_FIXES:
                chars.append(_OCR_FIXES[ch])
        result = "".join(chars)
    else:
        # No real digits — strip everything non-digit (likely produces "")
        result = "".join(c for c in cleaned if c.isdigit())

    return result if result else "0"


def apply_soft_rules(vote: str) -> float:
    """Return a confidence multiplier for a normalized vote string.

    The multiplier is used downstream (Phase 10) as a per-row penalty; it does
    **not** modify the vote value itself.

    +-------------------+------------+---------------------------------------+
    | Condition         | Multiplier | Rationale                             |
    +===================+============+=======================================+
    | Non-digit string  | 0.0        | Completely invalid                    |
    +-------------------+------------+---------------------------------------+
    | value > MAX_VOTE  | 0.1        | Impossible in any Thai constituency   |
    +-------------------+------------+---------------------------------------+
    | value < SOFT_LOW  | 0.5        | Suspicious — fringe candidate possible|
    +-------------------+------------+---------------------------------------+
    | Otherwise         | 1.0        | Plausible                             |
    +-------------------+------------+---------------------------------------+

    Parameters
    ----------
    vote:
        A normalized (digit-only) vote string as returned by
        :func:`normalize_votes`.

    Returns
    -------
    A float confidence multiplier in ``[0.0, 1.0]``.
    """
    if not vote or not vote.isdigit():
        return 0.0

    v = int(vote)
    if v > MAX_VOTE:
        return 0.1
    if v < SOFT_LOW_VOTE:
        return 0.5
    return 1.0


def apply_hard_rules(vote: str, fallback: str = "0") -> str:
    """Override truly impossible vote values; leave everything else unchanged.

    Only two categories are hard-overridden:

    - Non-digit strings (OCR garbage) → *fallback*.
    - Values exceeding ``MAX_VOTE`` (1 000 000) → *fallback*.

    Small-but-plausible values (< ``SOFT_LOW_VOTE``) are **not** overridden
    here — they receive a confidence penalty in :func:`apply_soft_rules`
    instead.  This preserves correct data for fringe candidates who genuinely
    receive very few votes.

    Parameters
    ----------
    vote:
        A normalized (digit-only) vote string.
    fallback:
        The replacement string when a hard rule triggers (default: ``"0"``).

    Returns
    -------
    The original *vote* string, or *fallback* if a hard rule fired.
    """
    if not vote or not vote.isdigit():
        logger.debug("Hard rule: non-digit vote %r → %r", vote, fallback)
        return fallback

    if int(vote) > MAX_VOTE:
        logger.warning(
            "Hard rule: vote %s > MAX_VOTE (%s) — impossible, replacing with %r",
            vote,
            MAX_VOTE,
            fallback,
        )
        return fallback

    return vote


def normalize_and_validate(raw: str, fallback: str = "0") -> str:
    """Normalize a raw OCR string and apply hard rule overrides in one step.

    Equivalent to ``apply_hard_rules(normalize_votes(raw), fallback)``.

    Parameters
    ----------
    raw:
        Raw vote-cell string from OCR output.
    fallback:
        Replacement used when the normalized value violates a hard rule.

    Returns
    -------
    A clean, validated Arabic digit string.
    """
    normalized = normalize_votes(raw)
    return apply_hard_rules(normalized, fallback)
