"""Phase 12 — Row Anchor Alignment.

If OCR misses a candidate row (e.g. row 3), all subsequent rows shift by one
position, causing cascading misalignment in the final submission.  This phase
detects and corrects such shifts using **candidate number anchors** — the
integer in the first cell of each table row.

The strategy:
- Parse each row's candidate number (from the first cell) alongside its vote.
- Any row whose candidate number is detected is *anchored* to that position.
- Rows with no detected candidate number are filled sequentially into the
  remaining empty slots.

Public API
----------
extract_anchored_rows(markdown)
    Parse ``(candidate_num | None, raw_cell, digit_string)`` from markdown.
anchor_align(anchored_rows, expected_count)
    Re-align a list of anchored rows into a fixed-length vote list.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from src.phase6_parse import SKIP_PATTERNS, extract_vote_cell
from src.phase7_thai_crosscheck import cross_check_vote
from src.phase8_normalize import apply_hard_rules, normalize_votes

logger = logging.getLogger(__name__)

# ── Type alias ────────────────────────────────────────────────────────────────

# (candidate_number | None, raw_vote_cell_text, digit_string)
AnchoredRow = tuple[Optional[int], str, str]


# ── Public API ────────────────────────────────────────────────────────────────


def extract_anchored_rows(markdown: str) -> list[AnchoredRow]:
    """Parse candidate-number-anchored rows from markdown table output.

    Scans the text line-by-line looking for pipe-delimited table rows (the
    same format produced by Typhoon OCR when it falls back from HTML).
    Consecutive lines sharing a ``|`` character are accumulated into a single
    logical row buffer to handle line-wrapped cells.

    For each qualifying row the function:

    1. Splits on ``|`` to extract cells.
    2. Skips separator lines (``---``, ``:::``).
    3. Skips rows matching ``SKIP_PATTERNS`` (headers, totals).
    4. Tries to extract a **candidate number** (1–2 ASCII/Thai digits) from
       the first cell.
    5. Extracts the vote count from the last qualifying cell.

    Parameters
    ----------
    markdown:
        Raw text string returned by Typhoon OCR (markdown format).

    Returns
    -------
    List of ``(candidate_number | None, raw_vote_cell, digit_string)`` tuples.
    ``candidate_number`` is ``None`` when the first cell does not look like a
    row index.

    Examples
    --------
    >>> rows = extract_anchored_rows("| 1 | สมชาย | 12345 |")
    >>> rows[0][0]  # candidate_number
    1
    """
    # Thai digits → ASCII for candidate-number extraction
    _thai = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

    rows: list[AnchoredRow] = []
    buffer = ""

    for line in markdown.splitlines():
        if "|" not in line:
            buffer = ""
            continue

        buffer += " " + line.strip()

        # Skip pure separator lines  (e.g. "|---|---|")
        if re.match(r"^[\s|:\-]+$", buffer.strip()):
            buffer = ""
            continue

        # Skip header / total rows
        if any(pat in buffer for pat in SKIP_PATTERNS):
            buffer = ""
            continue

        cells = [c.strip() for c in buffer.split("|") if c.strip()]
        if not cells:
            buffer = ""
            continue

        # Try to extract candidate number from first cell (≤ 2 digits)
        cand_num: Optional[int] = None
        first_normalised = cells[0].translate(_thai)
        first_digits = "".join(c for c in first_normalised if c.isdigit())
        if first_digits and len(first_digits) <= 2:
            cand_num = int(first_digits)

        raw, digits = extract_vote_cell(cells)
        if raw is None or digits is None:
            buffer = ""
            continue

        rows.append((cand_num, raw, digits))
        buffer = ""

    logger.debug("extract_anchored_rows: parsed %d rows from markdown.", len(rows))
    return rows


def anchor_align(
    anchored_rows: list[AnchoredRow],
    expected_count: int,
) -> list[str]:
    """Re-align anchored rows into a fixed-length vote list.

    Uses detected candidate numbers as *anchors* to place votes at their
    correct positions.  Rows without a detected candidate number are filled
    sequentially into the remaining empty (``"0"``) slots.

    Algorithm
    ---------
    1. Initialise *result* as ``["0"] * expected_count``.
    2. For every row that has a valid ``cand_num`` (1 ≤ num ≤ expected_count):
       a. Compute the vote via Phase 7 cross-check + Phase 8 hard rules.
       b. Place it at ``result[cand_num - 1]``.
       c. If the slot is already occupied (duplicate anchor), the **later**
          row wins — this is a conservative choice that favours the last
          seen value when numbering is ambiguous.
    3. Collect votes from unanchored rows in order.
    4. Fill remaining ``"0"`` slots sequentially with unanchored votes.

    Parameters
    ----------
    anchored_rows:
        Output of :func:`extract_anchored_rows`.
    expected_count:
        Number of candidates expected in the document.

    Returns
    -------
    A list of exactly *expected_count* vote strings.  Positions that could
    not be filled remain as ``"0"``.

    Examples
    --------
    >>> rows = [(1, "12345", "12345"), (None, "67890", "67890")]
    >>> anchor_align(rows, 3)
    ['12345', '67890', '0']
    >>> # Row-shift scenario: OCR missed row 2, rows 3–5 shifted up by one
    >>> rows = [(1, "11111", "11111"), (3, "33333", "33333"), (4, "44444", "44444")]
    >>> anchor_align(rows, 5)
    ['11111', '0', '33333', '44444', '0']
    """
    if expected_count <= 0:
        return []

    result: list[str] = ["0"] * expected_count

    # Pass 1 — place anchored rows at their known positions (first-wins).
    # When multiple pages contribute rows for the same candidate, keep the
    # first non-zero value encountered.  Later pages (e.g. summary tables)
    # are more prone to OCR errors and should not silently overwrite earlier
    # accurate reads.
    for cand_num, raw, digits in anchored_rows:
        vote = cross_check_vote(raw, normalize_votes(digits))
        vote = apply_hard_rules(vote)

        if cand_num is not None and 1 <= cand_num <= expected_count:
            if result[cand_num - 1] == "0":  # first-wins: skip already-filled slot
                result[cand_num - 1] = vote

    # Pass 2 — fill remaining slots with unanchored rows (in order)
    no_anchor_votes = [
        apply_hard_rules(cross_check_vote(raw, normalize_votes(digits)))
        for cand_num, raw, digits in anchored_rows
        if cand_num is None
    ]

    slot = 0
    for i in range(expected_count):
        if result[i] == "0" and slot < len(no_anchor_votes):
            result[i] = no_anchor_votes[slot]
            slot += 1

    logger.debug(
        "anchor_align: expected=%d, anchored=%d, no_anchor=%d, zeros_remaining=%d",
        expected_count,
        sum(1 for cn, _, __ in anchored_rows if cn is not None),
        len(no_anchor_votes),
        result.count("0"),
    )
    return result
