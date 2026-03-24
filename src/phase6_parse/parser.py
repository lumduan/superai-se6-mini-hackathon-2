"""Phase 6 — HTML Table Parsing with BeautifulSoup.

Primary parser for Typhoon OCR output.  Typhoon returns an HTML table; this
module uses BeautifulSoup to extract structured vote data from that HTML.
A markdown line-by-line parser is provided as a fallback when the OCR output
contains no ``<table>`` element.

Public API
----------
parse_html_table(html)
    Main entry point — parses HTML first, falls back to markdown.
parse_votes_from_markdown(markdown)
    Markdown fallback parser.
extract_vote_cell(cells)
    Return ``(raw_cell, digit_string)`` for the last cell that looks like a
    vote count (3–7 digits after Thai → Arabic normalisation).
has_consistent_column(digit_strings)
    Column consistency guard — high variance in digit lengths within a
    document suggests a wrong column was captured.
"""

from __future__ import annotations

import logging
import re
import statistics
from typing import List, Optional, Tuple

from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Thai keywords that indicate a header or total row — skip these rows.
SKIP_PATTERNS: list[str] = [
    "รวมคะแนน",
    "รวมทั้งสิ้น",
    "หมายเลข",
    "ชื่อ-สกุล",
    "พรรคการเมือง",
    "คะแนน",
]

# Translation table: Thai digits → ASCII digits
THAI_DIGIT_MAP: dict[int, int] = str.maketrans("๐๑๒๓๔๕๖๗๘๙", "0123456789")

# A candidate number occupies the first cell and is at most 2 digits long
# (constituencies have at most 99 registered candidates in practice).
MAX_CANDIDATE_NUM_DIGITS = 2

# Valid digit-string length range for a vote count cell.
# Minimum is 2 so that double-digit votes (e.g. candidate with 60 votes)
# are captured, while single-digit candidate numbers (1–9) in the first
# cell are never mis-selected as votes when the vote column is empty.
VOTE_DIGITS_MIN = 2
VOTE_DIGITS_MAX = 7

# Standard deviation threshold for the column consistency check.
# A stdev above this value indicates the extracted column is inconsistent
# across rows — likely the wrong column or severe misalignment.
COLUMN_STDEV_THRESHOLD = 2.0

# ── Type alias ────────────────────────────────────────────────────────────────

# (candidate_number | None, raw_vote_cell_text, digit_string)
ParsedRow = Tuple[Optional[int], str, str]


# ── Helpers ───────────────────────────────────────────────────────────────────


def _normalise(text: str) -> str:
    """Translate Thai digits to ASCII digits in *text*."""
    return text.translate(THAI_DIGIT_MAP)


def _candidate_number(cell_text: str) -> Optional[int]:
    """Extract a candidate number (1–2 digits) from the first cell text.

    Returns ``None`` when the cell does not look like a candidate row index.
    """
    digits = "".join(c for c in _normalise(cell_text) if c.isdigit())
    if digits and len(digits) <= MAX_CANDIDATE_NUM_DIGITS:
        return int(digits)
    return None


def _is_skip_row(cells: list[str]) -> bool:
    """Return ``True`` if the row should be skipped (header / total row)."""
    joined = " ".join(cells)
    return any(kw in joined for kw in SKIP_PATTERNS)


# ── Public API ────────────────────────────────────────────────────────────────


def extract_vote_cell(cells: list[str]) -> tuple[Optional[str], Optional[str]]:
    """Return ``(raw_cell, digit_string)`` for the last cell that looks like a
    vote count.

    A "vote count cell" is one that, after Thai → Arabic digit normalisation,
    yields a purely-digit string whose length is between
    ``VOTE_DIGITS_MIN`` and ``VOTE_DIGITS_MAX`` inclusive.

    Iterates cells **right-to-left** so that the rightmost qualifying cell is
    preferred — matching the physical layout of Form สส.6/1 where the vote
    count is always in the last column.

    Parameters
    ----------
    cells:
        List of cell text strings extracted from a single ``<tr>``.

    Returns
    -------
    ``(raw_cell, digit_string)`` if found, else ``(None, None)``.
    """
    for cell in reversed(cells):
        normalised = _normalise(cell)
        digits = "".join(c for c in normalised if c.isdigit())
        if VOTE_DIGITS_MIN <= len(digits) <= VOTE_DIGITS_MAX:
            return cell, digits
    return None, None


def has_consistent_column(digit_strings: list[str]) -> bool:
    """Check that the extracted vote-digit strings form a consistent column.

    High variance in digit lengths within a single document strongly suggests
    that the wrong column was captured (e.g. a candidate-number or page-number
    column) or that there is severe row misalignment.

    The check is skipped when fewer than 3 non-zero values are available —
    in that case ``True`` (consistent) is returned conservatively.

    Parameters
    ----------
    digit_strings:
        List of digit strings, one per extracted candidate row.

    Returns
    -------
    ``True`` if the column looks consistent, ``False`` otherwise.
    """
    lengths = [len(d) for d in digit_strings if d and d != "0"]
    if len(lengths) < 3:
        return True
    stdev = statistics.stdev(lengths)
    if stdev >= COLUMN_STDEV_THRESHOLD:
        logger.warning(
            "Column consistency check failed: digit-length stdev=%.2f (threshold=%.1f).",
            stdev,
            COLUMN_STDEV_THRESHOLD,
        )
        return False
    return True


def parse_votes_from_markdown(markdown: str) -> list[ParsedRow]:
    """Markdown fallback parser — used when Typhoon output contains no HTML.

    Scans the text line-by-line looking for pipe-delimited table rows.
    Consecutive lines sharing a ``|`` character are accumulated into a single
    logical row buffer to handle line-wrapped cells.  Separator lines
    (``---``, ``:::``) and header/total rows (matching ``SKIP_PATTERNS``) are
    discarded.

    Parameters
    ----------
    markdown:
        Raw text string returned by Typhoon OCR with no ``<table>`` element.

    Returns
    -------
    List of ``(candidate_number | None, raw_vote_cell, digit_string)`` tuples.
    """
    result: list[ParsedRow] = []
    buffer = ""

    for line in markdown.splitlines():
        if "|" not in line:
            buffer = ""
            continue

        buffer += " " + line.strip()

        # Skip pure separator lines (e.g. "|---|---|")
        if re.match(r"^[\s|:\-]+$", buffer.strip()):
            buffer = ""
            continue

        if _is_skip_row([buffer]):
            buffer = ""
            continue

        cells = [c.strip() for c in buffer.split("|") if c.strip()]
        if not cells:
            continue

        raw, digits = extract_vote_cell(cells)
        if raw is None or digits is None:
            continue

        result.append((None, raw, digits))
        buffer = ""

    logger.debug("Markdown fallback parsed %d rows.", len(result))
    return result


def parse_html_table(html: str) -> list[ParsedRow]:
    """Parse HTML table output from Typhoon OCR.

    Uses BeautifulSoup to extract structured vote data from the ``<table>``
    element(s) in *html*.  Falls back to :func:`parse_votes_from_markdown`
    when no ``<tr>`` elements are found — this handles the edge case where
    Typhoon falls back to plain markdown output.

    For each non-header, non-total row the function:

    1. Collects all ``<td>`` / ``<th>`` cell texts (``colspan`` is handled
       transparently by BeautifulSoup).
    2. Skips rows matching ``SKIP_PATTERNS`` (headers, total rows).
    3. Extracts a candidate number from the **first** cell (1–2 digits).
    4. Finds the **last** cell that looks like a vote count using
       :func:`extract_vote_cell`.

    Parameters
    ----------
    html:
        Raw HTML string returned by Typhoon OCR — may also be plain markdown.

    Returns
    -------
    List of ``(candidate_number | None, raw_vote_cell, digit_string)`` tuples,
    one entry per candidate row found in the table.  Empty list when no
    parseable rows are present.
    """
    soup = BeautifulSoup(html, "html.parser")
    rows = soup.find_all("tr")

    if not rows:
        logger.debug("No <tr> elements found — falling back to markdown parser.")
        return parse_votes_from_markdown(html)

    results: list[ParsedRow] = []

    for row in rows:
        cells = [td.get_text(strip=True) for td in row.find_all(["td", "th"])]
        if not cells:
            continue

        if _is_skip_row(cells):
            logger.debug("Skipping header/total row: %s", cells[:3])
            continue

        cand_num = _candidate_number(cells[0])

        raw, digits = extract_vote_cell(cells)
        if raw is None or digits is None:
            logger.debug("No vote cell found in row: %s", cells)
            continue

        results.append((cand_num, raw, digits))

    logger.debug("HTML table parsed %d candidate rows.", len(results))
    return results
