"""Tests for Phase 6 — HTML Table Parsing with BeautifulSoup.

Covers:
- parse_html_table: valid HTML table, header/total row skipping, colspan,
  Thai digits, multi-table, empty HTML, no-<tr> fallback to markdown.
- parse_votes_from_markdown: pipe-delimited rows, separator lines, skip rows.
- extract_vote_cell: rightmost-cell preference, Thai digit normalisation,
  out-of-range lengths.
- has_consistent_column: consistent / inconsistent digit-length distributions.
"""

from __future__ import annotations

import pytest

from src.phase6_parse.parser import (
    VOTE_DIGITS_MAX,
    VOTE_DIGITS_MIN,
    extract_vote_cell,
    has_consistent_column,
    parse_html_table,
    parse_votes_from_markdown,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_html_table(rows: list[list[str]]) -> str:
    """Build a minimal HTML table string from a list of row-cell lists."""
    trs = ""
    for cells in rows:
        tds = "".join(f"<td>{c}</td>" for c in cells)
        trs += f"<tr>{tds}</tr>\n"
    return f"<table>\n{trs}</table>"


# ── extract_vote_cell ─────────────────────────────────────────────────────────


class TestExtractVoteCell:
    def test_returns_last_qualifying_cell(self):
        cells = ["1", "นาย ก ข ค", "พรรค A", "12,345"]
        raw, digits = extract_vote_cell(cells)
        assert digits == "12345"
        assert raw == "12,345"

    def test_prefers_rightmost_cell(self):
        # Two cells with valid digit counts — rightmost wins.
        cells = ["5", "some name", "99999", "54321"]
        raw, digits = extract_vote_cell(cells)
        assert digits == "54321"

    def test_thai_digits_normalised(self):
        cells = ["3", "name", "พรรค", "๑๒,๓๔๕"]
        raw, digits = extract_vote_cell(cells)
        assert digits == "12345"

    def test_no_digit_cells_return_none(self):
        # No cell with >= VOTE_DIGITS_MIN digits → must return (None, None).
        # "1" is single-digit (< 2) and "--" has zero digits, so nothing qualifies.
        cells = ["1", "name", "party", "--"]
        raw, digits = extract_vote_cell(cells)
        assert raw is None
        assert digits is None

    def test_two_digit_vote_captured(self):
        # Two-digit vote (e.g. candidate 18 with 60 votes) — VOTE_DIGITS_MIN=2 captures it.
        cells = ["18", "name", "party", "60"]
        raw, digits = extract_vote_cell(cells)
        assert digits == "60"
        assert raw == "60"

    def test_too_long_digits_skipped(self):
        # 8 digits — above VOTE_DIGITS_MAX=7 — should not be selected.
        cells = ["2", "name", "12345678"]
        raw, digits = extract_vote_cell(cells)
        assert raw is None
        assert digits is None

    def test_empty_cells(self):
        raw, digits = extract_vote_cell([])
        assert raw is None
        assert digits is None

    def test_boundary_min(self):
        # Exactly VOTE_DIGITS_MIN=2 digits — must be accepted.
        cells = ["1", "xxx", "42"]
        raw, digits = extract_vote_cell(cells)
        assert digits == "42"
        assert len(digits) == VOTE_DIGITS_MIN

    def test_boundary_max(self):
        cells = ["1", "xxx", "1234567"]
        raw, digits = extract_vote_cell(cells)
        assert len(digits) == VOTE_DIGITS_MAX

    def test_mixed_text_and_digits(self):
        # Raw cell "12,345 (สองหมื่น...)" — digits extracted are "12345"
        cells = ["7", "full name", "12,345 (สองหมื่น...)"]
        raw, digits = extract_vote_cell(cells)
        assert digits == "12345"


# ── has_consistent_column ─────────────────────────────────────────────────────


class TestHasConsistentColumn:
    def test_consistent_same_length(self):
        strings = ["12345", "23456", "11111", "98765", "54321"]
        assert has_consistent_column(strings) is True

    def test_consistent_small_variance(self):
        # stdev of [5,5,5,4,5] = 0.44 — within threshold
        strings = ["12345", "23456", "11111", "9876", "54321"]
        assert has_consistent_column(strings) is True

    def test_inconsistent_high_variance(self):
        # Mix of 3-digit and 7-digit strings → stdev ≈ 2.19, above threshold 2.0
        strings = ["123", "1234567", "321", "9876543", "987", "5432198"]
        assert has_consistent_column(strings) is False

    def test_fewer_than_three_always_consistent(self):
        assert has_consistent_column([]) is True
        assert has_consistent_column(["123"]) is True
        assert has_consistent_column(["123", "456"]) is True

    def test_zeros_excluded_from_check(self):
        # "0" entries are excluded; remaining 2 values → skip check → True
        strings = ["0", "0", "12345"]
        assert has_consistent_column(strings) is True

    def test_all_zeros(self):
        # All zeros excluded → fewer than 3 → consistent
        assert has_consistent_column(["0", "0", "0", "0"]) is True


# ── parse_html_table ──────────────────────────────────────────────────────────


class TestParseHtmlTable:
    def test_basic_table(self):
        html = _make_html_table([
            ["1", "นาย กอ ขอ คอ", "พรรค ก", "10,000"],
            ["2", "นาย ขอ คอ งอ", "พรรค ข", "20,000"],
        ])
        results = parse_html_table(html)
        assert len(results) == 2
        cand, raw, digits = results[0]
        assert cand == 1
        assert digits == "10000"
        cand2, raw2, digits2 = results[1]
        assert cand2 == 2
        assert digits2 == "20000"

    def test_skips_header_row(self):
        html = _make_html_table([
            ["หมายเลข", "ชื่อ-สกุล", "พรรคการเมือง", "คะแนน"],
            ["1", "นาย ก", "พรรค A", "15,000"],
        ])
        results = parse_html_table(html)
        assert len(results) == 1
        assert results[0][2] == "15000"

    def test_skips_total_row(self):
        html = _make_html_table([
            ["1", "นาย ก", "พรรค A", "15,000"],
            ["รวมคะแนน", "", "", "15,000"],
        ])
        results = parse_html_table(html)
        assert len(results) == 1

    def test_skips_ruam_thang_sin_row(self):
        html = _make_html_table([
            ["1", "ผู้สมัคร", "พรรค", "8,500"],
            ["รวมทั้งสิ้น", "", "", "8,500"],
        ])
        results = parse_html_table(html)
        assert len(results) == 1

    def test_colspan_handled(self):
        # BeautifulSoup reads only the text; colspan doesn't change get_text.
        html = (
            "<table>"
            "<tr><td>1</td><td>some name</td><td colspan='2'>party</td><td>5,678</td></tr>"
            "</table>"
        )
        results = parse_html_table(html)
        assert len(results) == 1
        assert results[0][2] == "5678"

    def test_thai_digits_in_vote_cell(self):
        html = _make_html_table([
            ["๑", "ชื่อ", "พรรค", "๑๒,๓๔๕"],
        ])
        results = parse_html_table(html)
        assert len(results) == 1
        assert results[0][2] == "12345"

    def test_no_tr_falls_back_to_markdown(self):
        # Plain text with no <table> — should invoke markdown fallback.
        markdown = "| 1 | name | party | 11,111 |\n"
        results = parse_html_table(markdown)
        assert len(results) == 1
        assert results[0][2] == "11111"

    def test_empty_html(self):
        results = parse_html_table("")
        assert results == []

    def test_row_without_valid_vote_cell_skipped(self):
        html = _make_html_table([
            ["1", "name", "party", "N/A"],          # no digits
            ["2", "name", "party", "30,000"],
        ])
        results = parse_html_table(html)
        assert len(results) == 1
        assert results[0][2] == "30000"

    def test_candidate_number_none_for_large_first_cell(self):
        # First cell has 3+ digits → not a valid candidate number → None
        html = _make_html_table([
            ["100", "name", "party", "20,000"],
        ])
        results = parse_html_table(html)
        assert len(results) == 1
        cand, _, _ = results[0]
        assert cand is None

    def test_multiple_tables_parsed(self):
        # Two separate <table> blocks — both should be parsed.
        html = (
            _make_html_table([["1", "a", "p", "1,000"]])
            + _make_html_table([["2", "b", "q", "2,000"]])
        )
        results = parse_html_table(html)
        assert len(results) == 2

    def test_th_elements_treated_as_cells(self):
        html = (
            "<table>"
            "<tr><th>หมายเลข</th><th>ชื่อ</th><th>คะแนน</th></tr>"
            "<tr><td>1</td><td>ก ข ค</td><td>7,777</td></tr>"
            "</table>"
        )
        results = parse_html_table(html)
        # Header row is skipped (contains "หมายเลข" + "คะแนน")
        assert len(results) == 1
        assert results[0][2] == "7777"


# ── parse_votes_from_markdown ─────────────────────────────────────────────────


class TestParseVotesFromMarkdown:
    def test_basic_pipe_row(self):
        markdown = "| 1 | นาย กอ | พรรค A | 10,500 |"
        results = parse_votes_from_markdown(markdown)
        assert len(results) == 1
        assert results[0][2] == "10500"

    def test_separator_line_skipped(self):
        markdown = (
            "| 1 | name | party | 5,000 |\n"
            "|---|---|---|---|\n"
            "| 2 | name2 | party2 | 6,000 |\n"
        )
        results = parse_votes_from_markdown(markdown)
        assert len(results) == 2

    def test_skip_patterns_ignored(self):
        markdown = (
            "| หมายเลข | ชื่อ-สกุล | พรรคการเมือง | คะแนน |\n"
            "| 1 | นาย ก | พรรค | 9,999 |\n"
        )
        results = parse_votes_from_markdown(markdown)
        assert len(results) == 1
        assert results[0][2] == "9999"

    def test_non_pipe_line_resets_buffer(self):
        markdown = (
            "Some header text\n"
            "| 1 | name | party | 12,345 |\n"
        )
        results = parse_votes_from_markdown(markdown)
        assert len(results) == 1

    def test_empty_string(self):
        results = parse_votes_from_markdown("")
        assert results == []

    def test_thai_digit_in_markdown(self):
        markdown = "| ๓ | ชื่อ | พรรค | ๑๑,๑๑๑ |"
        results = parse_votes_from_markdown(markdown)
        assert len(results) == 1
        assert results[0][2] == "11111"

    def test_no_valid_vote_cell_in_row(self):
        # Row where vote column is "--" and no other cell has >= 2 digits → skipped.
        # Candidate number "1" has only 1 digit (< VOTE_DIGITS_MIN=2), so nothing qualifies.
        markdown = "| 1 | name | party | -- |"
        results = parse_votes_from_markdown(markdown)
        assert results == []

    def test_candidate_number_is_none_in_markdown_fallback(self):
        # Markdown fallback always returns None for candidate_number
        markdown = "| 5 | name | party | 25,000 |"
        results = parse_votes_from_markdown(markdown)
        assert results[0][0] is None
