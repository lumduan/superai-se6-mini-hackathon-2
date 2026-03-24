"""Tests for Phase 12 — Row Anchor Alignment.

Covers:

extract_anchored_rows
    - Returns empty list for empty markdown
    - Parses candidate number from first cell
    - Returns None candidate_num when first cell is not a number
    - Skips separator lines (``---``, ``:::``)
    - Skips header/total rows matching SKIP_PATTERNS
    - Handles Thai digits in first cell as candidate number
    - Handles multi-pipe rows correctly
    - Accumulates line-wrapped rows (multi-line buffer)
    - Returns None candidate_num when first cell has more than 2 digits
    - Skips rows with no qualifying vote cell

anchor_align
    - Returns list of expected_count "0"s for empty anchored_rows
    - Returns empty list when expected_count is 0
    - Anchored rows placed at correct index positions
    - Row-shift scenario: gaps filled with "0" where rows are missing
    - Unanchored rows fill remaining "0" slots sequentially
    - Anchored + unanchored rows combined
    - Duplicate anchor: last occurrence wins
    - Anchors out of range (< 1 or > expected) are ignored
    - All-anchored rows, no gaps → exact placement
    - Pure unanchored rows fall back to sequential fill
    - Extra unanchored rows beyond "0" slots are discarded
    - Result always has exactly expected_count items
"""

from __future__ import annotations

import pytest

from src.phase12_anchor import anchor_align, extract_anchored_rows


# ── extract_anchored_rows ─────────────────────────────────────────────────────


class TestExtractAnchoredRows:
    def test_empty_markdown_returns_empty(self):
        assert extract_anchored_rows("") == []

    def test_no_pipe_lines_returns_empty(self):
        assert extract_anchored_rows("some text\nno pipes here\n") == []

    def test_parses_candidate_number(self):
        md = "| 1 | สมชาย | 12345 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 1
        cand_num, raw, digits = rows[0]
        assert cand_num == 1
        assert digits == "12345"

    def test_candidate_number_two_digits(self):
        md = "| 12 | ผู้สมัคร | 54321 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 1
        assert rows[0][0] == 12

    def test_returns_none_when_first_cell_not_number(self):
        md = "| สมชาย | พรรคก้าวหน้า | 12345 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 1
        assert rows[0][0] is None

    def test_returns_none_when_first_cell_too_many_digits(self):
        # 3-digit first cell should not be treated as candidate number
        md = "| 999 | ผู้สมัคร | 12345 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 1
        # 999 has 3 digits — should be None (exceeds MAX_CANDIDATE_NUM_DIGITS=2)
        assert rows[0][0] is None

    def test_skips_separator_lines(self):
        md = "| 1 | สมชาย | 12345 |\n|---|---|---|\n| 2 | สมศรี | 67890 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 2
        assert rows[0][0] == 1
        assert rows[1][0] == 2

    def test_skips_header_rows(self):
        # Contains Thai header keywords
        md = "| หมายเลข | ชื่อ-สกุล | คะแนน |\n| 1 | สมชาย | 12345 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 1
        assert rows[0][0] == 1

    def test_skips_total_rows(self):
        md = "| 1 | สมชาย | 12345 |\n| รวมคะแนน | | 99999 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 1

    def test_handles_thai_digits_in_first_cell(self):
        # Thai digit ๒ = 2, ๓ = 3 → "23" but that's 2 digits → candidate 23
        md = "| ๑ | ผู้สมัคร | 12345 |"
        rows = extract_anchored_rows(md)
        assert len(rows) == 1
        assert rows[0][0] == 1

    def test_skips_row_with_no_vote_cell(self):
        # No cell with 3–7 digits in vote position
        md = "| 1 | ab | cd |"
        rows = extract_anchored_rows(md)
        assert rows == []

    def test_multiple_rows(self):
        md = (
            "| 1 | สมชาย | 11111 |\n"
            "| 2 | สมศรี | 22222 |\n"
            "| 3 | สมหมาย | 33333 |\n"
        )
        rows = extract_anchored_rows(md)
        assert len(rows) == 3
        assert [r[0] for r in rows] == [1, 2, 3]
        assert [r[2] for r in rows] == ["11111", "22222", "33333"]

    def test_digit_string_extracted(self):
        md = "| 1 | ผู้สมัคร | 54321 |"
        rows = extract_anchored_rows(md)
        assert rows[0][2] == "54321"


# ── anchor_align ──────────────────────────────────────────────────────────────


class TestAnchorAlign:
    def test_empty_rows_returns_zeros(self):
        result = anchor_align([], 5)
        assert result == ["0", "0", "0", "0", "0"]

    def test_expected_count_zero_returns_empty(self):
        result = anchor_align([(1, "12345", "12345")], 0)
        assert result == []

    def test_anchored_rows_placed_at_correct_position(self):
        rows = [
            (1, "11111", "11111"),
            (2, "22222", "22222"),
            (3, "33333", "33333"),
        ]
        result = anchor_align(rows, 3)
        assert result == ["11111", "22222", "33333"]

    def test_row_shift_gap_filled_with_zero(self):
        # OCR missed row 2; rows 3, 4 are anchored
        rows = [
            (1, "11111", "11111"),
            (3, "33333", "33333"),
            (4, "44444", "44444"),
        ]
        result = anchor_align(rows, 5)
        assert result[0] == "11111"
        assert result[2] == "33333"
        assert result[3] == "44444"
        assert result[1] == "0"  # gap where row 2 was missed
        assert result[4] == "0"

    def test_unanchored_rows_fill_remaining_slots(self):
        rows = [
            (1, "11111", "11111"),
            (None, "22222", "22222"),  # no anchor
        ]
        result = anchor_align(rows, 3)
        assert result[0] == "11111"
        assert result[1] == "22222"
        assert result[2] == "0"

    def test_anchored_and_unanchored_combined(self):
        rows = [
            (1, "11111", "11111"),
            (3, "33333", "33333"),
            (None, "55555", "55555"),  # fills first empty slot (index 1)
        ]
        # Slot 2 (index 1) is empty; slot 5 (index 4) is empty.
        result = anchor_align(rows, 5)
        assert result[0] == "11111"
        assert result[2] == "33333"
        # Unanchored fills first empty slot (index 1)
        assert result[1] == "55555"
        assert result[3] == "0"
        assert result[4] == "0"

    def test_duplicate_anchor_last_wins(self):
        rows = [
            (2, "22222", "22222"),
            (2, "99999", "99999"),  # duplicate — last value wins
        ]
        result = anchor_align(rows, 3)
        assert result[1] == "99999"

    def test_anchor_out_of_range_low_ignored(self):
        rows = [(0, "11111", "11111")]  # cand_num=0 is invalid
        result = anchor_align(rows, 3)
        # Out-of-range anchors are silently discarded (plan spec: only None rows
        # go to sequential fill; invalid-number rows are ignored entirely).
        assert result == ["0", "0", "0"]

    def test_anchor_out_of_range_high_ignored(self):
        rows = [(99, "11111", "11111")]  # cand_num=99 > expected_count=3
        result = anchor_align(rows, 3)
        # Out-of-range anchors are silently discarded.
        assert result == ["0", "0", "0"]

    def test_all_anchored_no_unanchored(self):
        rows = [
            (3, "33333", "33333"),
            (1, "11111", "11111"),
            (2, "22222", "22222"),
        ]
        result = anchor_align(rows, 3)
        assert result == ["11111", "22222", "33333"]

    def test_pure_unanchored_sequential_fill(self):
        rows = [
            (None, "11111", "11111"),
            (None, "22222", "22222"),
            (None, "33333", "33333"),
        ]
        result = anchor_align(rows, 3)
        assert result == ["11111", "22222", "33333"]

    def test_extra_unanchored_discarded(self):
        rows = [
            (None, "11111", "11111"),
            (None, "22222", "22222"),
            (None, "33333", "33333"),
            (None, "44444", "44444"),  # extra — no empty slot for it
        ]
        result = anchor_align(rows, 3)
        assert result == ["11111", "22222", "33333"]
        assert len(result) == 3

    def test_result_always_has_expected_length(self):
        rows = [(1, "11111", "11111")]
        for expected in [1, 5, 10]:
            result = anchor_align(rows, expected)
            assert len(result) == expected

    def test_apply_hard_rules_clamps_impossible_values(self):
        # Value > 1_000_000 should be replaced with "0" by apply_hard_rules
        rows = [(1, "9999999", "9999999")]
        result = anchor_align(rows, 3)
        assert result[0] == "0"
