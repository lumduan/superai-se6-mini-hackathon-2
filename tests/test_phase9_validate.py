"""Tests for Phase 9 — Row Structure Validation & Total-based Correction.

Covers:
- is_reasonable_distribution: empty list, all zeros, median < threshold,
  median == threshold, median > threshold, single-value list, mixed valid/invalid.
- extract_total_from_html: no HTML, no total row, total in last cell, Thai
  digits, colspan total row, multiple keyword rows (first match wins),
  digits outside 3–8 range ignored.
- extract_total_from_markdown: empty string, no keyword lines, total found,
  Thai digits translated, rightmost match on keyword line, short/long digit
  runs ignored.
- total_based_correction: no total (returns unchanged), sum matches (unchanged),
  gap too large (unchanged), single-row correction, adjusted value negative
  (skip), adjusted value > MAX_VOTE (skip), most-suspicious row selected,
  non-digit rows ignored when computing sum, non-digit rows not modified.
- validate_and_correct: distribution ok + correction, distribution fails +
  correction, distribution ok + no total.
"""

from __future__ import annotations

import pytest

from src.config import MAX_VOTE
from src.phase9_postprocess.validator import (
    extract_total_from_html,
    extract_total_from_markdown,
    is_reasonable_distribution,
    total_based_correction,
    validate_and_correct,
)


# ── is_reasonable_distribution ───────────────────────────────────────────────


class TestIsReasonableDistribution:
    def test_empty_list_returns_false(self):
        assert is_reasonable_distribution([]) is False

    def test_all_zeros_returns_false(self):
        assert is_reasonable_distribution(["0", "0", "0"]) is False

    def test_non_digit_strings_returns_false(self):
        assert is_reasonable_distribution(["abc", "xyz"]) is False

    def test_median_below_threshold_returns_false(self):
        # Values: 10, 20, 30 → median = 20 < 50
        assert is_reasonable_distribution(["10", "20", "30"]) is False

    def test_median_exactly_threshold_returns_true(self):
        # Values: 50, 50, 50 → median = 50 >= 50
        assert is_reasonable_distribution(["50", "50", "50"]) is True

    def test_median_above_threshold_returns_true(self):
        assert is_reasonable_distribution(["1234", "5678", "9012"]) is True

    def test_single_large_value_returns_true(self):
        assert is_reasonable_distribution(["12345"]) is True

    def test_single_value_below_threshold_returns_false(self):
        assert is_reasonable_distribution(["10"]) is False

    def test_mixed_zeros_and_real_votes(self):
        # Zeros are excluded from the median; large values should pass.
        assert is_reasonable_distribution(["0", "0", "1500", "2000"]) is True

    def test_typical_election_votes_returns_true(self):
        votes = ["10778", "8432", "7654", "6543", "9876"]
        assert is_reasonable_distribution(votes) is True

    def test_row_numbers_returns_false(self):
        # Row numbers 1–10 look like this after OCR garbage.
        votes = [str(i) for i in range(1, 11)]
        assert is_reasonable_distribution(votes) is False

    def test_returns_bool(self):
        result = is_reasonable_distribution(["1000"])
        assert isinstance(result, bool)


# ── extract_total_from_html ───────────────────────────────────────────────────


class TestExtractTotalFromHtml:
    def test_empty_string_returns_none(self):
        assert extract_total_from_html("") is None

    def test_whitespace_only_returns_none(self):
        assert extract_total_from_html("   ") is None

    def test_no_table_returns_none(self):
        assert extract_total_from_html("<p>No table here</p>") is None

    def test_no_total_row_returns_none(self):
        html = "<table><tr><td>1</td><td>John</td><td>12345</td></tr></table>"
        assert extract_total_from_html(html) is None

    def test_finds_total_in_last_cell(self):
        html = (
            "<table>"
            "<tr><td>1</td><td>Candidate A</td><td>12,345</td></tr>"
            "<tr><td colspan='2'>รวมคะแนนทั้งสิ้น</td><td>12,345</td></tr>"
            "</table>"
        )
        assert extract_total_from_html(html) == 12345

    def test_finds_total_with_second_keyword(self):
        html = (
            "<table>"
            "<tr><td>รวมคะแนน</td><td>77982</td></tr>"
            "</table>"
        )
        assert extract_total_from_html(html) == 77982

    def test_finds_total_with_third_keyword(self):
        html = (
            "<table>"
            "<tr><td colspan='3'>รวมทั้งสิ้น</td><td>50000</td></tr>"
            "</table>"
        )
        assert extract_total_from_html(html) == 50000

    def test_thai_digits_translated(self):
        html = (
            "<table>"
            "<tr><td>รวมคะแนน</td><td>๗๗,๙๘๒</td></tr>"
            "</table>"
        )
        assert extract_total_from_html(html) == 77982

    def test_ignores_cells_with_too_few_digits(self):
        # Cell has only 2 digits — should be ignored; next rightmost with 3–8
        # digits should be used.
        html = (
            "<table>"
            "<tr><td>รวมคะแนน</td><td>12</td><td>77982</td></tr>"
            "</table>"
        )
        assert extract_total_from_html(html) == 77982

    def test_ignores_cells_with_too_many_digits(self):
        # 9-digit cell should be ignored; 6-digit cell should be used.
        html = (
            "<table>"
            "<tr><td>รวมคะแนน</td><td>123456789</td><td>100000</td></tr>"
            "</table>"
        )
        assert extract_total_from_html(html) == 100000

    def test_returns_int(self):
        html = "<table><tr><td>รวมคะแนน</td><td>99999</td></tr></table>"
        result = extract_total_from_html(html)
        assert isinstance(result, int)

    def test_commas_in_number_stripped(self):
        html = "<table><tr><td>รวมคะแนนทั้งสิ้น</td><td>1,234,567</td></tr></table>"
        # 1234567 has 7 digits, within the 3–8 range
        assert extract_total_from_html(html) == 1234567


# ── extract_total_from_markdown ───────────────────────────────────────────────


class TestExtractTotalFromMarkdown:
    def test_empty_string_returns_none(self):
        assert extract_total_from_markdown("") is None

    def test_whitespace_only_returns_none(self):
        assert extract_total_from_markdown("  \n  ") is None

    def test_no_keyword_line_returns_none(self):
        markdown = "Candidate A | 12345\nCandidate B | 67890"
        assert extract_total_from_markdown(markdown) is None

    def test_finds_total_on_keyword_line(self):
        markdown = "รวมคะแนน | 77,982"
        assert extract_total_from_markdown(markdown) == 77982

    def test_finds_total_with_second_keyword(self):
        markdown = "| รวมทั้งสิ้น | | 55000 |"
        assert extract_total_from_markdown(markdown) == 55000

    def test_rightmost_number_used(self):
        # Line has two numbers — rightmost (last) should be used.
        markdown = "รวมคะแนน 3 คน | 99999"
        assert extract_total_from_markdown(markdown) == 99999

    def test_thai_digits_translated(self):
        markdown = "รวมคะแนนทั้งสิ้น | ๗๗,๙๘๒"
        assert extract_total_from_markdown(markdown) == 77982

    def test_ignores_non_keyword_lines(self):
        markdown = "12345\nรวมคะแนน | 88888\n67890"
        assert extract_total_from_markdown(markdown) == 88888

    def test_short_digit_run_ignored(self):
        # "12" is only 2 digits — below the 3-digit minimum.
        markdown = "รวมคะแนน | 12"
        assert extract_total_from_markdown(markdown) is None

    def test_long_digit_run_ignored(self):
        # "123456789" is 9 digits — above the 8-digit maximum.
        markdown = "รวมคะแนน | 123456789"
        assert extract_total_from_markdown(markdown) is None

    def test_returns_int(self):
        result = extract_total_from_markdown("รวมคะแนน | 99999")
        assert isinstance(result, int)


# ── total_based_correction ────────────────────────────────────────────────────


class TestTotalBasedCorrection:
    def test_no_total_returns_unchanged(self):
        votes = ["1000", "2000", "3000"]
        result = total_based_correction(votes, None)
        assert result == ["1000", "2000", "3000"]

    def test_no_total_does_not_mutate_input(self):
        votes = ["1000", "2000"]
        original = votes[:]
        total_based_correction(votes, None)
        assert votes == original

    def test_sum_matches_total_returns_unchanged(self):
        votes = ["1000", "2000", "3000"]
        result = total_based_correction(votes, 6000)
        assert result == ["1000", "2000", "3000"]

    def test_gap_too_large_returns_unchanged(self):
        # max_val = 10000; gap = 3000 > 10000 * 0.20 = 2000
        votes = ["5000", "10000", "3000"]
        result = total_based_correction(votes, 21000)
        assert result == ["5000", "10000", "3000"]

    def test_simple_positive_gap_corrected(self):
        # sum = 6000, total = 6100 → gap = 100
        # median = 2000; most suspicious = "1000" (lowest, soft_penalty highest)
        votes = ["1000", "2000", "3000"]
        result = total_based_correction(votes, 6100)
        # Row 0 ("1000") has highest soft penalty → adjusted to 1100
        assert result == ["1100", "2000", "3000"]

    def test_simple_negative_gap_corrected(self):
        # sum = 6000, total = 5900 → gap = -100
        votes = ["1000", "2000", "3000"]
        result = total_based_correction(votes, 5900)
        # Row 0 ("1000") is most suspicious → adjusted to 900
        assert result == ["900", "2000", "3000"]

    def test_adjusted_value_negative_skips_correction(self):
        # gap = -2000; most suspicious row value = 100 → 100 - 2000 = -1900 (invalid)
        votes = ["100", "50000", "60000"]
        result = total_based_correction(votes, 108100)
        assert result == ["100", "50000", "60000"]

    def test_adjusted_value_above_max_vote_skips_correction(self):
        # gap = +100; most suspicious row value = MAX_VOTE → adjusted > MAX_VOTE
        votes = [str(MAX_VOTE), "1000", "2000"]
        total = MAX_VOTE + 1000 + 2000 + 100
        result = total_based_correction(votes, total)
        # adjusted = MAX_VOTE + 100 > MAX_VOTE → skip
        assert result == [str(MAX_VOTE), "1000", "2000"]

    def test_non_digit_rows_not_counted_in_sum(self):
        # "abc" is non-digit — ignored in sum; "1000" + "2000" = 3000
        votes = ["1000", "abc", "2000"]
        result = total_based_correction(votes, 3100)
        # gap = 100; most suspicious among digit rows is "1000"
        assert result[1] == "abc"  # non-digit row unchanged
        assert sum(int(v) for v in result if v.isdigit()) == 3100

    def test_non_digit_rows_not_modified(self):
        votes = ["abc", "2000", "3000"]
        result = total_based_correction(votes, 5100)
        assert result[0] == "abc"

    def test_empty_votes_returns_empty(self):
        result = total_based_correction([], 1000)
        assert result == []

    def test_all_non_digit_returns_unchanged(self):
        votes = ["abc", "xyz"]
        result = total_based_correction(votes, 1000)
        assert result == ["abc", "xyz"]

    def test_returns_new_list_not_mutated_input(self):
        votes = ["1000", "2000", "3000"]
        result = total_based_correction(votes, 6100)
        assert votes == ["1000", "2000", "3000"]  # original unchanged
        assert result is not votes

    def test_exact_boundary_gap_allowed(self):
        # max_val = 10000; boundary gap = 2000 = 10000 * 0.20 exactly → corrected
        # votes: 5000, 3000, 10000 → sum = 18000
        # total = 20000 → gap = 2000
        votes = ["5000", "3000", "10000"]
        result = total_based_correction(votes, 20000)
        assert sum(int(v) for v in result) == 20000

    def test_single_vote_row_corrected(self):
        result = total_based_correction(["5000"], 5050)
        assert result == ["5050"]


# ── validate_and_correct ──────────────────────────────────────────────────────


class TestValidateAndCorrect:
    def test_reasonable_distribution_no_total(self):
        votes = ["1234", "5678", "9012"]
        corrected, dist_ok = validate_and_correct(votes, None)
        assert dist_ok is True
        assert corrected == ["1234", "5678", "9012"]

    def test_unreasonable_distribution_flagged(self):
        votes = ["1", "2", "3"]
        _, dist_ok = validate_and_correct(votes, None)
        assert dist_ok is False

    def test_correction_applied_when_total_given(self):
        votes = ["1000", "2000", "3000"]
        corrected, dist_ok = validate_and_correct(votes, 6100)
        assert dist_ok is True
        assert sum(int(v) for v in corrected) == 6100

    def test_distribution_fail_still_attempts_correction(self):
        # Even when distribution is bad, correction should still run.
        # median([1, 40, 1]) = 1 < 50 → distribution fails.
        # sum = 42, total = 48, gap = 6 ≤ 40 * 0.20 = 8 → correction applies.
        votes = ["1", "40", "1"]
        corrected, dist_ok = validate_and_correct(votes, 48)
        assert dist_ok is False
        assert sum(int(v) for v in corrected) == 48

    def test_returns_tuple_of_list_and_bool(self):
        result = validate_and_correct(["1000"], None)
        assert isinstance(result, tuple)
        assert len(result) == 2
        corrected, dist_ok = result
        assert isinstance(corrected, list)
        assert isinstance(dist_ok, bool)

    def test_empty_votes(self):
        corrected, dist_ok = validate_and_correct([])
        assert corrected == []
        assert dist_ok is False
