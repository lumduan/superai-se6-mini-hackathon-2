"""Tests for Phase 7 — Thai Text Cross-check with Digit-level Diff.

Covers:
- extract_thai_number_text: normal case, multi-word, empty parens, no parens.
- digit_distance: equal length, different length, identical strings.
- cross_check_vote: no Thai text, length match with low diff, length match with
  high diff, length diff of 1, length diff > 1, pythainlp failure fallback,
  partial regex parser path.
- _partial_thai_number: known Thai number words, zero result, raw-digit fallback.
"""

from __future__ import annotations

from unittest import mock

import pytest

from src.phase7_thai_crosscheck.crosscheck import (
    DIGIT_DIFF_THRESHOLD,
    _partial_thai_number,
    cross_check_vote,
    digit_distance,
    extract_thai_number_text,
)


# ── extract_thai_number_text ──────────────────────────────────────────────────


class TestExtractThaiNumberText:
    def test_basic_parenthesised_text(self):
        raw = "34,405 (สามหมื่นสี่พันสี่ร้อยห้า)"
        assert extract_thai_number_text(raw) == "สามหมื่นสี่พันสี่ร้อยห้า"

    def test_whitespace_stripped(self):
        raw = "12,000 (  หนึ่งหมื่นสองพัน  )"
        result = extract_thai_number_text(raw)
        assert result == "หนึ่งหมื่นสองพัน"

    def test_no_parentheses_returns_none(self):
        assert extract_thai_number_text("12,345") is None

    def test_empty_parens_returns_none(self):
        assert extract_thai_number_text("12,345 ()") is None

    def test_whitespace_only_parens_returns_none(self):
        assert extract_thai_number_text("12,345 (   )") is None

    def test_first_paren_group_extracted(self):
        # Only the first group should be returned.
        raw = "1,000 (หนึ่งพัน) (extra)"
        result = extract_thai_number_text(raw)
        assert result == "หนึ่งพัน"

    def test_arbitrary_text_in_parens(self):
        raw = "5,000 (ห้าพัน)"
        assert extract_thai_number_text(raw) == "ห้าพัน"

    def test_empty_string_returns_none(self):
        assert extract_thai_number_text("") is None


# ── digit_distance ────────────────────────────────────────────────────────────


class TestDigitDistance:
    def test_identical_strings(self):
        assert digit_distance("34405", "34405") == 0

    def test_single_digit_diff_same_length(self):
        assert digit_distance("34405", "34485") == 1

    def test_two_digit_diffs_same_length(self):
        assert digit_distance("34405", "34486") == 2

    def test_all_digits_differ(self):
        assert digit_distance("11111", "22222") == 5

    def test_different_lengths_one_longer(self):
        # Length diff = 1, positional diff over shared prefix.
        # "34405" vs "344050": length diff=1, shared "34405" matches → 1
        result = digit_distance("34405", "344050")
        assert result == 1

    def test_different_lengths_large_diff(self):
        # "100" vs "10000": length diff=2, shared "100" matches → 2
        result = digit_distance("100", "10000")
        assert result == 2

    def test_empty_strings(self):
        assert digit_distance("", "") == 0

    def test_one_empty(self):
        assert digit_distance("", "123") == 3

    def test_symmetric(self):
        assert digit_distance("12345", "12300") == digit_distance("12300", "12345")


# ── cross_check_vote ──────────────────────────────────────────────────────────


class TestCrossCheckVote:
    def test_no_thai_text_returns_digit_vote(self):
        # No parentheses → no Thai text → return original digit vote unchanged.
        assert cross_check_vote("12,345", "12345") == "12345"

    def test_empty_raw_cell_returns_digit_vote(self):
        assert cross_check_vote("", "5000") == "5000"

    def test_same_length_low_diff_keeps_ocr(self):
        # diff = 1 < DIGIT_DIFF_THRESHOLD(3) → keep OCR value
        # Mock pythainlp to return "34405" while OCR gives "34415" (diff=1)
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="34405",
        ):
            result = cross_check_vote("34,415 (สามหมื่นสี่พัน...)", "34415")
        assert result == "34415"

    def test_same_length_high_diff_uses_thai(self):
        # diff ≥ DIGIT_DIFF_THRESHOLD(3) → trust Thai text
        # Thai="34405", OCR="37785" → diff=3 (positions 1,2,3 differ)
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="34405",
        ):
            result = cross_check_vote("37,785 (สามหมื่นสี่พัน...)", "37785")
        assert result == "34405"

    def test_same_length_diff_two_keeps_ocr(self):
        # diff=2 < DIGIT_DIFF_THRESHOLD(3) → keep digit OCR (covers cand-11 bug)
        # Thai text OCR can also be wrong; only override on 3+ mismatches.
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="574",
        ):
            result = cross_check_vote("๖๙๔ (ห้าร้อยเจ็ดสิบสี่)", "694")
        assert result == "694"

    def test_length_diff_one_uses_thai(self):
        # len(thai)=5, len(ocr)=4 → diff=1 → trust Thai
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="34405",
        ):
            result = cross_check_vote("3,440 (สามหมื่นสี่พัน...)", "3440")
        assert result == "34405"

    def test_length_diff_greater_than_one_keeps_ocr(self):
        # len(thai)=6, len(ocr)=3 → diff=3 → keep OCR
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="123456",
        ):
            result = cross_check_vote("123 (xxx)", "123")
        assert result == "123"

    def test_thai_conversion_fails_returns_digit_vote(self):
        # When _convert_thai_text returns None, keep OCR value.
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value=None,
        ):
            result = cross_check_vote("12,345 (gobbledygook)", "12345")
        assert result == "12345"

    def test_exact_match_same_value(self):
        # Thai and OCR agree → diff=0 → keep OCR (same result either way)
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="12345",
        ):
            result = cross_check_vote("12,345 (หนึ่งหมื่น...)", "12345")
        assert result == "12345"

    def test_digit_diff_threshold_boundary_below(self):
        # diff = DIGIT_DIFF_THRESHOLD - 1 → keep OCR
        # diff = 2 when threshold = 3
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="12345",
        ):
            result = cross_check_vote("12344 (X)", "12344")
        # Thai="12345", OCR="12344", diff=1 < 2 → keep OCR
        # But here _convert_thai_text is mocked to "12345"
        # and raw_cell has no parens content that matters — extract just routes through.
        assert result == "12344"

    def test_digit_diff_threshold_at_threshold(self):
        # diff = 3 = DIGIT_DIFF_THRESHOLD(3) → use Thai
        # Thai="12345", OCR="19955": positions 1,2,3 differ → diff=3
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="12345",
        ):
            result = cross_check_vote("19,955 (xxx)", "19955")
        assert result == "12345"

    def test_digit_diff_below_threshold_keeps_ocr(self):
        # diff = 2 < DIGIT_DIFF_THRESHOLD(3) → keep OCR (not overridden)
        # Thai="12345", OCR="12155": positions 2 and 3 differ → diff=2
        with mock.patch(
            "src.phase7_thai_crosscheck.crosscheck._convert_thai_text",
            return_value="12345",
        ):
            result = cross_check_vote("12,155 (xxx)", "12155")
        assert result == "12155"


# ── _partial_thai_number ──────────────────────────────────────────────────────


class TestPartialThaiNumber:
    def test_simple_phan(self):
        # "ห้าพัน" = 5,000
        result = _partial_thai_number("ห้าพัน")
        assert result == "5000"

    def test_muen_phan(self):
        # "สามหมื่นสี่พัน" = 34,000
        result = _partial_thai_number("สามหมื่นสี่พัน")
        assert result == "34000"

    def test_roi(self):
        # "สามร้อย" = 300
        result = _partial_thai_number("สามร้อย")
        assert result == "300"

    def test_complex_number(self):
        # "สามหมื่นสี่พันสี่ร้อยห้า" = 34,405
        result = _partial_thai_number("สามหมื่นสี่พันสี่ร้อยห้า")
        assert result == "34405"

    def test_single_digit_word(self):
        # "เก้า" = 9
        result = _partial_thai_number("เก้า")
        assert result == "9"

    def test_no_recognised_word_returns_none(self):
        # No recognised Thai number words and no Arabic digits.
        result = _partial_thai_number("xyz abc")
        assert result is None

    def test_empty_string_returns_none(self):
        assert _partial_thai_number("") is None

    def test_raw_digit_fallback(self):
        # If Thai words are unrecognised but string contains Arabic digits,
        # return those digits as a last-resort fallback.
        result = _partial_thai_number("OCR junk 34405 more junk")
        assert result == "34405"

    def test_lan(self):
        # "หนึ่งล้าน" = 1,000,000
        result = _partial_thai_number("หนึ่งล้าน")
        assert result == "1000000"

    def test_san(self):
        # "สองแสน" = 200,000
        result = _partial_thai_number("สองแสน")
        assert result == "200000"


# ── Integration: cross_check with real pythainlp (if installed) ───────────────


class TestCrossCheckIntegration:
    """Integration tests that call pythainlp for real.

    These are skipped when pythainlp is not installed in the environment so
    the suite can still pass in minimal CI environments.
    """

    @pytest.fixture(autouse=True)
    def _skip_if_no_pythainlp(self):
        pytest.importorskip("pythainlp", reason="pythainlp not installed")

    def test_matching_values_no_correction(self):
        # "ห้าพัน" = 5000, OCR gives "5000" — no correction needed.
        result = cross_check_vote("5,000 (ห้าพัน)", "5000")
        assert result == "5000"

    def test_mismatched_value_corrected(self):
        # "ห้าพัน" = 5000, OCR gives "5500" — diff=1
        # diff(5000, 5500) = 1 < threshold 2 → keep OCR value
        result = cross_check_vote("5,500 (ห้าพัน)", "5500")
        # diff is 1 — below threshold — OCR kept
        assert result == "5500"

    def test_highly_mismatched_value_corrected(self):
        # "ห้าพัน" = 5000, OCR gives "5800" — diff=2 ≥ threshold → use Thai
        result = cross_check_vote("5,800 (ห้าพัน)", "5800")
        assert result == "5000"
