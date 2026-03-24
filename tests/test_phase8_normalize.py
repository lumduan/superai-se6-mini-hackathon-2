"""Tests for Phase 8 — Normalization & Hard Rule Overrides.

Covers:
- normalize_votes: empty/placeholder inputs, Thai digit translation, OCR fixes,
  mixed content, already-clean Arabic digits, no-digit garbage, comma/space
  stripping, combined Thai-digit + OCR-fix paths.
- apply_soft_rules: non-digit string, value > MAX_VOTE, value < SOFT_LOW_VOTE,
  boundary values, normal plausible votes.
- apply_hard_rules: non-digit string, value > MAX_VOTE, custom fallback,
  value <= MAX_VOTE kept unchanged, low-but-legal values kept unchanged.
- normalize_and_validate: end-to-end convenience wrapper.
"""

from __future__ import annotations

import pytest

from src.config import MAX_VOTE
from src.phase8_normalize.normalize import (
    SOFT_LOW_VOTE,
    apply_hard_rules,
    apply_soft_rules,
    normalize_and_validate,
    normalize_votes,
)


# ── normalize_votes ───────────────────────────────────────────────────────────


class TestNormalizeVotes:
    # ── Empty / placeholder inputs ─────────────────────────────────────────

    def test_empty_string_returns_zero(self):
        assert normalize_votes("") == "0"

    def test_whitespace_only_returns_zero(self):
        assert normalize_votes("   ") == "0"

    def test_hyphen_placeholder_returns_zero(self):
        assert normalize_votes("-") == "0"

    def test_em_dash_placeholder_returns_zero(self):
        assert normalize_votes("—") == "0"

    def test_en_dash_placeholder_returns_zero(self):
        assert normalize_votes("–") == "0"

    # ── Arabic digits — already clean ─────────────────────────────────────

    def test_clean_arabic_digits_unchanged(self):
        assert normalize_votes("34405") == "34405"

    def test_digits_with_comma_stripped(self):
        assert normalize_votes("34,405") == "34405"

    def test_digits_with_spaces_stripped(self):
        assert normalize_votes("34 405") == "34405"

    def test_single_digit(self):
        assert normalize_votes("7") == "7"

    def test_zero_string(self):
        assert normalize_votes("0") == "0"

    # ── Thai digit translation ─────────────────────────────────────────────

    def test_thai_digits_translated(self):
        assert normalize_votes("๓๔,๔๐๕") == "34405"

    def test_all_thai_digits(self):
        assert normalize_votes("๐๑๒๓๔๕๖๗๘๙") == "0123456789"

    def test_thai_digits_with_comma(self):
        assert normalize_votes("๑,๐๐๐") == "1000"

    def test_thai_zero(self):
        assert normalize_votes("๐") == "0"

    # ── OCR character fixes ────────────────────────────────────────────────

    def test_uppercase_O_replaced_with_zero(self):
        assert normalize_votes("3O,4O5") == "30405"

    def test_lowercase_o_replaced_with_zero(self):
        assert normalize_votes("3o405") == "30405"

    def test_lowercase_l_replaced_with_one(self):
        assert normalize_votes("l2345") == "12345"

    def test_uppercase_I_replaced_with_one(self):
        assert normalize_votes("I2345") == "12345"

    def test_all_ocr_fixes_combined(self):
        # "lO,OOO" — no real digit is present before OCR fixes, so the guard
        # condition (any real digit) is False and OCR fixes do NOT fire.
        # All the letters get stripped → empty → "0".
        assert normalize_votes("lO,OOO") == "0"

    def test_ocr_fix_not_applied_without_real_digits(self):
        # "O" alone — no real digit present — OCR fixes should NOT fire,
        # resulting in an empty digit string → "0"
        assert normalize_votes("O") == "0"

    def test_ocr_fix_not_applied_to_pure_letters(self):
        # Pure alphabetic garbage with no real digit → "0"
        assert normalize_votes("ABC") == "0"

    # ── Mixed real content ─────────────────────────────────────────────────

    def test_vote_cell_with_thai_text_in_parens(self):
        # Full Typhoon-style vote cell
        assert normalize_votes("34,405 (สามหมื่นสี่พันสี่ร้อยห้า)") == "34405"

    def test_digits_with_trailing_text(self):
        # Real digits are present, so OCR-fix mode fires.  The 'o' in 'votes'
        # is in _OCR_FIXES and is converted to '0'; other letters are discarded.
        assert normalize_votes("12345 votes") == "123450"

    def test_leading_nondigit_chars(self):
        # Real digits are present ('9's), so OCR-fix mode fires.  The 'o' in
        # 'votes' is converted to '0'; other non-OCR-fix letters are discarded.
        assert normalize_votes("votes: 999") == "0999"

    def test_number_one_million(self):
        assert normalize_votes("1,000,000") == "1000000"

    def test_number_zero_explicit(self):
        assert normalize_votes("0") == "0"

    # ── Returns string, never int ─────────────────────────────────────────

    def test_return_type_is_str(self):
        result = normalize_votes("12345")
        assert isinstance(result, str)

    def test_return_always_nonempty(self):
        # Even for pure garbage input, result must be non-empty
        assert normalize_votes("???!@#") != ""


# ── apply_soft_rules ─────────────────────────────────────────────────────────


class TestApplySoftRules:
    def test_non_digit_string_returns_zero(self):
        assert apply_soft_rules("abc") == 0.0

    def test_empty_string_returns_zero(self):
        assert apply_soft_rules("") == 0.0

    def test_value_above_max_vote_returns_low_confidence(self):
        over = str(MAX_VOTE + 1)
        assert apply_soft_rules(over) == 0.1

    def test_value_exactly_max_vote_returns_full_confidence(self):
        assert apply_soft_rules(str(MAX_VOTE)) == 1.0

    def test_value_below_soft_low_returns_half_confidence(self):
        below = str(SOFT_LOW_VOTE - 1)
        assert apply_soft_rules(below) == 0.5

    def test_value_exactly_soft_low_returns_full_confidence(self):
        assert apply_soft_rules(str(SOFT_LOW_VOTE)) == 1.0

    def test_zero_returns_half_confidence(self):
        # 0 < SOFT_LOW_VOTE → penalty
        assert apply_soft_rules("0") == 0.5

    def test_normal_vote_returns_full_confidence(self):
        assert apply_soft_rules("12345") == 1.0

    def test_large_valid_vote_returns_full_confidence(self):
        assert apply_soft_rules("999999") == 1.0

    def test_return_type_is_float(self):
        assert isinstance(apply_soft_rules("12345"), float)

    def test_string_with_leading_zeros_is_digit(self):
        # "00100" is all digits → treated as 100 by int() → full confidence
        assert apply_soft_rules("00100") == 1.0

    def test_mixed_alphanumeric_returns_zero_confidence(self):
        assert apply_soft_rules("1234a") == 0.0


# ── apply_hard_rules ─────────────────────────────────────────────────────────


class TestApplyHardRules:
    def test_clean_valid_vote_returned_unchanged(self):
        assert apply_hard_rules("12345") == "12345"

    def test_zero_returned_unchanged(self):
        assert apply_hard_rules("0") == "0"

    def test_value_at_max_vote_returned_unchanged(self):
        assert apply_hard_rules(str(MAX_VOTE)) == str(MAX_VOTE)

    def test_value_above_max_vote_replaced_with_fallback(self):
        over = str(MAX_VOTE + 1)
        assert apply_hard_rules(over) == "0"

    def test_value_above_max_vote_custom_fallback(self):
        over = str(MAX_VOTE + 1)
        assert apply_hard_rules(over, fallback="999") == "999"

    def test_non_digit_string_replaced_with_fallback(self):
        assert apply_hard_rules("abc") == "0"

    def test_empty_string_replaced_with_fallback(self):
        assert apply_hard_rules("") == "0"

    def test_mixed_alphanumeric_replaced_with_fallback(self):
        assert apply_hard_rules("123abc") == "0"

    def test_low_vote_not_overridden(self):
        # A vote of 5 is suspicious but legal — hard rules must NOT change it.
        assert apply_hard_rules("5") == "5"

    def test_value_one_below_soft_low_not_overridden(self):
        below = str(SOFT_LOW_VOTE - 1)
        assert apply_hard_rules(below) == below

    def test_return_type_is_str(self):
        assert isinstance(apply_hard_rules("12345"), str)

    def test_custom_fallback_used_for_non_digit(self):
        assert apply_hard_rules("garbage", fallback="X") == "X"


# ── normalize_and_validate ────────────────────────────────────────────────────


class TestNormalizeAndValidate:
    def test_normal_vote_cell(self):
        assert normalize_and_validate("34,405") == "34405"

    def test_thai_digits_normalized_and_valid(self):
        assert normalize_and_validate("๓๔,๔๐๕") == "34405"

    def test_placeholder_dash_returns_zero(self):
        assert normalize_and_validate("-") == "0"

    def test_impossibly_large_value_replaced(self):
        raw = str(MAX_VOTE + 99)
        assert normalize_and_validate(raw) == "0"

    def test_impossibly_large_thai_number_replaced(self):
        # Thai vote cell whose digit value exceeds MAX_VOTE
        assert normalize_and_validate("2,000,000") == "0"

    def test_custom_fallback_for_invalid(self):
        # normalize_votes("???") returns "0" (no digits), which passes hard rules.
        # To trigger the custom fallback, use a value that exceeds MAX_VOTE.
        assert normalize_and_validate(str(MAX_VOTE + 1), fallback="ERR") == "ERR"

    def test_low_legitimate_vote_not_overridden(self):
        # 3 votes — suspicious but legal; hard rules leave it alone
        assert normalize_and_validate("3") == "3"

    def test_ocr_garbage_with_no_digit_returns_zero(self):
        assert normalize_and_validate("OCR_FAIL") == "0"

    def test_ocr_letter_O_in_vote_is_fixed(self):
        # "1O,OOO" → normalize → "10000" → valid → kept
        assert normalize_and_validate("1O,OOO") == "10000"

    def test_full_typhoon_vote_cell(self):
        # Realistic Typhoon vote cell with Thai text in parens
        raw = "10,778 (หนึ่งหมื่นเจ็ดร้อยเจ็ดสิบแปด)"
        assert normalize_and_validate(raw) == "10778"
