"""Tests for Phase 10 — Per-row Confidence Scoring.

Covers:
- compute_row_confidence:
    non-digit string → 0.0
    empty string → 0.0
    high-value plausible vote → 1.0
    low-value suspicious vote (< SOFT_LOW_VOTE) → 0.5
    very short vote (< 3 digits) → penalised
    single digit → max(0, 0.5 - 0.3) = 0.2
    two-digit large enough soft-score minus length penalty
    exactly 3 digits → no length penalty applied
    position and total_expected params are accepted (reserved)

- compute_document_confidence:
    empty list → 0.0
    perfect list (right count, plausible distribution, no total) → near 1.0
    length mismatch → lower score
    all-zero votes → reduced score
    all-short votes → reduced score
    unreasonable distribution (low median) → reduced score
    ocr_total matches sum → bonus
    ocr_total mismatches sum → penalty
    result always clamped to [0.0, 1.0]

- needs_fallback:
    high confidence + exact count → False
    confidence below threshold → True
    count mismatch within tolerance → False (if confidence ok)
    count mismatch beyond tolerance → True
    both conditions True → True
"""

from __future__ import annotations

import pytest

from src.config import CONFIDENCE_THRESHOLD, MAX_VOTE
from src.phase10_confidence.scorer import (
    MISMATCH_TOLERANCE,
    _SHORT_LENGTH_PENALTY,
    _SHORT_LENGTH_THRESHOLD,
    compute_document_confidence,
    compute_row_confidence,
    needs_fallback,
)


# ── compute_row_confidence ────────────────────────────────────────────────────


class TestComputeRowConfidence:
    def test_non_digit_string_returns_zero(self):
        assert compute_row_confidence("abc", 0, 10) == 0.0

    def test_empty_string_returns_zero(self):
        assert compute_row_confidence("", 0, 10) == 0.0

    def test_plausible_large_vote_returns_one(self):
        # A five-digit vote is clearly plausible → soft_rules = 1.0, no short penalty
        assert compute_row_confidence("12345", 0, 10) == 1.0

    def test_low_value_vote_gets_soft_penalty(self):
        # Value < SOFT_LOW_VOTE (20) → apply_soft_rules = 0.5; len("5") = 1 < 3 → -0.3 → 0.2
        result = compute_row_confidence("5", 0, 10)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_vote_shorter_than_threshold_is_penalised(self):
        # "10" → apply_soft_rules: int("10") < SOFT_LOW_VOTE → 0.5; len=2 < 3 → -0.3 → 0.2
        result = compute_row_confidence("10", 0, 10)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_vote_exactly_at_length_threshold_no_penalty(self):
        # "100" → 3 digits, int=100 which is >= SOFT_LOW_VOTE → soft=1.0; len=3 >= 3 → no penalty
        result = compute_row_confidence("100", 0, 10)
        assert result == pytest.approx(1.0, abs=1e-9)

    def test_single_digit_zero_returns_zero_dot_zero(self):
        # "0" → apply_soft_rules("0"): int=0 < SOFT_LOW_VOTE → 0.5; len=1 < 3 → -0.3 → 0.2
        # But "0".isdigit() is True, so 0.5 - 0.3 = 0.2
        result = compute_row_confidence("0", 0, 10)
        assert result == pytest.approx(0.2, abs=1e-9)

    def test_impossible_value_very_low_score(self):
        # Value > MAX_VOTE → apply_soft_rules = 0.1; len > 3 → no short penalty
        big_vote = str(MAX_VOTE + 1)
        result = compute_row_confidence(big_vote, 0, 10)
        assert result == pytest.approx(0.1, abs=1e-9)

    def test_score_clamped_to_non_negative(self):
        # Even in the worst combination, score should never go below 0
        result = compute_row_confidence("1", 5, 10)
        assert result >= 0.0

    def test_position_and_total_accepted_as_reserved(self):
        # Different positions and totals should not change the score (reserved params)
        score_a = compute_row_confidence("12345", 0, 5)
        score_b = compute_row_confidence("12345", 4, 100)
        assert score_a == score_b


# ── compute_document_confidence ──────────────────────────────────────────────


class TestComputeDocumentConfidence:
    def test_empty_votes_returns_zero(self):
        assert compute_document_confidence([], 5) == 0.0

    def test_empty_votes_with_total_returns_zero(self):
        assert compute_document_confidence([], 5, ocr_total=12345) == 0.0

    def test_perfect_list_near_one(self):
        # 5 plausible votes, correct count, no total provided
        votes = ["12345", "23456", "34567", "45678", "56789"]
        result = compute_document_confidence(votes, 5)
        assert result >= 0.8

    def test_length_mismatch_lowers_score(self):
        # 2 votes instead of 10 expected
        votes = ["12345", "23456"]
        result = compute_document_confidence(votes, 10)
        result_perfect = compute_document_confidence(
            ["12345", "23456", "34567", "45678", "56789",
             "67890", "78901", "89012", "90123", "11111"],
            10,
        )
        assert result < result_perfect

    def test_all_zero_votes_lowers_score(self):
        votes_zeros = ["0"] * 5
        votes_good = ["12345"] * 5
        assert compute_document_confidence(votes_zeros, 5) < compute_document_confidence(votes_good, 5)

    def test_all_short_votes_lowers_score(self):
        votes_short = ["1"] * 5
        votes_good = ["12345"] * 5
        assert compute_document_confidence(votes_short, 5) < compute_document_confidence(votes_good, 5)

    def test_unreasonable_distribution_lowers_score(self):
        # Low values → unreasonable distribution penalty
        low_votes = ["10", "20", "15", "12", "18"]
        high_votes = ["1000", "2000", "1500", "1200", "1800"]
        score_low = compute_document_confidence(low_votes, 5)
        score_high = compute_document_confidence(high_votes, 5)
        assert score_low < score_high

    def test_total_match_gives_bonus(self):
        # Use 3 votes with expected=4 to create a small count-mismatch penalty
        # so the base score is < 1.0 and the bonus is actually observable.
        votes = ["1000", "2000", "3000"]
        total = 6000
        score_with_match = compute_document_confidence(votes, 4, ocr_total=total)
        score_no_total = compute_document_confidence(votes, 4, ocr_total=None)
        assert score_with_match > score_no_total

    def test_total_mismatch_gives_penalty(self):
        votes = ["1000", "2000", "3000"]
        wrong_total = 9999  # far from actual sum = 6000
        score_mismatch = compute_document_confidence(votes, 3, ocr_total=wrong_total)
        score_no_total = compute_document_confidence(votes, 3, ocr_total=None)
        assert score_mismatch < score_no_total

    def test_result_clamped_to_zero_one(self):
        # Pathological input — result must stay within [0, 1]
        votes = ["0"] * 20
        result = compute_document_confidence(votes, 3, ocr_total=999999)
        assert 0.0 <= result <= 1.0

    def test_total_near_match_within_tolerance_gives_bonus(self):
        # Sum = 10000, total = 10050 → error = 50/10050 ≈ 0.5% < 1% → bonus.
        # Use expected=4 to create a small count penalty so the base score
        # is below 1.0 and the bonus is actually observable.
        votes = ["3000", "3000", "4000"]
        total = 10050
        score_near = compute_document_confidence(votes, 4, ocr_total=total)
        score_no_total = compute_document_confidence(votes, 4, ocr_total=None)
        assert score_near > score_no_total

    def test_result_never_exceeds_one(self):
        # Even with the total match bonus, can't exceed 1.0
        votes = ["10000", "20000", "30000", "40000", "50000"]
        total = sum(int(v) for v in votes)
        result = compute_document_confidence(votes, 5, ocr_total=total)
        assert result <= 1.0


# ── needs_fallback ────────────────────────────────────────────────────────────


class TestNeedsFallback:
    def test_good_confidence_exact_count_no_fallback(self):
        votes = ["12345", "23456"]
        assert needs_fallback(votes, 2, 0.9) is False

    def test_confidence_below_threshold_triggers_fallback(self):
        votes = ["12345", "23456"]
        low_conf = CONFIDENCE_THRESHOLD - 0.01
        assert needs_fallback(votes, 2, low_conf) is True

    def test_confidence_exactly_threshold_no_fallback(self):
        votes = ["12345", "23456"]
        assert needs_fallback(votes, 2, CONFIDENCE_THRESHOLD) is False

    def test_count_mismatch_within_tolerance_no_fallback(self):
        # |3 - 2| = 1 ≤ MISMATCH_TOLERANCE=2
        votes = ["12345", "23456", "34567"]
        assert needs_fallback(votes, 2, 0.9) is False

    def test_count_mismatch_exactly_tolerance_no_fallback(self):
        # |4 - 2| = 2 == MISMATCH_TOLERANCE → not > tolerance → no fallback
        votes = ["12345", "23456", "34567", "45678"]
        assert needs_fallback(votes, 2, 0.9) is False

    def test_count_mismatch_beyond_tolerance_triggers_fallback(self):
        # |6 - 2| = 4 > MISMATCH_TOLERANCE=2
        votes = ["12345", "23456", "34567", "45678", "56789", "67890"]
        assert needs_fallback(votes, 2, 0.9) is True

    def test_both_conditions_true_triggers_fallback(self):
        votes = ["1", "2", "3", "4", "5", "6"]
        low_conf = CONFIDENCE_THRESHOLD - 0.1
        assert needs_fallback(votes, 2, low_conf) is True

    def test_empty_votes_beyond_tolerance_triggers_fallback(self):
        # |0 - 5| = 5 > MISMATCH_TOLERANCE
        assert needs_fallback([], 5, 0.9) is True

    def test_mismatch_tolerance_constant_is_two(self):
        assert MISMATCH_TOLERANCE == 2
