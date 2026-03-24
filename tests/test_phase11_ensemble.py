"""Tests for Phase 11 — Multi-pass Fallback OCR & Ensemble Voting.

Covers:
- normalize_length:
    shorter list is padded with "0"
    longer list is truncated
    exact-length list is unchanged

- preprocess_otsu:
    returns PIL Image
    raises FileNotFoundError for missing path
    output is grayscale (single channel via numpy)

- fallback_tesseract:
    returns list (empty if pytesseract unavailable)
    filters lines with 3–7 digits
    non-digit characters stripped

- apply_sanity_checks:
    value longer than 7 digits replaced with "0"
    leading zeros stripped ("01234" → "1234")
    plain "0" left unchanged
    clean values passed through unchanged
    empty list returns empty list

- ensemble_votes:
    empty candidates returns list of "0"s
    single candidate passthrough
    two candidates agree → unanimous result
    two candidates disagree → higher-confidence candidate wins
    agree_bonus favours values agreed by more passes
    length normalization applied inside ensemble
    result always has exactly expected length

- extract_votes_multipass:
    integration not tested (requires live Typhoon API key)
    _run_pass failure returns (0.0, []) and does not raise
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from src.phase11_ensemble import (
    apply_sanity_checks,
    ensemble_votes,
    fallback_tesseract,
    normalize_length,
    preprocess_otsu,
)
from src.phase11_ensemble.ensemble import _run_pass


# ── normalize_length ──────────────────────────────────────────────────────────


class TestNormalizeLength:
    def test_shorter_list_padded(self):
        result = normalize_length(["100", "200"], 4)
        assert result == ["100", "200", "0", "0"]

    def test_longer_list_truncated(self):
        result = normalize_length(["1", "2", "3", "4", "5"], 3)
        assert result == ["1", "2", "3"]

    def test_exact_length_unchanged(self):
        votes = ["100", "200", "300"]
        result = normalize_length(votes, 3)
        assert result == ["100", "200", "300"]

    def test_empty_list_padded(self):
        result = normalize_length([], 3)
        assert result == ["0", "0", "0"]

    def test_expected_zero_returns_empty(self):
        result = normalize_length(["100", "200"], 0)
        assert result == []

    def test_single_item_padded(self):
        result = normalize_length(["999"], 3)
        assert result == ["999", "0", "0"]


# ── preprocess_otsu ────────────────────────────────────────────────────────────


class TestPreprocessOtsu:
    def _make_temp_png(self) -> Path:
        """Create a small white PNG for testing."""
        img = Image.fromarray(np.full((100, 100), 200, dtype=np.uint8))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        return Path(tmp.name)

    def test_returns_pil_image(self):
        path = self._make_temp_png()
        try:
            result = preprocess_otsu(path)
            assert isinstance(result, Image.Image)
        finally:
            path.unlink(missing_ok=True)

    def test_output_size_matches_input(self):
        img = Image.fromarray(np.full((80, 120), 128, dtype=np.uint8))
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        img.save(tmp.name)
        tmp.close()
        path = Path(tmp.name)
        try:
            result = preprocess_otsu(path)
            assert result.size == (120, 80)
        finally:
            path.unlink(missing_ok=True)

    def test_raises_for_missing_file(self):
        with pytest.raises(FileNotFoundError):
            preprocess_otsu("/nonexistent/path/fake.png")

    def test_output_is_grayscale_or_binary(self):
        """Otsu output should have 2 unique values (binary) after thresholding."""
        path = self._make_temp_png()
        try:
            result = preprocess_otsu(path)
            arr = np.array(result)
            unique_vals = np.unique(arr)
            assert len(unique_vals) <= 2  # binary: 0 and 255
        finally:
            path.unlink(missing_ok=True)


# ── fallback_tesseract ─────────────────────────────────────────────────────────


class TestFallbackTesseract:
    def test_returns_list(self):
        """Should always return a list, even when pytesseract is unavailable."""
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            tmp_path = f.name
        img = Image.fromarray(np.full((50, 50), 255, dtype=np.uint8))
        img.save(tmp_path)
        try:
            result = fallback_tesseract(tmp_path)
            assert isinstance(result, list)
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_returns_empty_when_pytesseract_missing(self):
        """When pytesseract import fails, returns [] gracefully."""
        with patch.dict(sys.modules, {"pytesseract": None}):
            result = fallback_tesseract("/any/path.png")
        assert result == []

    def test_filters_lines_with_correct_digit_count(self):
        """Lines with 3–7 digits are kept; others are dropped."""
        fake_text = "\n".join([
            "ab 12 cd",         # 2 digits — dropped
            "12345 abc",        # 5 digits — kept
            "1234567 xx",       # 7 digits — kept
            "12345678",         # 8 digits — dropped
            "abc def ghi",      # 0 digits — dropped
            "1 2 345",          # 3 digits — kept
        ])
        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.return_value = fake_text
        with patch.dict(sys.modules, {"pytesseract": mock_pytesseract}):
            result = fallback_tesseract("/fake/path.png")
        assert result == ["12345", "1234567", "12345"]

    def test_strips_non_digit_chars(self):
        """Non-digit characters are stripped from the returned strings."""
        fake_text = "abc 1,234 def"  # 4 digits → kept; digits only = "1234"
        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.return_value = fake_text
        with patch.dict(sys.modules, {"pytesseract": mock_pytesseract}):
            result = fallback_tesseract("/fake/path.png")
        assert result == ["1234"]

    def test_handles_tesseract_exception(self):
        """If pytesseract.image_to_string raises, returns []."""
        mock_pytesseract = MagicMock()
        mock_pytesseract.image_to_string.side_effect = RuntimeError("tesseract crashed")
        with patch.dict(sys.modules, {"pytesseract": mock_pytesseract}):
            result = fallback_tesseract("/fake/path.png")
        assert result == []


# ── apply_sanity_checks ────────────────────────────────────────────────────────


class TestApplySanityChecks:
    def test_empty_list(self):
        assert apply_sanity_checks([]) == []

    def test_clean_values_unchanged(self):
        votes = ["100", "2345", "56789"]
        assert apply_sanity_checks(votes) == votes

    def test_more_than_7_digits_replaced_with_zero(self):
        assert apply_sanity_checks(["12345678"]) == ["0"]

    def test_exactly_7_digits_kept(self):
        assert apply_sanity_checks(["1234567"]) == ["1234567"]

    def test_leading_zeros_stripped(self):
        assert apply_sanity_checks(["01234"]) == ["1234"]

    def test_multiple_leading_zeros_stripped(self):
        assert apply_sanity_checks(["00099"]) == ["99"]

    def test_plain_zero_unchanged(self):
        assert apply_sanity_checks(["0"]) == ["0"]

    def test_all_zeros_stripped_to_single_zero(self):
        assert apply_sanity_checks(["000"]) == ["0"]

    def test_mix_of_cases(self):
        votes = ["12345678", "01234", "0", "100", "1234567"]
        expected = ["0", "1234", "0", "100", "1234567"]
        assert apply_sanity_checks(votes) == expected


# ── ensemble_votes ─────────────────────────────────────────────────────────────


class TestEnsembleVotes:
    def test_empty_candidates_returns_zeros(self):
        result = ensemble_votes([], 4)
        assert result == ["0", "0", "0", "0"]

    def test_single_candidate_passthrough(self):
        votes = ["100", "200", "300"]
        result = ensemble_votes([(1.0, votes)], 3)
        assert result == votes

    def test_unanimous_two_candidates(self):
        cands = [(0.9, ["111", "222"]), (0.8, ["111", "222"])]
        result = ensemble_votes(cands, 2)
        assert result == ["111", "222"]

    def test_higher_confidence_wins_disagreement(self):
        # Pass 1: conf=0.9, votes=["111", "222"]
        # Pass 2: conf=0.1, votes=["999", "888"]
        cands = [(0.9, ["111", "222"]), (0.1, ["999", "888"])]
        result = ensemble_votes(cands, 2)
        assert result == ["111", "222"]

    def test_majority_wins_three_candidates(self):
        # Two passes agree on "111", one outlier says "999"
        cands = [
            (0.8, ["111", "200"]),
            (0.8, ["111", "200"]),
            (0.3, ["999", "200"]),
        ]
        result = ensemble_votes(cands, 2)
        assert result[0] == "111"

    def test_result_has_exact_expected_length(self):
        cands = [(1.0, ["100", "200", "300", "400", "500"])]
        result = ensemble_votes(cands, 3)
        assert len(result) == 3

    def test_length_normalization_applied(self):
        # One candidate has too few votes — should be padded then voted
        cands = [
            (0.8, ["100"]),          # only 1 row, expected 3 → padded to ["100","0","0"]
            (0.8, ["100", "200", "300"]),
        ]
        result = ensemble_votes(cands, 3)
        assert len(result) == 3

    def test_all_zero_candidates(self):
        cands = [(0.5, ["0", "0"]), (0.5, ["0", "0"])]
        result = ensemble_votes(cands, 2)
        assert result == ["0", "0"]

    def test_agree_bonus_favours_consensus(self):
        # 3 passes agree on "500", 1 outlier pass says "1" with high confidence
        # but the consensus of 3 should still prevail due to agree_bonus
        cands = [
            (1.0, ["1"]),
            (0.4, ["500"]),
            (0.4, ["500"]),
            (0.4, ["500"]),
        ]
        result = ensemble_votes(cands, 1)
        # "500" has agree_bonus = 3/4 = 0.75; "1" has agree_bonus = 1/4 = 0.25
        # "500" weight ≈ 0.4 * 1.0 * 1.75 * 3 = 2.10
        # "1"   weight ≈ 1.0 * 0.0 = 0  (value "1" has row_conf ≈ 0.2 but low conf wins "500")
        # Depends on row_confidence values but consensus should dominate
        assert result[0] == "500"


# ── _run_pass (internal helper) ────────────────────────────────────────────────


class TestRunPass:
    def test_failure_returns_zero_conf_and_empty_votes(self):
        """If Typhoon OCR raises, _run_pass returns (0.0, []) without re-raising."""
        with patch("src.phase11_ensemble.ensemble.run_typhoon_ocr", side_effect=RuntimeError("API down")):
            conf, votes = _run_pass("test pass", "fake_path.png", 10, api_key="fake_key")
        assert conf == 0.0
        assert votes == []

    def test_successful_pass_returns_positive_confidence(self):
        """A valid HTML table from Typhoon should yield positive confidence."""
        mock_html = """<table>
<tr><td>1</td><td>สมชาย</td><td>12345</td></tr>
<tr><td>2</td><td>สมหญิง</td><td>23456</td></tr>
<tr><td>3</td><td>สมศักดิ์</td><td>34567</td></tr>
</table>"""
        with patch("src.phase11_ensemble.ensemble.run_typhoon_ocr", return_value=mock_html):
            conf, votes = _run_pass("test pass", "fake_path.png", 3, api_key="fake_key")
        assert conf > 0.0
        assert len(votes) == 3
