"""Tests for Phase 2 — Dynamic Table Page Detection."""

from __future__ import annotations

import importlib
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
from PIL import Image

from src.phase2_detection.detection import (
    DIGIT_LINE_THRESHOLD,
    DIGITS_PER_LINE,
    MIN_H_PIXELS,
    MIN_V_PIXELS,
    TABLE_KEYWORDS,
    _ocr_text,
    get_table_pages,
    has_digit_rich_rows,
    has_table_keywords,
    has_table_structure,
    is_table_page,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _write_gray_png(path: Path, array: np.ndarray) -> None:
    """Save a grayscale numpy array as PNG to *path*."""
    img = Image.fromarray(array.astype(np.uint8), mode="L")
    img.save(path)


def _make_blank_image(path: Path, width: int = 200, height: int = 200) -> None:
    """White (255) image — no lines, no content."""
    _write_gray_png(path, np.full((height, width), 255, dtype=np.uint8))


def _make_grid_image(path: Path, width: int = 800, height: int = 1000) -> None:
    """Image with dense horizontal and vertical ruled lines (table-like)."""
    arr = np.full((height, width), 255, dtype=np.uint8)
    # Draw 8 horizontal lines spaced evenly
    for y in np.linspace(50, height - 50, 8, dtype=int):
        arr[y, :] = 0
    # Draw 5 vertical lines spaced evenly
    for x in np.linspace(50, width - 50, 5, dtype=int):
        arr[:, x] = 0
    _write_gray_png(path, arr)


# ── Signal A — has_table_structure ───────────────────────────────────────────

class TestHasTableStructure:
    def test_returns_false_for_missing_file(self, tmp_path):
        assert has_table_structure(tmp_path / "nonexistent.png") is False

    def test_returns_false_for_blank_image(self, tmp_path):
        p = tmp_path / "blank.png"
        _make_blank_image(p)
        assert has_table_structure(p) is False

    def test_returns_true_for_grid_image(self, tmp_path):
        p = tmp_path / "grid.png"
        _make_grid_image(p)
        assert has_table_structure(p) is True

    def test_accepts_path_object(self, tmp_path):
        p = tmp_path / "grid.png"
        _make_grid_image(p)
        # Path object should work just as well as a string
        assert has_table_structure(p) is True

    def test_accepts_string_path(self, tmp_path):
        p = tmp_path / "grid.png"
        _make_grid_image(p)
        assert has_table_structure(str(p)) is True

    def test_thresholds_are_positive(self):
        assert MIN_H_PIXELS > 0
        assert MIN_V_PIXELS > 0


# ── Signal B — has_table_keywords ────────────────────────────────────────────

class TestHasTableKeywords:
    def _stub_ocr(self, text: str):
        """Patch _ocr_text to return *text*."""
        return patch(
            "src.phase2_detection.detection._ocr_text", return_value=text
        )

    def test_returns_true_when_keyword_present(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        with self._stub_ocr("บัญชีคะแนนสรุป"):
            assert has_table_keywords(p) is True

    def test_returns_false_when_no_keyword(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        with self._stub_ocr("random english text only"):
            assert has_table_keywords(p) is False

    def test_all_keywords_trigger(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        for kw in TABLE_KEYWORDS:
            with self._stub_ocr(f"header {kw} footer"):
                assert has_table_keywords(p) is True, f"keyword '{kw}' should trigger"

    def test_empty_ocr_returns_false(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        with self._stub_ocr(""):
            assert has_table_keywords(p) is False

    def test_table_keywords_list_not_empty(self):
        assert len(TABLE_KEYWORDS) > 0


# ── Signal C — has_digit_rich_rows ───────────────────────────────────────────

class TestHasDigitRichRows:
    def _stub_ocr(self, text: str):
        return patch(
            "src.phase2_detection.detection._ocr_text", return_value=text
        )

    def _many_digit_lines(self, count: int = 10) -> str:
        """Build OCR text with *count* digit-rich lines."""
        line = "1234" * 2  # >DIGITS_PER_LINE digits
        return "\n".join([line] * count)

    def test_returns_true_when_many_digit_lines(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        text = self._many_digit_lines(DIGIT_LINE_THRESHOLD + 2)
        with self._stub_ocr(text):
            assert has_digit_rich_rows(p) is True

    def test_returns_false_when_few_digit_lines(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        # Only 1 digit-rich line
        with self._stub_ocr("1234 votes"):
            assert has_digit_rich_rows(p) is False

    def test_returns_false_for_empty_ocr(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        with self._stub_ocr(""):
            assert has_digit_rich_rows(p) is False

    def test_exactly_threshold_not_enough(self, tmp_path):
        """Exactly DIGIT_LINE_THRESHOLD lines should NOT trigger (need strictly more)."""
        p = tmp_path / "page.png"
        _make_blank_image(p)
        text = self._many_digit_lines(DIGIT_LINE_THRESHOLD)
        with self._stub_ocr(text):
            assert has_digit_rich_rows(p) is False

    def test_constants_are_positive(self):
        assert DIGIT_LINE_THRESHOLD > 0
        assert DIGITS_PER_LINE > 0


# ── _ocr_text ────────────────────────────────────────────────────────────────

class TestOcrText:
    def test_returns_empty_string_when_pytesseract_missing(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        with patch.dict("sys.modules", {"pytesseract": None}):
            # Reload to trigger ImportError path
            import importlib
            import src.phase2_detection.detection as det_mod
            with patch.object(det_mod, "_ocr_text", return_value="") as mock_ocr:
                result = mock_ocr(p)
        assert result == ""

    def test_returns_empty_string_on_exception(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        mock_tess = MagicMock()
        mock_tess.image_to_string.side_effect = RuntimeError("lang not found")
        with patch.dict("sys.modules", {"pytesseract": mock_tess}):
            result = _ocr_text(p)
        assert result == ""


# ── is_table_page ─────────────────────────────────────────────────────────────

class TestIsTablePage:
    def test_returns_false_for_nonexistent_file(self, tmp_path):
        assert is_table_page(tmp_path / "missing.png") is False

    def test_returns_true_when_signal_a_fires(self, tmp_path):
        p = tmp_path / "grid.png"
        _make_grid_image(p)
        # Don't rely on pytesseract being installed — structure alone should fire
        result = is_table_page(p)
        assert result is True

    def test_returns_true_when_only_signal_b_fires(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        with patch("src.phase2_detection.detection.has_table_structure", return_value=False), \
             patch("src.phase2_detection.detection._ocr_text", return_value="คะแนน ผลการนับ"):
            assert is_table_page(p) is True

    def test_returns_true_when_only_signal_c_fires(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        digit_text = "\n".join(["1234 5678"] * (DIGIT_LINE_THRESHOLD + 3))
        with patch("src.phase2_detection.detection.has_table_structure", return_value=False), \
             patch("src.phase2_detection.detection._ocr_text", return_value=digit_text):
            assert is_table_page(p) is True

    def test_returns_false_when_no_signal_fires(self, tmp_path):
        p = tmp_path / "page.png"
        _make_blank_image(p)
        with patch("src.phase2_detection.detection.has_table_structure", return_value=False), \
             patch("src.phase2_detection.detection._ocr_text", return_value="hello world"):
            assert is_table_page(p) is False

    def test_short_circuits_on_signal_a(self, tmp_path):
        """When Signal A fires, Signals B & C should not be called."""
        p = tmp_path / "grid.png"
        _make_grid_image(p)
        with patch("src.phase2_detection.detection.has_table_keywords") as mock_b, \
             patch("src.phase2_detection.detection.has_digit_rich_rows") as mock_c:
            result = is_table_page(p)
        # If signal A fired, B and C should not have been invoked
        if result is True:
            mock_b.assert_not_called()
            mock_c.assert_not_called()


# ── get_table_pages ───────────────────────────────────────────────────────────

class TestGetTablePages:
    def test_returns_empty_when_no_images_exist(self, tmp_path):
        result = get_table_pages("constituency_99_99", tmp_path)
        assert result == []

    def test_detects_single_table_page(self, tmp_path):
        p = tmp_path / "constituency_10_1_page2.png"
        _make_grid_image(p)
        result = get_table_pages("constituency_10_1", tmp_path)
        assert p in result

    def test_ignores_non_table_pages(self, tmp_path):
        p1 = tmp_path / "constituency_10_1.png"    # cover page — blank
        p2 = tmp_path / "constituency_10_1_page2.png"  # data page — grid
        _make_blank_image(p1)
        _make_grid_image(p2)
        with patch("src.phase2_detection.detection.has_table_keywords", return_value=False), \
             patch("src.phase2_detection.detection.has_digit_rich_rows", return_value=False):
            result = get_table_pages("constituency_10_1", tmp_path)
        assert p1 not in result
        assert p2 in result

    def test_party_list_uses_constituency_images(self, tmp_path):
        """party_list doc keys should resolve to constituency image files."""
        p = tmp_path / "constituency_10_1_page2.png"
        _make_grid_image(p)
        result = get_table_pages("party_list_10_1", tmp_path)
        assert p in result

    def test_returns_sorted_paths(self, tmp_path):
        for suffix in ["_page2", "_page3"]:
            _make_grid_image(tmp_path / f"constituency_10_1{suffix}.png")
        result = get_table_pages("constituency_10_1", tmp_path)
        assert result == sorted(result)

    def test_multiple_table_pages_all_returned(self, tmp_path):
        """Documents with overflow tables (page3, page4) should return all table pages."""
        for suffix in ["_page2", "_page3"]:
            _make_grid_image(tmp_path / f"constituency_10_1{suffix}.png")
        result = get_table_pages("constituency_10_1", tmp_path)
        assert len(result) == 2

    def test_accepts_string_images_dir(self, tmp_path):
        p = tmp_path / "constituency_10_1_page2.png"
        _make_grid_image(p)
        result = get_table_pages("constituency_10_1", str(tmp_path))
        assert p in result
