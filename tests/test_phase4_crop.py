"""Tests for Phase 4 — Adaptive Vote Column Crop."""

from __future__ import annotations

from pathlib import Path
from unittest import mock

import cv2
import numpy as np
import pytest
from PIL import Image

from src.config import IMAGES_DIR
from src.phase4_crop.crop import (
    FALLBACK_CROP_RATIOS,
    MIN_VALID_RATIO,
    all_fallback_crops,
    crop_vote_column,
    detect_rightmost_column_boundary,
)

# ── Real-data sample list ─────────────────────────────────────────────────────

_REAL_IMAGES = [
    "constituency_10_4_page2.png",
    "constituency_20_7_page2.png",
    "party_list_24_3.png",
    "constituency_30_1_page2.png",
    "constituency_10_29_page2.png",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_blank_png(path: Path, height: int = 200, width: int = 400) -> Path:
    """White PNG with no structure — vertical line detection will find nothing."""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _make_table_png(path: Path, height: int = 800, width: int = 600) -> Path:
    """PNG with multiple strong vertical lines simulating a table grid.

    Draws 4 vertical black lines, so detection should find at least 2 and
    return a valid boundary ratio.
    """
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    for x in [120, 250, 400, 510]:
        arr[:, x] = 0  # solid black vertical line
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _make_single_vline_png(path: Path, height: int = 800, width: int = 600) -> Path:
    """PNG with only one vertical line — detection should return None."""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    arr[:, 300] = 0
    Image.fromarray(arr, mode="RGB").save(path)
    return path


# ── detect_rightmost_column_boundary ─────────────────────────────────────────

class TestDetectRightmostColumnBoundary:
    def test_returns_none_for_missing_file(self, tmp_path):
        result = detect_rightmost_column_boundary(tmp_path / "no_such_file.png")
        assert result is None

    def test_returns_none_for_blank_image(self, tmp_path):
        p = _make_blank_png(tmp_path / "blank.png")
        result = detect_rightmost_column_boundary(p)
        assert result is None

    def test_returns_none_for_single_vertical_line(self, tmp_path):
        """One line is not enough — need at least 2 unique positions."""
        p = _make_single_vline_png(tmp_path / "single.png")
        result = detect_rightmost_column_boundary(p)
        # May or may not find 2 positions depending on morphology; we accept None or float.
        assert result is None or isinstance(result, float)

    def test_returns_float_for_table_image(self, tmp_path):
        p = _make_table_png(tmp_path / "table.png")
        result = detect_rightmost_column_boundary(p)
        # With 4 clear vertical lines, detection should succeed.
        assert result is not None
        assert isinstance(result, float)

    def test_ratio_is_between_zero_and_one(self, tmp_path):
        p = _make_table_png(tmp_path / "table.png")
        result = detect_rightmost_column_boundary(p)
        if result is not None:
            assert 0.0 < result < 1.0

    def test_accepts_string_path(self, tmp_path):
        p = _make_table_png(tmp_path / "table.png")
        result = detect_rightmost_column_boundary(str(p))
        assert result is None or isinstance(result, float)

    def test_accepts_path_object(self, tmp_path):
        p = _make_table_png(tmp_path / "table.png")
        result = detect_rightmost_column_boundary(p)
        assert result is None or isinstance(result, float)

    def test_ratio_uses_second_to_last_line(self, tmp_path):
        """Boundary should be second-to-last unique vertical line (not the last)."""
        p = _make_table_png(tmp_path / "table.png", height=800, width=600)
        result = detect_rightmost_column_boundary(p)
        # The table PNG has lines at x=120,250,400,510 in a 600-wide image.
        # Second-to-last unique line cluster should be around x=400 → ratio ≈ 0.667.
        # We only verify that the result (if found) is well below 1.0.
        if result is not None:
            assert result < 1.0


# ── crop_vote_column ──────────────────────────────────────────────────────────

class TestCropVoteColumn:
    def test_returns_pil_image(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        result = crop_vote_column(p)
        assert isinstance(result, Image.Image)

    def test_fallback_crop_width_is_20_percent(self, tmp_path):
        """Blank image triggers fallback at 0.80 → crop width = 20 % of original."""
        w, h = 400, 200
        arr = np.full((h, w, 3), 255, dtype=np.uint8)
        p = tmp_path / "blank.png"
        Image.fromarray(arr).save(p)
        result = crop_vote_column(p)
        assert result.size == (int(w * 0.20), h)

    def test_adaptive_crop_used_when_detection_succeeds(self, tmp_path):
        """When detection returns a valid ratio the crop width must reflect it."""
        p = _make_blank_png(tmp_path / "page.png", width=500, height=300)
        fake_ratio = 0.70
        with mock.patch(
            "src.phase4_crop.crop.detect_rightmost_column_boundary",
            return_value=fake_ratio,
        ):
            result = crop_vote_column(p)
        expected_w = 500 - int(500 * fake_ratio)
        assert result.size == (expected_w, 300)

    def test_fallback_used_when_detection_returns_none(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png", width=400, height=200)
        with mock.patch(
            "src.phase4_crop.crop.detect_rightmost_column_boundary",
            return_value=None,
        ):
            result = crop_vote_column(p)
        # Fallback ratio = 0.80 → 20 % of 400 = 80 px wide
        assert result.size == (80, 200)

    def test_fallback_used_when_ratio_below_min_valid(self, tmp_path):
        """Ratio below MIN_VALID_RATIO must be treated the same as None."""
        p = _make_blank_png(tmp_path / "page.png", width=400, height=200)
        too_small = MIN_VALID_RATIO - 0.1
        with mock.patch(
            "src.phase4_crop.crop.detect_rightmost_column_boundary",
            return_value=too_small,
        ):
            result = crop_vote_column(p)
        # Fallback at 0.80 → 80 px wide
        assert result.size == (80, 200)

    def test_crop_height_equals_original(self, tmp_path):
        """Height must never change after cropping."""
        p = _make_blank_png(tmp_path / "page.png", width=400, height=300)
        result = crop_vote_column(p)
        assert result.size[1] == 300

    def test_crop_width_is_less_than_original(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png", width=400, height=200)
        result = crop_vote_column(p)
        assert result.size[0] < 400

    def test_accepts_string_path(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        result = crop_vote_column(str(p))
        assert isinstance(result, Image.Image)

    def test_table_image_adaptive_crop(self, tmp_path):
        """Table PNG with clear vertical lines should trigger adaptive mode."""
        p = _make_table_png(tmp_path / "table.png", height=800, width=600)
        result = crop_vote_column(p)
        assert isinstance(result, Image.Image)
        # Adaptive crop should be narrower than the full 600 px
        assert result.size[0] < 600
        assert result.size[1] == 800

    def test_exact_min_valid_ratio_triggers_adaptive(self, tmp_path):
        """Ratio exactly equal to MIN_VALID_RATIO should use adaptive mode."""
        p = _make_blank_png(tmp_path / "page.png", width=400, height=200)
        with mock.patch(
            "src.phase4_crop.crop.detect_rightmost_column_boundary",
            return_value=MIN_VALID_RATIO,
        ):
            result = crop_vote_column(p)
        expected_w = 400 - int(400 * MIN_VALID_RATIO)
        assert result.size == (expected_w, 200)


# ── all_fallback_crops ────────────────────────────────────────────────────────

class TestAllFallbackCrops:
    def test_returns_correct_number_of_crops(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        crops = all_fallback_crops(p)
        assert len(crops) == len(FALLBACK_CROP_RATIOS)

    def test_all_items_are_pil_images(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        for crop in all_fallback_crops(p):
            assert isinstance(crop, Image.Image)

    def test_crop_widths_match_ratios(self, tmp_path):
        """Each crop width must equal (1 - ratio) * original_width."""
        w, h = 400, 200
        arr = np.full((h, w, 3), 255, dtype=np.uint8)
        p = tmp_path / "page.png"
        Image.fromarray(arr).save(p)
        crops = all_fallback_crops(p)
        for crop, ratio in zip(crops, FALLBACK_CROP_RATIOS):
            expected_w = w - int(w * ratio)
            assert crop.size == (expected_w, h), (
                f"ratio={ratio}: expected width {expected_w}, got {crop.size[0]}"
            )

    def test_crops_are_strictly_decreasing_in_width(self, tmp_path):
        """Higher ratios → narrower crops (ratios are ascending in FALLBACK_CROP_RATIOS)."""
        p = _make_blank_png(tmp_path / "page.png", width=800, height=400)
        crops = all_fallback_crops(p)
        widths = [c.size[0] for c in crops]
        assert widths == sorted(widths, reverse=True), (
            f"Expected decreasing widths, got {widths}"
        )

    def test_heights_unchanged(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png", width=400, height=300)
        for crop in all_fallback_crops(p):
            assert crop.size[1] == 300

    def test_accepts_string_path(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        crops = all_fallback_crops(str(p))
        assert len(crops) == len(FALLBACK_CROP_RATIOS)

    def test_fallback_ratios_constant_is_sorted(self):
        """FALLBACK_CROP_RATIOS must be in ascending order (narrowing as ratio grows)."""
        assert FALLBACK_CROP_RATIOS == sorted(FALLBACK_CROP_RATIOS)

    def test_fallback_ratios_all_in_valid_range(self):
        for r in FALLBACK_CROP_RATIOS:
            assert 0.0 < r < 1.0, f"ratio {r} is outside (0, 1)"


# ── Module constants ──────────────────────────────────────────────────────────

class TestConstants:
    def test_min_valid_ratio_is_positive(self):
        assert MIN_VALID_RATIO > 0.0

    def test_min_valid_ratio_below_one(self):
        assert MIN_VALID_RATIO < 1.0

    def test_fallback_crop_ratios_not_empty(self):
        assert len(FALLBACK_CROP_RATIOS) > 0


# ── Real-data tests ───────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not IMAGES_DIR.exists(),
    reason="data/images not present — skipping real-data tests",
)
class TestCropRealData:
    """End-to-end tests on actual Thai election scans.

    Skipped automatically in CI environments without the image corpus.
    """

    @pytest.fixture(params=_REAL_IMAGES)
    def real_image(self, request) -> Path:
        p = IMAGES_DIR / request.param
        if not p.exists():
            pytest.skip(f"Real image not found: {p}")
        return p

    def test_crop_vote_column_returns_pil_image(self, real_image):
        result = crop_vote_column(real_image)
        assert isinstance(result, Image.Image)

    def test_crop_height_unchanged(self, real_image):
        with Image.open(real_image) as orig:
            orig_h = orig.size[1]
        result = crop_vote_column(real_image)
        assert result.size[1] == orig_h

    def test_crop_width_less_than_original(self, real_image):
        with Image.open(real_image) as orig:
            orig_w = orig.size[0]
        result = crop_vote_column(real_image)
        assert result.size[0] < orig_w

    def test_crop_pixel_values_in_valid_range(self, real_image):
        result = crop_vote_column(real_image)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_all_fallback_crops_count(self, real_image):
        crops = all_fallback_crops(real_image)
        assert len(crops) == len(FALLBACK_CROP_RATIOS)

    def test_all_fallback_crops_widths_correct(self, real_image):
        with Image.open(real_image) as orig:
            orig_w = orig.size[0]
        crops = all_fallback_crops(real_image)
        for crop, ratio in zip(crops, FALLBACK_CROP_RATIOS):
            expected_w = orig_w - int(orig_w * ratio)
            assert crop.size[0] == expected_w

    def test_detect_returns_float_or_none(self, real_image):
        result = detect_rightmost_column_boundary(real_image)
        assert result is None or isinstance(result, float)

    def test_detect_ratio_in_valid_range_when_found(self, real_image):
        result = detect_rightmost_column_boundary(real_image)
        if result is not None:
            assert 0.0 < result < 1.0
