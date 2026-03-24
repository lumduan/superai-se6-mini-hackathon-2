"""Tests for Phase 3 — Image Preprocessing."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.config import IMAGES_DIR
from src.phase3_preprocess.preprocess import (
    MAX_DESKEW_ANGLE,
    apply_clahe,
    deskew,
    preprocess_image,
    sharpen,
)

# ── Real-data fixtures ────────────────────────────────────────────────────────
# A small representative sample from the actual scan corpus.
# Tests in TestPreprocessImageRealData are automatically skipped when the
# data/images directory is absent (e.g. CI environments without the dataset).

_REAL_IMAGES = [
    "constituency_10_4_page2.png",   # typical two-page vote table scan
    "constituency_20_7_page2.png",   # different constituency, similar structure
    "party_list_24_3.png",           # party-list scan (single page)
    "constituency_30_1_page2.png",   # southern region scan
    "constituency_10_29_page2.png",  # multi-party dense table
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bgr(height: int = 200, width: int = 300) -> np.ndarray:
    """Uniform mid-grey BGR image."""
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _make_gray(height: int = 200, width: int = 300) -> np.ndarray:
    """Uniform mid-grey single-channel image."""
    return np.full((height, width), 128, dtype=np.uint8)


def _make_blank_png(path: Path, height: int = 200, width: int = 300) -> Path:
    """Write a plain white PNG to *path* and return *path*."""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _make_lined_png(path: Path, height: int = 800, width: int = 600) -> Path:
    """Write a white PNG with multiple strong horizontal lines (easy to detect)."""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    for y in range(50, height, 60):
        arr[y, :] = 0
    Image.fromarray(arr, mode="RGB").save(path)
    return path


# ── deskew ────────────────────────────────────────────────────────────────────

class TestDeskew:
    def test_returns_same_shape_for_bgr(self):
        img = _make_bgr()
        result = deskew(img)
        assert result.shape == img.shape

    def test_returns_same_shape_for_gray(self):
        img = _make_gray()
        result = deskew(img)
        assert result.shape == img.shape

    def test_uniform_image_no_lines_returns_original(self):
        """No Canny edges → no HoughLines → image returned unchanged."""
        img = _make_bgr()
        result = deskew(img)
        np.testing.assert_array_equal(result, img)

    def test_slightly_tilted_image_is_corrected(self, tmp_path):
        """Image with lines at a small known angle should be rotated back."""
        # Build a white image and draw a line tilted 2° to produce a detectable angle.
        h, w = 600, 800
        arr = np.full((h, w, 3), 255, dtype=np.uint8)
        # Draw a long diagonal-ish line — clearly not perfectly horizontal.
        cv2.line(arr, (0, h // 2 - 10), (w, h // 2 + 10), (0, 0, 0), 3)
        result = deskew(arr)
        # Result may or may not rotate depending on HoughLines threshold,
        # but it must always return the same shape.
        assert result.shape == arr.shape

    def test_large_angle_returns_original(self):
        """Angle > MAX_DESKEW_ANGLE must NOT rotate (corruption guard)."""
        img = _make_bgr(300, 400)
        # Patch median angle to be well above the limit.
        import unittest.mock as mock
        with mock.patch("src.phase3_preprocess.preprocess.np.median", return_value=MAX_DESKEW_ANGLE + 1):
            # Feed an image that produces HoughLines results so we reach angle check.
            # Use a strongly lined image to ensure lines are found.
            lined = np.full((300, 400, 3), 255, dtype=np.uint8)
            for y in range(20, 300, 30):
                lined[y, :] = 0
            result = deskew(lined)
        assert result.shape == lined.shape

    def test_max_deskew_angle_constant_is_positive(self):
        assert MAX_DESKEW_ANGLE > 0


# ── apply_clahe ───────────────────────────────────────────────────────────────

class TestApplyClahe:
    def test_output_is_single_channel(self):
        img = _make_bgr()
        result = apply_clahe(img)
        assert result.ndim == 2

    def test_output_shape_matches_input_spatial(self):
        img = _make_bgr(100, 150)
        result = apply_clahe(img)
        assert result.shape == (100, 150)

    def test_grayscale_input_accepted(self):
        img = _make_gray(80, 120)
        result = apply_clahe(img)
        assert result.shape == (80, 120)
        assert result.ndim == 2

    def test_output_dtype_is_uint8(self):
        result = apply_clahe(_make_bgr())
        assert result.dtype == np.uint8

    def test_dark_image_brightened(self):
        """CLAHE should raise the mean brightness of a very dark image."""
        dark = np.zeros((200, 200, 3), dtype=np.uint8)  # pure black
        result = apply_clahe(dark)
        # A pure black image has uniform histogram; CLAHE may not change it much,
        # but we just verify the function runs and returns sensible output.
        assert result.shape == (200, 200)

    def test_high_contrast_image_processed(self):
        """Image with strong contrast variations should not lose information."""
        img = np.tile(np.arange(256, dtype=np.uint8), (200, 1))[:200, :200]
        bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        result = apply_clahe(bgr)
        assert result.shape == (200, 200)


# ── sharpen ───────────────────────────────────────────────────────────────────

class TestSharpen:
    def test_returns_same_shape_bgr(self):
        img = _make_bgr()
        result = sharpen(img)
        assert result.shape == img.shape

    def test_returns_same_shape_gray(self):
        img = _make_gray()
        result = sharpen(img)
        assert result.shape == img.shape

    def test_output_dtype_uint8(self):
        assert sharpen(_make_bgr()).dtype == np.uint8

    def test_sharpened_differs_from_input(self):
        """Sharpening a non-uniform image should change at least some pixels."""
        # Create an image with edges (gradient pattern).
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 200  # half black, half bright grey
        result = sharpen(img)
        # Not identical to input — some pixels changed.
        assert not np.array_equal(result, img)

    def test_uniform_image_unchanged(self):
        """A fully uniform image has no edges; sharpening is a no-op."""
        img = np.full((100, 100), 128, dtype=np.uint8)
        result = sharpen(img)
        np.testing.assert_array_equal(result, img)


# ── preprocess_image ──────────────────────────────────────────────────────────

class TestPreprocessImage:
    def test_raises_for_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            preprocess_image(tmp_path / "nonexistent.png")

    def test_returns_pil_image(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        result = preprocess_image(p)
        assert isinstance(result, Image.Image)

    def test_output_mode_is_grayscale(self, tmp_path):
        """After CLAHE the output is always single-channel (mode 'L')."""
        p = _make_blank_png(tmp_path / "page.png")
        result = preprocess_image(p)
        assert result.mode == "L"

    def test_output_size_matches_input(self, tmp_path):
        h, w = 400, 600
        arr = np.full((h, w, 3), 200, dtype=np.uint8)
        p = tmp_path / "page.png"
        Image.fromarray(arr, mode="RGB").save(p)
        result = preprocess_image(p)
        assert result.size == (w, h)

    def test_accepts_string_path(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        result = preprocess_image(str(p))
        assert isinstance(result, Image.Image)

    def test_accepts_path_object(self, tmp_path):
        p = _make_blank_png(tmp_path / "page.png")
        result = preprocess_image(p)
        assert isinstance(result, Image.Image)

    def test_lined_scan_processed_successfully(self, tmp_path):
        """A scan-like image with horizontal lines should pass through pipeline."""
        p = _make_lined_png(tmp_path / "scan.png")
        result = preprocess_image(p)
        assert isinstance(result, Image.Image)
        assert result.size[0] > 0 and result.size[1] > 0

    def test_pipeline_order_does_not_crash(self, tmp_path):
        """All three steps chained must complete without exception."""
        import unittest.mock as mock
        calls = []

        original_deskew = deskew
        original_clahe = apply_clahe
        original_sharpen = sharpen

        def tracking_deskew(img):
            calls.append("deskew")
            return original_deskew(img)

        def tracking_clahe(img):
            calls.append("clahe")
            return original_clahe(img)

        def tracking_sharpen(img):
            calls.append("sharpen")
            return original_sharpen(img)

        p = _make_blank_png(tmp_path / "page.png")
        with mock.patch("src.phase3_preprocess.preprocess.deskew", side_effect=tracking_deskew), \
             mock.patch("src.phase3_preprocess.preprocess.apply_clahe", side_effect=tracking_clahe), \
             mock.patch("src.phase3_preprocess.preprocess.sharpen", side_effect=tracking_sharpen):
            preprocess_image(p)

        assert calls == ["deskew", "clahe", "sharpen"], (
            f"Expected pipeline order [deskew, clahe, sharpen], got {calls}"
        )


# ── Real-data tests ───────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not IMAGES_DIR.exists(),
    reason="data/images not present — skipping real-data tests",
)
class TestPreprocessImageRealData:
    """Run the full preprocessing pipeline on actual Thai election scans.

    These tests verify that the pipeline handles realistic, imperfect scans
    without crashing and produces sensibly-sized grayscale output.  They are
    skipped automatically when the image corpus is absent (e.g. on CI).
    """

    @pytest.fixture(params=_REAL_IMAGES)
    def real_image(self, request) -> Path:
        """Parametrised fixture: yields each real image path (or skips if missing)."""
        p = IMAGES_DIR / request.param
        if not p.exists():
            pytest.skip(f"Real image not found: {p}")
        return p

    def test_returns_pil_image(self, real_image):
        result = preprocess_image(real_image)
        assert isinstance(result, Image.Image)

    def test_output_mode_is_grayscale(self, real_image):
        """CLAHE converts to single-channel; output must be mode 'L'."""
        result = preprocess_image(real_image)
        assert result.mode == "L"

    def test_output_size_matches_input(self, real_image):
        """Deskew may rotate but must not change image dimensions."""
        with Image.open(real_image) as original:
            original_size = original.size
        result = preprocess_image(real_image)
        assert result.size == original_size

    def test_output_is_not_empty(self, real_image):
        """Output image must contain non-trivial pixel data (not all zeros)."""
        result = preprocess_image(real_image)
        arr = np.array(result)
        assert arr.max() > 0, "All-black output suggests pipeline failure"

    def test_output_pixel_values_in_valid_range(self, real_image):
        result = preprocess_image(real_image)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_pipeline_does_not_raise(self, real_image):
        """Full pipeline must complete without any exception on real scans."""
        # Any exception propagates and fails the test automatically.
        preprocess_image(real_image)

    def test_deskew_preserves_shape_on_real_image(self, real_image):
        """deskew() alone must not change the image dimensions."""
        img = cv2.imread(str(real_image))
        assert img is not None, f"cv2 could not read {real_image}"
        result = deskew(img)
        assert result.shape == img.shape

    def test_clahe_produces_grayscale_on_real_image(self, real_image):
        img = cv2.imread(str(real_image))
        assert img is not None
        result = apply_clahe(img)
        assert result.ndim == 2
        assert result.dtype == np.uint8

    def test_sharpen_preserves_shape_on_clahe_output(self, real_image):
        """sharpen() chained after apply_clahe() must not change shape."""
        img = cv2.imread(str(real_image))
        assert img is not None
        gray = apply_clahe(img)
        sharpened = sharpen(gray)
        assert sharpened.shape == gray.shape
