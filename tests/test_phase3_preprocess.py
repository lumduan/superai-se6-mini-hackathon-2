"""Tests for Phase 3 — Image Preprocessing.

Scope: sharpen only (unsharp mask).  No crop logic lives in this phase —
that belongs to Phase 4 (fallback only).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from src.config import IMAGES_DIR
from src.phase3_preprocess.preprocess import (
    preprocess_image,
    sharpen,
)

# ── Real-data fixtures ────────────────────────────────────────────────────────

_REAL_IMAGES = [
    "constituency_10_4_page2.png",
    "constituency_20_7_page2.png",
    "party_list_24_3.png",
    "constituency_30_1_page2.png",
    "constituency_10_29_page2.png",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_bgr(height: int = 200, width: int = 300) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def _make_gray(height: int = 200, width: int = 300) -> np.ndarray:
    return np.full((height, width), 128, dtype=np.uint8)


def _make_blank_png(path: Path, height: int = 200, width: int = 300) -> Path:
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _make_lined_png(path: Path, height: int = 800, width: int = 600) -> Path:
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    for y in range(50, height, 60):
        arr[y, :] = 0
    Image.fromarray(arr, mode="RGB").save(path)
    return path


# ── Boundary contract — no crop exported from Phase 3 ────────────────────────

class TestNoCropInPhase3:
    """Phase 3 must not export any crop functionality.

    Cropping lives exclusively in Phase 4.  These tests guard the boundary
    so a future edit cannot accidentally re-introduce crop logic here.
    """

    def test_no_crop_in_public_api(self):
        import src.phase3_preprocess as pkg

        for name in dir(pkg):
            assert "crop" not in name.lower(), (
                f"Phase 3 must not export crop-related names; found: {name!r}"
            )

    def test_preprocess_image_returns_full_size(self, tmp_path):
        """preprocess_image must not change image dimensions (no implicit crop)."""
        h, w = 400, 600
        arr = np.full((h, w, 3), 200, dtype=np.uint8)
        p = tmp_path / "page.png"
        Image.fromarray(arr, mode="RGB").save(p)
        result = preprocess_image(p)
        assert result.size == (w, h), (
            f"Expected ({w}, {h}), got {result.size} — phase 3 must not crop"
        )

    def test_preprocess_module_no_crop_import(self):
        """The preprocess module itself must not import from phase4_crop."""
        import importlib
        import sys

        mod = sys.modules.get("src.phase3_preprocess.preprocess")
        if mod is None:
            mod = importlib.import_module("src.phase3_preprocess.preprocess")

        crop_names = ["crop_vote_column", "detect_rightmost_column_boundary",
                      "all_fallback_crops", "FALLBACK_CROP_RATIOS"]
        for name in crop_names:
            assert not hasattr(mod, name), (
                f"Phase 3 preprocess module must not expose {name!r} from phase4_crop"
            )

    def test_no_deskew_or_clahe_exported(self):
        """deskew and apply_clahe are removed from Phase 3 scope."""
        import src.phase3_preprocess as pkg

        for removed in ("deskew", "apply_clahe", "MAX_DESKEW_ANGLE"):
            assert not hasattr(pkg, removed), (
                f"Phase 3 must not export {removed!r} — it was removed from scope"
            )


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
        img = np.zeros((100, 100), dtype=np.uint8)
        img[:, 50:] = 200
        result = sharpen(img)
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

    def test_output_mode_is_rgb(self, tmp_path):
        """Output is RGB (BGR converted to RGB)."""
        p = _make_blank_png(tmp_path / "page.png")
        result = preprocess_image(p)
        assert result.mode == "RGB"

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
        p = _make_lined_png(tmp_path / "scan.png")
        result = preprocess_image(p)
        assert isinstance(result, Image.Image)
        assert result.size[0] > 0 and result.size[1] > 0

    def test_pipeline_calls_sharpen(self, tmp_path):
        """preprocess_image must invoke sharpen."""
        import unittest.mock as mock

        calls = []
        original_sharpen = sharpen

        def tracking_sharpen(img):
            calls.append("sharpen")
            return original_sharpen(img)

        p = _make_blank_png(tmp_path / "page.png")
        with mock.patch("src.phase3_preprocess.preprocess.sharpen",
                        side_effect=tracking_sharpen):
            preprocess_image(p)

        assert calls == ["sharpen"], f"Expected ['sharpen'], got {calls}"


# ── Real-data tests ───────────────────────────────────────────────────────────

@pytest.mark.skipif(
    not IMAGES_DIR.exists(),
    reason="data/images not present — skipping real-data tests",
)
class TestPreprocessImageRealData:
    """Run the preprocessing pipeline on actual Thai election scans."""

    @pytest.fixture(params=_REAL_IMAGES)
    def real_image(self, request) -> Path:
        p = IMAGES_DIR / request.param
        if not p.exists():
            pytest.skip(f"Real image not found: {p}")
        return p

    def test_returns_pil_image(self, real_image):
        result = preprocess_image(real_image)
        assert isinstance(result, Image.Image)

    def test_output_mode_is_rgb(self, real_image):
        result = preprocess_image(real_image)
        assert result.mode == "RGB"

    def test_output_size_matches_input(self, real_image):
        with Image.open(real_image) as original:
            original_size = original.size
        result = preprocess_image(real_image)
        assert result.size == original_size

    def test_output_is_not_empty(self, real_image):
        result = preprocess_image(real_image)
        arr = np.array(result)
        assert arr.max() > 0

    def test_output_pixel_values_in_valid_range(self, real_image):
        result = preprocess_image(real_image)
        arr = np.array(result)
        assert arr.min() >= 0 and arr.max() <= 255

    def test_pipeline_does_not_raise(self, real_image):
        preprocess_image(real_image)

    def test_sharpen_preserves_shape_on_real_image(self, real_image):
        img = cv2.imread(str(real_image))
        assert img is not None
        result = sharpen(img)
        assert result.shape == img.shape
