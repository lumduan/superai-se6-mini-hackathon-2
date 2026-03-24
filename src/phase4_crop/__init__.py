"""Phase 4 — Adaptive Vote Column Crop."""

from .crop import (
    FALLBACK_CROP_RATIOS,
    MAX_VALID_RATIO,
    MIN_CROP_WIDTH_RATIO,
    MIN_VALID_RATIO,
    all_fallback_crops,
    crop_vote_column,
    detect_rightmost_column_boundary,
)

__all__ = [
    "FALLBACK_CROP_RATIOS",
    "MAX_VALID_RATIO",
    "MIN_CROP_WIDTH_RATIO",
    "MIN_VALID_RATIO",
    "all_fallback_crops",
    "crop_vote_column",
    "detect_rightmost_column_boundary",
]
