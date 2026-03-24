"""Phase 11 — Multi-pass Fallback OCR & Ensemble Voting."""

from src.phase11_ensemble.ensemble import (
    apply_sanity_checks,
    ensemble_votes,
    extract_votes_multipass,
    fallback_tesseract,
    normalize_length,
    preprocess_otsu,
)

__all__ = [
    "apply_sanity_checks",
    "ensemble_votes",
    "extract_votes_multipass",
    "fallback_tesseract",
    "normalize_length",
    "preprocess_otsu",
]
