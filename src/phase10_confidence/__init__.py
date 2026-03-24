"""Phase 10 — Per-row Confidence Scoring."""

from .scorer import (
    MISMATCH_TOLERANCE,
    compute_document_confidence,
    compute_row_confidence,
    needs_fallback,
)

__all__ = [
    "MISMATCH_TOLERANCE",
    "compute_document_confidence",
    "compute_row_confidence",
    "needs_fallback",
]
