"""Phase 8 — Normalization & Hard Rule Overrides."""

from .normalize import (
    SOFT_LOW_VOTE,
    apply_hard_rules,
    apply_soft_rules,
    normalize_and_validate,
    normalize_votes,
)

__all__ = [
    "SOFT_LOW_VOTE",
    "apply_hard_rules",
    "apply_soft_rules",
    "normalize_and_validate",
    "normalize_votes",
]
