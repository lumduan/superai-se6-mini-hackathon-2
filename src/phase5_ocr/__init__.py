"""Phase 5 — Full-Page Typhoon OCR (Primary Path)."""

from .ocr import (
    OCR_CALL_SLEEP,
    TYPHOON_MAX_TOKENS,
    TYPHOON_OCR_MODEL,
    TYPHOON_OCR_URL,
    TYPHOON_REPETITION_PENALTY,
    TYPHOON_TASK_TYPE,
    TYPHOON_TEMPERATURE,
    TYPHOON_TOP_P,
    run_full_page_ocr,
    run_typhoon_ocr,
)

__all__ = [
    "OCR_CALL_SLEEP",
    "TYPHOON_MAX_TOKENS",
    "TYPHOON_OCR_MODEL",
    "TYPHOON_OCR_URL",
    "TYPHOON_REPETITION_PENALTY",
    "TYPHOON_TASK_TYPE",
    "TYPHOON_TEMPERATURE",
    "TYPHOON_TOP_P",
    "run_full_page_ocr",
    "run_typhoon_ocr",
]
