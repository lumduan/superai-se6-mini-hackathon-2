"""Phase 5 — Full-Page Typhoon OCR (Primary Path).

Runs Typhoon OCR on the full preprocessed page — no crop.  The primary entry
point is ``run_full_page_ocr`` which applies Phase 3 preprocessing before
submitting the image to the API.  ``run_typhoon_ocr`` is also kept as the
lower-level helper (accepts PIL Image or path directly) for use in multi-pass
fallback (Phase 11).

Key behaviours
--------------
* ``run_full_page_ocr`` preprocesses the image via Phase 3 then calls
  ``run_typhoon_ocr`` — no crop.
* Accepts a PIL Image *or* a file path — PIL Images are written to a
  temporary PNG and cleaned up after the call.
* Inserts a ``sleep_between_calls`` pause before every API hit to stay inside
  the 2 req/s rate limit.
* Retries up to *retries* times with exponential back-off (1 s, 2 s, 4 s …)
  on any exception.
* API key is resolved from the ``api_key`` argument first, then the
  ``TYPHOON_OCR_API_KEY`` environment variable.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from pathlib import Path

import requests
from PIL import Image

from src.phase3_preprocess.preprocess import preprocess_image

logger = logging.getLogger(__name__)

# ── API constants ─────────────────────────────────────────────────────────────

TYPHOON_OCR_URL = "https://api.opentyphoon.ai/v1/ocr"
TYPHOON_OCR_MODEL = "typhoon-ocr"
TYPHOON_TASK_TYPE = "default"
TYPHOON_MAX_TOKENS = 16384
TYPHOON_TEMPERATURE = 0.1
TYPHOON_TOP_P = 0.6
TYPHOON_REPETITION_PENALTY = 1.2

# Minimum sleep between successive OCR calls to respect the 2 req/s rate limit.
OCR_CALL_SLEEP: float = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────


def _get_api_key(api_key: str | None = None) -> str:
    """Resolve the API key from argument or environment variable.

    Parameters
    ----------
    api_key:
        Explicit key.  ``None`` triggers env-var lookup.

    Returns
    -------
    Non-empty API key string.

    Raises
    ------
    ValueError
        When no key is available from either source.
    """
    key = api_key or os.environ.get("TYPHOON_OCR_API_KEY", "")
    if not key:
        raise ValueError(
            "Typhoon OCR API key not provided. "
            "Set the TYPHOON_OCR_API_KEY environment variable or pass api_key=."
        )
    return key


def _call_typhoon_api(path: str | Path, api_key: str) -> str:
    """Send *one* image to the Typhoon OCR endpoint and return extracted text.

    Parameters
    ----------
    path:
        Absolute path to the image file.
    api_key:
        Bearer token for the API.

    Returns
    -------
    Concatenated text extracted from all returned page results (may be ``""``
    if the API returns no successful pages).

    Raises
    ------
    requests.HTTPError
        When the API responds with a non-2xx status code.
    requests.RequestException
        On network-level errors (timeout, connection refused, …).
    """
    headers = {"Authorization": f"Bearer {api_key}"}
    data = {
        "model": TYPHOON_OCR_MODEL,
        "task_type": TYPHOON_TASK_TYPE,
        "max_tokens": str(TYPHOON_MAX_TOKENS),
        "temperature": str(TYPHOON_TEMPERATURE),
        "top_p": str(TYPHOON_TOP_P),
        "repetition_penalty": str(TYPHOON_REPETITION_PENALTY),
    }

    with open(path, "rb") as fh:
        response = requests.post(
            TYPHOON_OCR_URL,
            files={"file": fh},
            data=data,
            headers=headers,
        )

    response.raise_for_status()
    result = response.json()

    texts: list[str] = []
    for page_result in result.get("results", []):
        if page_result.get("success") and page_result.get("message"):
            content = page_result["message"]["choices"][0]["message"]["content"]
            try:
                parsed = json.loads(content)
                text = parsed.get("natural_text", content)
            except json.JSONDecodeError:
                text = content
            texts.append(text)
        elif not page_result.get("success"):
            logger.warning(
                "_call_typhoon_api: page error — %s",
                page_result.get("error", "unknown error"),
            )

    return "\n".join(texts)


# ── Public API ────────────────────────────────────────────────────────────────


def run_typhoon_ocr(
    image: Image.Image | str | Path,
    api_key: str | None = None,
    retries: int = 3,
    sleep_between_calls: float = OCR_CALL_SLEEP,
) -> str:
    """Run Typhoon OCR on a single image with automatic retry.

    Accepts either a :class:`PIL.Image.Image` or a file-system path.  PIL
    Images are written to a temporary PNG file before submission and cleaned
    up regardless of success or failure.

    A ``sleep_between_calls`` pause is applied *before each attempt* to stay
    within the 2 req/s rate limit.  After a failed attempt the function also
    waits ``2 ** attempt`` seconds (exponential back-off) before the next try.

    Parameters
    ----------
    image:
        PIL Image, or a ``str`` / ``Path`` pointing to the image file.
    api_key:
        Typhoon OCR API key.  Falls back to the ``TYPHOON_OCR_API_KEY`` env
        variable when ``None``.
    retries:
        Maximum number of attempts before giving up and returning ``""``.
    sleep_between_calls:
        Seconds to wait before every API call (rate-limit guard).  Set to
        ``0`` in tests to skip sleeping.

    Returns
    -------
    Extracted text string, or ``""`` when all attempts are exhausted.
    """
    key = _get_api_key(api_key)

    # Normalise input to a file path, writing a temp file for PIL Image inputs.
    cleanup = False
    if isinstance(image, (str, Path)):
        path = Path(image)
    else:
        tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
        tmp.close()
        path = Path(tmp.name)
        image.save(path)
        cleanup = True

    try:
        for attempt in range(retries):
            if attempt > 0:
                backoff = 2 ** (attempt - 1)
                logger.info(
                    "run_typhoon_ocr: attempt %d/%d failed — retrying in %ds",
                    attempt,
                    retries,
                    backoff,
                )
                time.sleep(backoff)

            time.sleep(sleep_between_calls)

            try:
                text = _call_typhoon_api(path, key)
                logger.debug(
                    "run_typhoon_ocr: success on attempt %d — %d chars returned",
                    attempt + 1,
                    len(text),
                )
                return text
            except Exception as exc:
                logger.warning(
                    "run_typhoon_ocr: attempt %d/%d — %s: %s",
                    attempt + 1,
                    retries,
                    type(exc).__name__,
                    exc,
                )
    finally:
        if cleanup and path.exists():
            path.unlink()

    logger.error("run_typhoon_ocr: all %d attempts exhausted — returning ''", retries)
    return ""


# ── Primary pipeline entry point ──────────────────────────────────────────────


def run_full_page_ocr(
    image_path: str | Path,
    api_key: str | None = None,
    retries: int = 3,
    sleep_between_calls: float = OCR_CALL_SLEEP,
) -> str:
    """Run Phase 3 preprocessing then Typhoon OCR on the full page.

    This is the **primary path** described in the master plan — no crop is
    performed.  The image is sharpened via Phase 3's ``preprocess_image``
    before being submitted to the Typhoon OCR API.

    Parameters
    ----------
    image_path:
        Path to the raw scan file (PNG or any OpenCV-readable format).
    api_key:
        Typhoon OCR API key.  Falls back to the ``TYPHOON_OCR_API_KEY`` env
        variable when ``None``.
    retries:
        Maximum number of OCR attempts before returning ``""``.
    sleep_between_calls:
        Seconds to wait before every API call (rate-limit guard).

    Returns
    -------
    Extracted text string (HTML table from Typhoon), or ``""`` on failure.
    """
    path = Path(image_path)
    logger.info("run_full_page_ocr: preprocessing %s", path.name)
    preprocessed = preprocess_image(path)
    logger.info("run_full_page_ocr: running OCR on preprocessed image")
    return run_typhoon_ocr(
        preprocessed,
        api_key=api_key,
        retries=retries,
        sleep_between_calls=sleep_between_calls,
    )
