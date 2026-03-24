"""Tests for Phase 5 — Typhoon OCR Extraction.

All network calls are mocked so these tests run offline without a real API key.
The ``sleep_between_calls=0`` kwarg is passed to every ``run_typhoon_ocr`` call
so tests do not stall on artificial sleeps.

Real-data tests (``TestOcrRealData``) require actual Thai election scan images
under ``data/images/`` **and** a valid ``TYPHOON_OCR_API_KEY`` environment
variable.  They are skipped automatically when either is absent.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
import requests
from PIL import Image

from src.config import IMAGES_DIR
from src.phase5_ocr.ocr import (
    OCR_CALL_SLEEP,
    TYPHOON_MAX_TOKENS,
    TYPHOON_OCR_MODEL,
    TYPHOON_OCR_URL,
    TYPHOON_REPETITION_PENALTY,
    TYPHOON_TASK_TYPE,
    TYPHOON_TEMPERATURE,
    TYPHOON_TOP_P,
    _call_typhoon_api,
    _get_api_key,
    run_typhoon_ocr,
)


# ── Fixtures & helpers ────────────────────────────────────────────────────────


def _make_png(path: Path, width: int = 100, height: int = 50) -> Path:
    """Create a small white PNG at *path*."""
    arr = np.full((height, width, 3), 255, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)
    return path


def _api_response(text: str, success: bool = True) -> dict:
    """Build a minimal mock Typhoon OCR JSON response."""
    if success:
        return {
            "results": [
                {
                    "success": True,
                    "message": {
                        "choices": [
                            {"message": {"content": text}}
                        ]
                    },
                }
            ]
        }
    return {"results": [{"success": False, "error": "simulated error"}]}


def _mock_response(json_body: dict, status_code: int = 200) -> mock.MagicMock:
    """Return a mock requests.Response with the given body and status."""
    resp = mock.MagicMock(spec=requests.Response)
    resp.status_code = status_code
    resp.json.return_value = json_body
    if status_code >= 400:
        resp.raise_for_status.side_effect = requests.HTTPError(
            response=resp, request=mock.MagicMock()
        )
    else:
        resp.raise_for_status.return_value = None
    return resp


# ── _get_api_key ──────────────────────────────────────────────────────────────


class TestGetApiKey:
    def test_returns_explicit_key(self):
        assert _get_api_key("my-key") == "my-key"

    def test_returns_env_var_key(self, monkeypatch):
        monkeypatch.setenv("TYPHOON_OCR_API_KEY", "env-key")
        assert _get_api_key() == "env-key"

    def test_explicit_key_takes_precedence(self, monkeypatch):
        monkeypatch.setenv("TYPHOON_OCR_API_KEY", "env-key")
        assert _get_api_key("explicit-key") == "explicit-key"

    def test_raises_when_no_key(self, monkeypatch):
        monkeypatch.delenv("TYPHOON_OCR_API_KEY", raising=False)
        with pytest.raises(ValueError, match="TYPHOON_OCR_API_KEY"):
            _get_api_key()

    def test_raises_on_empty_string(self, monkeypatch):
        monkeypatch.delenv("TYPHOON_OCR_API_KEY", raising=False)
        with pytest.raises(ValueError):
            _get_api_key("")


# ── _call_typhoon_api ─────────────────────────────────────────────────────────


class TestCallTyphoonApi:
    def test_returns_extracted_text(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        mock_resp = _mock_response(_api_response("1234\n5678"))
        with mock.patch("requests.post", return_value=mock_resp):
            result = _call_typhoon_api(img_path, "test-key")
        assert result == "1234\n5678"

    def test_uses_correct_url(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        mock_resp = _mock_response(_api_response("ok"))
        with mock.patch("requests.post", return_value=mock_resp) as mock_post:
            _call_typhoon_api(img_path, "test-key")
        assert mock_post.call_args[0][0] == TYPHOON_OCR_URL

    def test_sends_bearer_token(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        mock_resp = _mock_response(_api_response("ok"))
        with mock.patch("requests.post", return_value=mock_resp) as mock_post:
            _call_typhoon_api(img_path, "secret-key")
        headers = mock_post.call_args.kwargs["headers"]
        assert headers["Authorization"] == "Bearer secret-key"

    def test_sends_correct_model_params(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        mock_resp = _mock_response(_api_response("ok"))
        with mock.patch("requests.post", return_value=mock_resp) as mock_post:
            _call_typhoon_api(img_path, "key")
        data = mock_post.call_args.kwargs["data"]
        assert data["model"] == TYPHOON_OCR_MODEL
        assert data["task_type"] == TYPHOON_TASK_TYPE
        assert data["max_tokens"] == str(TYPHOON_MAX_TOKENS)
        assert data["temperature"] == str(TYPHOON_TEMPERATURE)
        assert data["top_p"] == str(TYPHOON_TOP_P)
        assert data["repetition_penalty"] == str(TYPHOON_REPETITION_PENALTY)

    def test_raises_on_http_error(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        mock_resp = _mock_response({}, status_code=429)
        with mock.patch("requests.post", return_value=mock_resp):
            with pytest.raises(requests.HTTPError):
                _call_typhoon_api(img_path, "key")

    def test_empty_string_when_no_successful_pages(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        mock_resp = _mock_response(_api_response("", success=False))
        with mock.patch("requests.post", return_value=mock_resp):
            result = _call_typhoon_api(img_path, "key")
        assert result == ""

    def test_parses_json_natural_text(self, tmp_path):
        """When API content is JSON, use the natural_text field."""
        img_path = _make_png(tmp_path / "page.png")
        json_content = json.dumps({"natural_text": "hello from json", "other": "x"})
        body = {
            "results": [
                {
                    "success": True,
                    "message": {
                        "choices": [{"message": {"content": json_content}}]
                    },
                }
            ]
        }
        mock_resp = _mock_response(body)
        with mock.patch("requests.post", return_value=mock_resp):
            result = _call_typhoon_api(img_path, "key")
        assert result == "hello from json"

    def test_falls_back_to_raw_content_when_not_json(self, tmp_path):
        """Non-JSON content is returned as-is."""
        img_path = _make_png(tmp_path / "page.png")
        raw_text = "plain text content"
        mock_resp = _mock_response(_api_response(raw_text))
        with mock.patch("requests.post", return_value=mock_resp):
            result = _call_typhoon_api(img_path, "key")
        assert result == raw_text

    def test_multiple_pages_joined_with_newline(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        body = {
            "results": [
                {
                    "success": True,
                    "message": {"choices": [{"message": {"content": "page1"}}]},
                },
                {
                    "success": True,
                    "message": {"choices": [{"message": {"content": "page2"}}]},
                },
            ]
        }
        mock_resp = _mock_response(body)
        with mock.patch("requests.post", return_value=mock_resp):
            result = _call_typhoon_api(img_path, "key")
        assert result == "page1\npage2"


# ── run_typhoon_ocr ───────────────────────────────────────────────────────────


class TestRunTyphoonOcr:
    def test_returns_text_on_success(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        with mock.patch(
            "src.phase5_ocr.ocr._call_typhoon_api", return_value="12345"
        ):
            result = run_typhoon_ocr(img_path, api_key="k", sleep_between_calls=0)
        assert result == "12345"

    def test_accepts_pil_image(self, tmp_path):
        img = Image.fromarray(np.full((50, 100, 3), 255, dtype=np.uint8), mode="RGB")
        with mock.patch(
            "src.phase5_ocr.ocr._call_typhoon_api", return_value="pil-ok"
        ) as mock_call:
            result = run_typhoon_ocr(img, api_key="k", sleep_between_calls=0)
        assert result == "pil-ok"
        # Verify a temp-file path was passed (not the PIL object itself)
        call_path = mock_call.call_args[0][0]
        assert isinstance(call_path, Path)
        assert call_path.suffix == ".png"

    def test_temp_file_cleaned_up_after_pil_input(self, tmp_path):
        """Temp PNG created for a PIL Image must be deleted on success."""
        img = Image.fromarray(np.full((50, 100, 3), 255, dtype=np.uint8), mode="RGB")
        captured_path: list[Path] = []

        def capture_and_return(path, key):
            captured_path.append(Path(path))
            return "ok"

        with mock.patch("src.phase5_ocr.ocr._call_typhoon_api", side_effect=capture_and_return):
            run_typhoon_ocr(img, api_key="k", sleep_between_calls=0)

        assert len(captured_path) == 1
        assert not captured_path[0].exists(), "Temp file should have been deleted"

    def test_temp_file_cleaned_up_on_failure(self):
        """Temp PNG must be deleted even when all retries fail."""
        img = Image.fromarray(np.full((50, 100, 3), 255, dtype=np.uint8), mode="RGB")
        captured_path: list[Path] = []

        def capture_and_raise(path, key):
            captured_path.append(Path(path))
            raise RuntimeError("boom")

        with mock.patch("src.phase5_ocr.ocr._call_typhoon_api", side_effect=capture_and_raise):
            result = run_typhoon_ocr(img, api_key="k", retries=1, sleep_between_calls=0)

        assert result == ""
        assert len(captured_path) == 1
        assert not captured_path[0].exists(), "Temp file should have been deleted"

    def test_retries_on_exception(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        call_count = 0

        def flaky(path, key):
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise requests.ConnectionError("network error")
            return "recovered"

        with mock.patch("src.phase5_ocr.ocr._call_typhoon_api", side_effect=flaky):
            with mock.patch("time.sleep"):
                result = run_typhoon_ocr(img_path, api_key="k", retries=3, sleep_between_calls=0)

        assert result == "recovered"
        assert call_count == 3

    def test_returns_empty_string_when_all_retries_fail(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        with mock.patch(
            "src.phase5_ocr.ocr._call_typhoon_api",
            side_effect=requests.ConnectionError("fail"),
        ):
            with mock.patch("time.sleep"):
                result = run_typhoon_ocr(img_path, api_key="k", retries=3, sleep_between_calls=0)
        assert result == ""

    def test_retry_count_respected(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        call_count = 0

        def always_fail(path, key):
            nonlocal call_count
            call_count += 1
            raise RuntimeError("always fail")

        with mock.patch("src.phase5_ocr.ocr._call_typhoon_api", side_effect=always_fail):
            with mock.patch("time.sleep"):
                run_typhoon_ocr(img_path, api_key="k", retries=2, sleep_between_calls=0)

        assert call_count == 2

    def test_uses_env_var_api_key(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TYPHOON_OCR_API_KEY", "env-key")
        img_path = _make_png(tmp_path / "page.png")
        with mock.patch(
            "src.phase5_ocr.ocr._call_typhoon_api", return_value="ok"
        ) as mock_call:
            run_typhoon_ocr(img_path, sleep_between_calls=0)
        assert mock_call.call_args[0][1] == "env-key"

    def test_raises_when_no_api_key(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TYPHOON_OCR_API_KEY", raising=False)
        img_path = _make_png(tmp_path / "page.png")
        with pytest.raises(ValueError, match="TYPHOON_OCR_API_KEY"):
            run_typhoon_ocr(img_path)

    def test_accepts_string_path(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        with mock.patch(
            "src.phase5_ocr.ocr._call_typhoon_api", return_value="str-path-ok"
        ):
            result = run_typhoon_ocr(str(img_path), api_key="k", sleep_between_calls=0)
        assert result == "str-path-ok"

    def test_accepts_path_object(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        with mock.patch(
            "src.phase5_ocr.ocr._call_typhoon_api", return_value="path-ok"
        ):
            result = run_typhoon_ocr(img_path, api_key="k", sleep_between_calls=0)
        assert result == "path-ok"

    def test_sleep_called_before_each_attempt(self, tmp_path):
        """sleep_between_calls must fire once per attempt (not only retries)."""
        img_path = _make_png(tmp_path / "page.png")
        sleep_duration = 0.123
        sleep_calls: list[float] = []

        def fake_sleep(secs):
            sleep_calls.append(secs)

        with mock.patch("src.phase5_ocr.ocr._call_typhoon_api", return_value="ok"):
            with mock.patch("time.sleep", side_effect=fake_sleep):
                run_typhoon_ocr(
                    img_path, api_key="k", retries=1, sleep_between_calls=sleep_duration
                )

        assert sleep_duration in sleep_calls

    def test_returns_string_type(self, tmp_path):
        img_path = _make_png(tmp_path / "page.png")
        with mock.patch("src.phase5_ocr.ocr._call_typhoon_api", return_value="text"):
            result = run_typhoon_ocr(img_path, api_key="k", sleep_between_calls=0)
        assert isinstance(result, str)


# ── Module constants ──────────────────────────────────────────────────────────


class TestModuleConstants:
    def test_ocr_url_is_string(self):
        assert isinstance(TYPHOON_OCR_URL, str)
        assert TYPHOON_OCR_URL.startswith("https://")

    def test_model_is_string(self):
        assert isinstance(TYPHOON_OCR_MODEL, str)
        assert TYPHOON_OCR_MODEL  # non-empty

    def test_task_type_is_string(self):
        assert isinstance(TYPHOON_TASK_TYPE, str)

    def test_max_tokens_positive(self):
        assert TYPHOON_MAX_TOKENS > 0

    def test_temperature_in_range(self):
        assert 0.0 <= TYPHOON_TEMPERATURE <= 1.0

    def test_top_p_in_range(self):
        assert 0.0 <= TYPHOON_TOP_P <= 1.0

    def test_repetition_penalty_positive(self):
        assert TYPHOON_REPETITION_PENALTY > 0

    def test_ocr_call_sleep_non_negative(self):
        assert OCR_CALL_SLEEP >= 0


# ── Real-data tests ───────────────────────────────────────────────────────────

_REAL_IMAGES = [
    "constituency_10_4_page2.png",
    "constituency_20_7_page2.png",
    "party_list_24_3.png",
    "constituency_30_1_page2.png",
    "constituency_10_29_page2.png",
]

_HAS_API_KEY = bool(os.environ.get("TYPHOON_OCR_API_KEY"))


@pytest.mark.skipif(
    not IMAGES_DIR.exists() or not _HAS_API_KEY,
    reason=(
        "Skipping real-data OCR tests: "
        "data/images not present or TYPHOON_OCR_API_KEY not set"
    ),
)
class TestOcrRealData:
    """End-to-end OCR tests on actual Thai election scans.

    These tests make live Typhoon OCR API calls and are skipped automatically
    in CI environments where the image corpus or API key is absent.
    """

    @pytest.fixture(params=_REAL_IMAGES)
    def real_image(self, request) -> Path:
        p = IMAGES_DIR / request.param
        if not p.exists():
            pytest.skip(f"Real image not found: {p}")
        return p

    def test_returns_string(self, real_image):
        result = run_typhoon_ocr(real_image)
        assert isinstance(result, str)

    def test_returns_non_empty_text(self, real_image):
        """A valid scan should always yield some extracted text."""
        result = run_typhoon_ocr(real_image)
        assert len(result.strip()) > 0

    def test_result_contains_digits(self, real_image):
        """Thai election scans must contain Arabic digit vote counts."""
        result = run_typhoon_ocr(real_image)
        assert any(ch.isdigit() for ch in result), (
            f"No digits found in OCR output for {real_image.name}"
        )

    def test_accepts_pil_image_real(self, real_image):
        """OCR via PIL Image input must produce the same non-empty output."""
        pil_img = Image.open(real_image)
        result = run_typhoon_ocr(pil_img)
        assert isinstance(result, str)
        assert len(result.strip()) > 0

    def test_no_temp_files_left_behind(self, real_image):
        """Temp PNG created for a PIL Image must be cleaned up after the call."""
        pil_img = Image.open(real_image)
        import tempfile
        original_mkstemp = tempfile.mkstemp
        created: list[str] = []

        def tracking_mkstemp(*args, **kwargs):
            fd, name = original_mkstemp(*args, **kwargs)
            created.append(name)
            return fd, name

        with mock.patch("tempfile.mkstemp", side_effect=tracking_mkstemp):
            # Use NamedTemporaryFile path tracking via the ocr module
            pass

        # Direct approach: capture path via _call_typhoon_api spy
        captured_paths: list[Path] = []
        original_call = __import__(
            "src.phase5_ocr.ocr", fromlist=["_call_typhoon_api"]
        )._call_typhoon_api

        def spy_call(path, key):
            captured_paths.append(Path(path))
            return original_call(path, key)

        with mock.patch("src.phase5_ocr.ocr._call_typhoon_api", side_effect=spy_call):
            run_typhoon_ocr(pil_img)

        for p in captured_paths:
            # The temp file (if it was a PIL input) should be deleted
            if p != real_image:
                assert not p.exists(), f"Temp file {p} was not cleaned up"
