"""Tests for runner.py — Phase 12 anchor alignment integration.

Covers the process_document function with mocked phase calls, verifying
that Phase 12 anchor alignment is correctly wired into the pipeline.

Tests use unittest.mock to avoid any real API calls or file I/O.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from src.phase1_mapping.mapping import DocumentGroup, SubmissionRow


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_group(doc_key: str, n_rows: int) -> DocumentGroup:
    """Build a DocumentGroup with *n_rows* SubmissionRows."""
    rows = [
        SubmissionRow(
            id=f"{doc_key}_{i}",
            id_type="constituency",
            province=10,
            district=1,
            row_number=i,
            doc_key=doc_key,
        )
        for i in range(1, n_rows + 1)
    ]
    return DocumentGroup(
        doc_key=doc_key,
        id_type="constituency",
        province=10,
        district=1,
        rows=rows,
    )


def _make_pages(n: int = 1) -> list[Path]:
    """Return a list of dummy page Paths."""
    return [Path(f"/fake/page{i}.png") for i in range(1, n + 1)]


# ── Test: no pages ────────────────────────────────────────────────────────────


class TestProcessDocumentNoPages:
    def test_returns_zeros_when_no_pages(self):
        from src.pipeline.runner import process_document

        group = _make_group("test_doc", 3)
        result = process_document("test_doc", group, [])

        assert result == {
            "test_doc_1": 0,
            "test_doc_2": 0,
            "test_doc_3": 0,
        }


# ── Test: Phase 2 table-page filtering ───────────────────────────────────────


class TestProcessDocumentPhase2:
    @patch("src.pipeline.runner.is_table_page", return_value=False)
    @patch("src.pipeline.runner.run_full_page_ocr", return_value="")
    def test_falls_back_to_all_pages_when_no_table_detected(
        self, mock_ocr, mock_detect
    ):
        """When Phase 2 detects no table pages, all pages are used as fallback."""
        from src.pipeline.runner import process_document

        group = _make_group("doc", 2)
        pages = _make_pages(2)

        result = process_document("doc", group, pages)

        # Phase 2 was called once per page
        assert mock_detect.call_count == 2
        # OCR was still attempted on fallback pages (returns "")
        assert mock_ocr.call_count == 2

    @patch("src.pipeline.runner.is_table_page", return_value=True)
    @patch("src.pipeline.runner.run_full_page_ocr", return_value="")
    def test_returns_zeros_when_ocr_empty(self, mock_ocr, mock_detect):
        """Empty OCR output → all votes remain 0."""
        from src.pipeline.runner import process_document

        group = _make_group("doc", 3)
        pages = _make_pages(1)

        result = process_document("doc", group, pages)
        assert all(v == 0 for v in result.values())


# ── Test: Phase 12 anchor alignment selection ─────────────────────────────────


class TestProcessDocumentPhase12:
    """Verify that Phase 12 anchor_align result is preferred when it has
    more non-zero votes than the sequential / ensemble base candidate."""

    def _common_patches(self):
        """Return a dict of patch targets → mock return values."""
        return {
            "src.pipeline.runner.is_table_page": True,
            "src.pipeline.runner.run_full_page_ocr": "<html>dummy</html>",
            "src.pipeline.runner.extract_total_from_html": None,
        }

    def test_anchor_preferred_when_more_nonzero(self):
        """Phase 12 result wins when it has more non-zero votes."""
        from src.pipeline.runner import process_document

        group = _make_group("doc", 3)
        pages = _make_pages(1)

        # Phase 6 returns 3 rows with candidate numbers
        parsed_rows = [
            (1, "11111", "11111"),
            (2, "22222", "22222"),
            (3, "33333", "33333"),
        ]
        # Sequential path produces 2 non-zero votes (one is "0")
        sequential = ["11111", "0", "33333"]
        # Anchor path produces 3 non-zero votes (better)
        anchor_result = ["11111", "22222", "33333"]

        with (
            patch("src.pipeline.runner.is_table_page", return_value=True),
            patch("src.pipeline.runner.run_full_page_ocr", return_value="<html/>"),
            patch("src.pipeline.runner.parse_html_table", return_value=parsed_rows),
            patch("src.pipeline.runner.extract_total_from_html", return_value=None),
            patch("src.pipeline.runner.cross_check_vote", side_effect=lambda raw, d: d),
            patch("src.pipeline.runner.apply_hard_rules", side_effect=lambda v: v),
            patch("src.pipeline.runner.validate_and_correct", return_value=(sequential, True)),
            patch("src.pipeline.runner.compute_document_confidence", return_value=0.9),
            patch("src.pipeline.runner.needs_fallback", return_value=False),
            patch("src.pipeline.runner.anchor_align", return_value=anchor_result),
        ):
            result = process_document("doc", group, pages)

        # Phase 12 result is selected (3 non-zero > 2 non-zero)
        assert result["doc_1"] == 11111
        assert result["doc_2"] == 22222
        assert result["doc_3"] == 33333

    def test_base_votes_preferred_when_anchor_has_fewer_nonzero(self):
        """Sequential/ensemble result wins when it has more non-zero votes."""
        from src.pipeline.runner import process_document

        group = _make_group("doc", 3)
        pages = _make_pages(1)

        parsed_rows = [
            (None, "11111", "11111"),
            (None, "22222", "22222"),
            (None, "33333", "33333"),
        ]
        # Sequential path: all 3 non-zero
        sequential = ["11111", "22222", "33333"]
        # Anchor path: only 1 non-zero (lost 2 rows with no anchor)
        anchor_result = ["11111", "0", "0"]

        with (
            patch("src.pipeline.runner.is_table_page", return_value=True),
            patch("src.pipeline.runner.run_full_page_ocr", return_value="<html/>"),
            patch("src.pipeline.runner.parse_html_table", return_value=parsed_rows),
            patch("src.pipeline.runner.extract_total_from_html", return_value=None),
            patch("src.pipeline.runner.cross_check_vote", side_effect=lambda raw, d: d),
            patch("src.pipeline.runner.apply_hard_rules", side_effect=lambda v: v),
            patch("src.pipeline.runner.validate_and_correct", return_value=(sequential, True)),
            patch("src.pipeline.runner.compute_document_confidence", return_value=0.9),
            patch("src.pipeline.runner.needs_fallback", return_value=False),
            patch("src.pipeline.runner.anchor_align", return_value=anchor_result),
        ):
            result = process_document("doc", group, pages)

        # Sequential result is kept (3 non-zero > 1 non-zero from anchor)
        assert result["doc_1"] == 11111
        assert result["doc_2"] == 22222
        assert result["doc_3"] == 33333

    def test_ensemble_used_as_base_when_fallback_triggered(self):
        """Phase 11 ensemble result is the base candidate when needs_fallback=True."""
        from src.pipeline.runner import process_document

        group = _make_group("doc", 3)
        pages = _make_pages(1)

        parsed_rows = [
            (1, "11111", "11111"),
            (2, "22222", "22222"),
            (3, "33333", "33333"),
        ]
        # Sequential low-quality
        sequential = ["111", "0", "0"]
        # Ensemble better
        ensemble = ["11111", "22222", "33333"]
        # Anchor same as ensemble
        anchor_result = ["11111", "22222", "33333"]

        with (
            patch("src.pipeline.runner.is_table_page", return_value=True),
            patch("src.pipeline.runner.run_full_page_ocr", return_value="<html/>"),
            patch("src.pipeline.runner.parse_html_table", return_value=parsed_rows),
            patch("src.pipeline.runner.extract_total_from_html", return_value=None),
            patch("src.pipeline.runner.cross_check_vote", side_effect=lambda raw, d: d),
            patch("src.pipeline.runner.apply_hard_rules", side_effect=lambda v: v),
            patch("src.pipeline.runner.validate_and_correct", return_value=(sequential, False)),
            patch("src.pipeline.runner.compute_document_confidence", return_value=0.3),
            patch("src.pipeline.runner.needs_fallback", return_value=True),
            patch("src.pipeline.runner.extract_votes_multipass", return_value=ensemble),
            patch("src.pipeline.runner.anchor_align", return_value=anchor_result),
            patch("src.pipeline.runner.USE_ENSEMBLE", True),
        ):
            result = process_document("doc", group, pages)

        assert result["doc_1"] == 11111
        assert result["doc_2"] == 22222
        assert result["doc_3"] == 33333

    def test_row_shift_corrected_by_phase12(self):
        """Anchor wins only when it has strictly more non-zero votes than sequential."""
        from src.pipeline.runner import process_document

        group = _make_group("doc", 5)
        pages = _make_pages(1)

        parsed_rows = [
            (1, "11111", "11111"),
            (3, "33333", "33333"),
            (4, "44444", "44444"),
            (5, "55555", "55555"),
        ]
        # Sequential: 4 non-zero
        sequential = ["11111", "33333", "44444", "55555", "0"]
        # Anchor: 5 non-zero (strictly more) — wins
        anchor_result = ["11111", "22222", "33333", "44444", "55555"]

        with (
            patch("src.pipeline.runner.is_table_page", return_value=True),
            patch("src.pipeline.runner.run_full_page_ocr", return_value="<html/>"),
            patch("src.pipeline.runner.parse_html_table", return_value=parsed_rows),
            patch("src.pipeline.runner.extract_total_from_html", return_value=None),
            patch("src.pipeline.runner.cross_check_vote", side_effect=lambda raw, d: d),
            patch("src.pipeline.runner.apply_hard_rules", side_effect=lambda v: v),
            patch("src.pipeline.runner.validate_and_correct", return_value=(sequential, True)),
            patch("src.pipeline.runner.compute_document_confidence", return_value=0.7),
            patch("src.pipeline.runner.needs_fallback", return_value=False),
            patch("src.pipeline.runner.anchor_align", return_value=anchor_result),
        ):
            result = process_document("doc", group, pages)

        # anchor_nonzero (5) > base_nonzero (4) → anchor wins
        assert result["doc_1"] == 11111
        assert result["doc_2"] == 22222
        assert result["doc_3"] == 33333
        assert result["doc_4"] == 44444
        assert result["doc_5"] == 55555

    def test_sequential_preferred_when_anchor_equal_nonzero(self):
        """Sequential wins when anchor has same non-zero count (physical order preserved)."""
        from src.pipeline.runner import process_document

        group = _make_group("doc", 5)
        pages = _make_pages(1)

        parsed_rows = [
            (1, "11111", "11111"),
            (3, "33333", "33333"),
            (4, "44444", "44444"),
            (5, "55555", "55555"),
        ]
        # Sequential: 4 non-zero (physical table order)
        sequential = ["11111", "33333", "44444", "55555", "0"]
        # Anchor: also 4 non-zero (reordered by ballot candidate number)
        anchor_result = ["11111", "0", "33333", "44444", "55555"]

        with (
            patch("src.pipeline.runner.is_table_page", return_value=True),
            patch("src.pipeline.runner.run_full_page_ocr", return_value="<html/>"),
            patch("src.pipeline.runner.parse_html_table", return_value=parsed_rows),
            patch("src.pipeline.runner.extract_total_from_html", return_value=None),
            patch("src.pipeline.runner.cross_check_vote", side_effect=lambda raw, d: d),
            patch("src.pipeline.runner.apply_hard_rules", side_effect=lambda v: v),
            patch("src.pipeline.runner.validate_and_correct", return_value=(sequential, True)),
            patch("src.pipeline.runner.compute_document_confidence", return_value=0.7),
            patch("src.pipeline.runner.needs_fallback", return_value=False),
            patch("src.pipeline.runner.anchor_align", return_value=anchor_result),
        ):
            result = process_document("doc", group, pages)

        # anchor_nonzero (4) == base_nonzero (4) → sequential wins (physical order)
        assert result["doc_1"] == 11111
        assert result["doc_2"] == 33333  # sequential order preserved
        assert result["doc_3"] == 44444
        assert result["doc_4"] == 55555
        assert result["doc_5"] == 0


# ── Test: run_pipeline limit / doc_keys params ────────────────────────────────


class TestRunPipelineFiltering:
    """Verify that limit and doc_keys correctly filter documents."""

    def _patched_pipeline_env(self):
        """Context managers for running run_pipeline without real I/O."""
        import contextlib

        @contextlib.contextmanager
        def ctx():
            with (
                patch(
                    "src.pipeline.runner.build_inventory",
                    return_value=(
                        {
                            "doc_a": _make_group("doc_a", 1),
                            "doc_b": _make_group("doc_b", 1),
                            "doc_c": _make_group("doc_c", 1),
                        },
                        {
                            "doc_a": _make_pages(),
                            "doc_b": _make_pages(),
                            "doc_c": _make_pages(),
                        },
                    ),
                ),
                patch("src.pipeline.runner.USE_CHECKPOINT", False),
                patch("src.pipeline.runner.process_document", return_value={"dummy_id": 0}),
                patch("src.pipeline.runner.save_csv"),
                patch("src.pipeline.runner.mark_done"),
                patch("src.pipeline.runner.save_checkpoint"),
            ):
                yield

        return ctx()

    def test_limit_restricts_document_count(self):
        from src.pipeline.runner import run_pipeline

        processed = []
        original_process = __import__(
            "src.pipeline.runner", fromlist=["process_document"]
        ).process_document

        with self._patched_pipeline_env():
            with patch(
                "src.pipeline.runner.process_document",
                side_effect=lambda key, *a, **kw: processed.append(key) or {},
            ):
                run_pipeline(limit=2)

        assert len(processed) == 2

    def test_doc_keys_restricts_to_specified_keys(self):
        from src.pipeline.runner import run_pipeline

        processed = []

        with self._patched_pipeline_env():
            with patch(
                "src.pipeline.runner.process_document",
                side_effect=lambda key, *a, **kw: processed.append(key) or {},
            ):
                run_pipeline(doc_keys=["doc_b"])

        assert processed == ["doc_b"]

    def test_unknown_doc_keys_are_skipped(self):
        from src.pipeline.runner import run_pipeline

        processed = []

        with self._patched_pipeline_env():
            with patch(
                "src.pipeline.runner.process_document",
                side_effect=lambda key, *a, **kw: processed.append(key) or {},
            ):
                run_pipeline(doc_keys=["doc_a", "nonexistent_key"])

        assert "doc_a" in processed
        assert "nonexistent_key" not in processed
