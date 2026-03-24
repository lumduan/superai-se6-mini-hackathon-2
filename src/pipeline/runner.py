"""Pipeline orchestrator — wires all phases together.

Each phase is imported and called in order.  Feature flags in config.py
let you skip or swap phases during development.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from src.config import (
    CHECKPOINT_FILE,
    PARALLEL_WORKERS,
    SUBMISSION_OUTPUT,
    SUBMISSION_TEMPLATE,
    USE_CHECKPOINT,
    USE_ENSEMBLE,
    IMAGES_DIR,
)
from src.phase1_mapping.mapping import DocumentGroup, build_inventory
from src.phase2_detection.detection import is_table_page
from src.phase5_ocr import run_full_page_ocr
from src.phase6_parse import parse_html_table
from src.phase7_thai_crosscheck import cross_check_vote
from src.phase8_normalize import apply_hard_rules, normalize_votes
from src.phase9_postprocess import extract_total_from_html, total_based_correction, validate_and_correct
from src.phase10_confidence import compute_document_confidence, needs_fallback
from src.phase11_ensemble import extract_votes_multipass
from src.phase12_anchor import anchor_align
from src.utils.checkpoint import is_done, load_checkpoint, mark_done, save_checkpoint
from src.utils.io import save_csv

import pandas as pd

logger = logging.getLogger(__name__)


def process_document(
    doc_key: str,
    group: DocumentGroup,
    pages: list[Path],
) -> dict[str, int]:
    """Process one document through all phases.  Returns {id: votes}."""
    logger.info("Processing %s (%d rows, %d pages)", doc_key, group.expected_row_count, len(pages))

    results: dict[str, int] = {row.id: 0 for row in group.rows}
    expected = group.expected_row_count

    if not pages:
        logger.warning("%s: no pages — returning zeros", doc_key)
        return results

    # Phase 2: filter to table pages only
    table_pages = [p for p in pages if is_table_page(p)]
    if not table_pages:
        logger.warning(
            "%s: no table pages detected — using all %d pages as fallback",
            doc_key, len(pages),
        )
        table_pages = pages

    # Accumulate parsed rows from all table pages
    # ParsedRow = (cand_num | None, raw_vote_cell, digit_string)
    all_parsed: list[tuple[Optional[int], str, str]] = []
    ocr_total: Optional[int] = None

    for page in table_pages:
        # Phase 5: full-page OCR (Phase 3 preprocessing is applied internally)
        html = run_full_page_ocr(page)
        if not html:
            logger.warning("%s: OCR returned empty for %s", doc_key, page.name)
            continue

        # Phase 6: parse HTML table → (cand_num|None, raw, digits) per row
        parsed = parse_html_table(html)
        all_parsed.extend(parsed)

        # Phase 9 helper: extract grand-total from first page that has one
        if ocr_total is None:
            ocr_total = extract_total_from_html(html)

    if not all_parsed:
        logger.warning("%s: no rows parsed from any page — returning zeros", doc_key)
        return results

    # Phase 7 + 8: sequential (naive) votes — no anchor correction
    sequential_votes: list[str] = []
    for _, raw, digits in all_parsed:
        v = cross_check_vote(raw, normalize_votes(digits))
        v = apply_hard_rules(v)
        sequential_votes.append(v)

    # Phase 9: distribution check + total-based correction
    sequential_votes, dist_ok = validate_and_correct(sequential_votes, ocr_total)
    if not dist_ok:
        logger.warning("%s: Phase 9 distribution check failed", doc_key)

    # Phase 10: confidence scoring
    confidence = compute_document_confidence(sequential_votes, expected, ocr_total)
    logger.info("%s: Phase 10 confidence=%.3f", doc_key, confidence)

    # Phase 11: multi-pass ensemble fallback when confidence is low
    ensemble_votes: Optional[list[str]] = None
    if USE_ENSEMBLE and needs_fallback(sequential_votes, expected, confidence):
        logger.info("%s: triggering Phase 11 ensemble fallback", doc_key)
        # Run ensemble on the primary (first) table page
        ensemble_votes = extract_votes_multipass(table_pages[0], expected)

    # Choose the base candidate: ensemble (if triggered) else sequential
    base_votes: list[str] = ensemble_votes if ensemble_votes is not None else sequential_votes

    # Phase 12: anchor alignment — re-align rows using candidate-number anchors
    # anchor_align accepts the same (cand_num|None, raw, digits) tuples from Phase 6
    anchor_votes = anchor_align(all_parsed, expected)

    anchor_nonzero = sum(v != "0" for v in anchor_votes)
    base_nonzero = sum(v != "0" for v in (base_votes[:expected] if len(base_votes) >= expected else base_votes))

    if anchor_nonzero > base_nonzero:
        logger.info(
            "%s: Phase 12 anchor alignment selected (anchor_nonzero=%d > base_nonzero=%d)",
            doc_key, anchor_nonzero, base_nonzero,
        )
        final_votes = anchor_votes
    else:
        logger.info(
            "%s: Phase 12 anchor alignment skipped (anchor_nonzero=%d <= base_nonzero=%d)",
            doc_key, anchor_nonzero, base_nonzero,
        )
        # Pad or truncate base_votes to exactly expected length
        if len(base_votes) < expected:
            base_votes = base_votes + ["0"] * (expected - len(base_votes))
        final_votes = base_votes[:expected]

    # Re-apply Phase 9 total-based correction on the final ordered votes.
    # Phase 9 was run earlier on sequential (unordered) votes; applying it
    # again here ensures the checksum correction targets the correct row
    # in the properly ordered final list.
    final_votes = list(total_based_correction(final_votes, ocr_total))

    # Map final votes back to submission row IDs
    for i, row in enumerate(group.rows):
        if i < len(final_votes):
            v = final_votes[i]
            results[row.id] = int(v) if isinstance(v, str) and v.isdigit() else 0

    return results


def run_pipeline(
    template_path: str | Path = SUBMISSION_TEMPLATE,
    images_dir: str | Path = IMAGES_DIR,
    output_path: str | Path = SUBMISSION_OUTPUT,
    checkpoint_file: str | Path = CHECKPOINT_FILE,
    limit: int | None = None,
    doc_keys: list[str] | None = None,
) -> None:
    """Main entry point — runs the full pipeline and writes submission.csv.

    Parameters
    ----------
    template_path:
        Path to the submission template CSV.
    images_dir:
        Directory containing PNG scan images.
    output_path:
        Where to write the output CSV.
    checkpoint_file:
        Path to the checkpoint JSON file (for resumable runs).
    limit:
        If set, process only the first *limit* documents (by sorted doc_key).
    doc_keys:
        If set, process only these specific document keys.
    """
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Phase 1 — build document inventory
    groups, pages = build_inventory(template_path, images_dir)
    all_keys = sorted(groups.keys())

    # Apply key filtering
    if doc_keys is not None:
        run_keys = [k for k in doc_keys if k in groups]
        missing = [k for k in doc_keys if k not in groups]
        if missing:
            logger.warning("doc_keys not found in inventory: %s", missing)
    elif limit is not None:
        run_keys = all_keys[:limit]
    else:
        run_keys = all_keys

    logger.info(
        "Running %d / %d documents%s",
        len(run_keys), len(all_keys),
        f" (limit={limit})" if limit else "",
    )

    # Checkpoint state
    state = load_checkpoint(checkpoint_file) if USE_CHECKPOINT else {"done": [], "results": {}}

    all_results: dict[str, int] = {}

    def _process(doc_key: str) -> tuple[str, dict]:
        if USE_CHECKPOINT and is_done(state, doc_key):
            logger.info("Skipping %s (already done)", doc_key)
            return doc_key, state["results"][doc_key]
        result = process_document(doc_key, groups[doc_key], pages[doc_key])
        return doc_key, result

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {pool.submit(_process, dk): dk for dk in run_keys}
        for future in as_completed(futures):
            doc_key, result = future.result()
            all_results.update(result)
            mark_done(state, doc_key, result)
            if USE_CHECKPOINT:
                save_checkpoint(checkpoint_file, state)

    # Build submission DataFrame
    rows = [{"id": k, "votes": v} for k, v in all_results.items()]
    submission = pd.DataFrame(rows)
    save_csv(submission, output_path)
    logger.info("Done — %d rows written to %s", len(submission), output_path)
