"""Pipeline orchestrator — wires all phases together.

Each phase is imported and called in order.  Feature flags in config.py
let you skip or swap phases during development.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import (
    CHECKPOINT_FILE,
    PARALLEL_WORKERS,
    SUBMISSION_OUTPUT,
    SUBMISSION_TEMPLATE,
    USE_CHECKPOINT,
    IMAGES_DIR,
)
from src.phase1_mapping.mapping import DocumentGroup, build_inventory
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

    # Placeholder — each phase will fill this in
    results: dict[str, int] = {row.id: 0 for row in group.rows}

    # TODO Phase 2: table page detection
    # TODO Phase 3: image preprocessing
    # TODO Phase 4: vote column crop
    # TODO Phase 5: OCR
    # TODO Phase 6: parsing
    # TODO Phase 7: Thai cross-check
    # TODO Phase 8: normalization
    # TODO Phase 9: postprocess / correction
    # TODO Phase 10: confidence scoring
    # TODO Phase 11: ensemble voting
    # TODO Phase 12: anchor alignment
    # TODO Phase 13: final row alignment

    return results


def run_pipeline(
    template_path: str | Path = SUBMISSION_TEMPLATE,
    images_dir: str | Path = IMAGES_DIR,
    output_path: str | Path = SUBMISSION_OUTPUT,
    checkpoint_file: str | Path = CHECKPOINT_FILE,
) -> None:
    """Main entry point — runs the full pipeline and writes submission.csv."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Phase 1 — inventory
    groups, pages = build_inventory(template_path, images_dir)

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
        futures = {pool.submit(_process, dk): dk for dk in groups}
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
    logger.info("Done — submission written to %s", output_path)
