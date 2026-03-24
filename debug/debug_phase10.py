"""Debug script — Phase 10: Per-row Confidence Scoring.

Two operating modes:

1. **Offline (default)** — reads cached Phase 9 validation results (``.csv``
   files) from ``debug/phase9/`` and computes per-row and document-level
   confidence scores for each document.  No API quota consumed.

2. **Live** — if ``--live`` is passed (and ``TYPHOON_OCR_API_KEY`` is set), it
   runs Phase 2 detection → Phase 3 preprocessing → Phase 5 OCR → Phase 6
   parsing → Phase 7 cross-check → Phase 8 normalization → Phase 9 validation
   → Phase 10 confidence scoring in one shot on real images.

Usage
-----
    # Offline mode — score whatever Phase 9 already saved:
    uv run debug/debug_phase10.py

    # Live mode — run full pipeline up to Phase 10:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase10.py --live

Output per document
-------------------
- ``debug/phase10/<stem>_confidence.csv`` — CSV with columns:
    candidate, vote, row_confidence, position

Console summary columns
-----------------------
    Document                #rows  doc_conf  fallback?  row_conf_min  row_conf_avg
"""

from __future__ import annotations

import csv
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGES_DIR
from src.phase10_confidence import (
    compute_document_confidence,
    compute_row_confidence,
    needs_fallback,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE9_DIR = ROOT / "debug" / "phase9"
OUT_DIR = ROOT / "debug" / "phase10"

# Offline: read these Phase 9 validation CSVs from debug/phase9/
OFFLINE_SOURCES = [
    "constituency_10_1_page2_validated.csv",
    "constituency_10_8_validated.csv",
    "constituency_10_8_page2_validated.csv",
    "constituency_10_11_validated.csv",
    "constituency_10_11_page2_validated.csv",
    "constituency_10_14_page2_validated.csv",
]

# Live: run OCR + parse + cross-check + normalize + validate + score on these images
LIVE_IMAGES = [
    "constituency_10_1_page2.png",
    "constituency_10_8.png",
    "constituency_10_8_page2.png",
    "constituency_10_11.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _compute_scores(
    rows: list[dict],
    ocr_total: int | None = None,
) -> tuple[list[dict], float]:
    """Compute per-row and document-level confidence for a list of rows.

    Parameters
    ----------
    rows:
        List of dicts; must have a ``corrected`` (or ``normalized``) field and
        optionally a ``candidate`` field.
    ocr_total:
        Grand total extracted from the OCR output (may be ``None``).

    Returns
    -------
    Tuple of:
        - List of per-row dicts with keys:
            candidate, vote, row_confidence, position
        - Document confidence float [0.0, 1.0]
    """
    votes = [row.get("corrected") or row.get("normalized", "0") for row in rows]
    expected = len(votes)

    doc_conf = compute_document_confidence(votes, expected, ocr_total)

    results: list[dict] = []
    for i, row in enumerate(rows):
        vote = votes[i]
        row_conf = compute_row_confidence(vote, i, expected)
        results.append(
            {
                "candidate": row.get("candidate", ""),
                "vote": vote,
                "row_confidence": f"{row_conf:.4f}",
                "position": i,
            }
        )
    return results, doc_conf


def _save_results(stem: str, results: list[dict]) -> None:
    """Write confidence scores to a CSV file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}_confidence.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate", "vote", "row_confidence", "position"],
        )
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved %d rows to %s", len(results), csv_path)


def _print_summary(
    label: str,
    results: list[dict],
    doc_conf: float,
    ocr_total: int | None,
) -> None:
    """Print a one-line summary for a document."""
    if not results:
        print(f"  {label:<45}  NO ROWS")
        return

    votes = [r["vote"] for r in results]
    expected = len(votes)
    row_confs = [float(r["row_confidence"]) for r in results]
    min_conf = min(row_confs)
    avg_conf = sum(row_confs) / len(row_confs)
    fallback = needs_fallback(votes, expected, doc_conf)

    total_str = f"total={ocr_total:,}" if ocr_total is not None else "total=n/a"
    fb_str = "✗ FALLBACK" if fallback else "✓ ok      "
    conf_bar = "█" * int(doc_conf * 10)

    print(
        f"  {label:<45}  {len(results):>4} rows  "
        f"doc={doc_conf:.3f} [{conf_bar:<10}]  "
        f"{fb_str}  row_min={min_conf:.3f}  row_avg={avg_conf:.3f}  {total_str}"
    )

    # Highlight low-confidence rows
    low_rows = [r for r in results if float(r["row_confidence"]) < 0.5]
    for r in low_rows:
        print(
            f"    [LOW] cand={r['candidate'] or '?':>3}  "
            f"vote={r['vote']:<10}  row_conf={r['row_confidence']}"
        )


# ── Offline mode ───────────────────────────────────────────────────────────────


def run_offline() -> None:
    """Score rows from cached Phase 9 validation CSV files."""
    print(f"\n{'─'*80}")
    print("  Mode: OFFLINE  (reading Phase 9 CSVs from debug/phase9/)")
    print(f"{'─'*80}")
    print(
        f"\n  {'Document':<45}  {'Rows':>4}  doc_conf  fallback  row stats\n"
    )

    found = 0
    for filename in OFFLINE_SOURCES:
        src = PHASE9_DIR / filename
        if not src.exists():
            print(
                f"  {filename:<45}  MISSING — run debug_phase9.py first"
            )
            continue

        stem = src.stem.replace("_validated", "")
        with src.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        if not rows:
            print(f"  {filename:<45}  EMPTY")
            continue

        results, doc_conf = _compute_scores(rows)
        _print_summary(filename, results, doc_conf, ocr_total=None)
        _save_results(stem, results)
        found += 1

    if found == 0:
        print(
            "\n  No Phase 9 CSVs found — run debug_phase9.py first.\n"
        )
    else:
        print(f"\n  Output saved to: {OUT_DIR}\n")


# ── Live mode ──────────────────────────────────────────────────────────────────


def run_live(api_key: str) -> None:
    """Run the full pipeline up to Phase 10 on real images."""
    from src.phase2_detection import detect_table_region
    from src.phase3_preprocess import preprocess_image
    from src.phase5_ocr import ocr_page
    from src.phase6_parse import parse_votes_from_html, parse_votes_from_markdown
    from src.phase7_thai_crosscheck import crosscheck_votes
    from src.phase8_normalize import normalize_and_validate
    from src.phase9_postprocess import extract_total_from_html, validate_and_correct

    print(f"\n{'─'*80}")
    print("  Mode: LIVE  (running full pipeline up to Phase 10)")
    print(f"{'─'*80}")
    print(
        f"\n  {'Image':<35}  {'Rows':>4}  doc_conf  fallback  row stats\n"
    )

    for image_name in LIVE_IMAGES:
        image_path = IMAGES_DIR / image_name
        if not image_path.exists():
            print(f"  {image_name:<35}  MISSING — image not found")
            continue

        name = image_path.stem
        stem = name

        try:
            # Phase 2 — detect table region
            region = detect_table_region(str(image_path))

            # Phase 3 — preprocess
            preprocessed = preprocess_image(str(image_path), region)

            # Phase 5 — OCR
            ocr_result = ocr_page(preprocessed, api_key=api_key)
            html = ocr_result.get("html", "")
            markdown = ocr_result.get("markdown", "")

            # Phase 6 — parse
            if html:
                parsed_rows = parse_votes_from_html(html)
            else:
                parsed_rows = parse_votes_from_markdown(markdown)

            # Phase 7 — cross-check
            checked_rows = crosscheck_votes(parsed_rows)

            # Phase 8 — normalize
            norm_rows = [
                {
                    **r,
                    "normalized": normalize_and_validate(r.get("raw", r.get("vote", "0"))),
                }
                for r in checked_rows
            ]

            # Phase 9 — validate & correct
            votes_in = [r.get("normalized", "0") for r in norm_rows]
            ocr_total = extract_total_from_html(html) if html else None
            corrected_votes, _ = validate_and_correct(votes_in, ocr_total)

            # Build rows for Phase 10
            phase10_rows = [
                {
                    "candidate": r.get("candidate", ""),
                    "corrected": corrected_votes[i] if i < len(corrected_votes) else "0",
                    "normalized": votes_in[i] if i < len(votes_in) else "0",
                }
                for i, r in enumerate(norm_rows)
            ]

            # Phase 10 — confidence scoring
            results, doc_conf = _compute_scores(phase10_rows, ocr_total)

            _print_summary(name, results, doc_conf, ocr_total)
            _save_results(stem, results)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing %s", image_name)
            print(f"  {name:<35}  ERROR — {exc}")

    print(f"\n  Output saved to: {OUT_DIR}\n")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    live = "--live" in sys.argv

    if live:
        api_key = os.environ.get("TYPHOON_OCR_API_KEY", "")
        if not api_key:
            print(
                "\nERROR: TYPHOON_OCR_API_KEY is not set.\n"
                "Export it before running in live mode:\n"
                "  export TYPHOON_OCR_API_KEY=your_api_key_here\n"
            )
            sys.exit(1)
        run_live(api_key)
    else:
        run_offline()


if __name__ == "__main__":
    main()
