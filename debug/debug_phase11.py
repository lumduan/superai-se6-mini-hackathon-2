"""Debug script — Phase 11: Multi-pass Fallback OCR & Ensemble Voting.

Two operating modes:

1. **Offline (default)** — reads cached Phase 10 confidence CSVs from
   ``debug/phase10/`` and simulates ensemble voting on documents that are
   flagged as needing a fallback (``needs_fallback == True``).  The
   simulation applies ``normalize_length`` + ``apply_sanity_checks`` without
   making API calls; it is used to verify the helper logic and output format.

2. **Live** — if ``--live`` is passed (and ``TYPHOON_OCR_API_KEY`` is set),
   runs the full multi-pass Phase 11 pipeline on real images, displaying a
   per-pass breakdown and the final ensemble result.  This consumes API quota.

Usage
-----
    # Offline mode — simulate ensembling from Phase 10 confidence CSVs:
    uv run debug/debug_phase11.py

    # Live mode — run full multi-pass on real images:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase11.py --live

Output per document (live mode only)
--------------------------------------
- ``debug/phase11/<stem>_ensemble.csv`` — CSV with columns:
    position, vote_before, vote_after, changed

Console summary columns
-----------------------
    Document                #rows  doc_conf  fallback?  ensemble→  #changed
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
    needs_fallback,
)
from src.phase11_ensemble import (
    apply_sanity_checks,
    extract_votes_multipass,
    normalize_length,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE10_DIR = ROOT / "debug" / "phase10"
OUT_DIR = ROOT / "debug" / "phase11"

# Offline: read Phase 10 confidence CSVs
OFFLINE_SOURCES = [
    "constituency_10_1_page2_confidence.csv",
    "constituency_10_8_confidence.csv",
    "constituency_10_8_page2_confidence.csv",
    "constituency_10_11_confidence.csv",
    "constituency_10_11_page2_confidence.csv",
    "constituency_10_14_page2_confidence.csv",
]

# Live: run full multi-pass ensemble on these images
LIVE_IMAGES = [
    "constituency_10_1_page2.png",
    "constituency_10_8.png",
    "constituency_10_8_page2.png",
    "constituency_10_11.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _print_banner(mode: str) -> None:
    print(f"\n{'─'*80}")
    print(f"  Mode: {mode}")
    print(f"{'─'*80}")


def _print_header() -> None:
    print(
        f"\n  {'Document':<45}  {'Rows':>5}  {'doc_conf':>8}  "
        f"{'fallback?':>10}  {'#changed':>8}\n"
    )


def _print_summary(
    label: str,
    votes_before: list[str],
    votes_after: list[str],
    doc_conf: float,
) -> None:
    """Print a one-line summary for one document."""
    if not votes_before:
        print(f"  {label:<45}  NO ROWS")
        return

    expected = len(votes_before)
    fb = needs_fallback(votes_before, expected, doc_conf)
    changed = sum(a != b for a, b in zip(votes_before, votes_after))

    fb_str = "✗ FALLBACK" if fb else "✓ ok      "
    conf_bar = "█" * int(doc_conf * 10)

    print(
        f"  {label:<45}  {expected:>5}  "
        f"doc={doc_conf:.3f} [{conf_bar:<10}]  "
        f"{fb_str}  changed={changed:>4}"
    )

    # Show rows where the ensemble changed the value
    for i, (before, after) in enumerate(zip(votes_before, votes_after)):
        if before != after:
            print(f"    [CHG] row={i:>3}  before={before:<10}  after={after:<10}")


def _save_results(stem: str, votes_before: list[str], votes_after: list[str]) -> None:
    """Write ensemble results to a CSV file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}_ensemble.csv"
    rows = [
        {
            "position": i,
            "vote_before": before,
            "vote_after": after,
            "changed": "1" if before != after else "0",
        }
        for i, (before, after) in enumerate(zip(votes_before, votes_after))
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["position", "vote_before", "vote_after", "changed"]
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows to %s", len(rows), csv_path)


# ── Offline mode ───────────────────────────────────────────────────────────────


def run_offline() -> None:
    """Simulate ensemble normalization using Phase 10 confidence CSV output."""
    _print_banner("OFFLINE  (reading Phase 10 CSVs from debug/phase10/)")
    _print_header()

    found = 0
    for filename in OFFLINE_SOURCES:
        src = PHASE10_DIR / filename
        if not src.exists():
            print(f"  {filename:<45}  MISSING — run debug_phase10.py first")
            continue

        with src.open(encoding="utf-8", newline="") as fh:
            rows = list(csv.DictReader(fh))

        if not rows:
            print(f"  {filename:<45}  EMPTY")
            continue

        votes = [r.get("vote", "0") for r in rows]
        expected = len(votes)
        row_confs = [float(r.get("row_confidence", "0")) for r in rows]

        # Document confidence: proxy using average row confidence
        doc_conf = sum(row_confs) / len(row_confs) if row_confs else 0.0

        # In offline mode we cannot re-run OCR — simulate what sanity checks
        # would change: normalize_length (no-op — already correct length) then
        # apply_sanity_checks to see which values would be corrected.
        votes_after = apply_sanity_checks(normalize_length(votes, expected))

        stem = src.stem.replace("_confidence", "")
        _print_summary(filename, votes, votes_after, doc_conf)
        _save_results(stem, votes, votes_after)
        found += 1

    if found == 0:
        print("\n  No Phase 10 CSVs found — run debug_phase10.py first.\n")
    else:
        print(f"\n  Output saved to: {OUT_DIR}\n")


# ── Live mode ──────────────────────────────────────────────────────────────────


def run_live(api_key: str) -> None:
    """Run the full multi-pass Phase 11 pipeline on real images."""
    from src.phase2_detection import detect_table_region
    from src.phase5_ocr import run_typhoon_ocr
    from src.phase6_parse import parse_html_table
    from src.phase7_thai_crosscheck import cross_check_vote
    from src.phase8_normalize import apply_hard_rules, normalize_votes
    from src.phase9_postprocess import extract_total_from_html, total_based_correction

    _print_banner("LIVE  (running full multi-pass Phase 11 on real images)")
    _print_header()

    for image_name in LIVE_IMAGES:
        image_path = IMAGES_DIR / image_name
        if not image_path.exists():
            print(f"  {image_name:<45}  MISSING — image not found")
            continue

        stem = image_path.stem
        logger.info("Processing %s ...", image_name)

        try:
            # Determine expected row count from a quick Phase 5+6 scan first
            # so we know how many rows to target in the ensemble passes.
            from src.phase3_preprocess import preprocess_image
            preprocessed = preprocess_image(str(image_path))
            html = run_typhoon_ocr(preprocessed, api_key=api_key)
            parsed = parse_html_table(html)
            ocr_total = extract_total_from_html(html)

            votes_before: list[str] = []
            for _, raw, digits in parsed:
                crossed = cross_check_vote(raw, normalize_votes(digits))
                hard = apply_hard_rules(crossed)
                votes_before.append(hard)
            votes_before = total_based_correction(votes_before, ocr_total)

            expected = len(votes_before)
            doc_conf = compute_document_confidence(votes_before, expected, ocr_total)

            print(
                f"\n  {image_name} — Phase 5 baseline: "
                f"{expected} rows, conf={doc_conf:.3f}"
            )

            if not needs_fallback(votes_before, expected, doc_conf):
                print(f"      → No fallback needed (conf={doc_conf:.3f}).\n")
                votes_after = apply_sanity_checks(normalize_length(votes_before, expected))
            else:
                print(f"      → Triggering Phase 11 multi-pass ensemble ...\n")
                votes_after = extract_votes_multipass(
                    str(image_path), expected, api_key=api_key
                )

            _print_summary(image_name, votes_before, votes_after, doc_conf)
            _save_results(stem, votes_before, votes_after)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing %s", image_name)
            print(f"  {stem:<45}  ERROR — {exc}")

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
