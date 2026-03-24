"""Debug script — Phase 9: Row Structure Validation & Total-based Correction.

Two operating modes:

1. **Offline (default)** — reads cached Phase 8 normalization results (``.csv``
   files) from ``debug/phase8/`` and runs Phase 9 distribution check and
   total-based correction on each document.  No API quota consumed.

2. **Live** — if ``--live`` is passed (and ``TYPHOON_OCR_API_KEY`` is set), it
   runs Phase 2 detection → Phase 3 preprocessing → Phase 5 OCR → Phase 6
   parsing → Phase 7 cross-check → Phase 8 normalization → Phase 9 validation
   in one shot on real images.

Usage
-----
    # Offline mode — validate whatever Phase 8 already saved:
    uv run debug/debug_phase9.py

    # Live mode — run full pipeline up to Phase 9:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase9.py --live

Output per document
-------------------
- ``debug/phase9/<stem>_validated.csv`` — CSV with columns:
    candidate, normalized, corrected, was_corrected, distribution_ok

Console summary columns
-----------------------
    Document                #rows  dist_ok  #corrected  ocr_total  computed_sum
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
from src.phase9_postprocess import (
    extract_total_from_html,
    is_reasonable_distribution,
    total_based_correction,
    validate_and_correct,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE8_DIR = ROOT / "debug" / "phase8"
OUT_DIR = ROOT / "debug" / "phase9"

# Offline: read these Phase 8 normalization CSVs from debug/phase8/
OFFLINE_SOURCES = [
    "constituency_10_1_page2_normalized.csv",
    "constituency_10_8_normalized.csv",
    "constituency_10_8_page2_normalized.csv",
    "constituency_10_11_normalized.csv",
    "constituency_10_11_page2_normalized.csv",
    "constituency_10_14_page2_normalized.csv",
]

# Live: run OCR + parse + cross-check + normalize + validate on these images
LIVE_IMAGES = [
    "constituency_10_1_page2.png",
    "constituency_10_8.png",
    "constituency_10_8_page2.png",
    "constituency_10_11.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run_validation(
    rows: list[dict],
    ocr_total: int | None = None,
) -> list[dict]:
    """Apply Phase 9 validation and total-based correction to a list of rows.

    Parameters
    ----------
    rows:
        List of dicts with at minimum a ``final`` field (the normalized vote
        from Phase 8) and a ``candidate`` field.
    ocr_total:
        Grand total extracted from the OCR output (may be ``None``).

    Returns
    -------
    List of dicts with keys:
        candidate, normalized, corrected, was_corrected, distribution_ok
    """
    votes = [row.get("final") or row.get("normalized", "0") for row in rows]

    corrected_votes, distribution_ok = validate_and_correct(votes, ocr_total)

    results: list[dict] = []
    for i, row in enumerate(rows):
        orig = votes[i]
        corr = corrected_votes[i] if i < len(corrected_votes) else orig
        results.append(
            {
                "candidate": row.get("candidate", ""),
                "normalized": orig,
                "corrected": corr,
                "was_corrected": "YES" if corr != orig else "",
                "distribution_ok": "YES" if distribution_ok else "NO",
            }
        )
    return results


def _save_results(stem: str, results: list[dict]) -> None:
    """Write validation results to a CSV file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}_validated.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate",
                "normalized",
                "corrected",
                "was_corrected",
                "distribution_ok",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved %d rows to %s", len(results), csv_path)


def _print_summary(
    label: str,
    results: list[dict],
    ocr_total: int | None,
) -> None:
    """Print a one-line summary for a document."""
    n_rows = len(results)
    corrected = [r for r in results if r["was_corrected"]]
    dist_ok = results[0]["distribution_ok"] == "YES" if results else False

    computed_sum = sum(
        int(r["normalized"])
        for r in results
        if r["normalized"].isdigit()
    )
    corrected_sum = sum(
        int(r["corrected"])
        for r in results
        if r["corrected"].isdigit()
    )

    dist_str = "✓ ok" if dist_ok else "✗ bad"
    total_str = f"total={ocr_total:,}" if ocr_total is not None else "total=n/a"
    sum_str = f"sum={computed_sum:,}"
    corr_sum_str = f"→{corrected_sum:,}" if corrected else ""
    corr_str = f"✎ {len(corrected)} corrected" if corrected else "  no corrections"

    print(
        f"  {label:<40} {n_rows:>5} rows  "
        f"dist={dist_str:<6}  {total_str:<18}  "
        f"{sum_str}{corr_sum_str:<14}  {corr_str}"
    )

    # Show detail for corrected rows
    for r in corrected:
        print(
            f"    cand={r['candidate'] or '?':>3}  "
            f"{r['normalized']:<12} → {r['corrected']:<12}  [CORRECTED]"
        )


# ── Offline mode ───────────────────────────────────────────────────────────────


def run_offline() -> None:
    """Validate rows from cached Phase 8 normalization CSV files."""
    print(f"\n{'─'*80}")
    print("  Mode: OFFLINE  (reading Phase 8 CSVs from debug/phase8/)")
    print(f"{'─'*80}")
    print(
        f"\n  {'Document':<40} {'Rows':>5}{'':4} Status\n"
    )

    found = 0
    for filename in OFFLINE_SOURCES:
        src = PHASE8_DIR / filename
        if not src.exists():
            print(
                f"  {filename:<40} MISSING — run debug_phase8.py first"
            )
            continue
        found += 1

        rows: list[dict] = []
        with src.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))

        stem = filename.replace("_normalized.csv", "")

        # Offline mode: no OCR total available from CSV alone.
        results = _run_validation(rows, ocr_total=None)
        _print_summary(stem, results, ocr_total=None)
        _save_results(stem, results)

    if found == 0:
        print(
            "\n  No Phase 8 CSV files found in debug/phase8/.\n"
            "  Run debug_phase8.py first, or use --live to call the OCR API.\n"
        )
        return

    print(f"\n  Output saved to: {OUT_DIR}\n")


# ── Live mode ──────────────────────────────────────────────────────────────────


def run_live(api_key: str) -> None:
    """Run Phase 2 → 3 → 5 → 6 → 7 → 8 → 9 on real images, then validate."""
    from src.phase2_detection.detection import is_table_page
    from src.phase5_ocr.ocr import run_full_page_ocr
    from src.phase6_parse.parser import parse_html_table
    from src.phase7_thai_crosscheck.crosscheck import (
        cross_check_vote,
        extract_thai_number_text,
    )
    from src.phase8_normalize import apply_hard_rules, apply_soft_rules, normalize_votes

    print(f"\n{'─'*80}")
    print("  Mode: LIVE  (OCR API → parse → cross-check → normalize → validate)")
    print(f"{'─'*80}")
    print(
        f"\n  {'Image':<35} {'Ph2':^5} {'OCR':^8} "
        f"{'Rows':>5}{'':4} Status\n"
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for name in LIVE_IMAGES:
        img_path = IMAGES_DIR / name
        if not img_path.exists():
            print(f"  {name:<35} MISSING image")
            continue

        stem = Path(name).stem

        # Phase 2 gate
        if not is_table_page(img_path):
            print(f"  {name:<35} ✗ not table     (skipped)")
            continue

        # Phase 5 — preprocess + OCR
        try:
            html = run_full_page_ocr(img_path, api_key=api_key, sleep_between_calls=0.5)
        except Exception as exc:
            logger.error("Phase 5 failed for %s: %s", name, exc)
            print(f"  {name:<35} ✓ TABLE    OCR ERROR: {exc}")
            continue

        # Extract OCR total before parsing (total row is skipped in parse_html_table)
        ocr_total = extract_total_from_html(html)

        # Phase 6 — parse
        parsed = parse_html_table(html)

        # Phase 7 → Phase 8 pipeline — build row dicts
        phase8_rows: list[dict] = []
        for cand, raw_cell, ocr_digits in parsed:
            thai_num = extract_thai_number_text(raw_cell)
            corrected_7 = cross_check_vote(raw_cell, ocr_digits)
            normalized = normalize_votes(corrected_7)
            final = apply_hard_rules(normalized)
            confidence = apply_soft_rules(normalized)
            phase8_rows.append(
                {
                    "candidate": str(cand) if cand is not None else "",
                    "final": final,
                    "normalized": normalized,
                    "confidence": f"{confidence:.2f}",
                }
            )

        # Phase 9 — validate + correct
        results = _run_validation(phase8_rows, ocr_total=ocr_total)
        n_corrected = sum(1 for r in results if r["was_corrected"])
        dist_ok = results[0]["distribution_ok"] == "YES" if results else False

        ocr_chars = len(html.strip())
        total_str = f"total={ocr_total:,}" if ocr_total is not None else "total=n/a"
        dist_str = "✓ ok" if dist_ok else "✗ bad"
        corr_str = f"✎ {n_corrected} corrected" if n_corrected else "  no corrections"

        print(
            f"  {name:<35} ✓ TABLE  {ocr_chars:>6}ch  "
            f"{len(results):>5} rows  dist={dist_str:<6}  "
            f"{total_str:<18}  {corr_str}"
        )

        # Show corrected rows
        for r in results:
            if r["was_corrected"]:
                print(
                    f"    cand={r['candidate'] or '?':>3}  "
                    f"{r['normalized']:<12} → {r['corrected']:<12}  [CORRECTED]"
                )

        _save_results(stem, results)

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
