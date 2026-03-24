"""Debug script — Phase 7: Thai Text Cross-check with Digit-level Diff.

Two operating modes:

1. **Offline (default)** — reads cached Phase 6 parse results (``.csv`` files)
   from ``debug/phase6/`` and runs the Phase 7 cross-check on each row.
   No API quota consumed.

2. **Live** — if ``--live`` is passed (and ``TYPHOON_OCR_API_KEY`` is set), it
   runs Phase 2 detection → Phase 3 preprocessing → Phase 5 OCR → Phase 6
   parsing → Phase 7 cross-check in one shot on real images.

Usage
-----
    # Offline mode — cross-check whatever Phase 6 already saved:
    uv run debug/debug_phase7.py

    # Live mode — run full pipeline up to Phase 7:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase7.py --live

Output per document
-------------------
- ``debug/phase7/<stem>_crosschecked.csv``  — CSV with columns:
    candidate, raw_cell, ocr_digits, thai_num, corrected, changed

Console summary columns
-----------------------
    Document                #rows  #corrected  sample_corrections
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
from src.phase6_parse.parser import has_consistent_column, parse_html_table
from src.phase7_thai_crosscheck.crosscheck import (
    cross_check_vote,
    extract_thai_number_text,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE6_DIR = ROOT / "debug" / "phase6"
PHASE5_DIR = ROOT / "debug" / "phase5"
OUT_DIR = ROOT / "debug" / "phase7"

# Offline: read these Phase 6 vote CSVs from debug/phase6/
OFFLINE_SOURCES = [
    "constituency_10_1_page2_votes.csv",
    "constituency_10_8_votes.csv",
    "constituency_10_8_page2_votes.csv",
    "constituency_10_11_votes.csv",
    "constituency_10_11_page2_votes.csv",
    "constituency_10_14_page2_votes.csv",
]

# Live: run OCR + parse on these images then cross-check
LIVE_IMAGES = [
    "constituency_10_1_page2.png",
    "constituency_10_8.png",
    "constituency_10_8_page2.png",
    "constituency_10_11.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run_crosscheck(
    rows: list[tuple[str | None, str, str]],
) -> list[dict]:
    """Apply Phase 7 cross-check to each (candidate, raw_cell, ocr_digits) row.

    Returns a list of dicts with keys:
        candidate, raw_cell, ocr_digits, thai_num, corrected, changed
    """
    results = []
    for cand, raw_cell, ocr_digits in rows:
        thai_num = extract_thai_number_text(raw_cell)
        corrected = cross_check_vote(raw_cell, ocr_digits)
        changed = corrected != ocr_digits
        results.append(
            {
                "candidate": cand if cand is not None else "",
                "raw_cell": raw_cell,
                "ocr_digits": ocr_digits,
                "thai_num": thai_num or "",
                "corrected": corrected,
                "changed": "YES" if changed else "",
            }
        )
    return results


def _save_results(stem: str, results: list[dict]) -> None:
    """Write cross-check results to a CSV file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}_crosschecked.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["candidate", "raw_cell", "ocr_digits", "thai_num", "corrected", "changed"],
        )
        writer.writeheader()
        writer.writerows(results)


def _print_summary(label: str, results: list[dict]) -> None:
    """Print a one-line summary for a document."""
    n_rows = len(results)
    changed = [r for r in results if r["changed"]]
    n_corrected = len(changed)
    corrections = [
        f"{r['ocr_digits']}→{r['corrected']}" for r in changed[:3]
    ]
    corrections_str = ", ".join(corrections)
    if n_corrected > 3:
        corrections_str += f" … (+{n_corrected - 3})"
    status = f"✓ {n_corrected} corrected" if n_corrected else "  no corrections"
    print(f"  {label:<35} {n_rows:>5} rows  {status:<20}  {corrections_str}")


# ── Offline mode ───────────────────────────────────────────────────────────────


def run_offline() -> None:
    """Cross-check rows from cached Phase 6 vote CSV files."""
    print(f"\n{'─'*70}")
    print("  Mode: OFFLINE  (reading Phase 6 CSVs from debug/phase6/)")
    print(f"{'─'*70}")
    print(
        f"\n  {'Document':<35} {'Rows':>5}{'':4} Status              Sample corrections\n"
    )

    found = 0
    for filename in OFFLINE_SOURCES:
        src = PHASE6_DIR / filename
        if not src.exists():
            print(
                f"  {filename:<35} MISSING — run debug_phase6.py first"
            )
            continue
        found += 1

        # Load Phase 6 rows: candidate, raw_cell, digits
        rows: list[tuple[str | None, str, str]] = []
        with src.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                cand_raw = row.get("candidate", "").strip()
                cand: str | None = cand_raw if cand_raw else None
                raw_cell = row.get("raw_cell", "")
                digits = row.get("digits", "")
                rows.append((cand, raw_cell, digits))

        stem = filename.replace("_votes.csv", "")
        results = _run_crosscheck(rows)
        _print_summary(stem, results)
        _save_results(stem, results)

        # Show detailed diff for any corrected rows
        changed = [r for r in results if r["changed"]]
        for r in changed:
            print(
                f"    cand={r['candidate'] or '?':>3}  "
                f"ocr={r['ocr_digits']:<10}  "
                f"thai_text={r['thai_num']:<30}  "
                f"→ {r['corrected']}"
            )

    if found == 0:
        print(
            "\n  No Phase 6 CSV files found in debug/phase6/.\n"
            "  Run debug_phase6.py first, or use --live to call the OCR API.\n"
        )
        return

    print(f"\n  Output saved to: {OUT_DIR}\n")


# ── Live mode ──────────────────────────────────────────────────────────────────


def run_live(api_key: str) -> None:
    """Run Phase 2 → 3 → 5 → 6 → 7 on real images, then cross-check."""
    from src.phase2_detection.detection import is_table_page
    from src.phase5_ocr.ocr import run_full_page_ocr

    print(f"\n{'─'*70}")
    print("  Mode: LIVE  (OCR API → parse → cross-check)")
    print(f"{'─'*70}")
    print(
        f"\n  {'Image':<35} {'Phase 2':^10} {'OCR':^8} "
        f"{'Rows':>5}{'':4} Status              Sample corrections\n"
    )

    PHASE5_DIR.mkdir(parents=True, exist_ok=True)

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

        # Cache OCR output
        cache_path = PHASE5_DIR / f"{stem}_ocr.txt"
        cache_path.write_text(html, encoding="utf-8")

        # Phase 6 — parse
        parsed = parse_html_table(html)

        # Phase 7 — cross-check
        results = _run_crosscheck(parsed)
        changed = [r for r in results if r["changed"]]

        ocr_chars = len(html.strip())
        status = f"✓ {len(changed)} corrected" if changed else "  no corrections"
        print(
            f"  {name:<35} ✓ TABLE  {ocr_chars:>6}ch  "
            f"{len(results):>5} rows  {status:<20}"
        )

        for r in changed:
            print(
                f"    cand={r['candidate'] or '?':>3}  "
                f"ocr={r['ocr_digits']:<10}  "
                f"thai_text={r['thai_num']:<30}  "
                f"→ {r['corrected']}"
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
