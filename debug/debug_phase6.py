"""Debug script — Phase 6: HTML Table Parsing with BeautifulSoup.

Two operating modes:

1. **Offline (default)** — reads cached OCR ``.txt`` files from ``debug/phase5/``
   so you can iterate on the parser without consuming API quota.

2. **Live** — if ``--live`` is passed (and ``TYPHOON_OCR_API_KEY`` is set) it
   runs Phase 2 detection + Phase 3 preprocessing + Phase 5 OCR on the sample
   images first, then immediately parses each result with Phase 6.

Usage
-----
    # Offline mode — parse whatever Phase 5 already saved:
    uv run debug/debug_phase6.py

    # Live mode — detect, OCR, then parse in one shot:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase6.py --live

Output per document
-------------------
- ``debug/phase6/<stem>_parsed.txt``  — human-readable parse table (TSV-ish)
- ``debug/phase6/<stem>_votes.csv``   — CSV with columns: candidate,raw_cell,digits

Console summary columns
-----------------------
    Image            #rows  consistent  sample_votes
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE5_DIR = ROOT / "debug" / "phase5"
OUT_DIR = ROOT / "debug" / "phase6"

# Offline: read these cached OCR files from debug/phase5/
OFFLINE_SOURCES = [
    "constituency_10_1_page2_ocr.txt",
    "constituency_10_8_ocr.txt",
    "constituency_10_8_page2_ocr.txt",
    "constituency_10_11_ocr.txt",
    "constituency_10_11_page2_ocr.txt",
    "constituency_10_14_page2_ocr.txt",
]

# Live: run OCR on these images then parse immediately
LIVE_IMAGES = [
    "constituency_10_1_page2.png",
    "constituency_10_8.png",
    "constituency_10_8_page2.png",
    "constituency_10_11.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _save_results(stem: str, results: list) -> None:
    """Write parsed rows to a human-readable .txt and a .csv file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Human-readable table
    txt_path = OUT_DIR / f"{stem}_parsed.txt"
    with txt_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# Phase 6 parse results — {stem}\n")
        fh.write(f"# rows: {len(results)}\n\n")
        fh.write(f"{'cand':>5}  {'digits':<10}  raw_cell\n")
        fh.write("-" * 60 + "\n")
        for cand, raw, digits in results:
            cand_str = str(cand) if cand is not None else "?"
            fh.write(f"{cand_str:>5}  {digits:<10}  {raw}\n")

    # CSV — easy to diff or load into pandas
    csv_path = OUT_DIR / f"{stem}_votes.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["candidate", "raw_cell", "digits"])
        for cand, raw, digits in results:
            writer.writerow([cand if cand is not None else "", raw, digits])


def _print_row(label: str, n_rows: int, consistent: bool, sample: list[str]) -> None:
    sample_str = ", ".join(sample[:5])
    if len(sample) > 5:
        sample_str += f" … (+{len(sample) - 5})"
    status = "✓" if consistent else "✗ INCONSISTENT"
    print(f"  {label:<35} {n_rows:>5} rows  {status:<16}  [{sample_str}]")


def _parse_and_report(stem: str, html: str) -> None:
    """Run Phase 6 on *html*, print summary, and save output files."""
    results = parse_html_table(html)
    digit_strings = [d for _, _, d in results]
    consistent = has_consistent_column(digit_strings)

    _print_row(stem, len(results), consistent, digit_strings)
    _save_results(stem, results)

    if not consistent:
        print(f"    ⚠  Column consistency check FAILED for {stem}")
        print(f"       digit lengths: {[len(d) for d in digit_strings]}")


# ── Offline mode ───────────────────────────────────────────────────────────────


def run_offline() -> None:
    """Parse cached Phase 5 OCR .txt files from debug/phase5/."""
    print(f"\n{'─'*70}")
    print("  Mode: OFFLINE  (reading cached OCR from debug/phase5/)")
    print(f"{'─'*70}")
    print(f"\n  {'Document':<35} {'Rows':>5}{'':4} Consistent        Sample digits\n")

    found = 0
    for filename in OFFLINE_SOURCES:
        src = PHASE5_DIR / filename
        if not src.exists():
            print(f"  {filename:<35} MISSING — run debug_phase5.py first")
            continue
        found += 1
        stem = filename.replace("_ocr.txt", "")
        html = src.read_text(encoding="utf-8")
        _parse_and_report(stem, html)

    if found == 0:
        print(
            "\n  No cached OCR files found in debug/phase5/.\n"
            "  Run debug_phase5.py first, or use --live to call the OCR API directly.\n"
        )
        return

    print(f"\n  Parsed output saved to: {OUT_DIR}\n")


# ── Live mode ──────────────────────────────────────────────────────────────────


def run_live(api_key: str) -> None:
    """Run Phase 2 → Phase 3 → Phase 5 → Phase 6 on real images."""
    from src.phase2_detection.detection import is_table_page
    from src.phase5_ocr.ocr import run_full_page_ocr

    print(f"\n{'─'*70}")
    print("  Mode: LIVE  (OCR API + parse)")
    print(f"{'─'*70}")
    print(f"\n  {'Image':<35} {'Phase 2':^10} {'OCR':^8} {'Rows':>5}{'':4} Consistent        Sample digits\n")

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

        # Cache the OCR output alongside existing phase5 artefacts
        cache_path = PHASE5_DIR / f"{stem}_ocr.txt"
        cache_path.write_text(html, encoding="utf-8")

        # Phase 6 — parse
        results = parse_html_table(html)
        digit_strings = [d for _, _, d in results]
        consistent = has_consistent_column(digit_strings)

        ocr_chars = len(html.strip())
        print(
            f"  {name:<35} ✓ TABLE  {ocr_chars:>6}ch  "
            f"{len(results):>5} rows  {'✓' if consistent else '✗ INCONSIST.':<16}"
            f"  [{', '.join(digit_strings[:4])}{'…' if len(digit_strings) > 4 else ''}]"
        )

        if not consistent:
            print(f"    ⚠  Column inconsistency — digit lengths: {[len(d) for d in digit_strings]}")

        _save_results(stem, results)

    print(f"\n  Parse output saved to: {OUT_DIR}\n")


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
