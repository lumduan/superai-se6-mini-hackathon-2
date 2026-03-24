"""Debug script — Phase 8: Normalization & Hard Rule Overrides.

Two operating modes:

1. **Offline (default)** — reads cached Phase 7 cross-check results (``.csv``
   files) from ``debug/phase7/`` and runs Phase 8 normalization + hard-rule
   overrides on each row.  No API quota consumed.

2. **Live** — if ``--live`` is passed (and ``TYPHOON_OCR_API_KEY`` is set), it
   runs Phase 2 detection → Phase 3 preprocessing → Phase 5 OCR → Phase 6
   parsing → Phase 7 cross-check → Phase 8 normalization in one shot on real
   images.

Usage
-----
    # Offline mode — normalize whatever Phase 7 already saved:
    uv run debug/debug_phase8.py

    # Live mode — run full pipeline up to Phase 8:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase8.py --live

Output per document
-------------------
- ``debug/phase8/<stem>_normalized.csv`` — CSV with columns:
    candidate, raw_corrected, normalized, confidence, hard_overridden

Console summary columns
-----------------------
    Document                #rows  #overridden  #penalized  sample_normalizations
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
from src.phase8_normalize import (
    apply_hard_rules,
    apply_soft_rules,
    normalize_votes,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE7_DIR = ROOT / "debug" / "phase7"
OUT_DIR = ROOT / "debug" / "phase8"

# Offline: read these Phase 7 cross-check CSVs from debug/phase7/
OFFLINE_SOURCES = [
    "constituency_10_1_page2_crosschecked.csv",
    "constituency_10_8_crosschecked.csv",
    "constituency_10_8_page2_crosschecked.csv",
    "constituency_10_11_crosschecked.csv",
    "constituency_10_11_page2_crosschecked.csv",
    "constituency_10_14_page2_crosschecked.csv",
]

# Live: run OCR + parse + cross-check on these images then normalize
LIVE_IMAGES = [
    "constituency_10_1_page2.png",
    "constituency_10_8.png",
    "constituency_10_8_page2.png",
    "constituency_10_11.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


# ── Helpers ────────────────────────────────────────────────────────────────────


def _run_normalization(
    rows: list[dict],
) -> list[dict]:
    """Apply Phase 8 normalization + soft/hard rules to each row.

    Parameters
    ----------
    rows:
        List of dicts with at minimum a ``corrected`` field (the digit string
        produced by Phase 7 cross-check) and a ``raw_cell`` field.

    Returns
    -------
    List of dicts with keys:
        candidate, raw_corrected, normalized, confidence, hard_overridden
    """
    results: list[dict] = []
    for row in rows:
        candidate = row.get("candidate", "")
        # Phase 7 output is the ``corrected`` column; fall back to ``ocr_digits``
        raw_corrected = row.get("corrected") or row.get("ocr_digits", "")

        normalized = normalize_votes(raw_corrected)
        confidence = apply_soft_rules(normalized)
        final = apply_hard_rules(normalized)
        hard_overridden = final != normalized

        results.append(
            {
                "candidate": candidate,
                "raw_corrected": raw_corrected,
                "normalized": normalized,
                "final": final,
                "confidence": f"{confidence:.2f}",
                "hard_overridden": "YES" if hard_overridden else "",
            }
        )
    return results


def _save_results(stem: str, results: list[dict]) -> None:
    """Write normalization results to a CSV file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}_normalized.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "candidate",
                "raw_corrected",
                "normalized",
                "final",
                "confidence",
                "hard_overridden",
            ],
        )
        writer.writeheader()
        writer.writerows(results)
    logger.info("Saved %d rows to %s", len(results), csv_path)


def _print_summary(label: str, results: list[dict]) -> None:
    """Print a one-line summary for a document."""
    n_rows = len(results)
    overridden = [r for r in results if r["hard_overridden"]]
    penalized = [r for r in results if float(r["confidence"]) < 1.0 and not r["hard_overridden"]]

    samples = [
        f"{r['raw_corrected']}→{r['normalized']}"
        for r in results
        if r["normalized"] != r["raw_corrected"]
    ][:3]
    samples_str = ", ".join(samples)
    if len(samples) == 3 and sum(
        1 for r in results if r["normalized"] != r["raw_corrected"]
    ) > 3:
        samples_str += "…"

    override_str = f"⚠ {len(overridden)} hard-overridden" if overridden else "  no overrides"
    penalty_str = f", {len(penalized)} penalized" if penalized else ""
    print(
        f"  {label:<40} {n_rows:>5} rows  "
        f"{override_str:<28}{penalty_str}"
        + (f"  samples: {samples_str}" if samples_str else "")
    )


# ── Offline mode ───────────────────────────────────────────────────────────────


def run_offline() -> None:
    """Normalize rows from cached Phase 7 cross-check CSV files."""
    print(f"\n{'─'*75}")
    print("  Mode: OFFLINE  (reading Phase 7 CSVs from debug/phase7/)")
    print(f"{'─'*75}")
    print(
        f"\n  {'Document':<40} {'Rows':>5}{'':4} Status\n"
    )

    found = 0
    for filename in OFFLINE_SOURCES:
        src = PHASE7_DIR / filename
        if not src.exists():
            print(
                f"  {filename:<40} MISSING — run debug_phase7.py first"
            )
            continue
        found += 1

        rows: list[dict] = []
        with src.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(dict(row))

        stem = filename.replace("_crosschecked.csv", "")
        results = _run_normalization(rows)
        _print_summary(stem, results)
        _save_results(stem, results)

        # Show detailed info for any overridden or penalized rows
        notable = [r for r in results if r["hard_overridden"] or float(r["confidence"]) < 1.0]
        for r in notable:
            tag = "OVERRIDE" if r["hard_overridden"] else f"conf={r['confidence']}"
            print(
                f"    cand={r['candidate'] or '?':>3}  "
                f"raw={r['raw_corrected']:<12}  "
                f"normalized={r['normalized']:<12}  "
                f"final={r['final']:<12}  [{tag}]"
            )

    if found == 0:
        print(
            "\n  No Phase 7 CSV files found in debug/phase7/.\n"
            "  Run debug_phase7.py first, or use --live to call the OCR API.\n"
        )
        return

    print(f"\n  Output saved to: {OUT_DIR}\n")


# ── Live mode ──────────────────────────────────────────────────────────────────


def run_live(api_key: str) -> None:
    """Run Phase 2 → 3 → 5 → 6 → 7 → 8 on real images, then normalize."""
    from src.phase2_detection.detection import is_table_page
    from src.phase5_ocr.ocr import run_full_page_ocr
    from src.phase6_parse.parser import parse_html_table
    from src.phase7_thai_crosscheck.crosscheck import (
        cross_check_vote,
        extract_thai_number_text,
    )

    print(f"\n{'─'*75}")
    print("  Mode: LIVE  (OCR API → parse → cross-check → normalize)")
    print(f"{'─'*75}")
    print(
        f"\n  {'Image':<35} {'Phase 2':^10} {'OCR':^8} "
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

        # Phase 6 — parse
        parsed = parse_html_table(html)

        # Phase 7 — cross-check, convert to row dicts
        phase7_rows: list[dict] = []
        for cand, raw_cell, ocr_digits in parsed:
            thai_num = extract_thai_number_text(raw_cell)
            corrected = cross_check_vote(raw_cell, ocr_digits)
            phase7_rows.append(
                {
                    "candidate": str(cand) if cand is not None else "",
                    "raw_cell": raw_cell,
                    "ocr_digits": ocr_digits,
                    "thai_num": thai_num or "",
                    "corrected": corrected,
                }
            )

        # Phase 8 — normalize
        results = _run_normalization(phase7_rows)
        overridden = [r for r in results if r["hard_overridden"]]
        penalized = [r for r in results if float(r["confidence"]) < 1.0 and not r["hard_overridden"]]

        ocr_chars = len(html.strip())
        override_str = f"⚠ {len(overridden)} overridden" if overridden else "  no overrides"
        print(
            f"  {name:<35} ✓ TABLE  {ocr_chars:>6}ch  "
            f"{len(results):>5} rows  {override_str}"
            + (f", {len(penalized)} penalized" if penalized else "")
        )

        for r in overridden:
            print(
                f"    cand={r['candidate'] or '?':>3}  "
                f"raw={r['raw_corrected']:<12}  "
                f"normalized={r['normalized']:<12}  "
                f"final={r['final']:<12}  [OVERRIDE]"
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
