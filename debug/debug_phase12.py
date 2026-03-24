"""Debug script — Phase 12: Row Anchor Alignment.

Two operating modes:

1. **Offline (default)** — reads Phase 5 OCR output from ``debug/phase5/``
   and runs anchor alignment on them, comparing the anchor-aligned result
   against naive sequential ordering.  Demonstrates row-shift detection and
   correction without any API calls.

2. **Live** — if ``--live`` is passed (and ``TYPHOON_OCR_API_KEY`` is set),
   runs Phase 5 OCR on real images, then applies Phase 12 anchor alignment,
   showing the full pipeline result.

Usage
-----
    # Offline mode — run anchor alignment on Phase 5 OCR cache:
    uv run debug/debug_phase12.py

    # Live mode — run full Phase 5 + 12 pipeline on real images:
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase12.py --live

Output per document
-------------------
- ``debug/phase12/<stem>_anchor.csv`` — CSV with columns:
    position, cand_num, sequential_vote, anchored_vote, shifted

Console summary columns
-----------------------
    Document                    #rows  #anchored  #shifted  #zeros
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
from src.phase6_parse import parse_html_table
from src.phase7_thai_crosscheck import cross_check_vote
from src.phase8_normalize import apply_hard_rules, normalize_votes
from src.phase12_anchor import anchor_align, extract_anchored_rows

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

PHASE5_DIR = ROOT / "debug" / "phase5"
OUT_DIR = ROOT / "debug" / "phase12"

# Offline: read Phase 5 OCR txt files
OFFLINE_SOURCES = [
    "constituency_10_1_page2_ocr.txt",
    "constituency_10_8_ocr.txt",
    "constituency_10_8_page2_ocr.txt",
    "constituency_10_11_ocr.txt",
    "constituency_10_11_page2_ocr.txt",
    "constituency_10_14_page2_ocr.txt",
]

# Live: run full Phase 5 OCR on these images
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
        f"\n  {'Document':<45}  {'Rows':>5}  {'#anchored':>9}  "
        f"{'#shifted':>8}  {'#zeros':>6}\n"
    )


def _naive_sequential(parsed_rows: list) -> list[str]:
    """Apply Phase 7+8 pipeline sequentially (no anchor correction)."""
    result = []
    for _, raw, digits in parsed_rows:
        vote = cross_check_vote(raw, normalize_votes(digits))
        vote = apply_hard_rules(vote)
        result.append(vote)
    return result


def _process_rows(parsed_rows: list, expected_count: int) -> tuple[list[str], list[str]]:
    """Return (sequential_votes, anchored_votes) for the given parsed rows."""
    sequential = _naive_sequential(parsed_rows)
    # Pad/truncate sequential to expected_count
    if len(sequential) < expected_count:
        sequential = sequential + ["0"] * (expected_count - len(sequential))
    sequential = sequential[:expected_count]

    anchored = anchor_align(parsed_rows, expected_count)
    return sequential, anchored


def _count_anchored(parsed_rows: list) -> int:
    """Count how many rows have a detected candidate number."""
    return sum(1 for cand_num, _, __ in parsed_rows if cand_num is not None)


def _print_summary(
    label: str,
    parsed_rows: list,
    sequential: list[str],
    anchored: list[str],
) -> None:
    """Print a one-line summary plus shift details."""
    n_rows = len(anchored)
    n_anchored = _count_anchored(parsed_rows)
    n_shifted = sum(a != s for a, s in zip(anchored, sequential))
    n_zeros = anchored.count("0")

    print(
        f"  {label:<45}  {n_rows:>5}  "
        f"{n_anchored:>9}  {n_shifted:>8}  {n_zeros:>6}"
    )

    # Show rows where anchor alignment changed the sequential value
    for i, (seq, anc) in enumerate(zip(sequential, anchored)):
        if seq != anc:
            # Find what candidate number claims this slot
            cand_str = "?"
            for cn, _, __ in parsed_rows:
                if cn == i + 1:
                    cand_str = str(cn)
                    break
            print(
                f"    [SHIFT] pos={i + 1:>2}  "
                f"sequential={seq:<10}  anchored={anc:<10}  "
                f"(cand#{cand_str})"
            )


def _save_results(
    stem: str,
    parsed_rows: list,
    sequential: list[str],
    anchored: list[str],
) -> None:
    """Write anchor alignment results to a CSV file."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / f"{stem}_anchor.csv"

    # Build candidate-number lookup for reporting
    cand_at_pos: dict[int, int] = {}
    for cn, _, __ in parsed_rows:
        if cn is not None:
            cand_at_pos[cn] = cn

    rows = [
        {
            "position": i + 1,
            "cand_num": cand_at_pos.get(i + 1, ""),
            "sequential_vote": seq,
            "anchored_vote": anc,
            "shifted": "1" if seq != anc else "0",
        }
        for i, (seq, anc) in enumerate(zip(sequential, anchored))
    ]

    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["position", "cand_num", "sequential_vote", "anchored_vote", "shifted"],
        )
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved %d rows to %s", len(rows), csv_path)


# ── Offline mode ───────────────────────────────────────────────────────────────


def run_offline() -> None:
    """Run anchor alignment on Phase 5 OCR txt files from debug/phase5/."""
    _print_banner("OFFLINE  (reading Phase 5 OCR outputs from debug/phase5/)")
    _print_header()

    found = 0
    for filename in OFFLINE_SOURCES:
        src = PHASE5_DIR / filename
        if not src.exists():
            print(f"  {filename:<45}  MISSING — run debug_phase5.py first")
            continue

        html = src.read_text(encoding="utf-8")
        # parse_html_table returns (cand_num | None, raw, digits) — same shape
        # as extract_anchored_rows output; anchor_align accepts both.
        parsed_rows = parse_html_table(html)

        if not parsed_rows:
            print(f"  {filename:<45}  EMPTY — no rows parsed")
            continue

        expected_count = len(parsed_rows)
        sequential, anchored = _process_rows(parsed_rows, expected_count)

        stem = src.stem.replace("_ocr", "")
        _print_summary(filename, parsed_rows, sequential, anchored)
        _save_results(stem, parsed_rows, sequential, anchored)
        found += 1

    if found == 0:
        print("\n  No Phase 5 OCR files found — run debug_phase5.py first.\n")
    else:
        print(f"\n  Output saved to: {OUT_DIR}\n")

    # ── Synthetic row-shift demo ───────────────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  Synthetic Row-Shift Demo")
    print(f"{'─'*80}")
    print(
        "\n  Simulates a document where OCR missed row 2, causing rows 3–5\n"
        "  to shift up. Candidate numbers allow Phase 12 to detect & correct.\n"
    )

    # Simulate: rows 1, 3, 4, 5 were parsed; row 2 was missed by OCR
    synthetic_rows = [
        (1, "11111", "11111"),
        (3, "33333", "33333"),
        (4, "44444", "44444"),
        (5, "55555", "55555"),
    ]
    expected = 5
    sequential, anchored = _process_rows(synthetic_rows, expected)

    print(f"  {'Position':>8}  {'Sequential':>12}  {'Anchored':>10}  {'Shifted?':>8}")
    print(f"  {'─'*46}")
    for i, (seq, anc) in enumerate(zip(sequential, anchored)):
        shifted = " <<< ROW SHIFT CORRECTED" if seq != anc else ""
        print(f"  {i + 1:>8}  {seq:>12}  {anc:>10}{shifted}")
    print()


# ── Live mode ──────────────────────────────────────────────────────────────────


def run_live(api_key: str) -> None:
    """Run Phase 5 OCR + Phase 12 anchor alignment on real images."""
    from src.phase3_preprocess import preprocess_image
    from src.phase5_ocr import run_typhoon_ocr

    _print_banner("LIVE  (running Phase 5 OCR + Phase 12 anchor alignment)")
    _print_header()

    for image_name in LIVE_IMAGES:
        image_path = IMAGES_DIR / image_name
        if not image_path.exists():
            print(f"  {image_name:<45}  MISSING — image not found")
            continue

        stem = image_path.stem
        logger.info("Processing %s ...", image_name)

        try:
            preprocessed = preprocess_image(str(image_path))
            html = run_typhoon_ocr(preprocessed, api_key=api_key)
            parsed_rows = parse_html_table(html)

            if not parsed_rows:
                print(f"  {image_name:<45}  EMPTY — no rows parsed")
                continue

            expected_count = len(parsed_rows)
            sequential, anchored = _process_rows(parsed_rows, expected_count)

            _print_summary(image_name, parsed_rows, sequential, anchored)
            _save_results(stem, parsed_rows, sequential, anchored)

        except Exception as exc:  # noqa: BLE001
            logger.exception("Error processing %s", image_name)
            print(f"  {image_name:<45}  ERROR — {exc}")

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
