"""Debug script — Phase 5: Full-Page Typhoon OCR (Primary Path).

Runs Phase 2 detection to identify table pages, applies Phase 3 preprocessing,
then calls ``run_full_page_ocr`` on each confirmed table page.  Saves the raw
OCR text output to ``debug/phase5/`` for inspection.

Usage
-----
    export TYPHOON_OCR_API_KEY=your_api_key_here
    uv run debug/debug_phase5.py

Requirements
------------
- ``TYPHOON_OCR_API_KEY`` environment variable must be set.
- Sample images must exist under ``data/images/``.

Output per image
----------------
- ``<stem>_ocr.txt``  — raw Typhoon OCR response (HTML table or plain text)

Sample images exercised (same table pages as debug_phase3.py)
-------------------------------------------------------------
- constituency_10_11.png
- constituency_10_8.png, constituency_10_1_page2.png, constituency_10_8_page2.png
- constituency_10_11_page2.png, constituency_10_14_page2.png
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGES_DIR
from src.phase2_detection.detection import is_table_page
from src.phase5_ocr.ocr import run_full_page_ocr

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "debug" / "phase5"

SAMPLE_IMAGES = [
    # Non-table pages — will be skipped after Phase 2 gate
    "constituency_10_1.png",
    "constituency_10_16.png",
    "constituency_10_1_page3.png",
    "constituency_10_12_page3.png",
    # Table pages — full-page OCR applied
    "constituency_10_11.png",
    "constituency_10_8.png",
    "constituency_10_1_page2.png",
    "constituency_10_8_page2.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


def run() -> None:
    api_key = os.environ.get("TYPHOON_OCR_API_KEY", "")
    if not api_key:
        print(
            "\nERROR: TYPHOON_OCR_API_KEY environment variable is not set.\n"
            "Export it before running this script:\n"
            "  export TYPHOON_OCR_API_KEY=your_api_key_here\n"
        )
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDebug output → {OUT_DIR}\n")
    print(f"{'Image':<35} {'Phase 2 gate':<16} {'OCR chars'}")
    print("-" * 70)

    for name in SAMPLE_IMAGES:
        img_path = IMAGES_DIR / name
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            print(f"  {name:<33} MISSING")
            continue

        stem = Path(name).stem

        # Phase 2 gate — only OCR confirmed table pages
        if not is_table_page(img_path):
            print(f"  {name:<33} ✗ not table     (skipped)")
            continue

        # Phase 5 — preprocess + full-page OCR
        try:
            text = run_full_page_ocr(img_path, api_key=api_key, sleep_between_calls=0.5)
        except Exception as exc:
            logger.error("Phase 5 failed for %s: %s", name, exc)
            print(f"  {name:<33} ✓ TABLE          ERROR: {exc}")
            continue

        out_file = OUT_DIR / f"{stem}_ocr.txt"
        out_file.write_text(text, encoding="utf-8")

        char_count = len(text.strip())
        status = f"{char_count} chars" if char_count > 0 else "EMPTY"
        print(f"  {name:<33} ✓ TABLE          {status}  → {out_file.name}")

    print(f"\nDone. OCR output saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    run()
