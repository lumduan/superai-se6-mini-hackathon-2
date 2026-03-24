"""Debug script — Phase 3: Image Preprocessing (deskew + CLAHE + sharpen).

Runs Phase 2 detection first to identify table pages, then applies the full
Phase 3 preprocessing pipeline to each confirmed table page.  Saves a
side-by-side comparison (original vs preprocessed) and the preprocessed
image alone to ``debug/phase3/``.

Usage
-----
    uv run debug/debug_phase3.py

Phase 3 scope: sharpen only (unsharp mask).  No deskew, no CLAHE.

Sample images exercised (10 images, same set as debug_phase2.py)
----------------------------------------------------------------
Non-table pages — skipped after Phase 2 gate:
- constituency_10_1.png, constituency_10_16.png
- constituency_10_1_page3.png, constituency_10_12_page3.png

Table pages — sharpened:
- constituency_10_11.png (small table at bottom)
- constituency_10_8.png, constituency_10_1_page2.png, constituency_10_8_page2.png
- constituency_10_11_page2.png, constituency_10_14_page2.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGES_DIR
from src.phase2_detection.detection import is_table_page
from src.phase3_preprocess.preprocess import preprocess_image

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "debug" / "phase3"

SAMPLE_IMAGES = [
    # Non-table pages — skipped after Phase 2 gate
    "constituency_10_1.png",
    "constituency_10_16.png",
    "constituency_10_1_page3.png",
    "constituency_10_12_page3.png",
    # Table pages — preprocessed
    "constituency_10_11.png",       # small table at bottom of cover page
    "constituency_10_8.png",
    "constituency_10_1_page2.png",
    "constituency_10_8_page2.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


def _side_by_side(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Stack two BGR images side-by-side."""
    h = max(left.shape[0], right.shape[0])
    pad_l = np.zeros((h - left.shape[0], left.shape[1], 3), dtype=np.uint8)
    pad_r = np.zeros((h - right.shape[0], right.shape[1], 3), dtype=np.uint8)
    l_full = np.vstack([left, pad_l])
    r_full = np.vstack([right, pad_r])
    combined = np.hstack([l_full, r_full])
    cv2.putText(combined, "original", (10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    cv2.putText(combined, "phase3 sharpened", (l_full.shape[1] + 10, combined.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 1)
    return combined


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDebug output → {OUT_DIR}\n")
    print(f"{'Image':<35} {'Phase 2 gate':<16} {'Phase 3 artefacts'}")
    print("-" * 75)

    for name in SAMPLE_IMAGES:
        img_path = IMAGES_DIR / name
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            print(f"  {name:<33} MISSING")
            continue

        stem = Path(name).stem

        # Phase 2 gate — only preprocess confirmed table pages
        if not is_table_page(img_path):
            print(f"  {name:<33} ✗ not table     (skipped)")
            continue

        # Phase 3 preprocessing
        try:
            preprocessed = preprocess_image(img_path)
        except Exception as exc:
            logger.error("Phase 3 failed for %s: %s", name, exc)
            print(f"  {name:<33} ✓ TABLE          ERROR: {exc}")
            continue

        # Save preprocessed image
        pre_out = OUT_DIR / f"{stem}_preprocessed.png"
        preprocessed.save(pre_out)

        # Save side-by-side comparison (original BGR vs sharpened RGB→BGR)
        orig_bgr = cv2.imread(str(img_path))
        pre_arr = np.array(preprocessed)                       # RGB from preprocess_image
        pre_bgr = cv2.cvtColor(pre_arr, cv2.COLOR_RGB2BGR)
        comparison = _side_by_side(orig_bgr, pre_bgr)

        max_w = 1600
        if comparison.shape[1] > max_w:
            scale = max_w / comparison.shape[1]
            comparison = cv2.resize(comparison, None, fx=scale, fy=scale)

        cmp_out = OUT_DIR / f"{stem}_comparison.png"
        cv2.imwrite(str(cmp_out), comparison)

        print(f"  {name:<33} ✓ TABLE          {pre_out.name}, {cmp_out.name}")

    print(f"\nDone. Artefacts saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    run()
