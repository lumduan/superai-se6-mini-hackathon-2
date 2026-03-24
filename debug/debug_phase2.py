"""Debug script — Phase 2: Dynamic Table Page Detection.

Runs Signal A (H×V intersection detection) on sample images and saves
visualisation artefacts to ``debug/phase2/``.  Each output image shows the
detected intersection pixels highlighted in red with a pass/fail label.

Usage
-----
    uv run debug/debug_phase2.py

Sample images exercised (10 images)
-------------------------------------
Cover pages (expect NOT table):
- constituency_10_1.png, constituency_10_16.png

Overflow / signature pages (expect NOT table):
- constituency_10_1_page3.png, constituency_10_12_page3.png

Table pages (expect IS table):
- constituency_10_11.png (has small 2-row table at bottom)
- constituency_10_8.png, constituency_10_1_page2.png, constituency_10_8_page2.png
- constituency_10_11_page2.png, constituency_10_14_page2.png
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import IMAGES_DIR
from src.phase2_detection.detection import (
    H_KERNEL_WIDTH_DIVISOR,
    MAX_V_KERNEL_HEIGHT,
    MIN_INTERSECTIONS,
    has_table_structure,
    is_table_page,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = ROOT / "debug" / "phase2"

SAMPLE_IMAGES = [
    # Cover pages — NOT table
    "constituency_10_1.png",
    "constituency_10_16.png",
    # Overflow / signature pages — NOT table
    "constituency_10_1_page3.png",
    "constituency_10_12_page3.png",
    # Table pages — IS table (mix of full tables and small 2-row table)
    "constituency_10_11.png",       # small table at bottom of cover page
    "constituency_10_8.png",
    "constituency_10_1_page2.png",
    "constituency_10_8_page2.png",
    "constituency_10_11_page2.png",
    "constituency_10_14_page2.png",
]


def _draw_intersections(img_gray: np.ndarray) -> np.ndarray:
    """Return BGR image with H×V intersection pixels highlighted in red.

    Uses the same kernel logic as detection.py (v_kernel capped at
    MAX_V_KERNEL_HEIGHT) so the label matches the detection result.
    """
    _, binary = cv2.threshold(img_gray, 180, 255, cv2.THRESH_BINARY_INV)

    h_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (img_gray.shape[1] // H_KERNEL_WIDTH_DIVISOR, 1)
    )
    v_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT, (1, min(img_gray.shape[0] // 8, MAX_V_KERNEL_HEIGHT))
    )

    h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel)
    intersections = cv2.bitwise_and(h_lines, v_lines)

    n = cv2.countNonZero(intersections)
    detected = n >= MIN_INTERSECTIONS

    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    vis[intersections > 0] = (0, 0, 255)  # red highlights

    label = f"Signal A: intersections={n}  threshold={MIN_INTERSECTIONS}  {'TABLE' if detected else 'NOT TABLE'}"
    color = (0, 200, 0) if detected else (0, 0, 200)
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return vis


def run() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nDebug output → {OUT_DIR}\n")
    print(f"{'Image':<40} {'Signal A (structure)':<22} {'is_table_page()'}")
    print("-" * 78)

    for name in SAMPLE_IMAGES:
        img_path = IMAGES_DIR / name
        if not img_path.exists():
            logger.warning("Image not found, skipping: %s", img_path)
            print(f"  {name:<38} MISSING")
            continue

        stem = Path(name).stem

        signal_a = has_table_structure(img_path)
        is_table = is_table_page(img_path)

        img_gray = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        vis = _draw_intersections(img_gray)

        # Scale down if very large (keep width ≤ 1200px for readability)
        max_w = 1200
        if vis.shape[1] > max_w:
            scale = max_w / vis.shape[1]
            vis = cv2.resize(vis, None, fx=scale, fy=scale)

        out_path = OUT_DIR / f"{stem}_detection.png"
        cv2.imwrite(str(out_path), vis)

        a_str = "✓ pass" if signal_a else "✗ fail"
        t_str = "✓ TABLE" if is_table else "✗ not table"
        print(f"  {name:<38} {a_str:<22} {t_str}")

    print(f"\nDone. Artefacts saved to: {OUT_DIR}\n")


if __name__ == "__main__":
    run()
