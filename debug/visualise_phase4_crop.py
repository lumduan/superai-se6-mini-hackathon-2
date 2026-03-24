#!/usr/bin/env python
"""Debug script — visualise Phase 4 adaptive vote column crop.

Runs the full Phase 4 crop pipeline on a small sample of real scans and
saves side-by-side comparison images (original | adaptive crop | all
fallback crops) into debug/phase4_crop_samples/.

Usage:
    python debug/visualise_phase4_crop.py

Output files (written to debug/phase4_crop_samples/):
    <name>_original.png          — original scan
    <name>_adaptive_crop.png     — result of crop_vote_column()
    <name>_fallback_<ratio>.png  — one file per fallback ratio
    <name>_comparison.png        — side-by-side strip of all crops
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageDraw, ImageFont

from src.config import IMAGES_DIR
from src.phase4_crop.crop import (
    FALLBACK_CROP_RATIOS,
    all_fallback_crops,
    crop_vote_column,
    detect_rightmost_column_boundary,
)

# ── Config ────────────────────────────────────────────────────────────────────

SAMPLE_IMAGES = [
    "constituency_10_4_page2.png",
    "constituency_20_7_page2.png",
    "party_list_24_3.png",
    "constituency_30_1_page2.png",
    "constituency_10_29_page2.png",
]

OUT_DIR = Path(__file__).parent / "phase4_crop_samples"
THUMB_HEIGHT = 600   # resize all crops to this height for the comparison strip
LABEL_HEIGHT = 28    # pixels reserved for text label below each thumbnail
FONT_SIZE = 16


# ── Helpers ───────────────────────────────────────────────────────────────────

def _thumb(img: Image.Image, height: int = THUMB_HEIGHT) -> Image.Image:
    """Resize *img* to *height* preserving aspect ratio."""
    ratio = height / img.size[1]
    new_w = max(1, int(img.size[0] * ratio))
    return img.resize((new_w, height), Image.LANCZOS)


def _label(img: Image.Image, text: str, font_size: int = FONT_SIZE) -> Image.Image:
    """Paste a white label strip with *text* below *img*."""
    labelled = Image.new("RGB", (img.size[0], img.size[1] + LABEL_HEIGHT), (240, 240, 240))
    labelled.paste(img.convert("RGB"), (0, 0))
    draw = ImageDraw.Draw(labelled)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except (OSError, IOError):
        font = ImageFont.load_default()
    draw.text((4, img.size[1] + 4), text, fill=(30, 30, 30), font=font)
    return labelled


def _hstack(images: list[Image.Image], gap: int = 6) -> Image.Image:
    """Stack *images* horizontally with a *gap*-px grey separator."""
    total_w = sum(im.size[0] for im in images) + gap * (len(images) - 1)
    max_h = max(im.size[1] for im in images)
    canvas = Image.new("RGB", (total_w, max_h), (180, 180, 180))
    x = 0
    for im in images:
        canvas.paste(im, (x, 0))
        x += im.size[0] + gap
    return canvas


def _draw_boundary_line(img: Image.Image, ratio: float | None) -> Image.Image:
    """Draw a red vertical line on *img* at *ratio* (if not None)."""
    out = img.convert("RGB")
    if ratio is None:
        return out
    x = int(out.size[0] * ratio)
    draw = ImageDraw.Draw(out)
    draw.line([(x, 0), (x, out.size[1])], fill=(220, 40, 40), width=3)
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {OUT_DIR}\n")

    for filename in SAMPLE_IMAGES:
        image_path = IMAGES_DIR / filename
        if not image_path.exists():
            print(f"  [SKIP] {filename} — not found in {IMAGES_DIR}")
            continue

        stem = image_path.stem
        print(f"Processing {filename} …")

        # Detect boundary ratio
        ratio = detect_rightmost_column_boundary(image_path)
        ratio_str = f"{ratio:.3f}" if ratio is not None else "None (fallback)"
        print(f"  detected boundary ratio: {ratio_str}")

        # Load original
        original = Image.open(image_path)
        print(f"  original size: {original.size}")

        # Save original with boundary overlay
        orig_with_line = _draw_boundary_line(original, ratio)
        orig_path = OUT_DIR / f"{stem}_original.png"
        orig_with_line.save(orig_path)
        print(f"  saved: {orig_path.name}")

        # Adaptive crop
        adaptive = crop_vote_column(image_path)
        adaptive_path = OUT_DIR / f"{stem}_adaptive_crop.png"
        adaptive.save(adaptive_path)
        print(f"  adaptive crop size: {adaptive.size} → saved: {adaptive_path.name}")

        # All fallback crops
        fallbacks = all_fallback_crops(image_path)
        for crop, r in zip(fallbacks, FALLBACK_CROP_RATIOS):
            fb_path = OUT_DIR / f"{stem}_fallback_{r:.2f}.png"
            crop.save(fb_path)
            print(f"  fallback {r:.2f} size: {crop.size} → saved: {fb_path.name}")

        # Comparison strip
        panels: list[Image.Image] = []

        # Panel 0: original (with boundary line), scaled
        panels.append(_label(
            _thumb(orig_with_line),
            f"original {original.size[0]}×{original.size[1]}"
            + (f"  boundary={ratio:.3f}" if ratio else "  boundary=None"),
        ))

        # Panel 1: adaptive crop
        mode = "adaptive" if (ratio is not None and ratio >= 0.5) else "fallback@0.80"
        panels.append(_label(
            _thumb(adaptive),
            f"{mode}  {adaptive.size[0]}×{adaptive.size[1]}",
        ))

        # Panels 2–5: fallback crops
        for crop, r in zip(fallbacks, FALLBACK_CROP_RATIOS):
            panels.append(_label(
                _thumb(crop),
                f"fallback@{r:.2f}  {crop.size[0]}×{crop.size[1]}",
            ))

        comparison = _hstack(panels)
        comp_path = OUT_DIR / f"{stem}_comparison.png"
        comparison.save(comp_path)
        print(f"  comparison strip: {comparison.size} → saved: {comp_path.name}\n")

    print("Done. Open the files in debug/phase4_crop_samples/ to review the crops.")


if __name__ == "__main__":
    main()
