# Thai Election OCR Pipeline

Super AI Engineer S6 — Mini Hackathon 2

Extract structured vote-count data from scanned Thai election documents (Form สส.6/1).

---

## Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 0 | Project structure & tooling | ✅ Done |
| Phase 1 | Data Inventory & ID Mapping | ✅ Done |
| Phase 2 | Dynamic Table Page Detection | 🔜 Next |
| Phase 3 | Image Preprocessing | 🔜 |
| Phase 4 | Adaptive Vote Column Crop | 🔜 |
| Phase 5 | Typhoon OCR Extraction | 🔜 |
| Phase 6 | Pattern-aware Parsing | 🔜 |
| Phase 7 | Thai Text Cross-check | 🔜 |
| Phase 8 | Normalization & Hard Rules | 🔜 |
| Phase 9 | Row Validation & Total Correction | 🔜 |
| Phase 10 | Per-row Confidence Scoring | 🔜 |
| Phase 11 | Multi-pass Fallback & Ensemble | 🔜 |
| Phase 12 | Row Anchor Alignment | 🔜 |
| Phase 13 | Final Row Alignment to IDs | 🔜 |
| Phase 14 | OCR Cache | 🔜 |
| Phase 15 | Checkpointing & Debug Mode | 🔜 |
| Phase 16 | Submission Generation & Validation | 🔜 |

---

## Project Structure

```
.
├── data/
│   ├── images/                  # raw input PNGs
│   ├── submission_template_v4.csv
│   └── interim/                 # intermediate results
├── src/
│   ├── config.py                # all constants / feature flags
│   ├── utils/
│   │   ├── io.py                # load/save helpers
│   │   ├── cache.py             # OCR cache
│   │   ├── checkpoint.py        # resume logic
│   │   └── debug.py             # per-stage debug artifacts
│   ├── phase1_mapping/          # Phase 1: ID mapping & inventory
│   ├── phase2_detection/        # Phase 2: table page detection
│   ├── phase3_preprocess/       # Phase 3: deskew / CLAHE / sharpen
│   ├── phase4_crop/             # Phase 4: vote column crop
│   ├── phase5_ocr/              # Phase 5: Typhoon + Tesseract OCR
│   ├── phase6_parse/            # Phase 6: pattern-aware parsing
│   ├── phase7_thai_crosscheck/  # Phase 7: Thai digit cross-check
│   ├── phase8_normalize/        # Phase 8: normalization & hard rules
│   ├── phase9_postprocess/      # Phase 9: correction & row validation
│   ├── phase10_confidence/      # Phase 10: confidence scoring
│   ├── phase11_ensemble/        # Phase 11: multi-pass ensemble
│   ├── phase12_anchor/          # Phase 12: anchor alignment
│   ├── phase13_align/           # Phase 13: final row alignment
│   └── pipeline/
│       └── runner.py            # main orchestration
├── scripts/
│   ├── run_all.py               # full pipeline entrypoint
│   ├── debug_single_doc.py      # inspect one document
│   └── benchmark.py            # measure Levenshtein score
├── tests/
│   └── test_phase1_mapping.py
├── debug/                       # per-document debug artifacts
├── .ocr_cache/                  # cached OCR results
├── checkpoint.json
└── submission.csv
```

---

## Quick Start

```bash
# Install dependencies
uv sync

# Run Phase 1 sanity check (inventory summary)
uv run python -c "
from src.phase1_mapping.mapping import build_inventory, print_inventory_summary
groups, pages = build_inventory()
print_inventory_summary(groups, pages)
"

# Debug a single document
uv run python scripts/debug_single_doc.py constituency_10_1 --verbose

# Run all tests
uv run pytest tests/ -v

# Run full pipeline (phases 2–16 are stubs until implemented)
uv run python main.py
```

---

## Phase 0 — Project Structure

Redesigned from a flat monolithic `main.py` to a modular per-phase layout.

**Key design decisions:**

- Each phase lives in its own `src/phaseN_*` package — isolated, independently testable
- `src/config.py` holds all constants and feature flags (no magic numbers scattered across files)
- `src/pipeline/runner.py` orchestrates all phases with `ThreadPoolExecutor` for parallel doc processing
- `src/utils/checkpoint.py` allows resuming an interrupted run
- `src/utils/cache.py` caches OCR results by image MD5 to avoid redundant API calls
- `debug/` stores per-document artifacts (raw OCR text, parsed JSON, etc.) when `DEBUG_MODE=True`

---

## Phase 1 — Data Inventory & ID Mapping

**What it does:**

1. Load `submission_template_v4.csv` (10 053 rows)
2. Parse each ID into `(id_type, province, district, row_number)`:
   - `constituency_{P}_{D}_{R}` — 1 502 rows
   - `party_list_{P}_{D}_{R}` — 8 551 rows
3. Group rows by `(id_type, province, district)` → `DocumentGroup`
4. Locate physical image files for each document (pages 1–4)

**Key insight discovered:** The template contains two ID families.
`party_list` entries share the same physical scan files as their `constituency` counterpart — the image lookup normalises `party_list_P_D` → `constituency_P_D` when searching `data/images/`.

**Output:** `DocumentGroup` objects with sorted `SubmissionRow` lists, ready for downstream phases.

---

## Evaluation Metric

**Levenshtein distance** (lower = better, 0 = perfect match).

Calculated string-by-string on the `votes` column.
Use `scripts/benchmark.py` to measure against a ground truth CSV.

---

## Feature Flags (`src/config.py`)

| Flag | Default | Effect |
|------|---------|--------|
| `USE_ENSEMBLE` | `True` | Enable multi-pass OCR ensemble |
| `USE_THAI_CHECK` | `True` | Enable Thai digit cross-check |
| `USE_CACHE` | `True` | Cache OCR results to `.ocr_cache/` |
| `USE_CHECKPOINT` | `True` | Resume interrupted runs |
| `DEBUG_MODE` | `False` | Save intermediate artifacts to `debug/` |
| `PARALLEL_WORKERS` | `4` | ThreadPoolExecutor worker count |
