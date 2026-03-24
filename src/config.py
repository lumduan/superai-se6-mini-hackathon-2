"""Central configuration — all constants and feature flags live here."""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
IMAGES_DIR = DATA_DIR / "images"
INTERIM_DIR = DATA_DIR / "interim"
DEBUG_DIR = ROOT_DIR / "debug"
CACHE_DIR = ROOT_DIR / ".ocr_cache"
SUBMISSION_TEMPLATE = DATA_DIR / "submission_template_v4.csv"
CHECKPOINT_FILE = ROOT_DIR / "checkpoint.json"
SUBMISSION_OUTPUT = ROOT_DIR / "submission.csv"

# ── Vote value constraints ─────────────────────────────────────────────────
MIN_VOTE = 0
MAX_VOTE = 1_000_000

# ── OCR / confidence ───────────────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.6
DIGIT_LENGTH_MIN = 1
DIGIT_LENGTH_MAX = 7

# ── Preprocessing ──────────────────────────────────────────────────────────
MAX_DESKEW_ANGLE = 5.0  # degrees — beyond this, skip rotation

# ── Feature flags (enable/disable phases for quick experimentation) ────────
USE_ENSEMBLE = True
USE_THAI_CHECK = True
USE_CACHE = True
USE_CHECKPOINT = True
DEBUG_MODE = False
PARALLEL_WORKERS = 4
