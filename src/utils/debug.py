"""Debug artifact helpers — save per-document intermediate outputs."""

import logging
from pathlib import Path

from src.config import DEBUG_DIR, DEBUG_MODE
from src.utils.io import save_json

logger = logging.getLogger(__name__)


def debug_dir(doc_key: str) -> Path:
    return DEBUG_DIR / doc_key


def debug_save_text(doc_key: str, stage: str, text: str) -> None:
    if not DEBUG_MODE:
        return
    path = debug_dir(doc_key) / f"{stage}.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    logger.debug("Debug saved: %s", path)


def debug_save_json(doc_key: str, stage: str, data: dict | list) -> None:
    if not DEBUG_MODE:
        return
    path = debug_dir(doc_key) / f"{stage}.json"
    save_json(data, path)
    logger.debug("Debug saved: %s", path)


def debug_save_image(doc_key: str, stage: str, image_path: str | Path) -> None:
    """Copy an intermediate image into the debug folder."""
    if not DEBUG_MODE:
        return
    import shutil
    dest = debug_dir(doc_key) / f"{stage}.png"
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, dest)
    logger.debug("Debug image saved: %s", dest)
