"""OCR result cache — avoid re-running expensive OCR on unchanged images."""

import hashlib
import logging
from pathlib import Path

from src.utils.io import load_json, save_json

logger = logging.getLogger(__name__)


def _image_hash(image_path: str | Path) -> str:
    with open(image_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def cache_key(image_path: str | Path, pass_id: str = "default") -> str:
    return f"{_image_hash(image_path)}_{pass_id}"


def load_cached(cache_dir: str | Path, key: str) -> dict | None:
    path = Path(cache_dir) / f"{key}.json"
    if path.exists():
        logger.debug("Cache hit: %s", key)
        return load_json(path)
    return None


def save_cached(cache_dir: str | Path, key: str, data: dict) -> None:
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    save_json(data, Path(cache_dir) / f"{key}.json")
    logger.debug("Cache saved: %s", key)
