"""I/O helpers — load/save files, path utilities."""

import json
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.debug("Loaded %s rows from %s", len(df), path)
    return df


def save_csv(df: pd.DataFrame, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Saved %s rows → %s", len(df), path)


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(data: dict | list, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def ensure_dirs(*paths: str | Path) -> None:
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)
