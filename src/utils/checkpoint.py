"""Checkpoint — save/resume pipeline progress per document."""

import logging
from pathlib import Path

from src.utils.io import load_json, save_json

logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_file: str | Path) -> dict:
    if Path(checkpoint_file).exists():
        data = load_json(checkpoint_file)
        logger.info("Resumed checkpoint: %s docs done", len(data.get("done", [])))
        return data
    return {"done": [], "results": {}}


def save_checkpoint(checkpoint_file: str | Path, state: dict) -> None:
    save_json(state, checkpoint_file)
    logger.debug("Checkpoint saved (%s done)", len(state.get("done", [])))


def mark_done(state: dict, doc_key: str, result: dict) -> dict:
    if doc_key not in state["done"]:
        state["done"].append(doc_key)
    state["results"][doc_key] = result
    return state


def is_done(state: dict, doc_key: str) -> bool:
    return doc_key in state["done"]
