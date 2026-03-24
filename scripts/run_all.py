#!/usr/bin/env python3
"""Full pipeline run — no prompts, no limits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.runner import run_pipeline

if __name__ == "__main__":
    run_pipeline()
