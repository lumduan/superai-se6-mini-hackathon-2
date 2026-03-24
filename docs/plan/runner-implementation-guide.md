# Runner Implementation Guide

## Overview

The pipeline runner has 2 modes:

| Mode | When to use | Entry point |
|------|-------------|-------------|
| **Full run** | Actual submission | `python scripts/run_all.py` |
| **Smoke test** | Testing the pipeline | `python scripts/smoke_test.py` |

---

## Architecture

```
scripts/
├── run_all.py          ← full run (no menu)
└── smoke_test.py       ← Rich menu-driven (for testing)

src/pipeline/
└── runner.py           ← core logic shared by both scripts
```

`runner.py` exposes a single function `run_pipeline()` that accepts these parameters:

```python
def run_pipeline(
    template_path: str | Path = SUBMISSION_TEMPLATE,
    images_dir: str | Path = IMAGES_DIR,
    output_path: str | Path = SUBMISSION_OUTPUT,
    checkpoint_file: str | Path = CHECKPOINT_FILE,
    limit: int | None = None,           # limit number of documents
    doc_keys: list[str] | None = None,  # specify docs directly
) -> None:
```

---

## `src/pipeline/runner.py`

```python
"""Pipeline orchestrator — wires all phases together."""
from __future__ import annotations
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from src.config import (
    CHECKPOINT_FILE,
    PARALLEL_WORKERS,
    SUBMISSION_OUTPUT,
    SUBMISSION_TEMPLATE,
    USE_CHECKPOINT,
    IMAGES_DIR,
)
from src.phase1_mapping.mapping import DocumentGroup, build_inventory
from src.utils.checkpoint import is_done, load_checkpoint, mark_done, save_checkpoint
from src.utils.io import save_csv
import pandas as pd

logger = logging.getLogger(__name__)


def process_document(
    doc_key: str,
    group: DocumentGroup,
    pages: list[Path],
) -> dict[str, int]:
    """Process one document through all phases. Returns {id: votes}."""
    logger.info(
        "Processing %s (%d rows, %d pages)",
        doc_key, group.expected_row_count, len(pages),
    )
    results: dict[str, int] = {row.id: 0 for row in group.rows}

    # TODO Phase 2: table page detection
    # TODO Phase 3: preprocess (deskew + CLAHE + sharpen)
    # TODO Phase 5: full-page OCR (primary)
    # TODO Phase 6: parse HTML table output
    # TODO Phase 7: Thai text cross-check
    # TODO Phase 8: normalize + hard rules
    # TODO Phase 9: row validate + total correction
    # TODO Phase 10: confidence scoring
    # TODO Phase 11: ensemble fallback OCR
    # TODO Phase 12: anchor alignment
    # TODO Phase 13: final row alignment → results

    return results


def run_pipeline(
    template_path: str | Path = SUBMISSION_TEMPLATE,
    images_dir: str | Path = IMAGES_DIR,
    output_path: str | Path = SUBMISSION_OUTPUT,
    checkpoint_file: str | Path = CHECKPOINT_FILE,
    limit: int | None = None,
    doc_keys: list[str] | None = None,
) -> None:
    """Main entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    groups, pages = build_inventory(template_path, images_dir)
    all_keys = sorted(groups.keys())

    # Filter keys by mode
    if doc_keys is not None:
        run_keys = [k for k in doc_keys if k in groups]
        missing = [k for k in doc_keys if k not in groups]
        if missing:
            logger.warning("doc_keys not found: %s", missing)
    elif limit is not None:
        run_keys = all_keys[:limit]
    else:
        run_keys = all_keys

    logger.info(
        "Running %d / %d documents%s",
        len(run_keys), len(all_keys),
        f" (limit={limit})" if limit else "",
    )

    state = (
        load_checkpoint(checkpoint_file)
        if USE_CHECKPOINT
        else {"done": [], "results": {}}
    )
    all_results: dict[str, int] = {}

    def _process(doc_key: str) -> tuple[str, dict]:
        if USE_CHECKPOINT and is_done(state, doc_key):
            logger.info("Skipping %s (already done)", doc_key)
            return doc_key, state["results"][doc_key]
        result = process_document(doc_key, groups[doc_key], pages[doc_key])
        return doc_key, result

    with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as pool:
        futures = {pool.submit(_process, dk): dk for dk in run_keys}
        for future in as_completed(futures):
            doc_key, result = future.result()
            all_results.update(result)
            mark_done(state, doc_key, result)
            if USE_CHECKPOINT:
                save_checkpoint(checkpoint_file, state)

    rows = [{"id": k, "votes": v} for k, v in all_results.items()]
    submission = pd.DataFrame(rows)
    save_csv(submission, output_path)
    logger.info("Done — %d rows written to %s", len(submission), output_path)
```

---

## `scripts/smoke_test.py` — Rich Menu

Install Rich first:

```bash
pip install rich
```

```python
#!/usr/bin/env python3
"""Smoke test — Rich interactive menu for testing the pipeline."""
from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt, Confirm
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from src.pipeline.runner import run_pipeline
from src.config import SUBMISSION_TEMPLATE, IMAGES_DIR

console = Console()


def show_header() -> None:
    console.print(Panel(
        Text("Thai Election OCR — Smoke Test", justify="center", style="bold cyan"),
        subtitle="Super AI Engineer S6",
        border_style="cyan",
    ))


def pick_mode() -> str:
    """Select main run mode."""
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column(style="bold yellow")
    table.add_column()
    table.add_row("1", "Quick test   — run first N docs")
    table.add_row("2", "Target test  — specify doc_key directly")
    table.add_row("3", "Full run     — run all (takes a long time)")
    table.add_row("q", "quit")
    console.print(table)
    return Prompt.ask("\nSelect mode", choices=["1", "2", "3", "q"], default="1")


def configure_quick() -> dict:
    """Mode 1: Quick test."""
    limit = IntPrompt.ask("Number of documents", default=10)
    output = Prompt.ask("Output file", default="smoke_output.csv")
    return {"limit": limit, "output_path": output}


def configure_target() -> dict:
    """Mode 2: Target specific docs."""
    console.print("\n[dim]Enter doc_keys separated by spaces, e.g. constituency_1_1 constituency_2_3[/dim]")
    raw = Prompt.ask("doc_keys")
    doc_keys = raw.strip().split()
    output = Prompt.ask("Output file", default="smoke_output.csv")
    return {"doc_keys": doc_keys, "output_path": output}


def configure_full() -> dict:
    """Mode 3: Full run."""
    confirmed = Confirm.ask(
        "[yellow]Full run takes a very long time. Are you sure?[/yellow]",
        default=False,
    )
    if not confirmed:
        return {}
    output = Prompt.ask("Output file", default="submission.csv")
    return {"output_path": output}


def show_summary(config: dict) -> None:
    """Display config before running."""
    table = Table(title="Run config", border_style="dim")
    table.add_column("Key", style="cyan")
    table.add_column("Value")
    for k, v in config.items():
        table.add_row(k, str(v))
    console.print(table)


def main() -> None:
    show_header()

    mode = pick_mode()
    if mode == "q":
        console.print("[dim]Exiting smoke test[/dim]")
        sys.exit(0)

    console.print()

    if mode == "1":
        config = configure_quick()
    elif mode == "2":
        config = configure_target()
    else:
        config = configure_full()
        if not config:
            console.print("[yellow]Cancelled[/yellow]")
            sys.exit(0)

    console.print()
    show_summary(config)
    console.print()

    if not Confirm.ask("Start running now?", default=True):
        console.print("[yellow]Cancelled[/yellow]")
        sys.exit(0)

    console.print("\n[bold green]Starting pipeline...[/bold green]\n")

    try:
        run_pipeline(**config)
        console.print(f"\n[bold green]✓ Done[/bold green] — output: [cyan]{config['output_path']}[/cyan]")
    except Exception as e:
        console.print(f"\n[bold red]✗ Error:[/bold red] {e}")
        raise


if __name__ == "__main__":
    main()
```

---

## `scripts/run_all.py` — Full Run (no menu)

```python
#!/usr/bin/env python3
"""Full pipeline run — no prompts, no limits."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.runner import run_pipeline

if __name__ == "__main__":
    run_pipeline()
```

---

## Smoke Test UX Flow

```
┌─────────────────────────────────────┐
│   Thai Election OCR — Smoke Test    │
└─────────────────────────────────────┘

  1  Quick test   — run first N docs
  2  Target test  — specify doc_key directly
  3  Full run     — run all (takes a long time)
  q  quit

Select mode [1]: _
```

### Mode 1 — Quick Test
```
Number of documents [10]: 20
Output file [smoke_output.csv]: _

┌─ Run config ─────────────────────┐
│ limit      │ 20                  │
│ output     │ smoke_output.csv    │
└──────────────────────────────────┘

Start running now? [Y/n]: _
```

### Mode 2 — Target Test
```
Enter doc_keys separated by spaces
doc_keys: constituency_1_1 constituency_2_3 constituency_5_1
Output file [smoke_output.csv]: _
```

---

## Dependencies

```toml
# pyproject.toml or requirements.txt
rich>=13.0
```

---

## Notes

- **Checkpoint**: The smoke test uses the same checkpoint as the full run. To run fresh, delete `checkpoint.json` first, or add a `--no-checkpoint` flag later.
- **Separate outputs**: Smoke test defaults to `smoke_output.csv`, full run to `submission.csv` — they do not overwrite each other.
- **PARALLEL_WORKERS**: If the Typhoon API rate limit is an issue, set `PARALLEL_WORKERS = 1` in `config.py` during testing.
