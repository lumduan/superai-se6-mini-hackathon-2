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
