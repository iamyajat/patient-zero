from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

from .config import GridConfig
from .metrics.basic import summarize
from .runner import run_grid

app = typer.Typer(no_args_is_help=True, help="Patient Zero: error-propagation tests for agent swarms.")


def _report(exp_dir: Path) -> None:
    with pd.option_context("display.width", 140, "display.max_columns", 20):
        print(summarize(exp_dir).to_string(index=False))


@app.command()
def run(
    layout: str = typer.Option("chain", help="chain | hub | mesh"),
    vector: str = typer.Option("v1"),
    condition: str = typer.Option("adversarial", help="benign | adversarial"),
    model: str = typer.Option("fake", help="openai | qwen | fake | provider:model_id"),
    n: int = typer.Option(3, help="number of tasks"),
    dataset: str = typer.Option("sample", help="sample | hotpotqa"),
    rounds: int = typer.Option(2, help="mesh rounds"),
    exp_id: str = typer.Option("smoke"),
    concurrency: int = typer.Option(4),
):
    """Run a single cell of the grid."""
    load_dotenv()
    cfg = GridConfig(
        exp_id=exp_id,
        dataset=dataset,
        n_tasks=n,
        vectors=[vector],
        conditions=[condition],
        layouts=[layout],
        models=[model],
        mesh_rounds=rounds,
        concurrency=concurrency,
    )
    asyncio.run(run_grid(cfg))
    _report(cfg.exp_dir)


@app.command()
def grid(config: Path):
    """Run every cell of a YAML grid config. Rerunning resumes where it stopped."""
    load_dotenv()
    cfg = GridConfig.from_yaml(config)
    asyncio.run(run_grid(cfg))
    _report(cfg.exp_dir)


@app.command()
def report(exp_dir: Path):
    """Print accuracy and adoption rate per cell for a finished experiment."""
    _report(exp_dir)
