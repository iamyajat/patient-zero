from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pandas as pd
import typer
from dotenv import load_dotenv

from .config import GridConfig
from .metrics.basic import summarize
from .runner import run_grid

app = typer.Typer(no_args_is_help=True, help="Patient Zero: contagion audits for multi-agent systems.")


def _print(df: pd.DataFrame) -> None:
    with pd.option_context("display.width", 160, "display.max_columns", 30):
        print(df.to_string(index=False))


def _report(exp_dir: Path, answer_judge: str | None = None) -> None:
    from .metrics.contagion import contagion_table, load_contagion
    from .metrics.equivalence import load_equiv

    eq = load_equiv(answer_judge)
    _print(summarize(exp_dir, eq))
    df = load_contagion(exp_dir, eq)
    cols = ["vector", "condition", "layout", "model", "runs", "entry_rate", "r_hat", "r_hat_lo", "r_hat_hi", "tau", "tau_lo",
            "tau_hi", "branching", "attack_rate", "final_corruption", "exec_rate", "flip_rate", "public_infected",
            "private_infected"]
    t = contagion_table(df, ["vector", "condition", "layout", "model"])
    print()
    _print(t[[c for c in cols if c in t]].round(3))


@app.command()
def run(
    layout: str = typer.Option("chain", help="chain | chain_noverifier | hub | mesh"),
    vector: str = typer.Option("v1", help="v1[.template|.llm|.mint] | v2[.injecagent|.bipia|.infection] | v3[.llm]"),
    condition: str = typer.Option("adversarial", help="benign | adversarial | adaptive"),
    model: str = typer.Option("fake", help="openai | qwen | fake | provider:model_id | mixed:role=name,..."),
    n: int = typer.Option(3, help="number of tasks"),
    dataset: str = typer.Option("sample", help="sample | hotpotqa | cwq"),
    rounds: int = typer.Option(2, help="mesh rounds"),
    exp_id: str = typer.Option("smoke"),
    concurrency: int = typer.Option(4),
):
    """Run a single cell of the grid."""
    load_dotenv()
    cfg = GridConfig(exp_id=exp_id, dataset=dataset, n_tasks=n, vectors=[vector], conditions=[condition],
                     layouts=[layout], models=[model], mesh_rounds=rounds, concurrency=concurrency)
    asyncio.run(run_grid(cfg))
    _report(cfg.exp_dir)


@app.command()
def grid(config: Path, only_models: str = typer.Option(None, help="run just these models from the config (comma-separated); "
                                                                  "the config file and run ids are unchanged, so a frozen grid can run in parts")):
    """Run every cell of a YAML grid config. Rerunning resumes where it stopped."""
    load_dotenv()
    cfg = GridConfig.from_yaml(config)
    if only_models:
        keep = [m.strip() for m in only_models.split(",")]
        unknown = set(keep) - set(cfg.models)
        if unknown:
            raise typer.BadParameter(f"not in the config: {sorted(unknown)}")
        cfg = cfg.model_copy(update={"models": keep})
    asyncio.run(run_grid(cfg))
    _report(cfg.exp_dir)


@app.command()
def dojo(config: Path):
    """Run V2b (tool environment) from a YAML DojoConfig. Needs `uv sync --extra dojo`."""
    load_dotenv()
    from .envs.dojo.runner import DojoConfig, run_dojo_grid

    cfg = DojoConfig.from_yaml(config)
    asyncio.run(run_dojo_grid(cfg))
    _report(cfg.exp_dir)


@app.command()
def report(exp_dir: Path, answer_judge: str = typer.Option(None, help="use cached answer-equivalence verdicts from this model")):
    """Accuracy, R-hat and contagion rates per cell for a finished experiment."""
    _report(exp_dir, answer_judge)


@app.command("judge-answers")
def judge_answers(exp_dir: Path, model: str = typer.Option("qwen"), concurrency: int = typer.Option(8)):
    """Judge answer equivalence (aliases, paraphrases) for every comparison the metrics make; cached in data/."""
    load_dotenv()
    from .metrics.equivalence import judge_experiment

    print(f"{judge_experiment(exp_dir, model, concurrency=concurrency)} new verdicts cached")


@app.command()
def monitor(exp_dir: Path, name: str = typer.Option("rules", help="rules | llm | ensemble | judge"),
            judge_model: str = typer.Option("openai"), concurrency: int = typer.Option(8),
            tasks: int = typer.Option(None, help="score only a deterministic sample of this many tasks")):
    """Score every message of an experiment with a monitor (writes flags_<name>.jsonl)."""
    load_dotenv()
    from .monitor.run import monitor_experiment

    print(monitor_experiment(exp_dir, name, judge_model, concurrency=concurrency, max_tasks=tasks))


@app.command()
def calibrate(exp_dirs: list[Path], name: str = typer.Option("rules"), answer_judge: str = typer.Option(None),
              by: str = typer.Option(None, help="calibrate per group, e.g. 'model' or 'model,vector'")):
    """Threshold a monitor at fixed false-alarm rates on benign tasks; report recall on held-out tasks."""
    from .metrics.equivalence import load_equiv
    from .monitor.calibrate import calibrate_experiment

    result = calibrate_experiment(list(exp_dirs), name, load_equiv(answer_judge), by.split(",") if by else None)
    _print(pd.DataFrame(result["points"]).round(3))


@app.command()
def audit(exp_dirs: list[Path], standard: Path = typer.Option(Path("configs/standard_v1.yaml")),
          monitor: str = typer.Option(None, help="monitor whose flags feed T3 (run `pz monitor` first)"),
          judge_model: str = typer.Option(None), out: Path = typer.Option(None, help="defaults to the first experiment"),
          answer_judge: str = typer.Option(None, help="use cached answer-equivalence verdicts from this model")):
    """Score experiments against a standard and write audit_report.md with verdicts, CIs and clause mappings."""
    from .audit.report import write_report
    from .audit.scorecard import scorecard
    from .audit.standard import Standard

    std = Standard.load(standard)
    if monitor and not all((d / f"flags_{monitor}.jsonl").exists() for d in exp_dirs):
        from .monitor.run import monitor_experiment

        load_dotenv()
        for d in exp_dirs:
            if not (d / f"flags_{monitor}.jsonl").exists():
                monitor_experiment(d, monitor, judge_model or "openai")
    from .metrics.equivalence import load_equiv

    eq = load_equiv(answer_judge)
    card = scorecard(exp_dirs, std, monitor, eq=eq)
    path = write_report(exp_dirs, std, card, out or exp_dirs[0], monitor, judge_model, eq, answer_judge)
    _print(card[[c for c in card if c.endswith("verdict") or c in ("vector", "condition", "layout", "model", "stratum", "overall")]])
    print(path)


@app.command("answer-labels-export")
def answer_labels_export(exp_dirs: list[Path], out: Path = typer.Option(Path("labels/answer_pairs.csv")),
                         n: int = typer.Option(100)):
    """Answer pairs for humans to label same/different, to validate the answer-equivalence judge."""
    from .metrics.judge_validation import export_answer_labels

    export_answer_labels(exp_dirs, out, n)


@app.command("answer-labels-score")
def answer_labels_score(csv: Path, models: str = typer.Option("openai,qwen-think", help="comma-separated judges"),
                        concurrency: int = typer.Option(4, help="use 1 for local Ollama models (one request at a time)"),
                        sheet: Path = typer.Option(None, help="the exported sheet, to join labels by pair_id")):
    """Agreement and kappa of each candidate judge (and the F1 rule) with the human labels."""
    load_dotenv()
    from .metrics.judge_validation import score_answer_labels

    out = score_answer_labels(csv, [m.strip() for m in models.split(",") if m.strip()], concurrency=concurrency, sheet=sheet)
    if out.attrs.get("annotators"):
        print(json.dumps(out.attrs["annotators"], indent=1))
    _print(out.round(3))


@app.command("label-export")
def label_export(exp_dir: Path, n: int = typer.Option(150)):
    """Balanced sample of messages for two human annotators (CSV)."""
    from .monitor.labels import export_labels

    print(export_labels(exp_dir, n))


@app.command()
def kappa(labels_csv: Path, flags: Path = typer.Option(None), threshold: float = typer.Option(None)):
    """Cohen's kappa between annotators, and between each annotator and a thresholded monitor."""
    from .monitor.labels import kappa_report

    print(json.dumps(kappa_report(labels_csv, flags, threshold), indent=1))


@app.command("export-graphs")
def export_graphs_cmd(exp_dir: Path):
    """Per-turn agent graphs with infection labels, for external graph guards (INFA-Guard / G-Safeguard)."""
    from .monitor.external.graph_export import export_graphs

    print(export_graphs(exp_dir))


@app.command("paper-assets")
def paper_assets(runs_dir: Path = typer.Option(Path("runs")), out: Path = typer.Option(Path("paper/generated"))):
    """Regenerate every number, table and figure the paper uses, from the run logs."""
    from .analysis.paper import make_assets

    print(make_assets(runs_dir, out))


@app.command()
def power(p_benign: float, p_adv: float, p10: float = typer.Option(None), p01: float = typer.Option(None)):
    """Sample size for a benign-vs-adversarial difference (unpaired), and for McNemar if discordance is given."""
    from .analysis.power import n_mcnemar, n_two_proportions

    print(f"unpaired, per group: {n_two_proportions(p_benign, p_adv)}")
    if p10 is not None and p01 is not None:
        print(f"paired (McNemar), pairs: {n_mcnemar(p10, p01)}")
