import asyncio
import json

from patient_zero.metrics.basic import summarize
from patient_zero.runner import run_grid


def test_grid_logs_and_resumes(cfg, tasks):
    c = cfg(conditions=["benign", "adversarial"], layouts=["chain", "hub", "mesh"])
    logger = asyncio.run(run_grid(c, tasks))
    runs = [json.loads(l) for l in logger.runs_path.read_text().splitlines()]
    msgs = [json.loads(l) for l in logger.messages_path.read_text().splitlines()]
    assert len(runs) == 2 * 3 * 3
    assert all(r["error"] is None for r in runs)
    assert len(msgs) == 2 * 3 * (4 + 5 + 8)
    assert all(m["injected_into_sender"] == (m["condition"] == "adversarial" and m["sender"] == "researcher") for m in msgs)

    asyncio.run(run_grid(c, tasks))  # resume: nothing new
    assert len(logger.runs_path.read_text().splitlines()) == len(runs)


def test_adversarial_marker_spreads_with_echo_model(cfg, tasks):
    c = cfg(conditions=["benign", "adversarial"], layouts=["chain"])
    asyncio.run(run_grid(c, tasks))
    s = summarize(c.exp_dir).set_index("condition")
    assert s.loc["adversarial", "adoption"] == 1.0  # echo model repeats everything it sees
    assert s.loc["benign", "adoption"] != s.loc["benign", "adoption"]  # NaN: nothing planted
    assert s.loc["adversarial", "n"] == 3
