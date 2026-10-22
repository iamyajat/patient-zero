import asyncio
import json
import types

from teaching.find_patient_zero import reference_solution
from teaching.find_patient_zero.generate import package, simulate
from teaching.find_patient_zero.grade import grade_part_a, grade_part_b, load_runs


def _data(tmp_path):
    runs, msgs = asyncio.run(simulate(tmp_path, secret=7, n_tasks=3))
    package(runs, msgs, tmp_path, secret=7, hidden_share=1.0)
    return (load_runs(tmp_path / "hidden" / "runs.jsonl"), json.loads((tmp_path / "instructor" / "truth.json").read_text()),
            json.loads((tmp_path / "instructor" / "hidden_conditions.json").read_text()))


def test_reference_passes_and_flag_everything_fails(tmp_path):
    runs, truth, conditions = _data(tmp_path)
    assert runs and all("condition" not in r for r in runs)  # hidden runs carry no labels
    assert all("injected_into_sender" not in m for r in runs for m in r["messages"])
    a = grade_part_a(reference_solution, runs, truth)
    assert a["passed"], a
    b = grade_part_b(reference_solution, runs, truth, conditions)
    assert b["within_budget"] and b["recall"] > 0, b
    everything = types.SimpleNamespace(THRESHOLD=0.5, score_message=lambda m, c: 1.0)
    bad = grade_part_b(everything, runs, truth, conditions)
    assert not bad["within_budget"] and bad["score"] == 0.0


def test_hidden_split_depends_on_secret(tmp_path):
    runs, msgs = asyncio.run(simulate(tmp_path, secret=7, n_tasks=3))
    package(runs, msgs, tmp_path / "a", secret=1)
    package(runs, msgs, tmp_path / "b", secret=2)
    ids = lambda d: {json.loads(l)["run_id"] for l in (d / "hidden" / "runs.jsonl").read_text().splitlines() if l}
    assert ids(tmp_path / "a") != ids(tmp_path / "b") or not ids(tmp_path / "a")
