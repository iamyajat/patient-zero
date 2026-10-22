"""Autograder. Point it at the hidden data and the instructor files:

  PZ_HIDDEN=teaching_data/hidden/runs.jsonl PZ_TRUTH=teaching_data/instructor/truth.json \
  PZ_CONDITIONS=teaching_data/instructor/hidden_conditions.json PZ_SUBMISSION=submission.py \
  uv run pytest teaching/find_patient_zero/autograder -q
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

from teaching.find_patient_zero.grade import grade_part_a, grade_part_b, load_runs


@pytest.fixture(scope="module")
def data():
    for var in ["PZ_HIDDEN", "PZ_TRUTH", "PZ_CONDITIONS", "PZ_SUBMISSION"]:
        if not os.environ.get(var):
            pytest.skip(f"{var} not set")
    spec = importlib.util.spec_from_file_location("submission", os.environ["PZ_SUBMISSION"])
    sub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sub)
    return (sub, load_runs(os.environ["PZ_HIDDEN"]), json.loads(Path(os.environ["PZ_TRUTH"]).read_text()),
            json.loads(Path(os.environ["PZ_CONDITIONS"]).read_text()))


def test_part_a_patient_zero_and_r_hat(data):
    sub, runs, truth, _ = data
    result = grade_part_a(sub, runs, truth)
    print(json.dumps(result, indent=1))
    assert result["passed"], result


def test_part_b_monitor_within_false_alarm_budget(data):
    sub, runs, truth, conditions = data
    result = grade_part_b(sub, runs, truth, conditions)
    print(json.dumps(result, indent=1))
    assert result["within_budget"], result
