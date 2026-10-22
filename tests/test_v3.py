import asyncio

import pytest

from patient_zero.attacks import make_attack
from patient_zero.layouts import LAYOUTS
from patient_zero.models import model_for
from patient_zero.runlog import RunMeta
from patient_zero.runner import RunSpec, run_one


def test_v3_override_targets_analyst_with_v1_false_answer(tasks):
    t = tasks[0]
    inj = make_attack("v3", tasks).apply(t)
    assert inj.target_role == "analyst" and inj.docs is None and not inj.extra_docs
    assert inj.marker == make_attack("v1", tasks).apply(t).marker
    assert inj.marker in inj.system_override and inj.meta["style"] == "v3:template"


def test_v3_rejects_researcher_as_adversary(tasks):
    with pytest.raises(ValueError):
        make_attack("v3", tasks, target_role="researcher")


def test_v3_override_reaches_only_the_adversary(tasks):
    t = tasks[0]
    inj = make_attack("v3", tasks).apply(t)
    meta = RunMeta(run_id="r", exp_id="t", layout="mesh", vector="v3", condition="adversarial", model="fake", task_id=t.id)
    graph = LAYOUTS["mesh"](model_for("fake"), rounds=2)
    run, msgs = asyncio.run(run_one(RunSpec(meta, t, inj, probe=True), graph, ["researcher"]))
    assert run.error is None
    flagged = {m.sender for m in msgs if m.injected_into_sender}
    assert flagged == {"analyst"}
    # researcher keeps its clean documents
    assert [d["text"] for d in run.role_docs["researcher"]] == t.docs
