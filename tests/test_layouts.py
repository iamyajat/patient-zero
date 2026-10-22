import asyncio

from patient_zero.layouts import LAYOUTS
from patient_zero.models import model_for
from patient_zero.runlog import RunMeta
from patient_zero.runner import RunSpec, run_one


def _run(layout, task, rounds=2, model="fake", probe=False, guard=None, injection=None):
    meta = RunMeta(run_id=f"r-{layout}", exp_id="t", layout=layout, vector="v1", condition="benign", model=model, task_id=task.id)
    graph = LAYOUTS[layout](model_for(model), rounds=rounds)
    return asyncio.run(run_one(RunSpec(meta, task, injection, probe=probe), graph, ["researcher"], guard))


def test_chain_each_agent_sees_previous(tasks):
    run, msgs = _run("chain", tasks[0])
    assert run.error is None
    assert [m.sender for m in msgs] == ["researcher", "analyst", "verifier", "writer"]
    assert msgs[0].visible_inputs == []
    for prev, cur in zip(msgs, msgs[1:]):
        assert cur.visible_inputs == [prev.id]
    assert [m.step for m in msgs] == [0, 1, 2, 3]
    assert run.final_answer == "fake"


def test_only_researcher_gets_docs(tasks):
    run, msgs = _run("chain", tasks[0])
    assert msgs[0].doc_inputs == [f"doc:{tasks[0].id}:0", f"doc:{tasks[0].id}:1"]
    assert all(m.doc_inputs == [] for m in msgs[1:])
    assert [d["text"] for d in run.role_docs["researcher"]] == tasks[0].docs


def test_chain_noverifier_has_three_hops(tasks):
    run, msgs = _run("chain_noverifier", tasks[0])
    assert [m.sender for m in msgs] == ["researcher", "analyst", "writer"]
    assert msgs[2].visible_inputs == [msgs[1].id]
    assert run.final_answer == "fake"


def test_hub_workers_see_only_dispatch(tasks):
    run, msgs = _run("hub", tasks[0])
    by_id = {m.id: m for m in msgs}
    dispatch = by_id["r-hub:coordinator_dispatch"]
    assert dispatch.sender == "writer" and dispatch.answer is None
    workers = [m for m in msgs if m.sender != "writer"]
    assert {m.sender for m in workers} == {"researcher", "analyst", "verifier"}
    assert all(m.visible_inputs == [dispatch.id] for m in workers)
    assert len({m.step for m in workers}) == 1  # parallel superstep
    final = by_id["r-hub:coordinator_final"]
    assert sorted(final.visible_inputs) == sorted(m.id for m in workers)
    assert run.n_messages == 5


def test_mesh_rounds_see_previous_round_of_others(tasks):
    run, msgs = _run("mesh", tasks[0], rounds=3)
    assert run.n_messages == 12
    for m in msgs:
        expected = {o.id for o in msgs if o.round == m.round - 1 and o.sender != m.sender}
        assert set(m.visible_inputs) == expected
    assert sorted({m.round for m in msgs}) == [0, 1, 2]


def test_mixed_model_routes_per_role(tasks):
    _, msgs = _run("chain", tasks[0], model="mixed:researcher=fake:a,verifier=fake:c,default=fake:b")
    tags = {m.sender: m.content.split(":", 1)[0] for m in msgs}
    assert tags == {"researcher": "Echo[a]", "analyst": "Echo[b]", "verifier": "Echo[c]", "writer": "Echo[b]"}


def test_probes_are_logged_but_never_seen(tasks):
    for layout in ["chain", "hub", "mesh"]:
        run, msgs = _run(layout, tasks[0], probe=True)
        probes = [m for m in msgs if m.phase == "probe"]
        public = [m for m in msgs if m.phase != "probe"]
        assert len(probes) == sum(m.answer is not None for m in public)  # no probe after the hub dispatch
        assert all(p.recipients == [] and p.id.endswith(":probe") for p in probes)
        probe_ids = {p.id for p in probes}
        assert not any(probe_ids & set(m.visible_inputs) for m in msgs)
        assert run.final_answer == "fake" and run.probe


class QuarantineResearcher:
    name = "test"
    threshold = 0.5

    async def score(self, message, docs, question=""):
        return 1.0 if message.sender == "researcher" else 0.0


def test_guard_quarantine_cuts_the_edge(tasks):
    run, msgs = _run("chain", tasks[0], guard=QuarantineResearcher())
    researcher, analyst = msgs[0], msgs[1]
    assert researcher.quarantined and researcher.guard_score == 1.0
    assert analyst.visible_inputs == []  # nothing older leaks through in place of the withheld message
    assert not analyst.quarantined
    assert run.guard == "test"
