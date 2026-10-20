from patient_zero.layouts import LAYOUTS
from patient_zero.models import get_chat_model
from patient_zero.runner import RunSpec, initial_state, run_one
from patient_zero.runlog import RunMeta


def _run(layout, task, rounds=2):
    meta = RunMeta(run_id=f"r-{layout}", exp_id="t", layout=layout, vector="v1", condition="benign", model="fake", task_id=task.id)
    graph = LAYOUTS[layout](get_chat_model("fake"), rounds=rounds)
    import asyncio

    return asyncio.run(run_one(RunSpec(meta, task, None), graph, ["researcher"]))


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
    _, msgs = _run("chain", tasks[0])
    assert "Source documents" in msgs[0].content
    assert all("Source documents:\n[1]" not in m.content.split("Messages from your teammates")[0] for m in msgs[1:])


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
