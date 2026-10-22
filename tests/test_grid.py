from patient_zero.config import GridConfig
from patient_zero.runner import expand_grid, fingerprint, subsample


def test_layouts_by_vector_and_repeats(tasks):
    cfg = GridConfig(
        exp_id="t", vectors=["v1"], conditions=["benign"], layouts=["chain", "hub"],
        layouts_by_vector={"v1": ["chain"]}, models=["fake"], repeats=3, repeat_fraction=0.34,
    )
    specs = expand_grid(cfg, {"v1": tasks})
    assert {s.meta.layout for s in specs} == {"chain"}
    repeated = subsample(tasks, 0.34)
    assert len(repeated) == 2
    assert len(specs) == len(tasks) + 2 * len(repeated)
    ids = [s.meta.run_id for s in specs]
    assert len(set(ids)) == len(ids)
    assert any(i.endswith("-s2") for i in ids) and not any(i.endswith("-s0") for i in ids)


def test_probe_subset_is_deterministic(tasks):
    cfg = GridConfig(exp_id="t", vectors=["v1"], conditions=["benign"], layouts=["mesh"], models=["fake"], probe_fraction=0.34)
    a = {s.meta.task_id for s in expand_grid(cfg, {"v1": tasks}) if s.probe}
    b = {s.meta.task_id for s in expand_grid(cfg, {"v1": tasks}) if s.probe}
    assert a == b and len(a) == 2


def test_fingerprint_tracks_model_layout_and_guard():
    base = fingerprint("fake", "chain", 2, None)
    assert base == fingerprint("fake", "chain", 2, None)
    assert base != fingerprint("fake:other", "chain", 2, None)
    assert base != fingerprint("fake", "chain_noverifier", 2, None)
    assert base != fingerprint("fake", "chain", 2, "rules")
    assert base == fingerprint("fake", "chain", 3, None)  # rounds only matter for mesh
    assert fingerprint("qwen", "mesh", 2, None) != fingerprint("qwen", "mesh", 3, None)


def test_probe_style_changes_fingerprint_and_wording(tasks):
    import asyncio

    from patient_zero.agent import PROBES
    from patient_zero.layouts import LAYOUTS
    from patient_zero.models import model_for
    from patient_zero.runlog import RunMeta
    from patient_zero.runner import RunSpec, run_one

    assert fingerprint("fake", "mesh", 2, None, probe_style="neutral") != fingerprint("fake", "mesh", 2, None)
    meta = RunMeta(run_id="r", exp_id="t", layout="chain", vector="v1", condition="benign", model="fake", task_id=tasks[0].id)
    graph = LAYOUTS["chain"](model_for("fake"))
    run, msgs = asyncio.run(run_one(RunSpec(meta, tasks[0], None, probe=True), graph, ["researcher"], probe_style="neutral"))
    probes = [m for m in msgs if m.phase == "probe"]
    assert probes and all(PROBES["neutral"] in p.content for p in probes)  # echo model repeats the probe question
    assert run.probe_style == "neutral"


def test_stuck_run_times_out_and_is_retried(tasks):
    import asyncio

    from patient_zero.runlog import RunMeta
    from patient_zero.runner import RunSpec, run_one

    class Stuck:
        async def ainvoke(self, state):
            await asyncio.sleep(10)

    meta = RunMeta(run_id="r", exp_id="t", layout="chain", vector="v1", condition="benign", model="fake", task_id=tasks[0].id)
    record, msgs = asyncio.run(run_one(RunSpec(meta, tasks[0], None), Stuck(), ["researcher"], timeout=0.05))
    assert record.error.startswith("TimeoutError") and msgs == []  # errored runs are not in completed_run_ids


def test_ollama_client_has_token_cap_and_timeout():
    from patient_zero.models import OLLAMA_MAX_TOKENS, get_chat_model

    m = get_chat_model("qwen")
    assert m.num_predict == OLLAMA_MAX_TOKENS and m.client_kwargs["timeout"] and m.reasoning is False
    assert get_chat_model("qwen-think").reasoning is True
