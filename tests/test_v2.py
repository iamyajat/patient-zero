import asyncio

import pytest

from patient_zero.attacks import make_attack
from patient_zero.attacks.v2_instruction import TEMPLATES, template

from .fakes import ScriptedToolModel


@pytest.mark.parametrize("family", sorted(TEMPLATES))
def test_v2a_payload_markers(tasks, family):
    t = tasks[0]
    inj = make_attack(f"v2.{family}", tasks).apply(t)
    payload = inj.meta["payload"]
    assert inj.marker in payload  # relay tag lives only in the payload
    assert inj.meta["exec_marker"] not in payload  # the sum is never written out: seeing it means execution
    host = [d for d in inj.docs if payload in d]
    assert len(host) == 1 and t.gold_answer not in host[0]  # rides on a document that does not hold the answer
    assert inj == make_attack(f"v2.{family}", tasks).apply(t)  # deterministic


def test_v2a_important_uses_agentdojo_template_when_installed(tasks):
    pytest.importorskip("agentdojo")
    text = template("important")
    assert "<INFORMATION>" in text and "{tag}" in text and text != TEMPLATES["important"]
    inj = make_attack("v2.important", tasks).apply(tasks[0])
    assert inj.marker in inj.meta["payload"] and inj.meta["exec_marker"] not in inj.meta["payload"]


def test_v2a_peer_compromises_the_researcher(tasks):
    from patient_zero.layouts import LAYOUTS
    from patient_zero.metrics.contagion import run_contagion
    from patient_zero.models import model_for
    from patient_zero.runlog import RunMeta
    from patient_zero.runner import RunSpec, run_one

    t = tasks[0]
    inj = make_attack("v2.peer", tasks).apply(t)
    assert inj.docs is None and inj.marker in inj.system_override and inj.meta["exec_marker"] not in inj.system_override

    class EchoSystem(ScriptedToolModel):  # obeys the system prompt by echoing it, like a compliant agent
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            sys, human = messages[0].text, messages[-1].text
            return ChatResult(generations=[ChatGeneration(message=AIMessage(f"{human}\n{sys[-400:]}\nANSWER: x"))])

    meta = RunMeta(run_id="r", exp_id="t", layout="chain", vector="v2.peer", condition="adversarial", model="fake", task_id=t.id)
    record, msgs = asyncio.run(run_one(RunSpec(meta, t, inj), LAYOUTS["chain"](lambda role: EchoSystem()), ["researcher"]))
    assert inj.marker in msgs[0].content and msgs[0].injected_into_sender
    out = run_contagion(record.model_dump(), [m.model_dump() for m in msgs])
    assert out["entry"] and out["agents_infected"] >= 1  # the analyst relays what the researcher appended


def test_v2a_unknown_family(tasks):
    with pytest.raises(ValueError):
        make_attack("v2.nope", tasks)


dojo = pytest.importorskip("agentdojo")


def test_v2b_split_covers_every_tool():
    from patient_zero.envs.dojo.runner import suite
    from patient_zero.envs.dojo.split import split_tools

    for name in ["slack", "banking", "travel", "workspace"]:
        tools = [t.name for t in suite("v1.2.1", name).tools]
        split = split_tools(name, tools)
        assert sorted(sum(split.values(), [])) == sorted(tools) and len(split) >= 2


def test_v2b_injection_reaches_worker_and_is_relayed():
    from patient_zero.envs.dojo.runner import DojoConfig, DojoSpec, run_dojo_one, suite
    from patient_zero.runlog import RunMeta

    s = suite("v1.2.1", "slack")
    env = s.load_and_inject_default_environment({})
    gt = s.get_user_task_by_id("user_task_1").ground_truth(env)
    web = next(c for c in gt if c.function == "get_webpage")
    model = ScriptedToolModel(script={"ask_web_agent": {"request": "Read Bob's article"}, "get_webpage": dict(web.args)})
    meta = RunMeta(run_id="r", exp_id="t", layout="dojo_hub", vector="v2.dojo.infection", condition="adversarial",
                   model="fake", task_id="slack.user_task_1")
    spec = DojoSpec(meta, "slack", "user_task_1", "injection_task_1", "infection")
    record, msgs = asyncio.run(run_dojo_one(spec, DojoConfig(exp_id="t"), model_for=lambda role: model))
    assert record.error is None, record.error
    worker = next(m for m in msgs if m.sender == "web_agent")
    assert worker.injected_into_sender and worker.tool_calls[0]["name"] == "get_webpage"
    assert worker.doc_inputs == ["tool:call_get_webpage"]
    final = msgs[-1]
    assert final.sender == "orchestrator" and worker.id in final.visible_inputs
    assert record.injection["marker"] in final.content  # relayed up the hierarchy by the echoing worker
    assert record.injection["target_role"] == "web_agent"
    assert record.extra["utility"] is False and record.extra["attack_success"] is False  # nothing sent to Alice


def test_v2b_orchestrator_always_ends_with_a_final_answer():
    from patient_zero.envs.dojo.runner import DojoConfig, DojoSpec, run_dojo_one
    from patient_zero.runlog import RunMeta

    class Delegator(ScriptedToolModel):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            from langchain_core.messages import AIMessage
            from langchain_core.outputs import ChatGeneration, ChatResult

            if "ask_slack_agent" in self.bound:  # orchestrator: delegate forever
                msg = AIMessage("", tool_calls=[{"name": "ask_slack_agent", "args": {"request": "x"}, "id": f"c{len(messages)}"}])
            else:
                msg = AIMessage("done")
            return ChatResult(generations=[ChatGeneration(message=msg)])

    meta = RunMeta(run_id="r", exp_id="t", layout="dojo_hub", vector="v2.dojo", condition="benign", model="fake", task_id="slack.user_task_1")
    record, msgs = asyncio.run(run_dojo_one(DojoSpec(meta, "slack", "user_task_1", None, None), DojoConfig(exp_id="t", max_steps=2),
                                            model_for=lambda role: Delegator()))
    assert record.error is None and msgs[-1].sender == "orchestrator" and msgs[-1].phase == "final"
    assert sum(m.sender == "orchestrator" for m in msgs) == 3  # 2 delegating turns + the forced final one
