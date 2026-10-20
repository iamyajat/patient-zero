"""The single agent-node factory every layout uses, so prompting, injection and logging are identical across layouts."""

from __future__ import annotations

import re
import time
from typing import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .roles import ANSWER_LINE, ROLE_PROMPTS
from .runlog import MessageRecord
from .state import SwarmState

VisibleFn = Callable[[list[MessageRecord]], list[MessageRecord]]

_ANSWER_RE = re.compile(r"^\s*\**ANSWER:?\**:?\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_answer(text: str) -> str | None:
    matches = _ANSWER_RE.findall(text)
    return matches[-1].strip("` *") if matches else None


def render_prompt(question: str, docs: list[str], seen: list[MessageRecord], instruction: str) -> str:
    parts = [f"Question: {question}"]
    if docs:
        parts.append("Source documents:\n" + "\n\n".join(f"[{i + 1}] {d}" for i, d in enumerate(docs)))
    if seen:
        parts.append("Messages from your teammates:\n" + "\n\n".join(f"[{m.sender}]: {m.content}" for m in seen))
    parts.append(instruction)
    return "\n\n".join(parts)


def make_agent_node(
    role: str,
    model: BaseChatModel,
    *,
    name: str,
    visible: VisibleFn,
    instruction: str,
    recipients: list[str],
    round: int = 0,
    phase: str = "work",
    answer: bool = True,
):
    async def node(state: SwarmState) -> dict:
        seen = visible(state["messages"])
        system = ROLE_PROMPTS[role]
        if override := state["role_overrides"].get(role):
            system += "\n\n" + override
        text = instruction + (" " + ANSWER_LINE if answer else "")
        prompt = render_prompt(state["task"].question, state["role_docs"].get(role, []), seen, text)

        t0 = time.perf_counter()
        reply = await model.ainvoke([SystemMessage(system), HumanMessage(prompt)])
        latency_ms = (time.perf_counter() - t0) * 1000
        usage = reply.usage_metadata or {}
        meta = state["meta"]
        content = reply.text

        record = MessageRecord(
            **meta.model_dump(),
            id=f"{meta.run_id}:{name}",
            ts=time.time(),
            step=max((m.step for m in state["messages"]), default=-1) + 1,
            round=round,
            phase=phase,
            sender=role,
            recipients=recipients,
            visible_inputs=[m.id for m in seen],
            content=content,
            answer=parse_answer(content) if answer else None,
            injected_into_sender=role in state["injected_roles"],
            prompt_tokens=usage.get("input_tokens"),
            completion_tokens=usage.get("output_tokens"),
            latency_ms=latency_ms,
        )
        return {"messages": [record]}

    return node
