from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from .runlog import MessageRecord, RunMeta
from .tasks.base import Task


class SwarmState(TypedDict):
    meta: RunMeta
    task: Task
    role_docs: dict[str, list[dict]]  # role -> [{"id", "text"}] it "retrieved"; V1/V2 plant content here
    role_overrides: dict[str, str]  # extra system prompt per role; V3 uses this
    injected_roles: list[str]
    probe: bool  # also ask each agent privately what it believes (V3 hidden-belief probe)
    probe_style: str  # "private" | "neutral" wording of the probe question
    guard: Any  # OnlineGuard or None; scores each message and quarantines it above threshold
    tagging: bool  # LLM Tagging defense: mark relayed messages with their provenance
    messages: Annotated[list[MessageRecord], operator.add]
