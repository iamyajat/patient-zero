from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from .runlog import MessageRecord, RunMeta
from .tasks.base import Task


class SwarmState(TypedDict):
    meta: RunMeta
    task: Task
    role_docs: dict[str, list[str]]  # documents each role "retrieved"; V1/V2 plant content here
    role_overrides: dict[str, str]  # extra system prompt per role; V3 uses this
    injected_roles: list[str]
    messages: Annotated[list[MessageRecord], operator.add]
