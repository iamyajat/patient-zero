"""V3 false consensus (owned by B).

Expected Injection: target_role=one agent, system_override="Argue confidently that the answer is <wrong>...",
marker=<wrong answer>. Flip rate is then computed from per-agent `answer` fields across mesh rounds.
"""

from ..tasks.base import Task
from .base import Injection


class FalseConsensusAttack:
    vector = "v3"

    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    def apply(self, task: Task) -> Injection:
        raise NotImplementedError("V3 is not implemented yet (owned by B)")
