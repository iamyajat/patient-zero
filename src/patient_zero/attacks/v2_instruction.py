"""V2 instruction relay (owned by B).

Expected Injection: target_role="researcher", extra_docs=[a document carrying an instruction aimed at the
*next* agent, adapted from InjecAgent / BIPIA / Prompt Infection], marker=a string that shows the
instruction was relayed or carried out (e.g. a canary token the instruction asks agents to repeat).
"""

from ..tasks.base import Task
from .base import Injection


class InstructionRelayAttack:
    vector = "v2"

    def __init__(self, tasks: list[Task]):
        self.tasks = tasks

    def apply(self, task: Task) -> Injection:
        raise NotImplementedError("V2 is not implemented yet (owned by B)")
