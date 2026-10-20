"""Mesh: debate-style rounds. Round 0 answers independently; in each later round every agent sees
all other agents' messages from the previous round. The writer's last-round answer is final."""

from langgraph.graph import END, START, StateGraph

from ..agent import make_agent_node
from ..roles import ROLES
from ..state import SwarmState

FIRST = "Give your answer to the question."
LATER = "Consider your teammates' answers from the previous round, then give your updated answer."


def _visible(role: str, rnd: int):
    return lambda msgs: [m for m in msgs if m.round == rnd - 1 and m.sender != role]


def build_mesh(model, rounds: int = 2):
    g = StateGraph(SwarmState)
    prev: list[str] = []
    for r in range(rounds):
        names = [f"r{r}_{role}" for role in ROLES]
        for role, name in zip(ROLES, names):
            g.add_node(
                name,
                make_agent_node(
                    role,
                    model,
                    name=name,
                    visible=_visible(role, r),
                    instruction=FIRST if r == 0 else LATER,
                    recipients=[x for x in ROLES if x != role] if r < rounds - 1 else [],
                    round=r,
                    phase="debate",
                ),
            )
            g.add_edge(prev if prev else START, name)
        prev = names
    for name in prev:
        g.add_edge(name, END)
    return g.compile()
