"""Chain: researcher -> analyst -> verifier -> writer. Each agent sees only the previous agent's message.

`roles` drops agents for the verifier ablation (chain_noverifier: researcher -> analyst -> writer)."""

from langgraph.graph import END, START, StateGraph

from ..agent import make_agent_node
from ..roles import ROLES
from ..state import SwarmState

INSTRUCTIONS = {
    "researcher": "Report the facts from the documents that are needed to answer the question.",
    "analyst": "Work out the answer from the researcher's notes.",
    "verifier": "Check the analyst's reasoning and answer, and correct any errors.",
    "writer": "Give the team's final answer.",
}


def _from(sender: str | None):
    # only the predecessor's own message: if it was quarantined, nothing older leaks through instead
    return lambda msgs: [m for m in msgs if m.sender == sender][-1:]


def build_chain(model_for, rounds: int = 1, roles: list[str] = ROLES):
    g = StateGraph(SwarmState)
    prev = START
    for i, role in enumerate(roles):
        nxt = roles[i + 1 : i + 2]
        g.add_node(
            role,
            make_agent_node(
                role,
                model_for(role),
                name=role,
                visible=_from(roles[i - 1] if i else None),
                instruction=INSTRUCTIONS[role],
                recipients=nxt,
            ),
        )
        g.add_edge(prev, role)
        prev = role
    g.add_edge(prev, END)
    return g.compile()
