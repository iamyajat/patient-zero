"""Chain: researcher -> analyst -> verifier -> writer. Each agent sees only the previous agent's message."""

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


def build_chain(model, rounds: int = 1):
    g = StateGraph(SwarmState)
    prev = START
    for i, role in enumerate(ROLES):
        nxt = ROLES[i + 1 : i + 2]
        g.add_node(
            role,
            make_agent_node(
                role,
                model,
                name=role,
                visible=lambda msgs: msgs[-1:],
                instruction=INSTRUCTIONS[role],
                recipients=nxt,
            ),
        )
        g.add_edge(prev, role)
        prev = role
    g.add_edge(prev, END)
    return g.compile()
