"""Hub: the writer doubles as coordinator so the swarm stays at four agents.

coordinator dispatches -> researcher, analyst, verifier work in parallel (each sees only the dispatch)
-> coordinator sees all three replies and writes the final answer.
"""

from langgraph.graph import END, START, StateGraph

from ..agent import make_agent_node
from ..state import SwarmState

COORDINATOR = "writer"
WORKERS = ["researcher", "analyst", "verifier"]

WORKER_INSTRUCTION = "Do the subtask the coordinator assigned to you."


def build_hub(model, rounds: int = 1):
    g = StateGraph(SwarmState)
    g.add_node(
        "coordinator_dispatch",
        make_agent_node(
            COORDINATOR,
            model,
            name="coordinator_dispatch",
            visible=lambda msgs: [],
            instruction=(
                "You are coordinating the team. Split the question into one subtask each for the Researcher "
                "(who has the source documents), the Analyst and the Verifier. Address each by role. Do not answer yet."
            ),
            recipients=WORKERS,
            phase="dispatch",
            answer=False,
        ),
    )
    g.add_edge(START, "coordinator_dispatch")
    for role in WORKERS:
        g.add_node(
            role,
            make_agent_node(
                role,
                model,
                name=role,
                visible=lambda msgs: [m for m in msgs if m.phase == "dispatch"],
                instruction=WORKER_INSTRUCTION,
                recipients=[COORDINATOR],
            ),
        )
        g.add_edge("coordinator_dispatch", role)
    g.add_node(
        "coordinator_final",
        make_agent_node(
            COORDINATOR,
            model,
            name="coordinator_final",
            visible=lambda msgs: [m for m in msgs if m.sender in WORKERS],
            instruction="Combine your workers' replies into the team's final answer.",
            recipients=[],
            phase="final",
        ),
    )
    g.add_edge(WORKERS, "coordinator_final")
    g.add_edge("coordinator_final", END)
    return g.compile()
