"""Export runs as per-turn agent graphs with infection labels, for training/evaluating external graph guards
(INFA-Guard, G-Safeguard: github.com/yjzscode/INFA-Guard).

That repository has no license and generates its own data with its own MAS framework, so nothing from it is
vendored here. This writes a neutral format; mapping it onto the repo's `generate_data/` inputs is a manual step:
  one JSON line per run: {run_id, layout, vector, condition, agents: [...],
    turns: [{turn, nodes: [{agent, text, infected, attacker}], edges: [{src, dst, text}]}]}
`turn` is the LangGraph superstep; an edge src->dst carries the src agent's message that dst saw.
"""

from __future__ import annotations

import json
from pathlib import Path

from ...metrics.basic import load_exp
from ...monitor.calibrate import message_labels


def export_graphs(exp_dir: str | Path) -> Path:
    exp_dir = Path(exp_dir)
    runs, msgs = load_exp(exp_dir)
    runs = runs[runs.error.isna()] if "error" in runs else runs
    labels = message_labels(runs, msgs).set_index("message_id")
    out = exp_dir / "graphs.jsonl"
    with out.open("w") as f:
        for run_id, g in msgs[msgs.phase != "probe"].groupby("run_id"):
            run = runs[runs.run_id == run_id]
            if run.empty:
                continue
            run = run.iloc[0]
            by_id = g.set_index("id")
            turns = []
            for step, t in g.groupby("step"):
                nodes = [{"agent": m.sender, "text": m.content, "attacker": bool(m.injected_into_sender),
                          "infected": bool(labels.infected.get(m.id, False))} for m in t.itertuples()]
                edges = [{"src": by_id.loc[p, "sender"], "dst": m.sender, "text": by_id.loc[p, "content"]}
                         for m in t.itertuples() for p in m.visible_inputs if p in by_id.index]
                turns.append({"turn": int(step), "nodes": nodes, "edges": edges})
            f.write(json.dumps({"run_id": run_id, "layout": run.layout, "vector": run.vector, "condition": run.condition,
                                "agents": sorted(g.sender.unique()), "turns": turns}) + "\n")
    return out
