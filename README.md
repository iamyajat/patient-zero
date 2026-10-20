# Patient Zero: AI Swarm Audit Framework

Tests whether an error that enters a multi-agent system through one agent gets caught or amplified.
A four-agent LangGraph swarm (researcher, analyst, verifier, writer) runs in three layouts (chain, hub, mesh)
on two models, with three attack vectors (V1 false-claim spread, V2 instruction relay, V3 false consensus).

## Setup

```bash
uv sync
cp .env.example .env          # add OPENAI_API_KEY
ollama pull qwen3:8b          # for the open-weight model
```

## Running

```bash
uv run pz run --layout chain --vector v1 --condition adversarial --model fake --n 3   # offline smoke test
uv run pz run --layout mesh --model qwen --dataset hotpotqa --n 3
uv run pz grid configs/pilot_v1.yaml      # resumable: rerun to retry errored runs
uv run pz report runs/pilot_v1
uv run pytest
```

Models: `openai` (gpt-6-luna), `qwen` (Ollama qwen3:8b), `fake` (offline echo model), or `openai:<id>` / `ollama:<id>`.

## Logs

`runs/<exp_id>/messages.jsonl` holds one line per agent message. `visible_inputs` lists the ids of the messages
that agent saw, which lets you trace a claim from agent to agent. `runs/<exp_id>/runs.jsonl` holds one line per
run, with the final answer, EM/F1 and the injection.

## Layout of the code

| Path | Owner | What |
| --- | --- | --- |
| `layouts/`, `agent.py`, `runlog.py`, `runner.py` | A | Swarm, layouts, logging, grid runner |
| `tasks/`, `attacks/`, `metrics/` | B | Datasets, V1–V3, metrics (V1 template and baseline metrics in place; V2/V3 stubbed) |
| `monitor/` | C + A | Monitor interface (`Monitor.inspect(messages) -> list[Flag]`) |

Layout decisions to confirm: in the hub layout the writer doubles as the coordinator, so the swarm stays at
four agents. Mesh runs 2 debate rounds by default. Only the researcher receives the source documents
(`doc_roles` in the config).
