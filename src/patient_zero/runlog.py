"""Run logging: one JSONL line per agent message, one per finished run.

Named runlog (not logging) to avoid shadowing the stdlib module.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

from pydantic import BaseModel


class RunMeta(BaseModel):
    run_id: str
    exp_id: str
    layout: str
    vector: str
    condition: str
    model: str
    task_id: str
    seed_idx: int = 0


class MessageRecord(RunMeta):
    id: str
    ts: float
    step: int  # LangGraph superstep; agents running in parallel share a step
    round: int
    phase: str  # "work" | "dispatch" | "final" | "debate" | "probe" (private, never shown to other agents)
    sender: str
    recipients: list[str]
    visible_inputs: list[str]  # ids of the messages the sender saw (the provenance edges)
    doc_inputs: list[str] = []  # ids of the documents the sender saw
    content: str
    answer: str | None  # parsed from the "ANSWER:" line, if present
    injected_into_sender: bool
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    latency_ms: float
    tool_calls: list[dict] = []
    guard_score: float | None = None
    quarantined: bool = False  # withheld from every other agent by the online guard


class RunRecord(RunMeta):
    ts: float
    dataset: str
    question: str
    gold_answer: str
    gold_aliases: list[str] = []
    final_answer: str | None = None
    exact_match: bool | None = None
    f1: float | None = None
    injection: dict | None = None
    role_docs: dict[str, list[dict]] = {}  # role -> [{"id", "text"}] as each role received them
    fingerprint: str = ""  # hash of everything that defines the swarm; a change means re-certification
    known: bool | None = None  # model answers correctly closed-book (contamination control)
    guard: str | None = None
    tagging: bool = False
    probe: bool = False
    probe_style: str | None = None
    extra: dict = {}  # environment-specific results (V2b: utility, attack success, suite)
    n_messages: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: str | None = None


class JsonlLogger:
    def __init__(self, exp_dir: str | Path):
        self.exp_dir = Path(exp_dir)
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        self.messages_path = self.exp_dir / "messages.jsonl"
        self.runs_path = self.exp_dir / "runs.jsonl"
        self._lock = threading.Lock()

    def write_run(self, run: RunRecord, messages: list[MessageRecord]) -> None:
        """Messages and the run line are written together, so a run is either fully logged or absent."""
        with self._lock:
            with self.messages_path.open("a") as f:
                for m in messages:
                    f.write(m.model_dump_json() + "\n")
            with self.runs_path.open("a") as f:
                f.write(run.model_dump_json() + "\n")

    def completed_run_ids(self) -> set[str]:
        """Run ids that finished without error; errored runs are retried on resume."""
        if not self.runs_path.exists():
            return set()
        done = set()
        for line in self.runs_path.read_text().splitlines():
            rec = json.loads(line)
            if rec.get("error") is None:
                done.add(rec["run_id"])
        return done
