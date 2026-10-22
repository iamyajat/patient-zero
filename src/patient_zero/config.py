from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

VECTORS = {
    "v1": "Cascading reliability failure",
    "v2": "Cascading infection",
    "v3": "False consensus",
}

DEFAULT_DATASETS = {"v1.mint": "cwq"}  # MINT passages only exist for CWQ


class GridConfig(BaseModel):
    exp_id: str
    dataset: str = "hotpotqa"
    n_tasks: int = 100
    seed: int = 0
    vectors: list[str] = ["v1"]
    conditions: list[str] = ["benign", "adversarial"]
    layouts: list[str] = ["chain", "hub", "mesh"]
    layouts_by_vector: dict[str, list[str]] | None = None  # e.g. mesh only for v3; overrides `layouts`
    tasks_from: dict[str, str] | None = None  # dataset per vector; overrides `dataset`
    models: list[str] = ["openai", "qwen"]  # model specs: name, provider:id or mixed:role=name,...
    mesh_rounds: int = 2
    doc_roles: list[str] = ["researcher"]  # roles that receive the task's documents
    repeats: int = 1  # extra seeds, run on a `repeat_fraction` subsample of tasks
    repeat_fraction: float = 0.2
    probe_fraction: float = 0.0  # share of tasks whose runs also log private-belief probes
    probe_style: str = "private"  # "private" (Huang et al. wording) | "neutral" (re-ask; control for wording effects)
    guard: str | None = None  # online guard: a monitor name (rules | llm | ensemble | judge); quarantines flagged messages
    guard_threshold: float | None = None  # from `pz calibrate` at the target false-alarm rate
    judge_model: str = "openai"  # LLM detectors; use a different family from the agents for the paper's runs
    tagging: bool = False  # LLM Tagging baseline defense
    adaptive_monitor: str = "rules"  # condition "adaptive": rewrite planted text this monitor flags
    adaptive_threshold: float = 0.5  # use the calibrated threshold at the target false-alarm rate
    adaptive_rounds: int = 2
    closed_book: bool = False  # mark each task `known` per model (answered correctly without documents)
    exclude_known: bool = False  # drop tasks any model already knows (keeps the design paired)
    answer_judge: str | None = None  # model for answer equivalence (aliases, paraphrases); None = F1 rule
    attack_model: str = "openai"  # generates LLM false facts (v1.llm) and adaptive rewrites
    attack_options: dict[str, dict] = {}  # per-vector kwargs, e.g. {"v1.mint": {"styles": ["hoax", "rumor"]}}
    data_dir: str = "data"
    concurrency: int = 8
    run_timeout: float = 900  # seconds per run; a run that exceeds it is logged as an error and retried on resume
    out_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> GridConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))

    @property
    def exp_dir(self) -> Path:
        return Path(self.out_dir) / self.exp_id

    def layouts_for(self, vector: str) -> list[str]:
        by = self.layouts_by_vector or {}
        return by.get(vector) or by.get(vector.split(".")[0]) or self.layouts

    def dataset_for(self, vector: str) -> str:
        by = {**DEFAULT_DATASETS, **(self.tasks_from or {})}
        return by.get(vector) or by.get(vector.split(".")[0]) or self.dataset
