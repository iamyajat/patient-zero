from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel


class GridConfig(BaseModel):
    exp_id: str
    dataset: str = "hotpotqa"
    n_tasks: int = 100
    seed: int = 0
    vectors: list[str] = ["v1"]
    conditions: list[str] = ["benign", "adversarial"]
    layouts: list[str] = ["chain", "hub", "mesh"]
    models: list[str] = ["openai", "qwen"]
    mesh_rounds: int = 2
    doc_roles: list[str] = ["researcher"]  # roles that receive the task's documents
    concurrency: int = 8
    out_dir: str = "runs"

    @classmethod
    def from_yaml(cls, path: str | Path) -> GridConfig:
        return cls(**yaml.safe_load(Path(path).read_text()))

    @property
    def exp_dir(self) -> Path:
        return Path(self.out_dir) / self.exp_id
