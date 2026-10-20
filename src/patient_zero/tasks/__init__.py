from .base import Task, TaskSource
from .hotpotqa import load_hotpotqa
from .sample import load_sample

DATASETS: dict[str, TaskSource] = {"hotpotqa": load_hotpotqa, "sample": load_sample}


def load_tasks(dataset: str, n: int, seed: int = 0) -> list[Task]:
    return DATASETS[dataset](n, seed=seed)
