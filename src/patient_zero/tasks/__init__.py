from .base import Task, TaskSource
from .cwq import load_cwq
from .hotpotqa import load_hotpotqa
from .sample import load_sample

DATASETS: dict[str, TaskSource] = {"hotpotqa": load_hotpotqa, "cwq": load_cwq, "sample": load_sample}


def load_tasks(dataset: str, n: int, seed: int = 0) -> list[Task]:
    return DATASETS[dataset](n, seed=seed)
