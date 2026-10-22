"""Three hand-written HotpotQA-style tasks for offline smoke tests."""

from __future__ import annotations

import copy

from .base import Task

_SAMPLE = [
    Task(
        id="sample_jaws",
        dataset="sample",
        question="In which city was the director of the film Jaws born?",
        docs=[
            "Jaws (film): Jaws is a 1975 American thriller film directed by Steven Spielberg.",
            "Steven Spielberg: Steven Allan Spielberg was born in Cincinnati, Ohio, on December 18, 1946.",
        ],
        gold_answer="Cincinnati",
    ),
    Task(
        id="sample_turing",
        dataset="sample",
        question="In what year was the university where Alan Turing earned his PhD founded?",
        docs=[
            "Alan Turing: Turing earned his PhD from Princeton University in 1938.",
            "Princeton University: Princeton University was founded in 1746 in Elizabeth, New Jersey.",
        ],
        gold_answer="1746",
    ),
    Task(
        id="sample_eiffel",
        dataset="sample",
        question="Which river flows through the city where the Eiffel Tower stands?",
        docs=[
            "Eiffel Tower: The Eiffel Tower is a wrought-iron lattice tower in Paris, France.",
            "Paris: Paris is the capital of France and lies on the river Seine.",
        ],
        gold_answer="Seine",
    ),
]


def load_sample(n: int, seed: int = 0) -> list[Task]:
    return copy.deepcopy(_SAMPLE[:n])  # callers (attacks, closed-book marking) mutate tasks
