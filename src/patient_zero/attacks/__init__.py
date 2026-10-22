from .base import Attack, Injection
from .mint import MintAttack
from .v1_false_fact import FalseFactAttack
from .v2_instruction import InstructionRelayAttack
from .v3_consensus import FalseConsensusAttack


def base_vector(vector: str) -> str:
    """`v1.llm` -> `v1`. The part after the dot picks a variant (the attack's source or payload family)."""
    return vector.split(".", 1)[0]


def make_attack(vector: str, tasks, model: str = "openai", data_dir: str = "data", **options) -> Attack:
    base, _, variant = vector.partition(".")
    if base == "v1":
        if variant == "mint":
            return MintAttack(tasks, **options)
        return FalseFactAttack(tasks, source=variant or "template", model=model, data_dir=data_dir, **options)
    if base == "v2":
        return InstructionRelayAttack(tasks, family=variant or "infection", **options)
    if base == "v3":
        return FalseConsensusAttack(tasks, source=variant or "template", model=model, data_dir=data_dir, **options)
    raise ValueError(f"unknown vector {vector!r}")
