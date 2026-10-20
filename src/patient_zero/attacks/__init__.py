from .base import Attack, Injection
from .v1_false_fact import FalseFactAttack
from .v2_instruction import InstructionRelayAttack
from .v3_consensus import FalseConsensusAttack

ATTACKS = {"v1": FalseFactAttack, "v2": InstructionRelayAttack, "v3": FalseConsensusAttack}


def make_attack(vector: str, tasks) -> Attack:
    return ATTACKS[vector](tasks)
