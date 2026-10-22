from functools import partial

from .chain import build_chain
from .hub import build_hub
from .mesh import build_mesh

LAYOUTS = {
    "chain": build_chain,
    "chain_noverifier": partial(build_chain, roles=["researcher", "analyst", "writer"]),
    "hub": build_hub,
    "mesh": build_mesh,
}

# source module per layout, hashed into the swarm fingerprint
LAYOUT_MODULES = {"chain": "chain", "chain_noverifier": "chain", "hub": "hub", "mesh": "mesh"}
