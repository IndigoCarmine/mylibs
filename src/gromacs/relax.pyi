# Type stub for relax.py -- staged soft-core LJ relaxation via OpenMM.
# The physics rationale lives in the relax.py module docstring and is not
# duplicated here, so the two cannot drift apart.

from openmm import CustomNonbondedForce, System
from openmm.app import Topology
from openmm.unit import Quantity, Unit

nanometer: Unit

STAGES: list[tuple[float, float]]
NONBONDED_CUTOFF: Quantity
MAX_ITERATIONS: int
TOLERANCE: Quantity
FIXED_ATOM_K: Quantity

def add_softcore_lj(system: System) -> CustomNonbondedForce: ...
def find_inter_itp(top_path: str) -> str | None: ...
def parse_inter_bonds(itp_path: str) -> list[tuple[int, int, float, float]]: ...
def add_intermolecular_bonds(system: System, itp_path: str) -> int: ...
def add_fixed_atoms(
    system: System,
    positions: Quantity,
    fixed_indices: list[int],
    k: Quantity = ...,
) -> int: ...
def write_gro(
    topology: Topology,
    positions: Quantity,
    box_vectors: Quantity,
    out_path: str,
    title: str,
) -> None: ...
def relax(
    gro_path: str,
    top_path: str,
    out_dir: str,
    include_dir: str | None = None,
    fixed_atom_k: Quantity = ...,
) -> str: ...
