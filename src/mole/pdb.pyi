from dataclasses import dataclass, field
from typing import Optional, override

import numpy as np
from scipy.spatial.transform import Rotation

from mole.molecules import AtomBase, IMolecule


@dataclass
class PdbAtom(AtomBase):
    atom_name: str
    residue_name: str
    residue_number: int
    chain_id: str = " "
    record_type: str = "ATOM"
    occupancy: float = 1.0
    temp_factor: float = 0.0
    element: str = ""
    alt_loc: str = " "
    insertion_code: str = " "
    raw_line: Optional[str] = None

    def __init__(
        self,
        symbol: str,
        index: int,
        coordinate: np.ndarray,
        atom_name: str,
        residue_name: str,
        residue_number: int,
        chain_id: str = " ",
        record_type: str = "ATOM",
        occupancy: float = 1.0,
        temp_factor: float = 0.0,
        element: str = "",
        alt_loc: str = " ",
        insertion_code: str = " ",
        raw_line: Optional[str] = None
    ) -> None: ...
    @override
    def __eq__(self, value: object) -> bool: ...
    @classmethod
    def cast_to_pdb(cls, atom: AtomBase) -> "PdbAtom": ...


@dataclass
class PdbFile(IMolecule[PdbAtom]):
    atoms: list[PdbAtom]
    header_lines: list[str] = field(default_factory=list)
    footer_lines: list[str] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)

    def __init__(
        self,
        atoms: list[PdbAtom],
        header_lines: Optional[list[str]] = None,
        footer_lines: Optional[list[str]] = None,
        connections: Optional[list[str]] = None,
    ) -> None: ...
    @override
    def get_children(self) -> list[PdbAtom]: ...
    @override
    def get_child(self, index: int) -> PdbAtom: ...
    @override
    def translate(self, coordinate: np.ndarray) -> None: ...
    @override
    def rotate(self, rotation: Rotation) -> None: ...
    @classmethod
    @override
    def make(cls, atoms: list[AtomBase]) -> "PdbFile": ...
    @classmethod
    def from_pdb_file(cls, path: str) -> "PdbFile": ...
    def save_pdb(self, path: str) -> None: ...
