from copy import deepcopy
from dataclasses import dataclass
from typing import override
from scipy.spatial.transform import Rotation
import numpy as np

from mole.molecules import AtomBase, IMolecule


class GroAtom(AtomBase):
    atom_name: str
    residue_name: str
    residue_number: int

    def __init__(
        self,
        atom_number: int,
        atom_name: str,
        residue_name: str,
        residue_number: int,
        coordinate: np.ndarray,
    ) -> None:

        self.atom_name = atom_name
        self.residue_name = residue_name
        self.residue_number = residue_number

        # atombase properties
        self.index = atom_number
        self.coordinate = coordinate
        self.symbol = self.atom_symbol

        # self.atom_info = ATOMS[atom_name[0]]

    @property
    def atom_symbol(self):
        return self.atom_name[0]

    def __str__(self) -> str:
        # (i5,2a5,i5,3f8.3,3f8.4)
        return f"{self.residue_number:>5}{self.residue_name:<5}{self.atom_name:>5}{self.index:>5}{self.coordinate[0]:>8.3f}{self.coordinate[1]:>8.3f}{self.coordinate[2]:>8.3f}"

    # factory method to create GroAtom object from a line in a gro file
    @classmethod
    def from_gro_line(cls, line: str):
        residue_number = int(line[:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        atom_number = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        return cls(
            atom_number,
            atom_name,
            residue_name,
            residue_number,
            np.array([x, y, z])
        )

    def __deepcopy__(self, memo) -> "GroAtom":
        return GroAtom(
            self.index,
            self.atom_name,
            self.residue_name,
            self.residue_number,
            self.coordinate.copy(),
        )


@dataclass
class GroFile(IMolecule[GroAtom]):
    title: str
    atoms: list[GroAtom]
    box_x: float
    box_y: float
    box_z: float

    box_angle_x: float = 90
    box_angle_y: float = 90
    box_angle_z: float = 90

    def __len__(self) -> int:
        return len(self.atoms)

    def find_atom(self, index: int) -> list[GroAtom]:
        return [atom for atom in self.atoms if atom.index == index]

    def generate_gro_text(self) -> list[str]:
        gro = []
        gro.append(self.title)
        gro.append(f"   {len(self.atoms)}")
        for atom in self.atoms:
            gro.append(str(atom))

        gro.append(f"{self.box_x} {self.box_y} {self.box_z}")
        return gro

    @classmethod
    def from_gro_text(cls, lines: list[str]) -> "GroFile":
        # first line is the title
        title = lines[0].strip()
        # second line is the number of atoms
        # last line is the box size
        atoms: list[GroAtom] = []
        working = 2
        for line in lines[2:-1]:
            atoms.append(GroAtom.from_gro_line(line))
            working += 1

        box_x = float(lines[-1].split()[0])
        box_y = float(lines[-1].split()[1])
        box_z = float(lines[-1].split()[2])

        return cls(title, atoms, box_x, box_y, box_z)

    @classmethod
    def from_gro_file(cls, file_path: str) -> "GroFile":
        with open(file_path, "r") as f:
            lines = f.readlines()
        return cls.from_gro_text(lines)

    def save_gro(self, file_path: str):
        with open(file_path, "w") as f:
            f.write("\n".join(self.generate_gro_text()))

    def renumber(self, start: int = 1):
        for i, atom in enumerate(self.atoms):
            atom.index = i + start

    def set_residue_number(self, number: int):
        for atom in self.atoms:
            atom.residue_number = number

    # --------------------- XYZ FILE input/output ---------------------

    def generate_xyz_text(self) -> list[str]:

        def format_float(f: float) -> str:
            return f"{f:12.6f}"

        NM_TO_ANGSTROM = 10
        xyz = []
        xyz.append(f"{len(self.atoms)}")
        xyz.append(self.title)
        for atom in self.atoms:
            xyz.append(
                f"{atom.atom_symbol} {format_float(atom.coordinate[0] * NM_TO_ANGSTROM)} {format_float(
                    atom.coordinate[1] * NM_TO_ANGSTROM)} {format_float(atom.coordinate[2] * NM_TO_ANGSTROM)}"
            )
        return xyz

    def save_xyz(self, file_path: str):
        with open(file_path, "w") as f:
            f.write("\n".join(self.generate_xyz_text()))

    def load_xyz_text(self, data: list[str], multiple_molecules=False):
        """
        load only coordinete data from xyz file
        """
        ANGSTROM_TO_NM = 0.1

        atomnum = int(data[0].strip() if data[0].strip().isnumeric() else 0)

        if not multiple_molecules:
            if atomnum != len(self.atoms):
                raise ValueError(
                    "number of atoms in gro file and xyz file are not equal"
                )

            for i in range(atomnum):
                line = data[i + 2].split()
                self.atoms[i].coordinate = (
                    np.array([float(line[1]), float(line[2]), float(line[3])])
                    * ANGSTROM_TO_NM
                )
        else:
            if atomnum % len(self.atoms) != 0:
                raise ValueError(
                    "number of atoms in xyz file is "
                    "not multiple of number of atoms in gro file"
                )
            molnum = atomnum // len(self.atoms)
            atoms = []
            for n in range(molnum):
                for i in range(len(self.atoms)):
                    line = data[n * len(self.atoms) + i + 2].split()
                    atom = deepcopy(self.atoms[i])
                    atom.index = atom.index + n * len(self.atoms)
                    atom.coordinate = (
                        np.array([float(line[1]), float(
                            line[2]), float(line[3])])
                        * ANGSTROM_TO_NM
                    )
                    atoms.append(atom)

            self.atoms = atoms

    def load_xyz_file(self, file_path: str, multiple_molecules=False):
        with open(file_path, "r") as f:
            lines = f.readlines()
        self.load_xyz_text(lines, multiple_molecules)

    @override
    def get_children(self) -> list[GroAtom]:
        return self.atoms

    @override
    def get_child(self, index: int) -> GroAtom:
        atoms = self.find_atom(index)

        if len(atoms) == 0:
            raise ValueError(f"Atom with index {index} not found")

        if len(atoms) > 1:
            raise ValueError(f"Multiple atoms with index {index} found")

        return atoms[0]

    @override
    def translate(self, coordinate: np.ndarray):
        for atom in self.atoms:
            atom.coordinate += coordinate

    @override
    def rotate(self, rotation: Rotation):
        for atom in self.atoms:
            atom.coordinate = rotation.apply(atom.coordinate)

    @override
    @classmethod
    def make(cls, atoms: list[AtomBase]) -> "GroFile":
        gro = cls("MOL", [], 0, 0, 0)
        if isinstance(atoms[0], GroAtom):
            gro.atoms = atoms
        else:
            gro.atoms = [
                GroAtom(atom.index, atom.symbol, "MOL", 1, atom.coordinate)
                for atom in atoms
            ]
        return gro


if __name__ == "__main__":
    grofile = GroFile.from_gro_file("QuinNaphTDP.gro")

    grofile.save_xyz("QuinNaphTDP.xyz")

    # grofile.load_xyz_file("QuinNaphTDP_mod.xyz")

    grofile.save_gro("QuinNaphTDP2.gro")
