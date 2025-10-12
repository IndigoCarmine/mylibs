"""
This module provides classes for parsing, manipulating, and generating GROMACS GRO files.
It includes functionalities for representing atoms and molecules in the GRO format,
and converting between GRO and XYZ file formats.
"""
from copy import deepcopy
from dataclasses import dataclass
from typing import override
from scipy.spatial.transform import Rotation
import numpy as np

from mole.molecules import AtomBase, IMolecule


class GroAtom(AtomBase):
    """
    Represents an atom in a GROMACS GRO file.
    Inherits from AtomBase and adds GRO-specific properties like residue name and number.
    """
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
        """
        Initializes a GroAtom object.
        Args:
            atom_number (int): The atom's index.
            atom_name (str): The atom's name (e.g., "C1", "O").
            residue_name (str): The residue's name (e.g., "MOL").
            residue_number (int): The residue's number.
            coordinate (np.ndarray): The 3D coordinates of the atom.
        """
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
        """
        Returns the chemical symbol of the atom (first character of atom_name).
        """
        return self.atom_name[0]

    def __str__(self) -> str:
        """
        Returns a string representation of the GroAtom in GRO file format.
        (i5,2a5,i5,3f8.3,3f8.4)
        """
        # (i5,2a5,i5,3f8.3,3f8.4)
        return f"{self.residue_number:>5}{self.residue_name:<5}{self.atom_name:>5}{self.index:>5}{self.coordinate[0]:>8.3f}{self.coordinate[1]:>8.3f}{self.coordinate[2]:>8.3f}"

    # factory method to create GroAtom object from a line in a gro file
    @classmethod
    def from_gro_line(cls, line: str):
        """
        Creates a GroAtom object by parsing a single line from a GRO file.
        Args:
            line (str): A line from a GRO file representing an atom.
        Returns:
            GroAtom: A new GroAtom instance.
        """
        residue_number = int(line[:5])
        residue_name = line[5:10].strip()
        atom_name = line[10:15].strip()
        atom_number = int(line[15:20])
        x = float(line[20:28])
        y = float(line[28:36])
        z = float(line[36:44])
        return cls(
            atom_number, atom_name, residue_name, residue_number, np.array([x, y, z])
        )

    def __deepcopy__(self, memo) -> "GroAtom":
        """
        Creates a deep copy of the GroAtom object.
        """
        return GroAtom(
            self.index,
            self.atom_name,
            self.residue_name,
            self.residue_number,
            self.coordinate.copy(),
        )


@dataclass
class GroFile(IMolecule[GroAtom]):
    """
    Represents a GROMACS GRO file, containing molecular structure and box information.
    Implements the IMolecule interface for common molecular operations.
    """
    title: str
    atoms: list[GroAtom]
    box_x: float
    box_y: float
    box_z: float

    box_angle_x: float = 90
    box_angle_y: float = 90
    box_angle_z: float = 90

    def __len__(self) -> int:
        """
        Returns the number of atoms in the GRO file.
        """
        return len(self.atoms)

    def find_atom(self, index: int) -> list[GroAtom]:
        """
        Finds atoms with a specific index.
        Args:
            index (int): The index of the atom to find.
        Returns:
            list[GroAtom]: A list of GroAtom objects matching the index.
        """
        return [atom for atom in self.atoms if atom.index == index]

    def generate_gro_text(self) -> list[str]:
        """
        Generates the content of the GRO file as a list of strings.
        Returns:
            list[str]: A list of strings representing the GRO file content.
        """
        gro: list[str] = []
        gro.append(self.title)
        gro.append(f"   {len(self.atoms)}")
        for atom in self.atoms:
            gro.append(str(atom))

        gro.append(f"{self.box_x} {self.box_y} {self.box_z}")
        return gro

    @classmethod
    def from_gro_text(cls, lines: list[str]) -> "GroFile":
        """
        Creates a GroFile object by parsing a list of strings representing GRO file content.
        Args:
            lines (list[str]): A list of strings from a GRO file.
        Returns:
            GroFile: A new GroFile instance.
        """
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
        """
        Creates a GroFile object by reading content from a GRO file.
        Args:
            file_path (str): The path to the GRO file.
        Returns:
            GroFile: A new GroFile instance.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        return cls.from_gro_text(lines)

    def save_gro(self, file_path: str):
        """
        Saves the current GroFile object to a GRO file.
        Args:
            file_path (str): The path to save the GRO file.
        """
        with open(file_path, "w") as f:
            f.write("\n".join(self.generate_gro_text()))

    def renumber(self, start: int = 1):
        """
        Renumber the atoms in the GRO file starting from a specified index.
        Args:
            start (int): The starting index for renumbering.
        """
        for i, atom in enumerate(self.atoms):
            atom.index = i + start

    def set_residue_number(self, number: int):
        """
        Sets the residue number for all atoms in the GRO file.
        Args:
            number (int): The new residue number.
        """
        for atom in self.atoms:
            atom.residue_number = number

    # --------------------- XYZ FILE input/output ---------------------

    def generate_xyz_text(self) -> list[str]:
        """
        Generates the content of the XYZ file as a list of strings.
        Returns:
            list[str]: A list of strings representing the XYZ file content.
        """
        def format_float(f: float) -> str:
            return f"{f:12.6f}"

        NM_TO_ANGSTROM = 10
        xyz: list[str] = []
        xyz.append(f"{len(self.atoms)}")
        xyz.append(self.title)
        for atom in self.atoms:
            xyz.append(
                f"{atom.atom_symbol} {
                    format_float(atom.coordinate[0] * NM_TO_ANGSTROM)
                } {format_float(atom.coordinate[1] * NM_TO_ANGSTROM)} {
                    format_float(atom.coordinate[2] * NM_TO_ANGSTROM)
                }"
            )
        return xyz

    def save_xyz(self, file_path: str):
        """
        Saves the current GroFile object to an XYZ file.
        Args:
            file_path (str): The path to save the XYZ file.
        """
        with open(file_path, "w") as f:
            f.write("\n".join(self.generate_xyz_text()))

    def load_xyz_text(self, data: list[str], multiple_molecules:bool=False):
        """
        Loads coordinate data from a list of strings representing an XYZ file.
        Args:
            data (list[str]): A list of strings from an XYZ file.
            multiple_molecules (bool): If True, handles multiple molecules in the XYZ file.
        Raises:
            ValueError: If the number of atoms in the XYZ file does not match or is not a multiple.
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
            atoms: list[GroAtom] = []
            for n in range(molnum):
                for i in range(len(self.atoms)):
                    line = data[n * len(self.atoms) + i + 2].split()
                    atom = deepcopy(self.atoms[i])
                    atom.index = atom.index + n * len(self.atoms)
                    atom.coordinate = (
                        np.array([float(line[1]), float(line[2]), float(line[3])])
                        * ANGSTROM_TO_NM
                    )
                    atoms.append(atom)

            self.atoms = atoms

    def load_xyz_file(self, file_path: str, multiple_molecules: bool = False):
        """
        Loads coordinate data from an XYZ file.
        Args:
            file_path (str): The path to the XYZ file.
            multiple_molecules (bool): If True, handles multiple molecules in the XYZ file.
        """
        with open(file_path, "r") as f:
            lines = f.readlines()
        self.load_xyz_text(lines, multiple_molecules)

    @override
    def get_children(self) -> list[GroAtom]:
        """
        Returns a list of all atoms in the GRO file.
        """
        return self.atoms

    @override
    def get_child(self, index: int) -> GroAtom:
        """
        Returns a specific atom by its index.
        Args:
            index (int): The index of the atom to retrieve.
        Returns:
            GroAtom: The GroAtom object.
        Raises:
            ValueError: If the atom is not found or multiple atoms with the same index exist.
        """
        atoms = self.find_atom(index)

        if len(atoms) == 0:
            raise ValueError(f"Atom with index {index} not found")

        if len(atoms) > 1:
            raise ValueError(f"Multiple atoms with index {index} found")

        return atoms[0]

    @override
    def translate(self, coordinate: np.ndarray):
        """
        Translates all atoms in the GRO file by a given vector.
        Args:
            coordinate (np.ndarray): The translation vector.
        """
        for atom in self.atoms:
            atom.coordinate += coordinate

    @override
    def rotate(self, rotation: Rotation):
        """
        Rotates all atoms in the GRO file by a given rotation.
        Args:
            rotation (Rotation): The rotation object.
        """
        for atom in self.atoms:
            atom.coordinate = rotation.apply(atom.coordinate)

    @override
    @classmethod
    def make(cls, atoms: list[AtomBase]) -> "GroFile":
        """
        Creates a GroFile object from a list of AtomBase objects.
        Args:
            atoms (list[AtomBase]): A list of AtomBase objects.
        Returns:
            GroFile: A new GroFile instance.
        """
        gro = cls("MOL", [], 0, 0, 0)
        if isinstance(atoms[0], GroAtom):
            gro.atoms = atoms # type: ignore
        else:
            gro.atoms = [
                GroAtom(atom.index, atom.symbol, "MOL", 1, atom.coordinate)
                for atom in atoms
            ]
        return gro

    def generate_ndx(self, path: str) -> None:
        """
        Generates a GROMACS index file (.ndx) based on residue numbers.
        Atoms are grouped by residue, and each group is written to the index file.
        Args:
            path (str): The path to save the index file.
        """
        if not path.endswith(".ndx"):
            path = path + ".ndx"

        with open(path, "w", newline="\n") as f:
            last_residue = 0
            line_length = 0
            for atom in self.atoms:
                if atom.residue_number != last_residue:
                    f.write(f"\n\n[ Mol{atom.residue_number} ]\n")
                    last_residue = atom.residue_number
                    line_length = 0
                f.write(f"{atom.index} ")
                line_length += 1
                if line_length >= 10:  # Split lines after 10 indices
                    f.write("\n")
                    line_length = 0



if __name__ == "__main__":
    grofile = GroFile.from_gro_file("QuinNaphTDP.gro")

    grofile.save_xyz("QuinNaphTDP.xyz")

    # grofile.load_xyz_file("QuinNaphTDP_mod.xyz")

    grofile.save_gro("QuinNaphTDP2.gro")
