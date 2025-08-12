"""
This module provides classes for handling molecular structures in XYZ format.
It includes functionalities for parsing, manipulating, and generating XYZ files,
as well as converting from MOL2 files and generating Gaussian input files.
"""

import copy
import os
from dataclasses import dataclass
from typing import Optional, cast, override
from scipy.optimize import curve_fit
import numpy as np
from scipy.spatial.transform import Rotation

from mole.molecules import AtomBase, IMolecule, Substructure

gaussian_param_default = """%NProcShared=24
%Chk={name}.chk
#p opt b3lyp/6-31+g(d,p) int=fine pop=full
"""


def format_float(f: float | int) -> str:
    """
    Formats a float or integer to a string with 6 decimal places.
    Args:
        f (float | int): The number to format.
    Returns:
        str: The formatted string.
    """
    return "{:.6f}".format(f)


@dataclass
class XyzAtom(AtomBase):
    """
    Represents an atom in an XYZ file.
    Inherits from AtomBase and adds an optional charge property.
    """

    charge: Optional[float] = None  # charge of atom

    @override
    def __eq__(self, value):
        if isinstance(value, XyzAtom):
            return super().__eq__(value) and self.charge == value.charge
        elif isinstance(value, AtomBase):
            return super().__eq__(value)
        return False


@dataclass
class XyzMolecule(IMolecule):
    """
    Represents a molecule in XYZ format.
    Implements the IMolecule interface for common molecular operations.
    """

    name: str  # custom name
    index: int
    children: list[XyzAtom]  # list of atoms or molecules

    @staticmethod
    def cast(mol: IMolecule) -> "XyzMolecule":
        """
        Casts a generic IMolecule object to an XyzMolecule object.
        Args:
            mol (IMolecule): The molecule to cast.
        Returns:
            XyzMolecule: The cast XyzMolecule object.
        """
        return cast("XyzMolecule", mol)

    @classmethod
    def make_from(cls, mol: IMolecule, name: str, index: int) -> "XyzMolecule":
        """
        Creates an XyzMolecule from a generic IMolecule object.
        Args:
            mol (IMolecule): The source molecule.
            name (str): The name for the new XyzMolecule.
            index (int): The index for the new XyzMolecule.
        Returns:
            XyzMolecule: A new XyzMolecule instance.
        """
        return cls(
            name,
            index,
            [
                XyzAtom(atom.symbol, atom.index, atom.coordinate)
                for atom in mol.get_children()
            ],
        )

    @override
    def translate(self, coordinate: np.ndarray):
        """
        Translates all atoms in the molecule by a given vector.
        Args:
            coordinate (np.ndarray): The translation vector.
        """
        for atom in self.children:
            atom.coordinate += coordinate

    @override
    def rotate(self, rotation: Rotation):
        """
        Rotates all atoms in the molecule by a given rotation.
        Args:
            rotation (Rotation): The rotation object.
        """
        for atom in self.children:
            atom.coordinate = rotation.apply(atom.coordinate)

    @override
    def get_children(self) -> list[XyzAtom]:
        """
        Returns a list of all atoms (children) in the molecule.
        """
        return self.children

    @override
    def get_child(self, index) -> XyzAtom:
        """
        Returns a specific atom (child) by its index.
        Args:
            index (int): The index of the atom to retrieve.
        Returns:
            XyzAtom: The XyzAtom object.
        """
        return self.children[index]

    @classmethod
    def make(cls, atoms: list[AtomBase]):
        """
        Factory method to create a new XyzMolecule instance from a list of AtomBase objects.
        Args:
            atoms (list[AtomBase]): A list of AtomBase objects.
        Returns:
            XyzMolecule: A new XyzMolecule instance.
        """
        return cls(
            "molecule",
            0,
            [XyzAtom(atom.symbol, atom.index, atom.coordinate) for atom in atoms],
        )

    @staticmethod
    def _plane(X: tuple[float, float, float], a, b, c, d) -> float:
        """
        Calculates the value of a plane equation at a given point.
        Args:
            X (tuple[float, float, float]): The 3D coordinates (x, y, z).
            a, b, c, d: Coefficients of the plane equation (ax + by + cz + d = 0).
        Returns:
            float: The value of the plane equation at X.
        """
        return a * X[0] + b * X[1] + c * X[2] + d

    def inner_move_to_xz_plane(self, target_atoms: list[XyzAtom]):
        """
        Moves the molecule so that a plane fitted to the target atoms aligns with the xz-plane.
        Args:
            target_atoms (list[XyzAtom]): A list of atoms to use for fitting the plane.
        """
        popt: list[float] = [0, 0, 0, 0]
        x = np.array([atom.coordinate for atom in target_atoms]).T
        y = np.array([0 for _ in target_atoms])
        popt, _ = curve_fit(f=XyzMolecule._plane, xdata=x, ydata=y)
        a, b, c, d = popt
        # fit to xy plane
        self.rotate(Rotation.from_euler("x", np.arctan(-b / c), degrees=True))
        self.rotate(Rotation.from_euler("y", np.arctan(-a / c), degrees=True))
        self.translate(np.array([0, 0, -d / c]))

    def get_inner_center(self) -> np.ndarray:
        """
        Calculates and returns the geometric center of the molecule (average of all atom coordinates).
        Returns:
            np.ndarray: The 3D coordinates of the center.
        """
        return np.mean([atom.coordinate for atom in self.children], axis=0)

    @classmethod
    def from_xyz_file(cls, path: str):
        """
        Loads an XyzMolecule object from an XYZ file.
        Args:
            path (str): The path to the XYZ file.
        Returns:
            XyzMolecule: A new XyzMolecule instance.
        """
        name: str | None = None
        atom_list: list[XyzAtom] = []
        lines = []
        with open(path, "r") as f:
            lines = f.readlines()
        name = lines[0].strip()
        lines = lines[2:]  # first two lines are comments
        lines = [line.strip() for line in lines]
        lines = [line for line in lines if line != ""]
        lines2 = [line.split() for line in lines]
        for line in lines2:
            atom_list.append(XyzAtom(line[0], 0, np.array(line[1:4], dtype=float)))

        return cls(name, 0, atom_list)

    @classmethod
    def from_mol2_file(cls, path: str):
        """
        Loads an XyzMolecule object from a MOL2 file.
        Parses atom information from the ATOM section of the MOL2 file.
        Args:
            path (str): The path to the MOL2 file.
        Returns:
            XyzMolecule: A new XyzMolecule instance.
        """
        name: str = os.path.basename(path).split(".")[0]
        atom_list: list[XyzAtom] = []
        lines = []
        with open(path, "r") as f:
            lines = f.readlines()
        # find @<TRIPOS>ATOM line and @... line after that
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>ATOM":
                lines = lines[i + 1 :]
                break
        for i, line in enumerate(lines):
            if line.strip()[0] == "@":
                lines = lines[:i]
                break

        for line in lines:
            # the following line is example of mol2 file
            #      14 C          -7.1834    9.1696   -1.7595 C.ar      1     ****    0.0000 # number of molecules

            # split line by space
            line_split = line.split()
            atom_list.append(
                XyzAtom(
                    line_split[1],
                    int(line_split[0]),
                    np.array(line_split[2:5], dtype=float),
                )
            )

        for atom in atom_list:
            # remove atom index from atom name
            atom.symbol = "".join([c for c in atom.symbol if not c.isdigit()])
        return cls(name, 0, atom_list)

    def save_xyz_text(self) -> str:
        """
        Generates the content of the XYZ file as a single string.
        Returns:
            str: The formatted XYZ file content.
        """
        # apply rotation and translation to atoms
        mol = copy.deepcopy(self)
        xyz_text = []
        xyz_text.append(str(self.children.__len__()))
        xyz_text.append(self.name)
        for atom in mol.children:
            xyz_text.append(
                atom.symbol + " " + " ".join([format_float(s) for s in atom.coordinate])
            )
        return "\n".join(xyz_text)

    def save_xyz(self, filename: str | None = None):
        """
        Saves the current XyzMolecule object to an XYZ file.
        Args:
            filename (str | None): The path to save the XYZ file. If None, saves as `name.xyz`.
        """
        if filename is None:  # if filename is not given, save as name.xyz
            filename = self.name + ".xyz"
        # apply rotation and translation to atoms
        mol = copy.deepcopy(self)
        if os.path.exists(str(filename)):
            os.remove(str(filename))

        with open(str(filename), "x") as f:
            f.write(str(self.children.__len__()) + "\n")
            f.write(self.name + "\n")
            for atom in mol.children:
                f.write(
                    atom.symbol
                    + " "
                    + " ".join([format_float(s) for s in atom.coordinate])
                    + "\n"
                )

    def generate_gaussian_input(
        self, dir_path: str, gaussan_pram: str | None = None
    ) -> None:
        """
        Generates a Gaussian input file (.gjf) for the molecule.
        Args:
            dir_path (str): The directory path to save the GJF file.
            gaussan_pram (str | None): Custom Gaussian parameters. If None, uses default.
        """
        if gaussan_pram is None:
            gaussan_pram = gaussian_param_default.format(name=self.name)

        gaussan_pram = str(gaussan_pram)

        f = open(dir_path + self.name + ".gjf", "w")
        f.write(gaussan_pram)
        f.write("\n")
        f.write(self.name + "\n")
        f.write("\n")
        f.write("0 1\n")
        for atom in self.children:
            f.write(
                atom.symbol
                + " "
                + " ".join([format_float(c) for c in atom.coordinate])
                + "\n"
            )
        f.write("\n")
        f.write("\n")
        f.close()

    def sizeofAtoms(self) -> int:
        """
        Returns the number of atoms in the molecule.
        """
        return len(self.children)


class XyzSubstructure(Substructure):
    """
    Represents a collection of XyzMolecule objects as a substructure.
    Provides methods for extracting XYZ files and generating Gaussian input files for the substructure.
    """

    def __init__(self, elements: list[XyzMolecule], name: str):
        """
        Initializes an XyzSubstructure object.
        Args:
            elements (list[XyzMolecule]): A list of XyzMolecule objects forming the substructure.
            name (str): The name of the substructure.
        """
        self.molecules: list[XyzMolecule] = elements
        self.name = name

    @classmethod
    def from_Substructure(cls, sub: Substructure):
        """
        Creates an XyzSubstructure from a generic Substructure object.
        Args:
            sub (Substructure): The source Substructure.
        Returns:
            XyzSubstructure: A new XyzSubstructure instance.
        """
        children: list[XyzMolecule] = sub.get_children()
        return cls(
            [XyzMolecule.make_from(mol, mol.name, mol.index) for mol in children], ""
        )

    def extract_xyz(self, filename: str) -> None:
        """
        Extracts all atoms from the substructure and saves them to a single XYZ file.
        Args:
            filename (str): The path to save the XYZ file.
        """
        temp_agg = copy.deepcopy(self)

        # make file
        f = open(filename, "w")

        # write number of atoms (first line)
        f.write(
            str(sum([molecule.sizeofAtoms() for molecule in temp_agg.molecules])) + "\n"
        )
        f.write("\n")
        for molecule in temp_agg.molecules:
            for atom in molecule.children:
                f.write(
                    atom.symbol
                    + " "
                    + " ".join([format_float(c) for c in atom.coordinate])
                    + "\n"
                )
        f.close()

    def generate_gaussian_input(
        self, dir_path: str, gaussan_pram: str | None = None, fragment=False
    ) -> None:
        """
        Generates a Gaussian input file (.gjf) for the substructure.
        Args:
            dir_path (str): The directory path to save the GJF file.
            gaussan_pram (str | None): Custom Gaussian parameters. If None, uses default.
            fragment (bool): If True, includes fragment information for each molecule.
        """

        if gaussan_pram is None:
            gaussan_pram = gaussian_param_default.format(name=self.name)
        gaussan_pram = str(gaussan_pram)
        temp_agg = copy.deepcopy(self)

        # set molcule name as fragment index
        for i, molecule in enumerate(temp_agg.molecules):
            molecule.name = str(i + 1)

        f = open(dir_path + self.name + ".gjf", "w")
        f.write(gaussan_pram)
        f.write("\n")
        f.write(self.name + "\n")
        f.write("\n")

        if fragment:
            # write charge and spin multiplicity
            f.write("0 1 " * (len(temp_agg.molecules) + 1) + "1 0\n")
            # write number of atoms
            for molecule in temp_agg.molecules:
                for atom in molecule.children:
                    f.write(
                        atom.symbol
                        + "(Fragment="
                        + molecule.name
                        + ") "
                        + " ".join([str(c) for c in atom.coordinate])
                        + "\n"
                    )
        else:
            # write charge and spin multiplicity
            f.write("0 1\n")
            for molecule in temp_agg.molecules:
                for atom in molecule.children:
                    f.write(
                        atom.symbol
                        + " "
                        + " ".join([format_float(c) for c in atom.coordinate])
                        + "\n"
                    )
        f.write("\n")
        f.write("\n")
        f.close()
