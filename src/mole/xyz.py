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
    return "{:.6f}".format(f)


@dataclass
class XyzAtom(AtomBase):
    """Atom class"""

    charge: Optional[float] = None  # charge of atom


@dataclass
class XyzMolecule(IMolecule):
    """Molecule class"""

    name: str  # custom name
    index: int
    children: list[XyzAtom]  # list of atoms or molecules

    @staticmethod
    def cast(mol: IMolecule) -> "XyzMolecule":
        return cast("XyzMolecule", mol)

    @classmethod
    def make_from(cls, mol: IMolecule, name: str, index: int) -> "XyzMolecule":
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
        move molecule
        """
        for atom in self.children:
            atom.coordinate += coordinate

    @override
    def rotate(self, rotation: Rotation):
        """
        rotate molecule
        """
        for atom in self.children:
            atom.coordinate = rotation.apply(atom.coordinate)

    @override
    def get_children(self) -> list[XyzAtom]:
        return self.children

    @override
    def get_child(self, index) -> XyzAtom:
        return self.children[index]

    @classmethod
    def make(cls, atoms: list[AtomBase]):
        return cls(
            "molecule",
            0,
            [XyzAtom(atom.symbol, atom.index, atom.coordinate)
             for atom in atoms],
        )

    @staticmethod
    def _plane(X: tuple[float, float, float], a, b, c, d) -> float:
        """plane function"""
        return a * X[0] + b * X[1] + c * X[2] + d

    def inner_move_to_xz_plane(self, target_atoms: list[XyzAtom]):
        """
        move molecule to xz plane
        fit plane to target_atoms and move molecule
        so that the plane is equal to xz plane
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
        return center of molecule
        it will be calculated as average of all atoms coordinate
        """
        return np.mean([atom.coordinate for atom in self.children], axis=0)

    @classmethod
    def from_xyz_file(cls, path: str):
        """
        load xyz file
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
            atom_list.append(
                XyzAtom(line[0], 0, np.array(line[1:4], dtype=float)))

        return cls(name, 0, atom_list)

    @classmethod
    def from_mol2_file(cls, path: str):
        """
        load mol2 file
        """
        name: str = os.path.basename(path).split(".")[0]
        atom_list: list[XyzAtom] = []
        lines = []
        with open(path, "r") as f:
            lines = f.readlines()
        # find @<TRIPOS>ATOM line and @... line after that
        for i, line in enumerate(lines):
            if line.strip() == "@<TRIPOS>ATOM":
                lines = lines[i + 1:]
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

    def save_xyz(self, filename: str | None = None):
        """
        save xyz file
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
        """return number of atoms"""
        return len(self.children)


class XyzSubstructure(Substructure):

    def __init__(self, elements: list[XyzMolecule], name: str):
        self.molecules: list[XyzMolecule] = elements
        self.name = name

    @classmethod
    def from_Substructure(cls, sub: Substructure):
        children: list[XyzMolecule] = sub.get_children()
        return cls(
            [XyzMolecule.make_from(mol, mol.name, mol.index)
             for mol in children], ""
        )

    def extract_xyz(self, filename: str) -> None:
        """extract xyz file"""
        temp_agg = copy.deepcopy(self)

        # make file
        f = open(filename, "w")

        # write number of atoms (first line)
        f.write(
            str(sum([molecule.sizeofAtoms()
                for molecule in temp_agg.molecules])) + "\n"
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
        generate gaussian input file

        fragment: if True, set fragment index like this
        (for Counterpoise method) :

        H (Fragment=1) 0.000000000000 0.000000000000 0.000000000000

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
            # write number of atoms
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
