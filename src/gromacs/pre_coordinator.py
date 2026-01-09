"""
This module provides functions for pre-coordinating molecular structures.
These functions are used to align and orient molecules in a specific way,
which can be useful for setting up GROMACS simulations or other molecular analyses.
"""

from scipy.spatial.transform import Rotation
from mole import molecules
from mole import xyz
import numpy as np
import rdkit.Chem as Chem


def pre_coordinate[T: molecules.IMolecule](
    molecule: T, topO: int, aromaticsideO: int, aromaticothersideO: int
) -> T:
    """
    Pre-coordinates a molecule by translating its 'top' atom to the origin,
    then rotating it so that a vector defined by two aromatic side atoms aligns with the x-axis,
    and finally rotating it to align another aromatic side atom with the xz-plane.
    Args:
        molecule (T): The molecule object to pre-coordinate.
        topO (int): Index of the 'top' atom.
        aromaticsideO (int): Index of an atom on the aromatic side.
        aromaticothersideO (int): Index of another atom on the aromatic side.
    Returns:
        T: The pre-coordinated molecule object.
    """
    top_coordinate = molecule.get_child(topO).coordinate
    molecule.translate(-top_coordinate)

    vector = (
        molecule.get_child(aromaticsideO).coordinate
        + molecule.get_child(aromaticothersideO).coordinate
    )
    print(vector)
    # C=O bond(Barbiturate) vector
    vector = vector.astype(float) / np.linalg.norm(vector)
    # move from vector to x axis
    rot = Rotation.align_vectors([[1, 0, 0]], [vector])[0]

    molecule.rotate(rot)

    # first O to next O vector
    vector = molecule.get_child(aromaticsideO).coordinate.astype(float)
    # rot to move vector to xz plane and rotate only x axis
    rot = Rotation.from_euler("x", np.arctan2(vector[1], vector[2]), degrees=False)
    molecule.rotate(rot)

    return molecule


def precooredinate2[T: molecules.IMolecule](
    molecule: T, topO: int, aromaticsideNH: int, aromaticothersideO: int
) -> T:
    """
    Pre-coordinates a molecule by translating its 'top' atom to the origin,
    then rotating it to align an aromatic side NH atom with the x-axis,
    and finally rotating it to align another aromatic side O atom with the xz-plane.
    Args:
        molecule (T): The molecule object to pre-coordinate.
        topO (int): Index of the 'top' atom.
        aromaticsideNH (int): Index of an aromatic side NH atom.
        aromaticothersideO (int): Index of another aromatic side O atom.
    Returns:
        T: The pre-coordinated molecule object.
    """

    # move top O to origin
    top_coordinate = molecule.get_child(topO).coordinate
    molecule.translate(-top_coordinate)

    # move aromatic side NH to x axis
    vector = molecule.get_child(aromaticsideNH).coordinate.astype(float)

    # move vector to x axis
    rot = Rotation.from_euler("z", -np.arctan2(vector[1], vector[0]), degrees=False)

    molecule.rotate(rot)

    # move aromatic side NH to x axis
    vector = molecule.get_child(aromaticsideNH).coordinate.astype(float)

    rot = Rotation.from_euler("y", np.arctan2(vector[2], vector[0]), degrees=False)
    molecule.rotate(rot)

    # move aromatic side O to xz plane
    vector = molecule.get_child(aromaticothersideO).coordinate.astype(float)

    print(-np.arctan2(vector[2], -vector[1]) * 180 / np.pi)
    # move vector to xz plane
    rot = Rotation.from_euler("x", np.arctan2(vector[2], -vector[1]), degrees=False)
    # rot = Rotation.from_euler("x", np.pi, degrees=False) * rot

    molecule.rotate(rot)

    return molecule


def get_substructure_match(
    molecule: molecules.IMolecule, substructure: molecules.IMolecule
) -> list[int]:
    """
    Finds the indices of atoms in the molecule that match the given substructure.
    Args:
        molecule (molecules.IMolecule): The molecule to search within.
        substructure (molecules.IMolecule): The substructure to match.
    Returns:
        list[int]: A list of atom indices in the molecule that match the substructure.
    """

    mol = Chem.MolFromXYZBlock(
        xyz.XyzMolecule.make_from(molecule, "A", 0).save_xyz_text()
    )
    submol = Chem.MolFromXYZBlock(
        xyz.XyzMolecule.make_from(substructure, "A", 0).save_xyz_text()
    )
    match = mol.GetSubstructMatch(submol)
    return list(match)
