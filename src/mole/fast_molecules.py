"""
This module provides a FastStructWapper class for efficient manipulation of molecular structures.
It enables rapid translation, rotation, and replication of molecules,
and includes a method for checking inter-molecular distances to detect clashes.
"""
import gc
from copy import deepcopy

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation

import mole.molecules as mol
import base_utils.cui_utils as cui


class FastStructWapper[A: mol.AtomBase]():
    """
    Move, rotate and replicate molecules in a fast way.
    and check if the molecules are too close to each other.
    """

    def __init__(self, molecule: mol.IMolecule[A]):
        """
        Initializes the FastStructWapper with a single molecule.
        Args:
            molecule (IMolecule[A]): The molecule to wrap.
        """
        self.molecule = molecule
        self.n_atoms = len(molecule.get_children())
        self.coordinates = [
            atom.coordinate for atom in molecule.get_children()]
        self.n_mols = 1

    @classmethod
    def from_substructure(cls, substructure: mol.Substructure
                          ) -> "FastStructWapper":
        """
        Creates a FastStructWapper from a Substructure object.
        Args:
            substructure (mol.Substructure): The substructure to wrap.
        Returns:
            FastStructWapper: A new FastStructWapper instance.
        """
        temp = cls(substructure.molecules[0])
        temp.n_mols = len(substructure.molecules)
        atoms = np.concatenate(
            [molecule.get_children() for molecule in substructure.molecules]
        )

        temp.coordinates = np.array([atom.coordinate for atom in atoms])
        temp.n_atoms = len(substructure.molecules[0].get_children())
        return temp

    def translate(self, coordinate: np.ndarray):
        """
        Translates all wrapped molecules by a given vector.
        Args:
            coordinate (np.ndarray): The translation vector.
        """
        self.coordinates += coordinate

    def rotate(self, rotation: Rotation):
        """
        Rotates all wrapped molecules around the origin by a given rotation.
        Args:
            rotation (Rotation): The rotation object.
        """
        self.coordinates = rotation.apply(self.coordinates)

    def generate_as_molecule(self, Type: type) -> mol.IMolecule[A]:
        """
        Generates a molecule object from the current coordinates.
        Args:
            Type (type): The type of molecule to generate (must be the same as the input molecule type).
        Returns:
            T: The generated molecule object.
        """
        old_atoms = self.molecule.get_children()
        n_old = len(old_atoms)
        atoms = []
        for i, coordinate in enumerate(self.coordinates):
            atoms.append(deepcopy(old_atoms[i % n_old]))
            atoms[-1].coordinate = coordinate
        return Type.make(atoms)

    def reset(self):
        """
        Resets the coordinates to the original molecule's coordinates.
        """
        self.coordinates = [
            atom.coordinate for atom in self.molecule.get_children()]
        self.n_mols = 1

    def replicate(self, n: int):
        """
        Replicates the wrapped molecule `n` times.
        Args:
            n (int): The number of times to replicate the molecule.
        """
        self.coordinates = np.tile(self.coordinates, (n, 1))

        # somehow, the memory is not released after the operation. Probably?
        gc.collect()

        self.n_mols = n

    def translate_one(self, index: int, coordinate: np.ndarray):
        """
        Translates a single molecule at the specified index by a given vector.
        Args:
            index (int): The index of the molecule to translate.
            coordinate (np.ndarray): The translation vector.
        Raises:
            ValueError: If the index is out of range.
        """
        if index >= self.n_mols:
            raise ValueError("index out of range")
        self.coordinates[
            index * self.n_atoms: (index + 1) * self.n_atoms
        ] += coordinate

    def rotate_one(self, index: int, rotation: Rotation):
        """
        Rotates a single molecule at the specified index by a given rotation.
        Args:
            index (int): The index of the molecule to rotate.
            rotation (Rotation): The rotation object.
        Raises:
            ValueError: If the index is out of range.
        """
        if index >= self.n_mols:
            raise ValueError("index out of range")

        self.coordinates[index * self.n_atoms: (index + 1) * self.n_atoms] = (
            rotation.apply(
                self.coordinates[index *
                                 self.n_atoms: (index + 1) * self.n_atoms]
            )
        )

    def linear_translate(self, vector: np.ndarray):
        """
        Translates each molecule by a multiple of the given vector,
        where the multiple is determined by the molecule's index.
        e.g., if the vector is [1,0,0], the first molecule is translated by [0,0,0],
        the second by [1,0,0], the third by [2,0,0], and so on.
        Args:
            vector (np.ndarray): The base translation vector.
        """

        for i in range(self.n_mols):
            self.translate_one(i, vector * i)

    def linear_rotate(self, rotation: Rotation):
        """
        Rotates each molecule by a multiple of the given rotation,
        where the multiple is determined by the molecule's index.
        e.g., if the rotation is a 90-degree rotation around the z-axis,
        the first molecule is rotated by 0 degrees, the second by 90 degrees,
        the third by 180 degrees, and so on.
        Args:
            rotation (Rotation): The base rotation object.
        """
        for i in range(self.n_mols):
            self.rotate_one(i, rotation**i)

    # def reset(self):
    #     for i in range(self.n_mols):
    #         self.co

    def is_too_close(self, distance: float = 0.09953893710503443) -> bool:
        """
        Checks if any two atoms from different molecules are closer than a specified distance.
        Args:
            distance (float): The minimum allowed distance between atoms from different molecules.
        Returns:
            bool: True if any atoms are too close, False otherwise.
        """

        if self.n_mols == 1:
            cui.warning("Only one molecule. The test is meaningless.")
            cui.warning("Do you mistakenly call this function?")
            return False

        # pairwise distance
        dist: np.ndarray = distance_matrix(self.coordinates, self.coordinates)
        # check if there is any distance less than 0.1 except for the same atom

        # consider the distance between the atoms in the same molecule.
        mol_identity = (
            np.identity(self.n_mols) * 1000
        )  # big number (Covalent bond at 1000 nm is impossible!)
        wrap = np.kron(mol_identity, np.ones((self.n_atoms, self.n_atoms)))
        """
        wrap is like this when n_mols = 2, n_atoms = 3:
        1000 1000 1000 0    0    0
        1000 1000 1000 0    0    0
        1000 1000 1000 0    0    0
        0    0    0    1000 1000 1000
        0    0    0    1000 1000 1000
        0    0    0    1000 1000 1000

        """

        dist = (
            dist + wrap
            # if the atoms are in the same molecule,
            # the distance is added by 1000.
        )
        """
        dist is like this when n_mols = 2, n_atoms = 3:
        1000 1000 1000 0.x  0.x  0.x
        1000 1000 1000 0.x  0.x  0.x
        1000 1000 1000 0.x  0.x  0.x
        0.x  0.x  0.x  1000 1000 1000
        0.x  0.x  0.x  1000 1000 1000
        0.x  0.x  0.x  1000 1000 1000

        finally, evaluate only zero elements in wrap matrix
        (it means the distance between the atoms in different molecules)
        """

        a = dist.flatten()

        print(np.min(a))
        return np.any(a < distance)
