import gc
from copy import deepcopy

import numpy as np
from scipy.spatial import distance_matrix
from scipy.spatial.transform import Rotation

import mole.molecules as mol
import base_utils.cui_utils as cui


class FastStructWapper[T: mol.IMolecule]():
    """
    Move, rotate and replicate molecules in a fast way.
    and check if the molecules are too close to each other.
    """

    def __init__(self, molecule: T):
        self.molecule = molecule
        self.n_atoms = len(molecule.get_children())
        self.coordinates = [atom.coordinate for atom in molecule.get_children()]
        self.n_mols = 1

    @classmethod
    def from_substructure(cls, substructure: mol.Substructure) -> "FastStructWapper":
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
        Translate all the molecules by the same vector.
        """
        self.coordinates += coordinate

    def rotate(self, rotation: Rotation):
        """
        Rotate all the molecules by the same rotation (origin).
        """
        self.coordinates = rotation.apply(self.coordinates)

    def generate_as_molecule(self, Type: type) -> T:
        """
        T: type of the molecule. (It should be same as the type of input molecule) it need for calling make function. TODO: rewrite better.
        generate a molecule object from the current coordinates.

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
        regenerate the coordinates from the original molecule.
        """
        self.coordinates = [atom.coordinate for atom in self.molecule.get_children()]
        self.n_mols = 1

    def replicate(self, n: int):
        """
        replicate the molecule n times.
        """
        self.coordinates = np.tile(self.coordinates, (n, 1))

        # somehow, the memory is not released after the operation. Probably?
        gc.collect()

        self.n_mols = n

    def translate_one(self, index: int, coordinate: np.ndarray):
        """
        Translate the molecule at the index by the vector.
        if the index is out of range, raise ValueError.
        """
        if index >= self.n_mols:
            raise ValueError("index out of range")
        self.coordinates[
            index * self.n_atoms : (index + 1) * self.n_atoms
        ] += coordinate

    def rotate_one(self, index: int, rotation: Rotation):
        """
        Rotate the molecule at the index by the rotation.
        if the index is out of range, raise ValueError.
        """
        if index >= self.n_mols:
            raise ValueError("index out of range")

        self.coordinates[index * self.n_atoms : (index + 1) * self.n_atoms] = (
            rotation.apply(
                self.coordinates[index * self.n_atoms : (index + 1) * self.n_atoms]
            )
        )

    def linear_translate(self, vector: np.ndarray):
        """
        Translate all the molecules by the vector multiplied by the index.
        e.g)
        if the vector is [1,0,0],
        the first molecule is translated by [0,0,0],
        the second molecule is translated by [1,0,0],
        the third molecule is translated by [2,0,0]...
        """

        for i in range(self.n_mols):
            self.translate_one(i, vector * i)

    def linear_rotate(self, rotation: Rotation):
        """
        Rotate all the molecules by the rotation multiplied by the index.
        e.g)
        if the rotation is 90 degree rotation around z axis,
        the first molecule is rotated by 0 degree,
        the second molecule is rotated by 90 degree,
        the third molecule is rotated by 180 degree...
        """
        for i in range(self.n_mols):
            self.rotate_one(i, rotation**i)

    # def reset(self):
    #     for i in range(self.n_mols):
    #         self.co

    def is_too_close(self, distance: float = 0.09953893710503443) -> bool:
        """
        Check if the molecules are too close to each other.
        distance: the minimum distance between the atoms.
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
        )  # if the atoms are in the same molecule, the distance is added by 1000.
        """
        dist is like this when n_mols = 2, n_atoms = 3:
        
        1000 1000 1000 0.x  0.x  0.x
        1000 1000 1000 0.x  0.x  0.x
        1000 1000 1000 0.x  0.x  0.x
        0.x  0.x  0.x  1000 1000 1000
        0.x  0.x  0.x  1000 1000 1000
        0.x  0.x  0.x  1000 1000 1000

        finally, evaluate only zero elements in wrap matrix (it means the distance between the atoms in different molecules)
        """

        a = dist.flatten()

        print(np.min(a))
        return np.any(a < distance)
