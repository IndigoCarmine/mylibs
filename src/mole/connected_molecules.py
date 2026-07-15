""" """

import src.mole.molecules as mol
import numpy as np
from scipy.spatial.transform import Rotation
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

# move Interfaces


class ConnectedMolecule:
    atom_positions: np.ndarray
    symbols: list[str]
    graph: Chem.Mol  # RDKit Mol whose bonds encode atom connectivity

    def _generate_bond_graph(self) -> Chem.Mol:
        """
        Infers bond connectivity from the current atom positions and symbols.
        Returns:
            Chem.Mol: An RDKit Mol with bonds determined from the 3D geometry.
        """
        xyz_block = "\n".join(
            [str(len(self.symbols)), ""]
            + [
                f"{symbol} {position[0]:.6f} {position[1]:.6f} {position[2]:.6f}"
                for symbol, position in zip(self.symbols, self.atom_positions)
            ]
        )
        rd_mol = Chem.MolFromXYZBlock(xyz_block)
        rdDetermineBonds.DetermineBonds(rd_mol, charge=0)
        return rd_mol

    def _atoms_on_far_side(self, fixed_index: int, moving_index: int) -> list[int]:
        """
        Traverses the bond graph starting from moving_index without crossing back
        through fixed_index, collecting every atom on that side of the bond.
        Args:
            fixed_index (int): The index of the atom that stays fixed (not included in the result).
            moving_index (int): The index of the atom to start the traversal from.
        Returns:
            list[int]: Indices of the atoms on moving_index's side of the bond.
        """
        visited = {fixed_index}
        stack = [moving_index]
        reached: list[int] = []
        while stack:
            current = stack.pop()
            if current in visited:
                continue
            visited.add(current)
            reached.append(current)
            for neighbor in self.find_connected_atoms(current):
                if neighbor not in visited:
                    stack.append(neighbor)
        return reached

    def rotate_with_bond(self, bond: tuple[int, int], angle: float):
        """
        Dihedral rotation around a bond defined by two atoms in the molecule.
        Args:
            bond (tuple[int, int]): A tuple representing the indices of the two atoms forming the bond. first index is fixed, second index is rotated
            angle (float): The angle in degrees to rotate around the bond.
        """
        fixed_index, moving_index = bond
        axis_origin = self.atom_positions[fixed_index]
        axis_vector = self.atom_positions[moving_index] - axis_origin
        axis_vector = axis_vector / np.linalg.norm(axis_vector)

        rotation = Rotation.from_rotvec(np.radians(angle) * axis_vector)
        for index in self._atoms_on_far_side(fixed_index, moving_index):
            relative = self.atom_positions[index] - axis_origin
            self.atom_positions[index] = axis_origin + rotation.apply(relative)

    def load(self, molecule: mol.IMolecule):
        """
        Loads a molecule into the ConnectedMolecule instance.
        Args:
            molecule (mol.Molecule): The molecule to load.
        """
        children = molecule.get_children()
        self.atom_positions = np.array([atom.coordinate for atom in children])
        self.symbols = [atom.symbol for atom in children]
        self.graph = self._generate_bond_graph()

    def export(self, base_molecule: mol.IMolecule):
        """
        Exports the current state of the ConnectedMolecule to a base molecule.
        Args:
            base_molecule (mol.Molecule): The base molecule to export to.
        """
        if self.atom_positions.shape[0] != len(base_molecule.get_children()):
            raise ValueError("The number of atom positions does not match the number of atoms in the base molecule.")

        for atom, position in zip(base_molecule.get_children(), self.atom_positions):
            atom.coordinate = position

    def find_connected_atoms(self, atom_index: int) -> list[int]:
        """
        Finds all atoms directly bonded to a given atom in the molecule.
        Args:
            atom_index (int): The index of the atom to find connections for.
        Returns:
            list[int]: A list of indices of connected atoms.
        """
        atom = self.graph.GetAtomWithIdx(atom_index)
        return [neighbor.GetIdx() for neighbor in atom.GetNeighbors()]
