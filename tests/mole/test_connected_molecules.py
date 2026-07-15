import os
import unittest

import numpy as np

from src.mole.connected_molecules import ConnectedMolecule
from src.mole.xyz import XyzMolecule

TEST_DIR = os.path.dirname(__file__)


class TestConnectedMoleculeRotation(unittest.TestCase):
    def test_rotate_with_bond_and_save(self):
        input_path = os.path.join(TEST_DIR, "PhNaph.xyz")
        output_path = os.path.join(TEST_DIR, "PhNaph_rot.xyz")

        molecule = XyzMolecule.from_xyz_file(input_path)
        original_positions = np.array([atom.coordinate for atom in molecule.get_children()])

        connected = ConnectedMolecule()
        connected.load(molecule)
        connected.rotate_with_bond((10, 8), 90)
        connected.export(molecule)

        new_positions = np.array([atom.coordinate for atom in molecule.get_children()])

        # the fixed atom (index 10) must stay exactly where it was
        np.testing.assert_array_almost_equal(new_positions[10], original_positions[10])

        # the bond length between the rotated bond's atoms must be preserved
        original_bond_length = np.linalg.norm(original_positions[10] - original_positions[8])
        new_bond_length = np.linalg.norm(new_positions[10] - new_positions[8])
        self.assertAlmostEqual(new_bond_length, original_bond_length, places=5)

        # atoms on the phenyl ring side (attached through atom 10) must not move
        unchanged_indices = [10, 11, 12, 13, 14, 15, 23, 24, 25, 26, 27]
        for i in unchanged_indices:
            np.testing.assert_array_almost_equal(new_positions[i], original_positions[i])

        # atoms on the naphthalene side must have actually rotated
        changed_indices = [0, 1, 2, 3, 4, 5, 6, 7, 9, 16, 17, 18, 19, 20, 21, 22]
        for i in changed_indices:
            self.assertFalse(
                np.allclose(new_positions[i], original_positions[i]),
                msg=f"atom {i} was expected to move but did not",
            )

        molecule.name = "PhNaph_rot"
        if os.path.exists(output_path):
            os.remove(output_path)
        molecule.save_xyz(output_path)

        self.assertTrue(os.path.exists(output_path))
        saved = XyzMolecule.from_xyz_file(output_path)
        saved_positions = np.array([atom.coordinate for atom in saved.get_children()])
        np.testing.assert_array_almost_equal(saved_positions, new_positions, decimal=5)


if __name__ == "__main__":
    unittest.main()
