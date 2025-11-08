import unittest
import sys
import os
import tempfile
import numpy as np

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from mole.xyz import XyzMolecule, XyzAtom

class TestXyzMolecule(unittest.TestCase):

    def test_from_xyz_file_and_save_xyz_text(self):
        """
        Tests the from_xyz_file and save_xyz_text methods.
        """
        xyz_content = """2
test molecule
C 1.0 2.0 3.0
H 4.0 5.0 6.0
"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".xyz") as tmpfile:
            tmpfile.write(xyz_content)
            filepath = tmpfile.name

        molecule = XyzMolecule.from_xyz_file(filepath)

        self.assertEqual(molecule.name, "test molecule")
        self.assertEqual(len(molecule.children), 2)
        self.assertEqual(molecule.children[0].symbol, "C")
        self.assertTrue(np.array_equal(molecule.children[0].coordinate, np.array([1.0, 2.0, 3.0])))
        self.assertEqual(molecule.children[1].symbol, "H")
        self.assertTrue(np.array_equal(molecule.children[1].coordinate, np.array([4.0, 5.0, 6.0])))

        saved_text = molecule.save_xyz_text()
        expected_text = "2\ntest molecule\nC 1.000000 2.000000 3.000000\nH 4.000000 5.000000 6.000000"
        self.assertEqual(saved_text, expected_text)

        os.remove(filepath)

if __name__ == '__main__':
    unittest.main()
