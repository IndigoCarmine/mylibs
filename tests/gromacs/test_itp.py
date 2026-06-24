import unittest
import sys
import os
import tempfile

from src.gromacs.itp import generate_inermolecular_interactions

class TestItp(unittest.TestCase):

    def test_generate_inermolecular_interactions(self):
        """
        Tests the generate_inermolecular_interactions function.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            outfile_path = tmpfile.name

        generate_inermolecular_interactions(
            natoms=10,
            nmols=6,
            bonds=[(1, 10)],
            nmols_in_rosette=6,
            outfile_path=outfile_path
        )

        with open(outfile_path, 'r') as f:
            lines = f.readlines()

        expected_lines = [
            "[ intermolecular_interactions ]\n",
            "[ bonds ]\n",
            ";  ai    aj funct   length    k\n",
            "     1     10   6     0.300     5000\n",
            "    11     20   6     0.300     5000\n",
            "    21     30   6     0.300     5000\n",
            "    31     40   6     0.300     5000\n",
            "    41     50   6     0.300     5000\n",
            "    51      0   6     0.300     5000\n"
        ]

        self.assertEqual(lines, expected_lines)

        os.remove(outfile_path)

    def test_generate_inermolecular_interactions_multiple_rosettes(self):
        """
        Bonds in the second rosette must reference that rosette's own atoms,
        not collapse back onto the first rosette.
        """
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmpfile:
            outfile_path = tmpfile.name

        bonds = generate_inermolecular_interactions(
            natoms=10,
            nmols=12,
            bonds=[(1, 10)],
            nmols_in_rosette=6,
            outfile_path=outfile_path,
        )

        expected_bonds = [
            # rosette 0 (molecules 0-5), with ring closure on the last molecule
            (1, 10), (11, 20), (21, 30), (31, 40), (41, 50), (51, 0),
            # rosette 1 (molecules 6-11), offset to its own atoms (61-120)
            (61, 70), (71, 80), (81, 90), (91, 100), (101, 110), (111, 60),
        ]
        self.assertEqual(bonds, expected_bonds)

        os.remove(outfile_path)

if __name__ == '__main__':
    unittest.main()
