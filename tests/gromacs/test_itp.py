import unittest
import sys
import os
import tempfile

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from gromacs.itp import generate_inermolecular_interactions

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
            "    51      1   6     0.300     5000\n"
        ]

        self.assertEqual(lines, expected_lines)

        os.remove(outfile_path)

if __name__ == '__main__':
    unittest.main()
