import sys
import unittest
import numpy as np

from src.mole.xyz import XyzMolecule, XyzAtom
from src.gromacs.pre_coordinator import (
    get_substructure_match,
    pre_coordinate,
    precooredinate2,
)


class TestPreCoordinator(unittest.TestCase):
    def test_get_substructure_match(self):
        # Create a water molecule
        h1 = XyzAtom("H", 1, np.array([0.957, 0.0, 0.0]))
        o = XyzAtom("O", 8, np.array([0.0, 0.0, 0.0]))
        h2 = XyzAtom("H", 1, np.array([0.0, 0.957, 0.0]))
        water = XyzMolecule("water", 1, [h1, o, h2])

        # Create a substructure (an oxygen atom)
        substructure = XyzMolecule(
            "sub", 1, [XyzAtom("O", 8, np.array([0.0, 0.0, 0.0]))]
        )

        # Get the match
        match = get_substructure_match(water, substructure)

        # Assert that the correct index is found
        self.assertEqual(match, [1])

    def test_pre_coordinate(self):
        # Create a molecule
        c1 = XyzAtom("C", 6, np.array([1.0, 0.0, 0.0]))
        o1 = XyzAtom("O", 8, np.array([1.0, 1.0, 0.0]))
        o2 = XyzAtom("O", 8, np.array([-1.0, 1.0, 0.0]))
        molecule = XyzMolecule("test", 1, [c1, o1, o2])

        # Pre-coordinate the molecule
        pre_coordinate(molecule, 0, 1, 2)

        # Assert that the top atom is at the origin
        np.testing.assert_array_almost_equal(
            molecule.get_child(0).coordinate, np.array([0.0, 0.0, 0.0])
        )

        # Assert that the vector resulting from the sum of the two side atoms' coordinates is aligned with the x-axis.
        # The function under test uses vector addition to determine the orientation vector,
        # so the test does the same to verify the result.
        side_vector = (
            molecule.get_child(1).coordinate + molecule.get_child(2).coordinate
        )
        side_vector /= np.linalg.norm(side_vector)
        np.testing.assert_array_almost_equal(side_vector, np.array([1.0, 0.0, 0.0]))

        # Assert that the first side atom is in the xz-plane
        self.assertAlmostEqual(molecule.get_child(1).coordinate[1], 0.0)

    @unittest.expectedFailure
    def test_precooredinate2(self):
        # Create a molecule
        c1 = XyzAtom("C", 6, np.array([1.0, 0.0, 0.0]))
        n1 = XyzAtom("N", 7, np.array([1.0, 1.0, 0.0]))
        o1 = XyzAtom("O", 8, np.array([-1.0, 1.0, 0.0]))
        molecule = XyzMolecule("test", 1, [c1, n1, o1])

        # Pre-coordinate the molecule
        precooredinate2(molecule, 0, 1, 2)

        # Assert that the top atom is at the origin
        np.testing.assert_array_almost_equal(
            molecule.get_child(0).coordinate, np.array([0.0, 0.0, 0.0])
        )

        # Assert that the NH atom is aligned with the x-axis
        nh_vector = molecule.get_child(1).coordinate
        self.assertAlmostEqual(nh_vector[1], 0.0)
        self.assertAlmostEqual(nh_vector[2], 0.0)
        self.assertGreater(nh_vector[0], 0.0)

        # Assert that the O atom is in the xz-plane
        self.assertAlmostEqual(molecule.get_child(2).coordinate[1], 0.0)
