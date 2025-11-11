import unittest
import sys
import os

from src.DFT.gjf import CalculationType

class TestCalculationType(unittest.TestCase):

    def test_from_string(self):
        """
        Tests the from_string method of the CalculationType class.
        """
        input_string = "#P opt B3LYP\\6-31G*\n"
        expected_calculation_type = CalculationType(
            basis_set="6-31G*",
            functional="B3LYP",
            workflow="opt"
        )
        actual_calculation_type = CalculationType.from_string(input_string)
        self.assertEqual(actual_calculation_type, expected_calculation_type)

    def test_from_string_no_basis_set(self):
        """
        Tests the from_string method with no basis set.
        """
        input_string = "#P opt B3LYP\n"
        with self.assertRaises(ValueError):
            CalculationType.from_string(input_string)


    def test_from_string_invalid_string(self):
        """
        Tests the from_string method with an invalid input string.
        """
        input_string = "Invalid string"
        with self.assertRaises(ValueError):
            CalculationType.from_string(input_string)

if __name__ == '__main__':
    unittest.main()
