import os
import sys
import unittest

from src.gromacs.calculation import (
    EM,
    MD,
    MDType,
    Calculation,
    save_json,
    load_json,
    RuntimeSolvation,
    Solvation,
    SolvationSCP216,
    FileControl
)


class TestCalculationJson(unittest.TestCase):
    def test_json_save_and_load(self):
        calculations: list[Calculation] = [
            EM(calculation_name="em_1"),
            MD(
                calculation_name="md_1",
                type=MDType.v_rescale_c_rescale,
                nsteps=50000,
                temperature=298.15,
            ),
            RuntimeSolvation(calculation_name="rs_1"),
            Solvation(calculation_name="sol_1"),
            SolvationSCP216(calculation_name="scp_1"),
            FileControl(calculation_name="fc_1", command="echo 'hello'")
        ]
        filepath = "test.json"
        save_json(calculations, filepath)
        loaded_calculations = load_json(filepath)

        self.assertEqual(len(calculations), len(loaded_calculations))
        for i in range(len(calculations)):
            original = calculations[i]
            loaded = loaded_calculations[i]
            # Custom assertion for deep comparison
            self.assertDictEqual(original.__dict__, loaded.__dict__)

        os.remove(filepath)


if __name__ == "__main__":
    unittest.main()
