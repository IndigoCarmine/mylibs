import os
import tempfile
import unittest

import numpy as np

from src.gromacs.calculation import MDType, default_file_content
from src.gromacs.metad import (
    KB_KJ_PER_MOL_K,
    MDWithPlumed,
    _atomlist,
    build_twist_plumed,
    fold_fes_to_period,
    load_fes,
)


class TestAtomList(unittest.TestCase):
    def test_ranges_and_singletons(self):
        self.assertEqual(_atomlist([1, 2, 3, 5, 7, 8]), "1-3,5,7-8")

    def test_sorts_and_dedup_order(self):
        self.assertEqual(_atomlist([3, 1, 2]), "1-3")


class TestBuildTwistPlumed(unittest.TestCase):
    def test_contains_cv_and_metad(self):
        txt = build_twist_plumed([1, 2, 3], [4, 5, 6], [1], [4], height=1.5)
        self.assertIn("TORSION ATOMS=spokeLo,cenLo,cenUp,spokeUp", txt)
        self.assertIn("METAD", txt)
        self.assertIn("HEIGHT=1.5", txt)
        self.assertIn("cenLo:   CENTER ATOMS=1-3", txt)


class TestMDWithPlumed(unittest.TestCase):
    def _step(self, **kw):
        return MDWithPlumed(
            type=MDType.v_rescale_c_rescale,
            calculation_name="metad",
            nsteps=1000,
            nstout=100,
            gen_vel="no",
            **kw,
        )

    def test_inline_plumed_content(self):
        step = self._step(plumed_content="dummy plumed input\n")
        files = step.generate()
        self.assertEqual(files["plumed.dat"], "dummy plumed input\n")
        # The shared mdrun.sh template auto-detects plumed.dat itself; the step
        # must reuse it verbatim, not hand-edit it or double-inject -plumed.
        self.assertEqual(files["mdrun.sh"], default_file_content("mdrun.sh"))
        self.assertIn("setting.mdp", files)

    def test_plumed_from_file_and_additional_files(self):
        with tempfile.TemporaryDirectory() as d:
            pf = os.path.join(d, "plumed.dat")
            with open(pf, "w") as f:
                f.write("from file\n")
            ref = os.path.join(d, "ref.gro")
            with open(ref, "w") as f:
                f.write("reference\n")
            step = self._step(plumed_file=pf, additional_files=[ref])
            files = step.generate()
        self.assertEqual(files["plumed.dat"], "from file\n")
        self.assertEqual(files["ref.gro"], "reference\n")

    def test_requires_exactly_one_source(self):
        with self.assertRaises(ValueError):
            self._step().generate()
        with self.assertRaises(ValueError):
            self._step(plumed_content="x", plumed_file="y").generate()


class TestFoldFes(unittest.TestCase):
    def test_load_fes(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "fes.dat")
            with open(p, "w") as f:
                f.write("# comment\n0.0 1.0\n1.0 2.0\n")
            cv, fes = load_fes(p)
        np.testing.assert_allclose(cv, [0.0, 1.0])
        np.testing.assert_allclose(fes, [1.0, 2.0])

    def test_fold_periodicity_and_minzero(self):
        # A CV over -pi..pi with a single well repeated every 60 deg should fold
        # onto one 60-deg sector with its minimum shifted to zero.
        cv = np.linspace(-np.pi, np.pi, 3600)
        deg = np.degrees(cv)
        fes = 1.0 - np.cos(np.radians(6.0 * deg))  # period = 60 deg
        centers, folded = fold_fes_to_period(cv, fes, period=60.0, nbins=60)
        self.assertEqual(len(centers), 60)
        finite = folded[np.isfinite(folded)]
        self.assertAlmostEqual(float(np.min(finite)), 0.0, places=6)
        self.assertLessEqual(float(centers[-1]), 60.0)

    def test_kt_constant(self):
        self.assertAlmostEqual(KB_KJ_PER_MOL_K * 300.0, 2.4943, places=3)


if __name__ == "__main__":
    unittest.main()
