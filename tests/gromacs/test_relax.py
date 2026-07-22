import os
import tempfile
import unittest

from openmm import CustomNonbondedForce, HarmonicBondForce, NonbondedForce, System, unit
import openmm.app as app

from src.gromacs.relax import (
    add_fixed_atoms,
    add_intermolecular_bonds,
    add_softcore_lj,
    find_inter_itp,
    write_gro,
)

nanometer = unit.nanometer


class TestFindInterItp(unittest.TestCase):
    def test_finds_include_inside_ifdef_inter(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = os.path.join(tmpdir, "system.top")
            with open(top_path, "w") as f:
                f.write(
                    "#include \"forcefield.itp\"\n"
                    "#ifdef INTER\n"
                    "#include \"inter.itp\"\n"
                    "#endif\n"
                )
            with open(os.path.join(tmpdir, "inter.itp"), "w") as f:
                f.write("[ intermolecular_interactions ]\n")

            result = find_inter_itp(top_path)
            self.assertEqual(result, os.path.join(tmpdir, "inter.itp"))

    def test_returns_none_without_ifdef_inter_block(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            top_path = os.path.join(tmpdir, "system.top")
            with open(top_path, "w") as f:
                f.write("#include \"forcefield.itp\"\n")

            self.assertIsNone(find_inter_itp(top_path))


class TestAddFixedAtoms(unittest.TestCase):
    def _make_system(self, n):
        system = System()
        for _ in range(n):
            system.addParticle(1.0)
        return system

    def test_pins_atoms_at_their_positions(self):
        system = self._make_system(3)
        positions = [(0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (4.0, 5.0, 6.0)] * nanometer

        count = add_fixed_atoms(system, positions, [0, 2])

        self.assertEqual(count, 2)
        force = system.getForce(system.getNumForces() - 1)
        self.assertEqual(force.getNumParticles(), 2)
        index, params = force.getParticleParameters(0)
        self.assertEqual(index, 0)
        self.assertEqual(tuple(params), (0.0, 0.0, 0.0))
        index, params = force.getParticleParameters(1)
        self.assertEqual(index, 2)
        self.assertEqual(tuple(params), (4.0, 5.0, 6.0))

    def test_deduplicates_repeated_indices(self):
        system = self._make_system(3)
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (2.0, 2.0, 2.0)] * nanometer

        count = add_fixed_atoms(system, positions, [1, 1, 1])

        self.assertEqual(count, 1)

    def test_empty_indices_adds_no_force(self):
        system = self._make_system(2)
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] * nanometer

        count = add_fixed_atoms(system, positions, [])

        self.assertEqual(count, 0)
        self.assertEqual(system.getNumForces(), 0)

    def test_out_of_range_index_raises(self):
        system = self._make_system(2)
        positions = [(0.0, 0.0, 0.0), (1.0, 1.0, 1.0)] * nanometer

        with self.assertRaises(ValueError):
            add_fixed_atoms(system, positions, [5])


class TestAddIntermolecularBonds(unittest.TestCase):
    def _write_itp(self, tmpdir, body):
        itp_path = os.path.join(tmpdir, "inter.itp")
        with open(itp_path, "w") as f:
            f.write(body)
        return itp_path

    def test_adds_type6_bond(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                ";  ai    aj funct   length    k\n"
                "     1     10   6     0.300     5000\n",
            )
            system = System()
            for _ in range(10):
                system.addParticle(1.0)

            count = add_intermolecular_bonds(system, itp_path)

            self.assertEqual(count, 1)
            force = next(f for f in system.getForces() if isinstance(f, HarmonicBondForce))
            p1, p2, length, k = force.getBondParameters(0)
            self.assertEqual((p1, p2), (0, 9))
            self.assertAlmostEqual(length.value_in_unit(nanometer), 0.300)
            self.assertAlmostEqual(k.value_in_unit(unit.kilojoule_per_mole / nanometer**2), 5000)

    def test_rejects_unsupported_funct(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "     1     10   1     0.300     5000\n",
            )
            system = System()
            for _ in range(10):
                system.addParticle(1.0)

            with self.assertRaises(ValueError):
                add_intermolecular_bonds(system, itp_path)

    def test_no_bonds_section_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(tmpdir, "[ intermolecular_interactions ]\n")
            system = System()
            system.addParticle(1.0)

            with self.assertRaises(ValueError):
                add_intermolecular_bonds(system, itp_path)

    def test_atom_outside_system_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            itp_path = self._write_itp(
                tmpdir,
                "[ intermolecular_interactions ]\n"
                "[ bonds ]\n"
                "     1     10   6     0.300     5000\n",
            )
            system = System()
            for _ in range(5):  # too few particles for atom 10
                system.addParticle(1.0)

            with self.assertRaises(ValueError):
                add_intermolecular_bonds(system, itp_path)


class TestAddSoftcoreLj(unittest.TestCase):
    def test_moves_lj_into_softcore_force_and_zeroes_original(self):
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(0.5, 0.3, 1.0)  # charged particle
        nb.addParticle(0.0, 0.3, 1.0)  # neutral particle
        system.addForce(nb)

        softcore = add_softcore_lj(system)

        self.assertIsInstance(softcore, CustomNonbondedForce)
        self.assertEqual(softcore.getNumParticles(), 2)

        # LJ and charge zeroed out of the original NonbondedForce ...
        charge0, sigma0, epsilon0 = nb.getParticleParameters(0)
        self.assertEqual(epsilon0.value_in_unit(unit.kilojoule_per_mole), 0.0)
        self.assertEqual(charge0.value_in_unit(unit.elementary_charge), 0.0)
        # ... but restored via a lambda_q parameter offset instead.
        _, index, charge, _, _ = nb.getParticleParameterOffset(0)
        self.assertEqual(index, 0)
        self.assertAlmostEqual(charge, 0.5)

        # ... and moved into the softcore force's per-particle parameters.
        sigma, epsilon = softcore.getParticleParameters(0)
        self.assertAlmostEqual(sigma, 0.3)
        self.assertAlmostEqual(epsilon, 1.0)

    def test_only_charged_particles_get_lambda_q_offset(self):
        system = System()
        system.addParticle(1.0)
        system.addParticle(1.0)
        nb = NonbondedForce()
        nb.addParticle(0.5, 0.3, 1.0)
        nb.addParticle(0.0, 0.3, 1.0)
        system.addForce(nb)

        add_softcore_lj(system)

        self.assertEqual(nb.getNumParticleParameterOffsets(), 1)


class TestWriteGro(unittest.TestCase):
    def test_writes_expected_fields(self):
        top = app.Topology()
        chain = top.addChain()
        residue = top.addResidue("MOL", chain, id="1")
        top.addAtom("C1", app.element.carbon, residue)
        top.addAtom("C2", app.element.carbon, residue)

        positions = [(0.0, 0.0, 0.0), (1.234, 2.345, 3.456)] * nanometer
        box_vectors = [(2, 0, 0), (0, 2, 0), (0, 0, 2)] * nanometer

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.gro")
            write_gro(top, positions, box_vectors, out_path, "test title")

            with open(out_path, "r") as f:
                lines = f.readlines()

        self.assertEqual(lines[0].rstrip("\n"), "test title")
        self.assertEqual(lines[1].strip(), "2")
        self.assertEqual(
            lines[3].rstrip("\n"),
            "    1MOL     C2    2   1.234   2.345   3.456",
        )
        self.assertEqual(lines[4].split(), ["2.00000", "2.00000", "2.00000"])


if __name__ == "__main__":
    unittest.main()
