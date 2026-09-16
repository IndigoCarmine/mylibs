"""
Tests for the solvation step's topology rewriting.

The bug these guard against: resuming an interrupted solvation applied the
topology edits a second time, so `topo.top` ended up with duplicated
`#include "MCH.itp"` and `<solvent> <count>` lines and grompp refused it.
The rewriting must therefore be idempotent -- N runs must leave exactly what
one run leaves.
"""

import glob
import os
import subprocess
import sys
import tempfile
import unittest

from gromacs.calculation import (
    SolvationMCH,
    SolvationSCP216,
    copy_inherited_files_script,
    default_file_content,
)

COUNT = "MCH             432\n"
INCLUDE = '#include "MCH.itp"\n'

# [ molecules ] is followed by another section, which is the case GROMACS' own
# "append at the end of the file" behaviour gets wrong. The molecule names are
# deliberately not pure letters: "MOL1"/"PEG_A" used to be mistaken for the end
# of the section.
TOPO_INTERMOLECULAR = """; test topology
#include "oplsaa.ff/forcefield.itp"
#include "MOL.itp"

[ system ]
SYSTEMNAME

[ molecules ]
; Compound       mols
MOL1            100
PEG_A           2

[ intermolecular_interactions ]
[ bonds ]
1 2 6 0.5 1000
"""

# [ molecules ] runs to the end of the file.
TOPO_AT_EOF = """#include "MOL.itp"

[ system ]
SYSTEMNAME

[ molecules ]
; Compound       mols
MOL             100
"""

# GROMACS accepts section headers without inner spaces.
TOPO_NO_SPACES = """#include "MOL.itp"

[system]
SYSTEMNAME

[molecules]
MOL             100
"""

# What a second solvation step in the same chain inherits.
TOPO_ALREADY_SOLVATED = """#include "MOL.itp"
#include "MCH.itp"

[ system ]
SYSTEMNAME

[ molecules ]
MOL             100
MCH             100
"""


class SolvationScriptTestCase(unittest.TestCase):
    """Runs the generated top_mod.py the way the generated mdrun.sh does."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name
        self.addCleanup(self._tmp.cleanup)
        self.write("top_mod.py", default_file_content("top_mod.py"))

    def path(self, name):
        return os.path.join(self.dir, name)

    def write(self, name, content):
        with open(self.path(name), "w", newline="") as f:
            f.write(content)

    def read(self, name):
        with open(self.path(name), "r", newline="") as f:
            return f.read()

    def setup_topology(self, topology, dummy="MCH 432\n"):
        self.write("topo.top", topology)
        self.write("dummy.top", dummy)

    def run_top_mod(self, *args):
        return subprocess.run(
            [sys.executable, "top_mod.py"] + list(args),
            cwd=self.dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        )

    def run_ok(self, *args):
        result = self.run_top_mod(*args)
        self.assertEqual(result.returncode, 0, result.stdout)
        return result


class TestTopModPlacement(SolvationScriptTestCase):
    def test_count_goes_inside_the_molecules_section(self):
        self.setup_topology(TOPO_INTERMOLECULAR)
        self.run_ok("--include", "MCH.itp")
        lines = self.read("topo.top").splitlines(True)

        self.assertEqual(lines.count(COUNT), 1)
        self.assertEqual(lines.count(INCLUDE), 1)
        # after the molecules already listed, before the next section
        self.assertLess(lines.index("PEG_A           2\n"), lines.index(COUNT))
        self.assertLess(lines.index(COUNT), lines.index("[ intermolecular_interactions ]\n"))
        # and the include goes just above [ system ]
        self.assertEqual(lines.index(INCLUDE) + 1, lines.index("[ system ]\n"))

    def test_molecules_section_at_end_of_file(self):
        self.setup_topology(TOPO_AT_EOF)
        self.run_ok("--include", "MCH.itp")
        self.assertTrue(self.read("topo.top").endswith(COUNT))

    def test_headers_without_inner_spaces(self):
        self.setup_topology(TOPO_NO_SPACES)
        self.run_ok("--include", "MCH.itp")
        lines = self.read("topo.top").splitlines(True)
        self.assertEqual(lines.count(COUNT), 1)
        self.assertEqual(lines.index(INCLUDE) + 1, lines.index("[system]\n"))

    def test_second_solvation_step_adds_a_count_but_not_a_second_include(self):
        self.setup_topology(TOPO_ALREADY_SOLVATED)
        self.run_ok("--include", "MCH.itp")
        lines = self.read("topo.top").splitlines(True)
        self.assertEqual(lines.count(INCLUDE), 1)
        self.assertEqual(lines.count(COUNT), 1)
        self.assertEqual(lines.count("MCH             100\n"), 1)

    def test_last_count_in_dummy_top_wins(self):
        # gmx solvate -p appends, so a re-run leaves the older count behind.
        self.setup_topology(TOPO_AT_EOF, dummy="MCH 111\nMCH 432\n")
        self.run_ok("--include", "MCH.itp")
        self.assertEqual(self.read("topo.top").count(COUNT), 1)
        self.assertNotIn("111", self.read("topo.top"))

    def test_without_include_option_no_include_is_added(self):
        self.setup_topology(TOPO_AT_EOF, dummy="SOL 9000\n")
        self.run_ok()
        content = self.read("topo.top")
        self.assertIn("SOL             9000\n", content)
        self.assertNotIn("#include \"MCH.itp\"", content)


class TestTopModIdempotency(SolvationScriptTestCase):
    """The actual regression: resuming must not rewrite the topology twice."""

    def assert_repeated_runs_match(self, topology):
        self.setup_topology(topology)
        self.run_ok("--include", "MCH.itp")
        once = self.read("topo.top")

        self.run_ok("--include", "MCH.itp")
        self.run_ok("--include", "MCH.itp")
        self.assertEqual(self.read("topo.top"), once)
        self.assertEqual(once.count(INCLUDE), 1)
        self.assertEqual(once.count(COUNT), 1)

    def test_three_runs_equal_one_run_with_a_following_section(self):
        self.assert_repeated_runs_match(TOPO_INTERMOLECULAR)

    def test_three_runs_equal_one_run_with_molecules_at_end_of_file(self):
        self.assert_repeated_runs_match(TOPO_AT_EOF)

    def test_three_runs_equal_one_run_on_an_already_solvated_topology(self):
        self.assert_repeated_runs_match(TOPO_ALREADY_SOLVATED)

    def test_resume_after_a_second_solvate_run(self):
        # The real resume shape: solvate ran again and appended to dummy.top.
        self.setup_topology(TOPO_INTERMOLECULAR)
        self.run_ok("--include", "MCH.itp")
        once = self.read("topo.top")

        with open(self.path("dummy.top"), "a", newline="") as f:
            f.write("MCH 432\n")
        self.run_ok("--include", "MCH.itp")

        self.assertEqual(self.read("topo.top"), once)

    def test_rerun_with_a_new_count_replaces_the_old_one(self):
        # e.g. the user changed -scale and re-ran the step.
        self.setup_topology(TOPO_INTERMOLECULAR)
        self.run_ok("--include", "MCH.itp")

        self.write("dummy.top", "MCH 500\n")
        self.run_ok("--include", "MCH.itp")

        content = self.read("topo.top")
        self.assertIn("MCH             500\n", content)
        self.assertNotIn(COUNT, content)
        self.assertEqual(content.count(INCLUDE), 1)

    def test_snapshot_does_not_match_the_top_glob(self):
        # copy.sh copies *.top forward; the snapshot must stay behind.
        self.setup_topology(TOPO_INTERMOLECULAR)
        self.run_ok("--include", "MCH.itp")
        self.assertTrue(os.path.exists(self.path("topo.top.orig")))
        self.assertEqual(
            sorted(os.path.basename(p) for p in glob.glob(self.path("*.top"))),
            ["dummy.top", "topo.top"],
        )


class TestTopModFailures(SolvationScriptTestCase):
    def test_missing_count_leaves_the_topology_alone(self):
        # This used to rename topo.top away before reading dummy.top, leaving
        # the directory with no topology at all -- unrecoverable on resume.
        self.setup_topology(TOPO_AT_EOF, dummy="")
        result = self.run_top_mod("--include", "MCH.itp")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.read("topo.top"), TOPO_AT_EOF)

    def test_missing_dummy_file_leaves_the_topology_alone(self):
        self.write("topo.top", TOPO_AT_EOF)
        result = self.run_top_mod("--include", "MCH.itp")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.read("topo.top"), TOPO_AT_EOF)

    def test_hand_edited_topology_is_refused_but_force_rebuilds(self):
        self.setup_topology(TOPO_AT_EOF)
        self.run_ok("--include", "MCH.itp")
        edited = self.read("topo.top") + "; a hand written note\n; and another\n"
        self.write("topo.top", edited)

        result = self.run_top_mod("--include", "MCH.itp")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.read("topo.top"), edited)

        self.run_ok("--include", "MCH.itp", "--force")
        rebuilt = self.read("topo.top")
        self.assertEqual(rebuilt.count(COUNT), 1)
        self.assertNotIn("hand written note", rebuilt)

    def test_a_directory_corrupted_by_the_old_scripts_is_refused(self):
        # No snapshot exists in such a directory (the old scripts never made
        # one), so the duplicated include is the only evidence left.
        corrupted = TOPO_AT_EOF.replace(
            "[ system ]", INCLUDE + INCLUDE + "[ system ]"
        ).replace("MOL             100\n", "MOL             100\n" + COUNT + COUNT)
        self.setup_topology(corrupted)

        result = self.run_top_mod("--include", "MCH.itp")
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(self.read("topo.top"), corrupted)
        self.assertFalse(os.path.exists(self.path("topo.top.orig")))

    def test_topology_without_a_molecules_section_is_refused(self):
        self.setup_topology("[ system ]\nSYSTEMNAME\n")
        result = self.run_top_mod("--include", "MCH.itp")
        self.assertNotEqual(result.returncode, 0)


class TestSolvationGenerators(unittest.TestCase):
    def test_mch_no_longer_ships_the_second_rewriting_script(self):
        files = SolvationMCH().generate()
        self.assertNotIn("add_mchitp.py", files)
        self.assertIn("top_mod.py", files)

    def test_output_gro_is_published_after_the_topology_is_committed(self):
        script = SolvationMCH().generate()["mdrun.sh"]
        self.assertIn("set -e", script)
        self.assertIn(": > dummy.top", script)
        # gmx must not write output.gro directly: run.sh treats it as the
        # "this step is finished" marker when a calculation is resumed.
        self.assertNotIn("-o output.gro", script)
        self.assertLess(script.index("python top_mod.py"), script.index("mv solvated.gro output.gro"))

    def test_spc216_uses_the_same_restart_safe_shape(self):
        script = SolvationSCP216().generate()["mdrun.sh"]
        self.assertIn(": > dummy.top", script)
        self.assertNotIn("-o output.gro", script)
        self.assertNotIn("--include", script)

    def test_copy_script_leaves_scratch_topologies_behind(self):
        script = copy_inherited_files_script("1_em")
        self.assertNotIn("cp *.top", script)
        self.assertIn("dummy.top", script)


if __name__ == "__main__":
    unittest.main()
