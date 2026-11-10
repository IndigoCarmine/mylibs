import os
import tempfile
import unittest
import warnings
import xml.etree.ElementTree as ET

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    from gromacs.calculation import EM, MD, FileControl, MDType, SolvationMCH
    from gromacs.drawio import save_drawio, to_drawio


def _steps() -> list:
    return [
        EM(nsteps=30000, emtol=100, defines=["INTER"]),
        SolvationMCH(),
        MD(type=MDType.v_rescale_only_nvt, calculation_name="nvt", nsteps=10_000_000, defines=["INTER"]),
        MD(type=MDType.v_rescale_c_rescale, calculation_name="npt", nsteps=5_000_000, gen_vel="no",
           temperature=310),
    ]


def _labels(text: str) -> dict[str, str]:
    root = ET.fromstring(text)
    return {c.get("id", ""): c.get("value", "") for c in root.iter("mxCell") if c.get("vertex") == "1"}


class TestDrawio(unittest.TestCase):
    def test_cards(self) -> None:
        labels = _labels(to_drawio(_steps(), define_labels={"INTER": "Intermolecular Interaction"},
                                   none_label="なし"))
        self.assertEqual(labels["title"], "MD Calculation Workflow")
        self.assertEqual(
            labels["step0"],
            "Energy Minimization<hr>nsteps: 30,000<br>emtol: 100.0 kJ/mol/nm<br>defines: Intermolecular Interaction",
        )
        self.assertTrue(labels["step1"].startswith("Solvation (MCH) gmx solvate"))
        self.assertIn("nsteps: 10,000,000 (20.0 ns)", labels["step2"])
        self.assertIn("temperature: 310.0 K", labels["step3"])
        self.assertIn("defines: なし", labels["step3"])

    def test_edges_and_layout(self) -> None:
        root = ET.fromstring(to_drawio(_steps(), per_column=3, title=None))
        edges = [(c.get("source"), c.get("target")) for c in root.iter("mxCell") if c.get("edge") == "1"]
        self.assertEqual(edges, [("step0", "step1"), ("step1", "step2"), ("step2", "step3")])
        geo = {c.get("id"): c.find("mxGeometry") for c in root.iter("mxCell") if c.get("vertex") == "1"}
        self.assertNotIn("title", geo)
        g0, g3 = geo["step0"], geo["step3"]
        assert g0 is not None and g3 is not None
        self.assertEqual(g0.get("y"), g3.get("y"))  # 4th card starts the second column
        self.assertLess(float(g0.get("x", 0)), float(g3.get("x", 0)))

    def test_escaping(self) -> None:
        labels = _labels(to_drawio([FileControl("fc", "cat a.gro > output.gro\necho done")], show_names=True))
        self.assertEqual(labels["step0"], "File Control [fc]<hr>cat a.gro &gt; output.gro")

    def test_save_does_not_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "w.drawio")
            save_drawio(_steps(), path)
            with self.assertRaises(FileExistsError):
                save_drawio(_steps(), path)
            save_drawio(_steps()[:1], path, overwrite=True)
            with open(path, encoding="utf-8") as f:
                self.assertEqual(len(_labels(f.read())), 2)


if __name__ == "__main__":
    unittest.main()
