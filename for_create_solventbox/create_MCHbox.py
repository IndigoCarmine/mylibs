import os
import sys
from pathlib import Path
import shutil

# add src path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from gromacs import calculation as calc
from mole import gro
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def main():
    if not os.path.exists("MCH_solventbox"):
        os.makedirs("MCH_solventbox")

    cell_size = 3  # nm
    gro_path = "MCH.gro"
    grofile = gro.GroFile("MCH", [], cell_size, cell_size, cell_size)
    grofile.save_gro(gro_path)

    calcs: list[calc.Calclation] = [
        calc.RuntimeSolvation(
            calculation_name="solv",
            rate=0.8,
        ).check(np.array([cell_size, cell_size, cell_size])),
        calc.EM(
            nsteps=30000000000,
            emtol=100,
            calculation_name="em",
        ),
        calc.MD(
            type=calc.MDType.v_rescale_c_rescale,
            calculation_name="md_equilibration",
            nsteps=150000,
            nstout=5000,
            gen_vel="yes",
            temperature=300,
        ),
    ]

    calc.launch(calcs, gro_path, "MCH_solventbox", overwrite=calc.OverwriteType.full_overwrite)
    # copy topology file
    shutil.copy("topo.top", "MCH_solventbox/0_solv/topo.top")


def extract_gro_and_remove():
    gro_file = "MCH_solventbox/2_md_equilibration/output.gro"
    shutil.copy(gro_file, "../src/gromacs/DefaultFiles/MCH_solventbox.gro")
    # shutil.rmtree("MCH_solventbox")


if __name__ == "__main__":
    # main()
    extract_gro_and_remove()
