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
    if not os.path.exists("MCH_test"):
        os.makedirs("MCH_test")

    cell_size = 20  # nm
    gro_path = "cell.gro"
    grofile = gro.GroFile("none", [], cell_size, cell_size, cell_size)
    grofile.save_gro(gro_path)

    calcs: list[calc.Calclation] = [
        calc.SolvationMCH(
            calculation_name="solv",
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

    dir = "MCH_test"
    calc.launch(calcs, gro_path, dir, overwrite=calc.OverwriteType.full_overwrite)
    # copy topology file
    shutil.copy("topo.top", dir + "/0_solv/topo.top")


if __name__ == "__main__":
    main()
