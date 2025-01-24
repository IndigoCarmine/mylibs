# NOW IT S FOR ONLY MCH!!!!!!!!!!!

# for python 3.6.8 (it is server version)

import os
import subprocess
import argparse
import sys

parser = argparse.ArgumentParser(
    description="Insert molecules into a GROMACS configuration file."
)
parser.add_argument("input_gro", help="Input gro file")
parser.add_argument("solvent", help="Solventname")
parser.add_argument("output_gro", help="Output gro file")
parser.add_argument("rate", help="rate of molecules to insert")
parser.add_argument("t", help="Number of tries to insert molecules")
argments = parser.parse_args()

solvent_gro = argments.solvent + ".gro"
volume = None  # nm^3


with open("input.gro", "r") as file:
    # last line is cell x y z
    cellsize = file.readlines()[-1]
    volume = (
        float(cellsize.split()[0])
        * float(cellsize.split()[1])
        * float(cellsize.split()[2])
    )


print("The volume of the cell is", volume, "nm^3")
if argments.solvent == "MCH":
    density = 0.77  # g/cm^3
    mass = 98.186  # g/mol
    mass_den = mass / density  # cm^3/mol = nm^3/mol * 10e21
    print("MCH is", mass_den, "cm^3/mol")
    print("is", mass_den * 10e21, "nm^3/mol")
else:
    raise ValueError("Invalid solvent")

Na = 6.022 * 100  # *10e21 # Avogadro's number
nmol = int(volume / mass_den * float(argments.rate) * Na)  # number of molecules

print("I will fill the cell with", nmol, "molecules")


shell = """
export OMP_NUM_THREADS=48
# Enable alias expansion
shopt -s expand_aliases
############################ Gromacs ##################################
source /short/cfga5932/App/gromacs/bin/GMXRC
alias gmx=/short/cfga5932/App/gromacs/bin/gmx_mpi
############################ Gromacs ##################################
gmx insert-molecules -f {input_gro} -ci {solvent_gro} -nmol {nmol} -try {t} -o {output_gro}
"""

proc: subprocess.Popen | None = None

if "3.6.8" in sys.version:
    print("It is server")
    proc = subprocess.Popen(
        [
            "bash",
            "-c",
            shell.format(
                input_gro=argments.input_gro,
                solvent_gro=solvent_gro,
                nmol=nmol,
                t=argments.t,
                output_gro=argments.output_gro,
            ),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
    )
else:
    import shutil

    gmx = None
    if shutil.which("gmx_mpi"):
        gmx = "gmx_mpi"
    elif shutil.which("gmx_d"):
        gmx = "gmx_d"
    elif shutil.which("gmx"):
        gmx = "gmx"
    else:
        raise ValueError("Could not find GROMACS executable")

    proc = subprocess.Popen(
        [
            gmx,
            "insert-molecules",
            "-f",
            argments.input_gro,
            "-ci",
            solvent_gro,
            "-nmol",
            str(nmol),
            "-try",
            argments.t,
            "-o",
            argments.output_gro,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )


residues = None

while True:
    if proc.stdout is None:
        break
    line = proc.stdout.readline()
    if "(now" in line:
        print(line, end="")
    elif "Added" in line:
        residues = int(line.split()[1])

    if not line and proc.poll() is not None:
        break


if residues is not None:
    print(f"Number of residues: {residues}")
else:
    print("Could not find the number of residues in the output.")


# modify the topology file
"""
...
(molecule topology data)
...

#include "MCH.itp"     # added by folowing script

[ system ]
SYSTEMNAME

[ molecules ]
; Compound       mols
MOL             100
MCH             432    # added by folowing script

...
[ intermolecular_interactions ]
...
"""

top = r"topo.top"
# rename topo to topo_old
top_old = r"topo_old.top"
os.rename(top, top_old)
is_molecules_section = False

with open(top_old, "r") as file:
    with open(top, "w") as newfile:
        for line in file:
            if line.startswith("[ system ]"):
                newfile.write('#include "MCH.itp"\n')
                newfile.write(line)
            elif line.startswith("[ molecules ]"):
                newfile.write(line)
                is_molecules_section = True
            elif (
                line.startswith("[") and is_molecules_section
            ):  # end of molecules section
                newfile.write("MCH             {}\n".format(residues))
                newfile.write(line)
                is_molecules_section = False
            else:
                newfile.write(line)
        # if the molecules section is at the end of the file
        if is_molecules_section:
            newfile.write("MCH             {}\n".format(residues))
            is_molecules_section = False
