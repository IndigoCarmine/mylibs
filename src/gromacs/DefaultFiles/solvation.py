import os
import subprocess
import argparse

# NOW IT S FOR ONLY MCH!!!!!!!!!!!

parser = argparse.ArgumentParser(
    description="Insert molecules into a GROMACS configuration file."
)
parser.add_argument("input_gro", help="Input gro file")
parser.add_argument("solvent_gro", help="Solvent gro file")
parser.add_argument("output_gro", help="Output gro file")
parser.add_argument("nmol", help="Number of molecules to insert")
parser.add_argument("t", help="Number of tries to insert molecules")
argments = parser.parse_args()


proc = subprocess.Popen(
    [
        "gmx",
        "insert-molecules",
        "-f",
        argments.input_gro,
        "-ci",
        argments.solvent_gro,
        "-nmol",
        argments.nmol,
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
    if "Added" in line:
        residues = int(line.split()[1])

    if not line and proc.poll() is not None:
        break


if residues is not None:
    print(f"Number of residues: {residues}")
else:
    print("Could not find the number of residues in the output.")


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
