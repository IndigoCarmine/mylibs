"""
This script modifies a GROMACS topology file (topo.top) by adding a new molecule definition
"""

# for python 3.6.8 (it is server version)

import os


# modify the topology file
"""
...
(molecule topology data)
...


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


dummy = "dummy.top"
# Written by GROMACS. In the GROMACS code, the modification is simply done
# by adding one line at the end of the file, not at the end of the [molecules] section.
# However, if we use the [intermolecular_interactions] section, we need to add the line
# before that section instead.
# I believe this is a bug in GROMACS, since it does not insert the line within the
# [molecules] section. Why should I have to make such a radical change to the code?


top = r"topo.top"
# rename topo to topo_old
top_old = r"topo_old.top"
os.rename(top, top_old)
is_molecules_section = False

additional_line = ""
try:
    with open(dummy, "r") as f:
        additional_line = f.readline()
except FileNotFoundError:
    pass

if not additional_line:
    print(
        f"dummy file '{dummy}' not found. Please create it with the number of residues."
    )
    exit(1)


with open(top_old, "r") as file:
    with open(top, "w") as newfile:
        for line in file:
            if line.startswith("[ molecules ]"):
                newfile.write(line)
                is_molecules_section = True
            elif (
                line.startswith("[") and is_molecules_section
            ):  # end of molecules section
                newfile.write(additional_line)
                newfile.write(line)
                is_molecules_section = False
            else:
                newfile.write(line)
        # if the molecules section is at the end of the file
        if is_molecules_section:
            newfile.write(additional_line)
            is_molecules_section = False
