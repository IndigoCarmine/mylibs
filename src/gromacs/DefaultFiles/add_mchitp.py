import os

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
...
[ intermolecular_interactions ]
...
"""

top = r"topo.top"
# rename topo to topo_old
top_old = r"topo_old.top"
if os.path.exists(top_old):
    os.remove(top_old)
os.rename(top, top_old)
is_molecules_section = False
with open(top_old, "r") as file:
    with open(top, "w") as newfile:
        for line in file:
            if line.startswith("[ system ]"):
                newfile.write('#include "MCH.itp"\n')
                newfile.write(line)
            else:
                newfile.write(line)
