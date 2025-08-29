# This script generates intermolecular bond definitions for a system of multiple monomers.
# Each monomer is read from a GRO file, and specified atom pairs are converted to global indices.
# Bonds are created sequentially, with ring closures applied every 'nmols_in_rosette' monomers.
# The resulting bonds are written in GROMACS [ intermolecular_interactions ] format
# (ai aj funct length k), ready to be appended to the topology file.
def generate_inermolecular_interactions(
    natoms: int,
    nmols: int,
    bonds: list[tuple[int, int]],
    nmols_in_rosette: int = 6,
    bondleng: float = 0.3,  # nm
    k: float = 5000,  # kJ/(mol*nm^2)
    outfile_path: str = "intermolecular_bond.itp",
):
    grobalbond = []
    # generate global bond list
    for i in range(nmols):
        for b in bonds:
            if (i + 1) % nmols_in_rosette == 0:
                # ring closure
                grobalbond.append(
                    (b[0] + i * natoms, b[1] + (i - nmols_in_rosette) * natoms)
                )
            else:
                grobalbond.append((b[0] + i * natoms, b[1] + i * natoms))

    #   135   286     1     0.3   5000
    with open(outfile_path, "w", newline="\n") as f:
        f.write("[ intermolecular_interactions ]\n")
        f.write("[ bonds ]\n")
        f.write(";  ai    aj funct   length    k\n")
        for b in grobalbond:
            f.write(f"{b[0]:6d} {b[1]:6d}   6    {bondleng:6.3f}  {k:6.0f}\n")
