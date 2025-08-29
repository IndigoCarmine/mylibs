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
): ...
