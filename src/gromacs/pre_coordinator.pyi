import mole.molecules as molecules

def pre_coordinate[T: molecules.IMolecule](
    molecule: T, topO: int, aromaticsideO: int, aromaticothersideO: int
) -> T: ...
def precooredinate2[T: molecules.IMolecule](
    molecule: T, topO: int, aromaticsideNH: int, aromaticothersideO: int
) -> T: ...
def get_substructure_match(
    molecule: molecules.IMolecule, substructure: molecules.IMolecule
) -> list[int]: ...
