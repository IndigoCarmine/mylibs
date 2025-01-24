import mole.molecules as i

def make_rosette[
    A: i.AtomBase, T: i.IMolecule
](monomer: T, n: int, size: float, angle: float, degree: bool = True) -> i.Substructure[
    A, T
]: ...
def make_rosette2[
    A: i.AtomBase, T: i.IMolecule
](monomer: T, n: int, size: float) -> i.Substructure[A, T]: ...
def make_half_rosette2[
    A: i.AtomBase, T: i.IMolecule
](monomer: T, n: int, size: float) -> i.Substructure[A, T]: ...
def make_oligorosette[
    A: i.AtomBase, T: i.IMolecule
](
    rosette: T,
    n: int,
    length: float,
    angle: float,
    slip: float = 0,
    degree: bool = True,
) -> i.Substructure[A, T]: ...
