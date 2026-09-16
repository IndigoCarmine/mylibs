import mole.molecules as molecules


def make_rosette[
    A: molecules.AtomBase, T: molecules.IMolecule
](monomer: T, n: int, size: float, angle: float, degree: bool = True
  ) -> molecules.Substructure[A, T]: ...


def make_rosette2[
    A: molecules.AtomBase, T: molecules.IMolecule
](monomer: T, n: int, size: float) -> molecules.Substructure[A, T]: ...


def make_half_rosette2[
    A: molecules.AtomBase, T: molecules.IMolecule
](monomer: T, n: int, size: float) -> molecules.Substructure[A, T]: ...


def make_oligorosette[
    A: molecules.AtomBase, T: molecules.IMolecule
](
    rosette: T,
    n: int,
    length: float,
    angle: float,
    slip: float = 0,
    degree: bool = True,
) -> molecules.Substructure[A, T]: ...
