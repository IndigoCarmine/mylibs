import mole.molecules as i

def pre_coordinate[
    T: i.IMolecule
](molecule: T, topO: int, aromaticsideO: int, aromaticothersideO: int) -> T: ...
def precooredinate2[
    T: i.IMolecule
](molecule: T, topO: int, aromaticsideNH: int, aromaticothersideO: int) -> T: ...
