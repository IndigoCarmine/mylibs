import copy

import numpy as np
import mole.molecules as molecules
from scipy.spatial.transform import Rotation


def make_rosette[
    T: molecules.IMolecule
](
    monomer: T, n: int, size: float, angle: float, degree: bool = True
) -> molecules.Substructure[T]:
    """
    make rosette from monomer
    n: number of monomer
    size: diamiter of circle setting monomer top atom
    angle: angle between monomer Otop=C vector and normal vector of the circle
    """
    rosette = molecules.Substructure(
        [copy.deepcopy(monomer) for i in range(n)])

    radius = size / 2
    # move monomer to circle
    for i in range(n):
        rosette.get_children()[i].rotate(
            Rotation.from_euler("y", -angle, degrees=degree)
        )
        # move head atom to circle
        rosette.get_children()[i].translate(np.array([radius, 0, 0]))
        # rotate monomer in a radial direction
        rosette.get_children()[i].rotate(
            Rotation.from_euler("y", 360 / n * i, degrees=True)
        )
    return rosette
    # for debug
    # rosette.extract_xyz('')


def make_rosette2[T: molecules.IMolecule](
        monomer: T, n: int, size: float
) -> molecules.Substructure[T]:
    """
    make rosette from monomer
    n: number of monomer
    size: diamiter of circle setting monomer top atom
    """
    rosette: molecules.Substructure[T] = molecules.Substructure(
        [copy.deepcopy(monomer) for i in range(n)]
    )

    # move monomer to circle
    for i in range(n):
        # move head atom to circle
        monomer: T = rosette.get_children()[i]
        monomer.translate(np.array([size, 0, 0]))
        # rotate monomer in a radial direction
        monomer.rotate(Rotation.from_euler("z", 360 / n * i, degrees=True))
    return rosette
    # for debug
    # rosette.extract_xyz('')


def make_half_rosette2[
    T: molecules.IMolecule
](monomer: T, n: int, size: float) -> molecules.Substructure[T]:
    """
    make rosette from monomer
    n: number of monomer (half of rosette)
    size: diamiter of circle setting monomer top atom
    """
    rosette: molecules.Substructure[T] = molecules.Substructure(
        [copy.deepcopy(monomer) for i in range(n)]
    )

    # move monomer to circle
    for i in range(n):
        # move head atom to circle
        monomer: T = rosette.get_children()[i]
        monomer.translate(np.array([size, 0, 0]))
        # rotate monomer in a radial direction
        monomer.rotate(Rotation.from_euler("z", 180 / n * i, degrees=True))
    return rosette
    # for debug
    # rosette.extract_xyz('')


def make_oligorosette[
    T: molecules.IMolecule
](
    rosette: T,
    n: int,
    length: float,
    angle: float,
    slip: float = 0,
    degree: bool = True,
) -> molecules.Substructure[T]:
    """
    make stacked rosette from rosette
    n: number of rosette
    length: length of stacking
    angle: how much rotate rosette
    slip: slip distance of rosette
    """
    oligorosette = molecules.Substructure[T](
        [copy.deepcopy(rosette) for i in range(n)])

    # distance between supramolecular polymer axis and rosette.
    radius = 0
    if slip != 0:
        radius = (slip / 2) / (np.sin((angle / 2) * np.pi / 180))

    # move rosette to stacking
    for i, rosette in enumerate(oligorosette.get_children()):
        # move y direction
        rosette.translate(np.array([0, 0, length * i]))

        # move rosette to slip. It will be helix
        rosette.rotate(Rotation.from_euler("Z", angle * i, degrees=degree))
        rosette.translate(
            np.array(
                [
                    radius * np.sin(i * angle * np.pi / 180),
                    radius * np.cos(i * angle * np.pi / 180),
                    0,
                ]
            )
        )

    return oligorosette
