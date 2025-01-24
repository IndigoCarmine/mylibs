from scipy.spatial.transform import Rotation
import mole.molecules as i
import numpy as np


def pre_coordinate[
    T: i.IMolecule
](molecule: T, topO: int, aromaticsideO: int, aromaticothersideO: int) -> T:
    """
    pre-coordinate molecule
    """
    top_coordinate = molecule.get_child(topO).coordinate
    molecule.translate(-top_coordinate)

    vector = (
        molecule.get_child(aromaticsideO).coordinate
        + molecule.get_child(aromaticothersideO).coordinate
    )
    print(vector)
    # C=O bond(Barbiturate) vector
    vector = vector.astype(float) / np.linalg.norm(vector)
    # move from vector to x axis
    rot = Rotation.align_vectors([[1, 0, 0]], [vector])[0]

    molecule.rotate(rot)

    # first O to next O vector
    vector = molecule.get_child(aromaticsideO).coordinate.astype(float)
    # rot to move vector to xz plane and rotate only x axis
    rot = Rotation.from_euler("x", np.arctan2(vector[1], vector[2]), degrees=False)
    molecule.rotate(rot)

    return molecule


def precooredinate2[
    T: i.IMolecule
](molecule: T, topO: int, aromaticsideNH: int, aromaticothersideO: int) -> T:
    """
    pre-coordinate molecule
    """

    # move top O to origin
    top_coordinate = molecule.get_child(topO).coordinate
    molecule.translate(-top_coordinate)

    # move aromatic side NH to x axis
    vector = molecule.get_child(aromaticsideNH).coordinate.astype(float)

    # move vector to x axis
    rot = Rotation.from_euler("z", -np.arctan2(vector[1], vector[0]), degrees=False)

    molecule.rotate(rot)

    # move aromatic side NH to x axis
    vector = molecule.get_child(aromaticsideNH).coordinate.astype(float)

    rot = Rotation.from_euler("y", np.arctan2(vector[2], vector[0]), degrees=False)
    molecule.rotate(rot)

    # move aromatic side O to xz plane
    vector = molecule.get_child(aromaticothersideO).coordinate.astype(float)

    print(-np.arctan2(vector[2], -vector[1]) * 180 / np.pi)
    # move vector to xz plane
    rot = Rotation.from_euler("x", np.arctan2(vector[2], -vector[1]), degrees=False)
    # rot = Rotation.from_euler("x", np.pi, degrees=False) * rot

    molecule.rotate(rot)

    return molecule
