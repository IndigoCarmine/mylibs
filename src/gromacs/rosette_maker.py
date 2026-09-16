"""
This module provides functions for constructing complex molecular assemblies,
specifically rosette and oligorosette structures, from individual molecular monomers.
"""
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
    Generates a rosette structure from a given monomer.
    The monomers are arranged in a circle, with each monomer rotated by a specified angle.
    Args:
        monomer (T): The monomer molecule to use.
        n (int): The number of monomers in the rosette.
        size (float): The diameter of the circle on which the monomer's top atom is placed.
        angle (float): The angle (in degrees or radians) between the monomer's Otop=C vector
                       and the normal vector of the circle.
        degree (bool): If True, `angle` is interpreted as degrees; otherwise, radians.
    Returns:
        molecules.Substructure[T]: A Substructure object representing the rosette.
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
    Generates a rosette structure from a given monomer.
    The monomers are arranged in a circle, with each monomer rotated radially.
    Args:
        monomer (T): The monomer molecule to use.
        n (int): The number of monomers in the rosette.
        size (float): The radius of the circle on which the monomer is placed.
    Returns:
        molecules.Substructure[T]: A Substructure object representing the rosette.
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
    Generates a half-rosette structure from a given monomer.
    The monomers are arranged in a semi-circle.
    Args:
        monomer (T): The monomer molecule to use.
        n (int): The number of monomers in the half-rosette.
        size (float): The radius of the semi-circle on which the monomer is placed.
    Returns:
        molecules.Substructure[T]: A Substructure object representing the half-rosette.
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
    Creates a stacked oligorosette structure from a single rosette.
    The rosettes are stacked along an axis, with optional rotation and slip to form a helix.
    Args:
        rosette (T): The base rosette molecule to stack.
        n (int): The number of rosettes in the oligorosette.
        length (float): The stacking distance between each rosette.
        angle (float): The rotation angle between each stacked rosette.
        slip (float): The slip distance between each stacked rosette, contributing to a helical structure.
        degree (bool): If True, `angle` is interpreted as degrees; otherwise, radians.
    Returns:
        molecules.Substructure[T]: A Substructure object representing the oligorosette.
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
