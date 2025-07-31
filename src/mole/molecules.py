"""
This module defines abstract base classes and interfaces for representing and manipulating molecular structures.
It provides a foundation for handling atoms, molecules, and substructures in a generic way.
"""
from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

# move Interfaces


class IObject(metaclass=ABCMeta):
    """
    Abstract base class for objects that can be translated and rotated in 3D space.
    """

    @abstractmethod
    def __init__(self):
        """
        Initializes the object. Must be implemented by subclasses.
        """
        raise NotImplementedError("This is an abstract class")

    @abstractmethod
    def translate(self, coordinate: np.ndarray):
        """
        Translates the object by a given coordinate vector.
        Args:
            coordinate (np.ndarray): The translation vector.
        """
        pass

    @abstractmethod
    def rotate(self, rotation: Rotation):
        """
        Rotates the object by a given rotation.
        Args:
            rotation (Rotation): The rotation object.
        """
        pass


@dataclass
class AtomBase:
    """
    Base class for representing an atom.
    Contains fundamental properties like symbol, index, and 3D coordinate.
    """
    symbol: str
    index: int
    coordinate: np.ndarray

    @classmethod
    def cast(cls, atom: "AtomBase") -> "AtomBase":
        """
        Casts an existing AtomBase object to a new AtomBase instance.
        Args:
            atom (AtomBase): The atom to cast.
        Returns:
            AtomBase: A new AtomBase instance with the same properties.
        """
        return cls(atom.symbol, atom.index, atom.coordinate)


class IMolecule[Atom: AtomBase](IObject):
    """
    Interface for molecular objects.
    Defines methods for accessing constituent atoms and creating new molecular instances.
    """
    @abstractmethod
    def get_children(self) -> list[Atom]:
        """
        Returns a list of all atoms (children) in the molecule.
        """
        pass

    @abstractmethod
    def get_child(self, index: int) -> Atom:
        """
        Returns a specific atom (child) by its index.
        Args:
            index (int): The index of the atom to retrieve.
        """
        pass

    @classmethod
    def make(cls, atoms: list[Atom]):
        """
        Factory method to create a new molecule instance from a list of atoms.
        """
        pass

    def part(self, start: int, end: int) -> "IMolecule[Atom]":
        """
        Returns a new molecule instance representing a part of the current molecule.
        Args:
            start (int): The starting index of the atoms to include.
            end (int): The ending index of the atoms to include (exclusive).
        Returns:
            IMolecule[Atom]: A new molecule object containing the specified part.
        """
        return self.make(self.get_children()[start:end])


class Substructure[A: AtomBase, T: IMolecule[A]](IObject, Iterable[T]):
    """
    Represents a collection of molecules as a substructure.
    Allows for collective translation and rotation of its constituent molecules.
    """
    molecules: list[T]

    def __init__(self, elements: list[T]):
        """
        Initializes a Substructure object.
        Args:
            elements (list[T]): A list of molecules forming the substructure.
        """
        self.molecules = elements

    def get_children(self) -> list[T]:
        """
        Returns the list of molecules within the substructure.
        """
        return self.molecules

    def translate(self, coordinate: np.ndarray):
        """
        Translates all molecules in the substructure by a given vector.
        Args:
            coordinate (np.ndarray): The translation vector.
        """
        for element in self.molecules:
            element.translate(coordinate)

    def rotate(self, rotation: Rotation):
        """
        Rotates all molecules in the substructure by a given rotation.
        Args:
            rotation (Rotation): The rotation object.
        """
        for element in self.molecules:
            element.rotate(rotation)

    def __iter__(self):
        """
        Enables iteration over the molecules in the substructure.
        """
        return iter(self.molecules)

    def press(self) -> list[A]:
        """
        Concatenates all atoms from all molecules in the substructure into a single list.
        Returns:
            list[A]: A flattened list of all atoms.
        """
        return [atom for molecule in self.molecules
                for atom in molecule.get_children()]

    def as_molecule(self, type: type[IMolecule]) -> T:
        """
        Converts the substructure into a single molecule of a specified type.
        Args:
            type (type[IMolecule]): The type of molecule to create.
        Returns:
            T: A new molecule object representing the combined substructure.
        """
        pressed = self.press()
        return type.make(pressed)
