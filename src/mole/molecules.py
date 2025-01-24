from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation

# move Interfaces


class IObject(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        raise NotImplementedError("This is an abstract class")

    @abstractmethod
    def translate(self, coordinate: np.ndarray):
        pass

    @abstractmethod
    def rotate(self, rotation: Rotation):
        pass


@dataclass
class AtomBase:
    symbol: str
    index: int
    coordinate: np.ndarray

    @classmethod
    def cast(cls, atom: "AtomBase") -> "AtomBase":
        return cls(atom.symbol, atom.index, atom.coordinate)


class IMolecule[Atom: AtomBase](IObject):
    @abstractmethod
    def get_children(self) -> list[Atom]:
        pass

    @abstractmethod
    def get_child(self, index: int) -> Atom:
        pass

    @classmethod
    def make(cls, atoms: list[Atom]):
        pass

    def part(self, start: int, end: int) -> "IMolecule[Atom]":
        """
        return a part of the molecule
        """
        return self.make(self.get_children()[start:end])


class Substructure[A: AtomBase, T: IMolecule[A]](IObject, Iterable[T]):
    molecules: list[T]

    def __init__(self, elements: list[T]):
        self.molecules = elements

    def get_children(self) -> list[T]:
        return self.molecules

    def translate(self, coordinate: np.ndarray):
        for element in self.molecules:
            element.translate(coordinate)

    def rotate(self, rotation: Rotation):
        for element in self.molecules:
            element.rotate(rotation)

    def __iter__(self):
        return iter(self.molecules)

    def press(self) -> list[A]:
        """
        concatenate all molecules as one molecule
        """
        return [atom for molecule in self.molecules for atom in molecule.get_children()]

    def as_molecule(self, type: type[IMolecule]) -> T:
        """
        concatenate all molecules as one molecule
        type is the type of the molecule
        """
        pressed = self.press()
        return type.make(pressed)
