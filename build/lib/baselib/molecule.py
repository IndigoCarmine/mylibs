import abc
from copy import deepcopy
from re import S
from typing import List, Optional, Self, override

import numpy as np
import numpy.typing as npt


class Atom(abc.ABC):
    @property
    @abc.abstractmethod
    def atom_symbol(self)->str:
        pass

    @property
    @abc.abstractmethod
    def atom_name(self)->str:
        pass

    @property
    @abc.abstractmethod
    def position(self)->np.ndarray:
        pass

    @abc.abstractmethod
    def translate(self, vector: np.ndarray):
        pass
    @abc.abstractmethod
    def rotate(self, matrix: np.ndarray):
        pass


class Objct:
    def __init__(self):
        self.positions: npt.NDArray[np.float32] = np.array(3)
        self.rotation: npt.NDArray[np.float32] = np.zeros((3, 3), dtype=np.float32)
        
    @abc.abstractmethod
    def translate(self, vector: npt.NDArray[np.float32]):
        pass
    @abc.abstractmethod
    def rotate(self, matrix:np.matrix):
        pass

    @abc.abstractmethod
    def clsName(self)->str:
        pass


class Molecule(Objct,abc.ABC):
    
    @property
    @abc.abstractmethod
    def atoms(self):
        pass

    @property
    @abc.abstractmethod
    def center(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @override
    def translate(self, vector: npt.NDArray[np.float32]):
        self.positions += vector

    @override
    def rotate(self, matrix: np.matrix):
        self.rotation = matrix @ self.rotation
    
    @override
    def clsName(self)->str:
        return "Molecule"

    def flatten(self) -> Self:
        '''
        Return a copy of the molecule
        '''
        return deepcopy(self)

class Structure(Objct):

    def __init__(self):
        self.name = ''
        self.positions: np.ndarray = np.array(3)
        self.rotation: np.matrix = np.matrix(np.zeros((3,3)))
        self.structures: list[Structure|Molecule] = []
    

    @override
    def translate(self, vector: np.ndarray):
        self.positions += vector
    
    @override
    def rotate(self, rotation: np.matrix):
        self.rotation = np.matrix(rotation @ self.rotation)
    
    @override
    def clsName(self)->str:
        return "Structure"
        
    def inner_translate(self, vector: np.ndarray):
        for structure in self.structures:
            structure.translate(vector)
        
    def inner_rotate(self, matrix: np.matrix):
        for structure in self.structures:
            structure.rotate(matrix)
        
        
    def add_structure(self, structure:Self|Molecule):
        self.structures.append(structure)
    
    def remove_structure(self, structure:Self):
        self.structures.remove(structure)


    def flatten(self)->Self:
        '''
        copy the structure and all substructures and return a flat structure
        '''
        new_structure = self.__class__()
        new_structure.positions = self.positions

        for structure in self.structures:
            if isinstance(structure, Structure):
                for substructure in structure.flatten().structures:
                    st = deepcopy(substructure)
                    new_structure.add_structure(substructure)  # type: ignore
            elif isinstance(structure, Molecule):
                st = deepcopy(structure)
                st.rotate(self.rotation)
                st.translate(self.positions)
                new_structure.add_structure(st) 
        return new_structure
    
    # def slice(self, nstruct:int)->Optional[Self]:
    #     '''
    #     slice the structure and all substructures and return a new structure
    #     '''
    #     if len(self.structures)%nstruct != 0:
    #         return None
        
    #     new_structure = self.__class__()
    #     new_structure.positions = self.positions
    #     new_structure.rotation = self.rotation
        
    #     return new_structure