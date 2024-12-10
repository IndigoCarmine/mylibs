import abc
import enum
from _typeshed import Incomplete
from abc import ABC, abstractmethod

import numpy as np
from core.cui_utils import format_return_char as format_return_char
from dataclasses import dataclass

def defaut_file_content(name: str) -> str: ...

class Calclation(ABC, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    @abstractmethod
    def generate(self) -> dict[str, str]: ...
    @property
    @abstractmethod
    def name(self) -> str: ...

class EM(Calclation):
    nsteps: Incomplete
    emtol: Incomplete
    calculation_name: Incomplete
    def __init__(self, nsteps: int = 3000, emtol: float = 300, name: str = 'em') -> None: ...
    @property
    def name(self) -> str: ...
    def generate(self) -> dict[str, str]: ...

class MDType(enum.Enum):
    v_rescale_c_rescale = 1
    nose_hoover_parinello_rahman = 2
    berendsen = 3

@dataclass
class MD(Calclation):
    type:MDType
    calculation_name:str
    nsteps:int = 10000
    gen_vel:str = "yes"
    temperature:float = 300
    @property
    def name(self) -> str: ...
    def generate(self) -> dict[str, str]: ...


class RuntimeSolvation(Calclation):
    '''
    solvation (calculate number of molecules at runtime from the cell size)
    '''
    def __init__(self, solvent:str = "MCH", name:str = "solvation", rate:float = 1.0, ntry:int = 300):...

    @override
    def generate(self) -> dict[str,str]:...
    @override
    @property
    def name(self) -> str:...
    

class Solvation(Calclation):
    '''
    solvation
    '''
    def __init__(self, solvent:str = "MCH", name:str = "solvation",nmol:int = 100, ntry:int = 300):...
    @classmethod
    def from_cell_size(cls, cell_size:np.ndarray,name:str = "solvation", solvent:str = "MCH",  rate:float = 1.0):...
    @property
    def name(self) -> str:...
    def generate(self) -> dict[str,str]:...



def copy_file_script(extension: str, destination: str) -> str: ...
def copy_inherited_files_script(destination: str) -> str: ...

class OvereriteType(enum.Enum):
    '''
    no : do not overwrite the working directory. If the directory already exists, raise an error
    full_overwrite : remove the folder and recreate it
    add_calculation : add the calculation if the calculation folder not exists. If the folder exists, skip generating the calculation.
    '''
    no = 0
    full_overwrite = 1
    add_calculation = 2


def launch(calculations: list[Calclation], input_gro: str, working_dir: str, overwrite: bool = OvereriteType): ...
def test() -> None: ...
def main() -> None: ...
