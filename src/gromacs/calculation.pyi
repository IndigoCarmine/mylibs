import abc
import enum
from abc import ABC, abstractmethod
from typing import override

import numpy as np
from base_utils.cui_utils import format_return_char as format_return_char
from dataclasses import dataclass

def defaut_file_content(name: str) -> str: ...

class Calclation(ABC, metaclass=abc.ABCMeta):
    def __init__(self) -> None: ...
    @abstractmethod
    def generate(self) -> dict[str, str]: ...
    @property
    @abstractmethod
    def name(self) -> str: ...

@dataclass(kw_only=True)
class EM(Calclation):
    nsteps: int = 3000
    emtol: float = 300
    calculation_name: str = "em"
    defines: list[str] = []
    maxwarn: int = (
        0  # maximum number of warnings (if you dont understand this parameter, Set it to 0!)
    )
    useRestraint: bool = False

    @property
    def name(self) -> str: ...
    def generate(self) -> dict[str, str]: ...

class MDType(enum.Enum):
    v_rescale_c_rescale = 1
    nose_hoover_parinello_rahman = 2
    berendsen = 3

@dataclass(kw_only=True)
class MD(Calclation):
    type: MDType
    calculation_name: str
    nsteps: int = 10000
    nstout: int = 1000  # frequency to write the coordinates to the trajectory file
    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = []
    maxwarn: int = (
        0  # maximum number of warnings (if you dont understand this parameter, Set it to 0!)
    )
    useRestraint: bool = False
    useSemiisotropic: bool = False
    @property
    def name(self) -> str: ...
    def generate(self) -> dict[str, str]: ...

class RuntimeSolvation(Calclation):
    """
    solvation (calculate number of molecules at runtime from the cell size)
    """

    solvent: str
    rate: float
    ntry: int
    calculation_name: str

    def __init__(
        self,
        *,
        solvent: str = "MCH",
        name: str = "solvation",
        rate: float = 1.0,
        ntry: int = 300,
    ): ...
    @override
    def generate(self) -> dict[str, str]: ...
    @override
    @property
    def name(self) -> str: ...
    def check(self, cell_size: np.ndarray) -> "RuntimeSolvation":
        """
        print the number of molecules to be added to the cell
        """
        volume = np.prod(cell_size)  # nm^3

        Na = 6.022 * 100  # *10e21 # Avogadro's number

        match self.solvent:
            case "MCH":
                density = 0.77  # g/cm^3
                mass = 98.186  # g/mol
                mass_den = mass / density  # cm^3/mol = nm^3/mol * 10e21

                nmol = int(volume / mass_den * self.rate * Na)  # number of molecules
                natoms = nmol * 20  # number of atoms

            case _:
                raise ValueError("Invalid solvent")

        print("I will fill the cell with", nmol, "molecules")
        print("The cell contains", natoms, "atoms")

        return self

class Solvation(Calclation):
    """
    solvation
    """

    def __init__(
        self,
        solvent: str = "MCH",
        name: str = "solvation",
        nmol: int = 100,
        ntry: int = 300,
    ): ...
    @classmethod
    def from_cell_size(
        cls,
        cell_size: np.ndarray,
        name: str = "solvation",
        solvent: str = "MCH",
        rate: float = 1.0,
    ): ...
    @property
    def name(self) -> str: ...
    def generate(self) -> dict[str, str]: ...

def copy_file_script(extension: str, destination: str) -> str: ...
def copy_inherited_files_script(destination: str) -> str: ...

class OverwriteType(enum.Enum):
    """
    no : do not overwrite the working directory. If the directory already exists, raise an error
    full_overwrite : remove the folder and recreate it
    add_calculation : add the calculation if the calculation folder not exists. If the folder exists, skip generating the calculation.
    """

    no = 0
    full_overwrite = 1
    add_calculation = 2

def launch(
    calculations: list[Calclation],
    input_gro: str,
    working_dir: str,
    overwrite: OverwriteType = OverwriteType.no,
): ...
def generate_stepbystep_runfile(
    init_structures: list[str],
    calculation_name_and_isparaleljob: tuple[list[str], bool],
    calc_path: str,
): ...
