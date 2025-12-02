import abc
import enum
from abc import ABC, abstractmethod
from typing import override

import numpy as np
from dataclasses import dataclass, field

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

    # maximum number of warnings
    # (if you dont understand this parameter, Set it to 0!)
    maxwarn: int = 0
    useRestraint: bool = False

    @property
    def name(self) -> str: ...
    def generate(self) -> dict[str, str]: ...

class MDType(enum.Enum):
    v_rescale_c_rescale = 1
    v_rescale_only_nvt = 2
    nose_hoover_parinello_rahman = 3
    berendsen = 4

@dataclass(kw_only=True)
class MD(Calclation):
    type: MDType
    calculation_name: str
    nsteps: int = 10000

    # frequency to write the coordinates to the trajectory file
    nstout: int = 1000
    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = []
    # maximum number of warnings
    # (if you dont understand this parameter, Set it to 0!)
    maxwarn: int = 0
    useRestraint: bool = False
    useSemiisotropic: bool = False
    @property
    def name(self) -> str: ...
    def generate(self) -> dict[str, str]: ...

class RuntimeSolvation(Calclation):
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

                # number of molecules
                nmol = int(volume / mass_den * self.rate * Na)
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

class SolvationSCP216(Calclation):
    def __init__(self, name: str = "solvation"): ...
    @override
    def generate(self) -> dict[str, str]: ...
    @property
    @abstractmethod
    def name(self) -> str: ...

class FileControl(Calclation):
    def __init__(self, name: str, command: str) -> None: ...
    @override
    def generate(self) -> dict[str, str]: ...
    @classmethod
    def remove_MCH(cls, name: str) -> "FileControl": ...
    @classmethod
    def cell_resizeing(
        cls, name: str, x: float, y: float, z: float
    ) -> "FileControl": ...

@dataclass(kw_only=True)
class BarMethod(Calclation):
    calculation_name: str
    nsteps: int = 10000

    # frequency to write the coordinates to the trajectory file
    nstout: int = 1000

    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = field(default_factory=list)

    # maximum number of warnings
    # (if you dont understand this parameter, Set it to 0!)
    maxwarn: int = 0
    useRestraint: bool = False
    useSemiisotropic: bool = False
    additional_mdp_parameters: dict[str, str | float] = field(default_factory=dict)

    vdw_lambdas: list[float]
    coul_lambdas: list[float]
    bonded_lambdas: list[float]
    restraint_lambdas: list[float]
    mass_lambdas: list[float]
    temperature_lambdas: list[float]

    couple_moltype: str = "System"
    couple_lamda0: str = "vdw"
    couple_lamda1: str = "none"

    nstdhdl: int = 100

def copy_file_script(extension: str, destination: str) -> str: ...
def copy_inherited_files_script(destination: str) -> str: ...

class OverwriteType(enum.Enum):
    """
    no : do not overwrite the working directory.
    If the directory already exists, raise an error
    full_overwrite : remove the folder and recreate it
    add_calculation : add the calculation
    if the calculation folder not exists.
    If the folder exists, skip generating the calculation.
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
    calculation_name_and_isparaleljob: list[tuple[str, bool]],
    calc_path: str,
): ...
