"""
This module defines abstract and concrete classes for setting up and managing GROMACS simulations.
It includes functionalities for energy minimization (EM), molecular dynamics (MD),
solvation, and file control operations, along with utilities for generating simulation scripts.
"""

from abc import ABC, abstractmethod
import warnings
from pydantic.dataclasses import dataclass, Field
import dataclasses
import enum
import inspect
import os
import shutil
from typing import override

import json
from . import mdp
import numpy as np


def defaut_file_content(name: str) -> str:
    """
    Reads the content of a default file from the DefaultFiles directory.

    Args:
        name (str): The name of the file to read.

    Returns:
        str: The content of the file as a string.

    Raises:
        FileNotFoundError: If the DefaultFiles directory does not exist or the file is not found.
    """
    DefaultFile_dir = os.path.join(os.path.dirname(__file__), "DefaultFiles")
    if not os.path.exists(DefaultFile_dir):
        raise FileNotFoundError("DefaultFiles directory not found")
    with open(os.path.join(DefaultFile_dir, name), "r") as f:
        return f.read()


class Calculation(ABC):
    """
    Abstract base class for all GROMACS calculation types.
    Defines the interface for generating calculation files and retrieving the calculation name.
    """

    @abstractmethod
    def generate(self) -> dict[str, str]:
        """
        Generates the necessary files for the calculation.
        Returns:
            dict[str, str]: A dictionary where keys are file names and values are file contents.
        """
        raise NotImplementedError(
            "This method must be implemented. " + "Abstract method was called"
        )

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Returns the name of the calculation.
        """
        pass


@dataclass(kw_only=True)
class EM(Calculation):
    """
    Class representing an energy minimization (EM) calculation.
    Attributes:
        nsteps (int): Number of steps for the energy minimization. Default is 3000.
        emtol (float): Energy minimization tolerance. Default is 300.
        calculation_name (str): Name of the calculation. Default is "em".
        defines (list[str]): List of defines for the calculation. Default is an empty list.
        maxwarn (int): Maximum number of warnings allowed. Default is 0.
        useRestraint (bool): Flag to indicate if restraints should be used. Default is False.
    Properties:
        name (str): Returns the name of the calculation.
    Methods:
        generate() -> dict[str, str]: Generates the necessary files for the energy minimization calculation.
    """

    nsteps: int = 3000
    emtol: float = 300
    calculation_name: str = "em"
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = (
        # maximum number of warnings
        # (if you dont understand this parameter, Set it to 0!)
        0
    )
    useRestraint: bool = False

    @property
    def name(self) -> str:
        return self.calculation_name

    @override
    def generate(self) -> dict[str, str]:
        options_txt = " -maxwarn " + str(self.maxwarn)
        if self.useRestraint:
            options_txt += " -r input.gro"

        if len(self.defines) == 0:
            return {
                "setting.mdp": mdp.MDParameters(mdp.EM_MDP)
                .add_or_update("nsteps", self.nsteps)
                .add_or_update("emtol", self.emtol)
                .export(),
                "grommp.sh": defaut_file_content("grommp.sh").format(
                    options=options_txt
                ),
                "mdrun.sh": defaut_file_content("mdrun.sh"),
                "ovito.sh": defaut_file_content("em_ovito.sh"),
            }
        else:
            return {
                "setting.mdp": mdp.MDParameters(mdp.EM_MDP)
                .add_or_update("nsteps", self.nsteps)
                .add_or_update("emtol", self.emtol)
                .add_or_update(
                    "define", " ".join(["-D" + define for define in self.defines])
                )
                .export(),
                "grommp.sh": defaut_file_content("grommp.sh").format(
                    options=options_txt
                ),
                "mdrun.sh": defaut_file_content("mdrun.sh"),
                "ovito.sh": defaut_file_content("em_ovito.sh"),
            }


class MDType(enum.Enum):
    """
    Defines different types of Molecular Dynamics (MD) simulations.
    """

    v_rescale_c_rescale = 1
    v_rescale_only_nvt = 2
    nose_hoover_parinello_rahman = 3
    berendsen = 4


@dataclass(kw_only=True)
class MD(Calculation):
    """
    A class to represent a Molecular Dynamics (MD) calculation.
    Attributes:
    ----------
    type : MDType
        The type of MD calculation.
    calculation_name : str
        The name of the calculation.
    nsteps : int, optional
        Number of steps for the MD simulation (default is 10000).
    nstout : int, optional
        Frequency to write the coordinates to the trajectory file (default is 1000).
    gen_vel : str, optional
        Whether to generate velocities ("yes" or "no") (default is "yes").
    temperature : float, optional
        Temperature for the simulation (default is 300).
    defines : list[str], optional
        List of defines for the MD parameters (default is an empty list).
    maxwarn : int, optional
        Maximum number of warnings (default is 0).
    useRestraint : bool, optional
        Whether to use restraints (default is False).
    useSemiisotropic : bool, optional
        Whether to use semiisotropic pressure coupling (default is False).
    additional_mdp_parameters : dict[str, str | float], optional
        Additional parameters for the MD simulation (default is an empty dictionary).
    Methods:
    --------
    name() -> str:
        Returns the name of the calculation.
    generate() -> dict[str, str]:
        Generates the necessary files for the MD calculation based on the type.
    """

    type: MDType
    calculation_name: str
    nsteps: int = 10000

    # frequency to write the coordinates to the trajectory file
    nstout: int = 1000

    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = dataclasses.field(default_factory=list)

    # maximum number of warnings
    # (if you dont understand this parameter, Set it to 0!)
    maxwarn: int = 0
    useRestraint: bool = False
    useSemiisotropic: bool = False
    additional_mdp_parameters: dict[str, str | float] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
        """
        Initializes the MD object and performs input validation.
        Prints the estimated time span of the MD simulation.
        """
        print(
            "time span of the MD ",
            self.calculation_name,
            " is",
            self.nsteps * 0.002,
            "ps or",
            self.nsteps * 0.002 / 1000,
            "ns.",
        )

        # input validation
        if self.gen_vel not in ["yes", "no"]:
            raise ValueError("Invalid gen_vel value")
        if self.temperature < 0:
            raise ValueError("Invalid temperature value")

        if self.maxwarn != 0:
            warnings.warn("maxwarn is not 0. Do you know what you are doing?")

    @property
    def name(self) -> str:
        return self.calculation_name

    @override
    def generate(self) -> dict[str, str]:
        """
        Generates the necessary files for the MD calculation based on the specified MDType.
        Returns:
            dict[str, str]: A dictionary where keys are file names and values are file contents.
        Raises:
            NotImplementedError: If the selected MDType is not yet implemented.
        """
        match self.type:
            case MDType.v_rescale_c_rescale:
                options_txt = " -maxwarn " + str(self.maxwarn)
                mdp_file = (
                    mdp.MDParameters(mdp.V_RESCALE_C_RESCALE_MDP)
                    .add_or_update("nsteps", self.nsteps)
                    .add_or_update("nstxout", self.nstout)
                    .add_or_update("nstvout", self.nstout)
                    .add_or_update("nstfout", self.nstout)
                    .add_or_update("nstenergy", self.nstout)
                    .add_or_update("gen_vel", self.gen_vel)
                    .add_or_update("ref_t", self.temperature)
                    .add_or_update("gen_temp", self.temperature)
                )
                for key, value in self.additional_mdp_parameters.items():
                    mdp_file.add_or_update(key, str(value))
                if self.useRestraint:
                    options_txt += " -r input.gro"
                    mdp_file.add_or_update("refcoord_scaling", "all")
                if len(self.defines) != 0:
                    mdp_file.add_or_update(
                        "define", " ".join(["-D" + define for define in self.defines])
                    )
                if self.useSemiisotropic:
                    mdp_file.add_or_update("pcoupltype", "semiisotropic")
                    mdp_file.add_or_update(
                        "ref_p", " ".join([str(mdp_file.get("ref_p")) for _ in range(2)])
                    )
                    mdp_file.add_or_update(
                        "compressibility",
                        " ".join([str(mdp_file.get("compressibility")) for _ in range(2)]),
                    )

                return {
                    "setting.mdp": mdp_file.export(),
                    "grommp.sh": defaut_file_content("grommp.sh").format(
                        options=options_txt
                    ),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh"),
                }

            case MDType.v_rescale_only_nvt:
                options_txt = " -maxwarn " + str(self.maxwarn)
                mdp_file = (
                    mdp.MDParameters(mdp.V_RESCALE_ONLY_NVT_MDP)
                    .add_or_update("nsteps", self.nsteps)
                    .add_or_update("nstxout", self.nstout)
                    .add_or_update("nstvout", self.nstout)
                    .add_or_update("nstfout", self.nstout)
                    .add_or_update("nstenergy", self.nstout)
                    .add_or_update("gen_vel", self.gen_vel)
                    .add_or_update("ref_t", self.temperature)
                    .add_or_update("gen_temp", self.temperature)
                )
                for key, value in self.additional_mdp_parameters.items():
                    mdp_file.add_or_update(key, str(value))
                if self.useRestraint:
                    options_txt += " -r input.gro"
                    mdp_file.add_or_update("refcoord_scaling", "all")
                if len(self.defines) != 0:
                    mdp_file.add_or_update(
                        "define", " ".join(["-D" + define for define in self.defines])
                    )
                if self.useSemiisotropic:
                    warnings.warn(
                        "Semiisotropic is not supported in NVT. Do you understand NVT ensemble?"
                    )

                return {
                    "setting.mdp": mdp_file.export(),
                    "grommp.sh": defaut_file_content("grommp.sh").format(
                        options=options_txt
                    ),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh"),
                }

            case MDType.nose_hoover_parinello_rahman:
                # raise NotImplementedError("parameters are not linked to the mdp file")
                options_txt = " -maxwarn " + str(self.maxwarn)
                mdp_file = (
                    mdp.MDParameters(mdp.NOSE_HOOVER_PARINELLO_RAHMAN_MDP)
                    .add_or_update("nsteps", self.nsteps)
                    .add_or_update("nstxout", self.nstout)
                    .add_or_update("nstvout", self.nstout)
                    .add_or_update("nstfout", self.nstout)
                    .add_or_update("nstenergy", self.nstout)
                    .add_or_update("gen_vel", self.gen_vel)
                    .add_or_update("ref_t", self.temperature)
                    .add_or_update("gen_temp", self.temperature)
                )
                return {
                    "setting.mdp": mdp_file.export(),
                    "grommp.sh": defaut_file_content("grommp.sh").format(
                        options=options_txt
                    ),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh"),
                }
            case MDType.berendsen:
                raise NotImplementedError("parameters are not linked to the mdp file")
                return {
                    "setting.mdp": defaut_file_content("berendsen.mdp"),
                    "grommp.sh": defaut_file_content("grommp.sh").format(
                        options=options_txt
                    ),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh"),
                }


class RuntimeSolvation(Calculation):
    """
    A class to represent a runtime solvation calculation.
    (runtime solvation is original keyword in the library
    and is for inserting the solvent into the cell)
    Attributes:
        solvent (str): The type of solvent used in the calculation.
        rate (float): The rate of solvation.
        ntry (int): The number of attempts for solvation.
        calculation_name (str): The name of the calculation.
    Methods:
        __init__(solvent: str = "MCH", name: str = "solvation", rate: float = 1.0, ntry: int = 300):
            Initializes the RuntimeSolvation object with the given parameters.
        generate() -> dict[str, str]:
            Generates the necessary files for the solvation calculation.
        name() -> str:
            Returns the name of the calculation.
        check(cell_size: np.ndarray) -> "RuntimeSolvation":
            Prints the number of molecules to be added to the cell and the number of atoms.
    """

    solvent: str
    rate: float
    ntry: int
    calculation_name: str

    def __init__(
        self,
        *,
        solvent: str = "MCH",
        calculation_name: str = "solvation",
        rate: float = 1.0,
        ntry: int = 300,
    ):
        """
        Initializes the RuntimeSolvation object with the given parameters.
        Args:
            solvent (str): The type of solvent (e.g., "MCH").
            name (str): The name of the calculation.
            rate (float): The rate of solvation.
            ntry (int): The number of attempts for solvation.
        Raises:
            ValueError: If an invalid solvent is specified.
        """
        match solvent:
            case "MCH":
                self.solvent = "MCH"
            case "H2O":
                raise ValueError("H2O should be used spc216.gro (SolvateSCP216 class)")
            case _:
                raise ValueError("Invalid solvent")

        self.calculation_name = calculation_name
        self.rate = rate
        self.ntry = ntry

    @override
    def generate(self) -> dict[str, str]:
        """
        Generates the necessary files for the runtime solvation calculation.
        Returns:
            dict[str, str]: A dictionary where keys are file names and values are file contents.
        """
        return {
            "mdrun.sh": defaut_file_content("runtime_solvation.sh")
            .replace("SOLVENT", self.solvent)
            .replace("RATE", str(self.rate))
            .replace("TRY", str(self.ntry)),
            f"{self.solvent}.itp": defaut_file_content(f"{self.solvent}.itp"),
            f"{self.solvent}.gro": defaut_file_content(f"{self.solvent}.gro"),
            "runtime_solvation.py": defaut_file_content("runtime_solvation.py"),
            "grommp.sh": 'echo "this is a dummy file for automation"',
        }

    @override
    @property
    def name(self) -> str:
        return self.calculation_name

    def check(self, cell_size: np.ndarray) -> "RuntimeSolvation":
        """
        Prints the number of molecules to be added to the cell and the number of atoms.
        Args:
            cell_size (np.ndarray): The dimensions of the simulation cell.
        Returns:
            RuntimeSolvation: The current instance of RuntimeSolvation.
        Raises:
            ValueError: If an invalid solvent is specified.
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


class Solvation(Calculation):
    """
    A class to represent a solvation calculation, allowing control over the number of solvent molecules.
    """

    def __init__(
        self,
        solvent: str = "MCH",
        calculation_name: str = "solvation",
        nmol: int = 100,
        ntry: int = 300,
    ):
        """
        Initializes the Solvation object.
        Args:
            solvent (str): The type of solvent (e.g., "MCH").
            name (str): The name of the calculation.
            nmol (int): The number of solvent molecules to add.
            ntry (int): The number of attempts for solvation.
        Raises:
            ValueError: If an invalid solvent is specified.
        """
        match solvent:
            case "MCH":
                self.solvent = "MCH"
            case "H2O":
                raise ValueError("H2O should be used spc216.gro (SolvateSCP216 class)")
            case _:
                raise ValueError("Invalid solvent")

        self.calculation_name = calculation_name
        self.nmol = nmol
        self.ntry = ntry

    @classmethod
    def from_cell_size(
        cls,
        cell_size: np.ndarray,
        name: str = "solvation",
        solvent: str = "MCH",
        rate: float = 1.0,
    ):
        """
        Creates a Solvation instance by calculating the number of solvent molecules
        needed to fill a given cell size at a specified rate.
        Args:
            cell_size (np.ndarray): The dimensions of the simulation cell.
            name (str): The name of the calculation.
            solvent (str): The type of solvent.
            rate (float): The rate of solvent filling (1.0 means filled to density).
        Returns:
            Solvation: A new Solvation object.
        Raises:
            ValueError: If an invalid solvent is specified.
        """
        volume = np.prod(cell_size)  # nm^3
        print("The volume of the cell is", volume, "nm^3")
        match solvent:
            case "MCH":
                density = 0.77  # g/cm^3
                mass = 98.186  # g/mol
                mass_den = mass / density  # cm^3/mol = nm^3/mol * 10e21
                print("MCH is", mass_den, "cm^3/mol")
                print("is", mass_den * 10e21, "nm^3/mol")

            case _:
                raise ValueError("Invalid solvent")

        Na = 6.022 * 100  # *10e21 # Avogadro's number
        nmol = int(volume / mass_den * rate * Na)  # number of molecules

        print("I will fill the cell with", nmol, "molecules")

        return cls(solvent, name=name, nmol=nmol, ntry=300)

    @property
    def name(self) -> str:
        return self.calculation_name

    @override
    def generate(self) -> dict[str, str]:
        """
        Generates the necessary files for the solvation calculation.
        Returns:
            dict[str, str]: A dictionary where keys are file names and values are file contents.
        """
        return {
            "mdrun.sh": defaut_file_content("solvation.sh")
            .replace("SOLVENT", self.solvent)
            .replace("NMOL", str(self.nmol))
            .replace("TRY", str(self.ntry)),
            f"{self.solvent}.itp": defaut_file_content(f"{self.solvent}.itp"),
            f"{self.solvent}.gro": defaut_file_content(f"{self.solvent}.gro"),
            "solvation.py": defaut_file_content("solvation.py"),
            "grommp.sh": 'echo "this is a dummy file for automation"',
        }


class SolvationSCP216(Calculation):
    """
    A class to represent a solvation calculation using the SPC216 water model.
    Attributes:
        name (str): The name of the calculation.
    """

    def __init__(self, calculation_name: str = "solvation"):
        """
        Initializes the SolvationSCP216 object.
        Args:
            calculation_name (str): The name of the calculation.
        """
        self.calculation_name = calculation_name

    @override
    def generate(self) -> dict[str, str]:
        """
        Generates the necessary files for the SPC216 solvation calculation.
        Returns:
            dict[str, str]: A dictionary where keys are file names and values are file contents.
        """
        return {
            "dummy.top": "",
            "grommp.sh": "echo 'this is a dummy file for automation'",
            "top_mod.py": defaut_file_content("top_mod.py"),
            "mdrun.sh": _gmx_alias
            + "\n\n\n"
            + "inner_gmx solvate -cp input.gro -cs spc216.gro -o output.gro -p dummy.top \n\n\n"
            + "python top_mod.py",
        }

    @property
    @override
    def name(self) -> str:
        return self.calculation_name


_gmx_alias = """
# for supporting all gmx (gmx_d, gmx_mpi, gmx) commands
# Enable alias expansion
shopt -s expand_aliases
if command -v gmx_d &> /dev/null
then
    alias inner_gmx=gmx_d
elif command -v gmx_mpi &> /dev/null
then
    alias inner_gmx=gmx_mpi
elif command -v gmx &> /dev/null
then
    alias inner_gmx=gmx
else
    echo "No gromacs installation found."
    exit 1
fi
# end of alias support


"""


class FileControl(Calculation):
    """
    A class for performing file manipulation operations within GROMACS workflows.
    It allows defining custom shell commands to be executed.
    """

    def __init__(self, calculation_name: str, command: str):
        """
        Initializes the FileControl object.
        Args:
            calculation_name (str): The name of the file control operation.
            command (str): The shell command to execute.
        """
        self.calculation_name = calculation_name
        self.command = command

    @override
    def generate(self) -> dict[str, str]:
        """
        Generates the necessary files for the file control operation.
        Returns:
            dict[str, str]: A dictionary where keys are file names and values are file contents.
        """
        command = _gmx_alias + self.command
        return {
            "grommp.sh": 'echo "this is a dummy file for automation"',
            "mdrun.sh": command,
        }

    @property
    def name(self) -> str:
        return self.calculation_name

    @classmethod
    def remove_MCH(cls, name: str):
        """
        Creates a FileControl instance to remove MCH molecules from a GROMACS system.
        Args:
            name (str): The name of the removal operation.
        Returns:
            FileControl: A FileControl object configured for MCH removal.
        """
        commands = []
        commands.append(
            "{ echo -e '!rMCH'; echo -e 'q'; } | inner_gmx make_ndx -f input.gro -o withoutMCH.ndx"
        )
        commands.append(
            "echo '!MCH' | inner_gmx trjconv -f input.gro -s input.gro -o output.gro -n withoutMCH.ndx"
        )

        # remove MCH from the topology file
        commands.append("sed -i '/MCH/d' topol.top")
        # remove lines
        return cls(name, "\n".join(commands))

    @classmethod
    def cell_resizeing(cls, name: str, x: float, y: float, z: float):
        """
        Creates a FileControl instance to resize the simulation cell.
        Args:
            name (str): The name of the cell resizing operation.
            x (float): New X dimension of the cell.
            y (float): New Y dimension of the cell.
            z (float): New Z dimension of the cell.
        Returns:
            FileControl: A FileControl object configured for cell resizing.
        """
        commands = []
        commands.append(
            "{ echo -e '!rMCH'; echo -e 'q'; } | inner_gmx make_ndx -f input.gro -o withoutMCH.ndn"
        )
        commands.append(
            "echo '!MCH' | inner_gmx trjconv -f input.gro -s input.gro -o output.gro -n withoutMCH.ndx"
        )

        # remove MCH from the topology file
        commands.append("sed -i '/MCH/d' topo.top")

        # resize the cell
        commands.append(
            "echo 1 | inner_gmx editconf -f output.gro -o output.gro -box {x} {y} {z}".format(
                x=x, y=y, z=z
            )
        )
        # remove lines
        return cls(name, "\n".join(commands))


@dataclass(kw_only=True)
class BarMethod(Calculation):
    """
    A class to represent a Free Energy Perturbation (FEP) calculation using the BAR method.
    Attributes:
        type (MDType): The type of MD calculation.
        calculation_name (str): The name of the calculation.
        nsteps (int): Number of steps for the MD simulation.
        nstout (int): Frequency to write coordinates to the trajectory file.
        gen_vel (str): Whether to generate velocities.
        temperature (float): Temperature for the simulation.
        defines (list[str]): List of defines for the MD parameters.
        maxwarn (int): Maximum number of warnings.
        useRestraint (bool): Whether to use restraints.
        useSemiisotropic (bool): Whether to use semiisotropic pressure coupling.
        additional_mdp_parameters (dict[str, str | float]): Additional parameters for the MD simulation.
        vdw_lambdas (list[float]): List of lambda values for van der Waals interactions.
        coul_lambdas (list[float]): List of lambda values for Coulombic interactions.
        bonded_lambdas (list[float]): List of lambda values for bonded interactions.
        restraint_lambdas (list[float]): List of lambda values for restraints.
        mass_lambdas (list[float]): List of lambda values for mass.
        temperature_lambdas (list[float]): List of lambda values for temperature.
        couple_moltype (str): Molecule type to couple.
        couple_lamda0 (str): Lambda state 0 coupling type.
        couple_lamda1 (str): Lambda state 1 coupling type.
        nstdhdl (int): Frequency to write dH/dL to the energy file.
    """

    type: MDType
    calculation_name: str
    nsteps: int = 10000

    # frequency to write the coordinates to the trajectory file
    nstout: int = 1000

    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = Field(default_factory=list)

    # maximum number of warnings
    # (if you dont understand this parameter, Set it to 0!)
    maxwarn: int = 0
    useRestraint: bool = False
    useSemiisotropic: bool = False
    additional_mdp_parameters: dict[str, str | float] = Field(default_factory=dict)

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

    def __post_init__(self):
        """
        Initializes the BarMethod object and performs input validation.
        Prints the estimated time span of the MD simulation.
        Raises ValueError if input parameters are invalid.
        """
        print(
            "time span of the MD ",
            self.calculation_name,
            " is",
            self.nsteps * 0.002,
            "ps or",
            self.nsteps * 0.002 / 1000,
            "ns.",
        )

        # input validation
        if self.gen_vel not in ["yes", "no"]:
            raise ValueError("Invalid gen_vel value")
        if self.temperature < 0:
            raise ValueError("Invalid temperature value")

        if self.maxwarn != 0:
            warnings.warn("maxwarn is not 0. Do you know what you are doing?")

        nlambdas = len(self.vdw_lambdas)
        if nlambdas == 0:
            raise ValueError("vdw_lambdas is empty")

        if len(self.coul_lambdas) != nlambdas:
            raise ValueError("coul_lambdas is not same length as vdw_lambdas")
        if len(self.bonded_lambdas) != nlambdas:
            raise ValueError("bonded_lambdas is not same length as vdw_lambdas")
        if len(self.restraint_lambdas) != nlambdas:
            raise ValueError("restraint_lambdas is not same length as vdw_lambdas")
        if len(self.mass_lambdas) != nlambdas:
            raise ValueError("mass_lambdas is not same length as vdw_lambdas")
        if len(self.temperature_lambdas) != nlambdas:
            raise ValueError("temperature_lambdas is not same length as vdw_lambdas")


def copy_file_script(extension: str, destination: str) -> str:
    """
    Generates a bash command to copy files with a specific extension to a destination directory.
    Args:
        extension (str): The file extension (e.g., "top", "itp").
        destination (str): The destination directory.
    Returns:
        str: The bash copy command.
    """
    return f"cp *.{extension} ../{destination}"


def copy_inherited_files_script(destination: str) -> str:
    """
    Generates a bash script to copy inherited GROMACS files (topology, itp, gro) to a new calculation directory.
    Args:
        destination (str): The destination directory for the copied files.
    Returns:
        str: The bash script content.
    """
    scripts = [
        copy_file_script("top", destination),
        copy_file_script("itp", destination),
        f"cp output.gro ../{destination}/input.gro",
    ]
    return "\n".join(scripts)


class OverwriteType(enum.Enum):
    """
    Defines strategies for handling existing working directories during calculation setup.
    - `no`: Do not overwrite; raise an error if the directory exists.
    - `full_overwrite`: Remove the existing directory and recreate it.
    - `add_calculation`: Add the calculation if the folder does not exist; skip if it does.
    """

    no = 0
    full_overwrite = 1
    add_calculation = 2


def launch(
    calculations: list[Calculation],
    input_gro: str,
    working_dir: str,
    overwrite: OverwriteType = OverwriteType.no,
):
    """
    Launches a series of GROMACS calculations.
    Creates working directories, generates necessary files, and sets up run scripts.
    Args:
        calculations (list[Calculation]): A list of Calculation objects to run.
        input_gro (str): The path to the initial input .gro file.
        working_dir (str): The base directory where calculation folders will be created.
        overwrite (OverwriteType): The strategy for handling existing working directories.
    Raises:
        ValueError: If duplicate calculation names are found or if a directory exists and overwrite is set to `no`.
    """
    names = [calculation.name for calculation in calculations]
    # check if there are any duplicate names
    if len(names) != len(set(names)):
        raise ValueError("Duplicate names")

    for i in range(len(calculations)):
        calculation = calculations[i]
        dirname = os.path.join(working_dir, str(i) + "_" + calculation.name)
        # create folder in the working directory
        if os.path.exists(os.path.join(dirname)):
            match overwrite:
                case OverwriteType.no:
                    raise ValueError("Working directory already exists", dirname)
                case OverwriteType.full_overwrite:
                    # remove the folder and its content
                    for file in os.listdir(dirname):
                        os.remove(os.path.join(dirname, file))
                    os.rmdir(dirname)
                case OverwriteType.add_calculation:
                    continue

        os.mkdir(dirname)

        # generate files
        files = calculation.generate()
        for name, content in files.items():
            with open(os.path.join(dirname, name), "w", newline="\n") as f:
                f.write(content)

        script_content = """#!/bin/bash

if [ -f "output.gro" ]; then
    echo "output.gro already exists. This calculation is finished."
    exit 0
fi

if ls ./*.pdb >/dev/null 2>&1; then
    echo "this calculation has problems and this is already calculated."
    exit 1
fi
"""

        # create a script to run the calculation
        with open(os.path.join(dirname, "run.sh"), "w", newline="\n") as f:
            f.write(script_content)
            f.write("bash grommp.sh\n")
            f.write("bash mdrun.sh\n")
            if i != len(calculations) - 1:
                f.write(". copy.sh\n")

        if i == len(calculations) - 1:
            """last calculation"""
            break

        # addtionally, if there is a next calculation,
        # create a script to copy inherited files
        with open(os.path.join(dirname, "copy.sh"), "w", newline="\n") as f:
            f.write(
                copy_inherited_files_script(
                    str(i + 1) + "_" + (calculations[i + 1].name)
                )
            )
            f.write(f"\necho {calculation.name} is done")
            f.write(f"\necho Next calculation is {calculations[i + 1].name}")

    # copy input file to the first calculation
    shutil.copy2(
        input_gro, os.path.join(working_dir, "0_" + calculations[0].name, "input.gro")
    )

    # create a script to all the calculations

    with open(os.path.join(working_dir, "run.sh"), "w", newline="\n") as f:
        f.write("#!/bin/bash\n")
        f.write("set -e\n")
        for i, calc in enumerate(calculations):
            f.write(f"cd {str(i) + '_' + calc.name}\n")
            f.write("bash run.sh\n")
            f.write("cd ..\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
        f.write("echo All calculations are done")


def generate_stepbystep_runfile(
    init_structures: list[str],
    calculation_name_and_isparaleljob: list[tuple[str, bool]],
    calc_path: str,
):
    """
    Generates step-by-step run scripts for a series of calculations,
    optionally supporting parallel execution.
    Args:
        init_structures (list[str]): List of initial structure file paths.
        calculation_name_and_isparaleljob (tuple[list[str], bool]): A tuple containing
            a list of calculation names and a boolean indicating if the job is parallel.
        calc_path (str): The base path for the calculation directories.
    """
    for calculation_name, do_paralel in calculation_name_and_isparaleljob:
        with open(
            os.path.join(calc_path, f"{calculation_name}.sh"), "w", newline="\n"
        ) as f:
            if do_paralel:
                f.write("echo 'Starting paralel job'\n")
            for file in init_structures:
                f.write(f"cd {file}/{calculation_name}\n")
                if do_paralel:
                    f.write("bash run.sh &\n")
                else:
                    f.write("bash run.sh\n")
                f.write("cd ../..\n")
                f.write("\n")
            if do_paralel:
                f.write("wait\n")
            f.write("\n")

    with open(os.path.join(calc_path, "stebystep.sh"), "w", newline="\n") as f:
        for calculation_name, _ in calculation_name_and_isparaleljob:
            f.write(f"bash {calculation_name}.sh\n")
            f.write("\n")

        f.write("echo All calculations are done")


def generate_batch_execution_script(
    init_name: list[str],
    base_dir: str,
    calc_path: str | None = None,
    command: str = "bash run.sh",
) -> None:
    """
    Generates a batch execution script to run commands across multiple initial structure directories.
    Args:
        init_name (list[str]): A list of initial structure names (subdirectories within `base_dir`).
        base_dir (str): The base directory containing the initial structure directories.
        calc_path (str | None): The name of the calculation subdirectory within each initial structure directory.
                                If None, the command runs in the initial structure directory.
        command (str): The command to run for each calculation.
    """
    with open(os.path.join(base_dir, "run.sh"), "w", newline="\n") as f:
        for i, file in enumerate(init_name):
            f.write(f"cd {os.path.basename(file).split('.')[0]}\n")

            if calc_path is not None:
                f.write(f"cd {calc_path}\n")

            f.write(command + "\n")

            if calc_path is not None:
                f.write("cd ..\n")

            f.write("cd ..\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")

        f.write("echo All calculations are done")


class CalculationJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            d = dataclasses.asdict(o)
            d["__class__"] = o.__class__.__name__
            return d
        if isinstance(o, enum.Enum):
            return o.value
        if isinstance(o, Calculation):
            d = o.__dict__.copy()
            d["__class__"] = o.__class__.__name__
            return d

        return super().default(o)


def save_json(calculations: list[Calculation], filepath: str):
    with open(filepath, "w") as f:
        json.dump(calculations, f, cls=CalculationJSONEncoder, indent=4)


CALCULATION_REGISTRY = {cls.__name__: cls for cls in Calculation.__subclasses__()}


def load_json(filepath: str) -> list[Calculation]:
    with open(filepath, "r") as f:
        data = json.load(f)

    calculations = []
    for item in data:
        class_name = item.pop("__class__")
        cls = CALCULATION_REGISTRY.get(class_name)

        if cls is None:
            raise ValueError(f"Unknown class: {class_name}")

        # Get the constructor parameters and their types
        sig = inspect.signature(cls)
        params = sig.parameters

        # Iterate over the item's keys and convert enum values
        for key, value in item.items():
            if key in params:
                param = params[key]
                param_type = param.annotation
                if inspect.isclass(param_type) and issubclass(param_type, enum.Enum):
                    item[key] = param_type(value)

        calculations.append(cls(**item))

    return calculations
