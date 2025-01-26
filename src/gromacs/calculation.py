from abc import ABC, abstractmethod
import warnings
from pydantic.dataclasses import dataclass
import dataclasses
import enum
import os
import shutil
from typing import override

import gromacs.mdp as mdp
import numpy as np


def defaut_file_content(name: str) -> str:
    DefaultFile_dir = os.path.join(os.path.dirname(__file__), "DefaultFiles")
    if not os.path.exists(DefaultFile_dir):
        raise FileNotFoundError("DefaultFiles directory not found")
    with open(os.path.join(DefaultFile_dir, name), "r") as f:
        return f.read()


class Calclation(ABC):
    def __init__(self):
        raise Exception("Abstract class cannot be instantiated")

    @abstractmethod
    def generate(self) -> dict[str, str]:
        Exception("This method must be implemented. Abstract method was called")
        """
        dict[str,str] : key is the name of the file, value is the content of the file 
        """
        return {}

    @property
    @abstractmethod
    def name(self) -> str:
        pass


@dataclass(kw_only=True)
class EM(Calclation):
    """
    energy minimization
    """

    nsteps: int = 3000
    emtol: float = 300
    calculation_name: str = "em"
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = (
        0  # maximum number of warnings (if you dont understand this parameter, Set it to 0!)
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
    v_rescale_c_rescale = 1
    nose_hoover_parinello_rahman = 2
    berendsen = 3


@dataclass(kw_only=True)
class MD(Calclation):
    """
    molecular dynamics
    """

    type: MDType
    calculation_name: str
    nsteps: int = 10000
    nstout: int = 1000  # frequency to write the coordinates to the trajectory file
    gen_vel: str = "yes"
    temperature: float = 300
    defines: list[str] = dataclasses.field(default_factory=list)
    maxwarn: int = (
        0  # maximum number of warnings (if you dont understand this parameter, Set it to 0!)
    )
    useRestraint: bool = False
    useSemiisotropic: bool = False
    additional_mdp_parameters: dict[str, str | float] = dataclasses.field(
        default_factory=dict
    )

    def __post_init__(self):
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
                        "ref_p", " ".join([mdp_file.get("ref_p") for _ in range(2)])
                    )
                    mdp_file.add_or_update(
                        "compressibility",
                        " ".join([mdp_file.get("compressibility") for _ in range(2)]),
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
                raise NotImplementedError("parrameters are not linked to the mdp file")
                return {
                    "setting.mdp": defaut_file_content(
                        "nose_hoover_parinello_rahman.mdp"
                    ),
                    "grommp.sh": defaut_file_content("grommp.sh").format(
                        options=options_txt
                    ),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh"),
                }
            case MDType.berendsen:
                raise NotImplementedError("parrameters are not linked to the mdp file")
                return {
                    "setting.mdp": defaut_file_content("berendsen.mdp"),
                    "grommp.sh": defaut_file_content("grommp.sh").format(
                        options=options_txt
                    ),
                    "mdrun.sh": defaut_file_content("mdrun.sh"),
                    "generate_xtc.sh": defaut_file_content("generate_xtc.sh"),
                    "ovito.sh": defaut_file_content("ovito.sh"),
                }


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
    ):
        match solvent:
            case "MCH":
                self.solvent = "MCH"
            case _:
                raise ValueError("Invalid solvent")

        self.calculation_name = name
        self.rate = rate
        self.ntry = ntry

    @override
    def generate(self) -> dict[str, str]:
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
    ):
        match solvent:
            case "MCH":
                self.solvent = "MCH"
            case _:
                raise ValueError("Invalid solvent")

        self.calculation_name = name
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
        try to fill the cell with the solvent as much as density of the solvent
        rate : the rate of the solvent filling in the cell (when 1.0, the cell is filled with the solvent as much as the density of the solvent)
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


def copy_file_script(extension: str, destination: str) -> str:
    return f"cp *.{extension} ../{destination}"


def copy_inherited_files_script(destination: str) -> str:
    scripts = [
        copy_file_script("top", destination),
        copy_file_script("itp", destination),
        f"cp output.gro ../{destination}/input.gro",
    ]
    return "\n".join(scripts)


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
):
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
                case OvereriteType.no:
                    raise ValueError("Working directory already exists", dirname)
                case OvereriteType.full_overwrite:
                    # remove the folder and its content
                    for file in os.listdir(dirname):
                        os.remove(os.path.join(dirname, file))
                    os.rmdir(dirname)
                case OvereriteType.add_calculation:
                    continue

        os.mkdir(dirname)

        # generate files
        files = calculation.generate()
        for name, content in files.items():
            with open(os.path.join(dirname, name), "w", newline="\n") as f:
                f.write(content)

        # create a script to run the calculation
        with open(os.path.join(dirname, "run.sh"), "w", newline="\n") as f:
            f.write(f"bash grommp.sh\n")
            f.write(f"bash mdrun.sh\n")
            if i != len(calculations) - 1:
                f.write(f". copy.sh\n")

        if i == len(calculations) - 1:
            """last calculation"""
            break

        # addtionally, if there is a next calculation, create a script to copy inherited files
        # create a script to copy inherited files
        with open(os.path.join(dirname, "copy.sh"), "w", newline="\n") as f:
            f.write(
                copy_inherited_files_script(
                    str(i + 1) + "_" + (calculations[i + 1].name)
                )
            )
            f.write(f"\necho {calculation.name} is done")
            f.write(f"\necho Next calculation is {calculations[i+1].name}")

    # copy input file to the first calculation
    shutil.copy2(
        input_gro, os.path.join(working_dir, "0_" + calculations[0].name, "input.gro")
    )

    # create a script to all the calculations

    with open(os.path.join(working_dir, "run.sh"), "w", newline="\n") as f:
        for i, calc in enumerate(calculations):
            f.write(f"cd {str(i) + '_' + calc.name}\n")
            f.write(f"bash run.sh\n")
            f.write("cd ..\n")
            f.write("\n")
            f.write("\n")
            f.write("\n")
        f.write("echo All calculations are done")


def generate_stepbystep_runfile(
    init_structures: list[str],
    calculation_name_and_isparaleljob: tuple[list[str], bool],
    calc_path: str,
):

    for calculation_name, do_paralel in calculation_name_and_isparaleljob:
        with open(
            os.path.join(calc_path, f"{calculation_name}.sh"), "w", newline="\n"
        ) as f:
            if do_paralel:
                f.write("echo 'Starting paralel job'\n")
            for file in init_structures:
                f.write(f"cd {file}/{calculation_name}\n")
                if do_paralel:
                    f.write(f"bash run.sh &\n")
                else:
                    f.write(f"bash run.sh\n")
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
