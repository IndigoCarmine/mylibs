"""
This module provides classes for generating Gaussian Job Files (GJF) for DFT calculations.
It includes classes to define allocation properties, calculation types, and the overall GJF structure.
"""
import dataclasses
from src.mole import xyz


@dataclasses.dataclass
class AllocationProperty:
    """
    Represents allocation properties for a Gaussian job, such as memory and number of processors.
    """
    memory: int = 0  # GB
    processors: int = 0

    def __str__(self):
        """
        Returns a string representation of the allocation properties in Gaussian input format.
        """
        return f"%mem={self.memory}GB\n%NProcShared={self.processors}\n"

    @classmethod
    def from_string(cls, string: str):
        """
        Parses allocation properties from a string and returns an AllocationProperty object.
        Raises ValueError if the string format is invalid.
        """
        lines = string.splitlines()
        memory = 0
        processors = 0
        for line in lines:
            if line.startswith("%mem="):
                memory = int(line.split("=")[1].replace("GB", "").strip())
            elif line.startswith("%NProcShared="):
                processors = int(line.split("=")[1].strip())

        if memory == 0 or processors == 0:
            raise ValueError("Invalid allocation property string")

        return cls(memory=memory, processors=processors)


@dataclasses.dataclass
class CalculationType:
    """
    Represents the type of calculation for a Gaussian job, including basis set, functional, and workflow.
    """
    basis_set: str = "6-31G*"
    functional: str = "B3LYP"
    workflow: str = "opt"

    def __str__(self):
        """
        Returns a string representation of the calculation type in Gaussian input format.
        """
        return (
            "#P " + self.workflow + " " + self.functional + "\\" + self.basis_set + "\n"
        )

    @classmethod
    def from_string(cls, string: str):
        """
        Parses calculation type from a string and returns a CalculationType object.
        """
        lines = string.strip().splitlines()
        basis_set = ""
        functional = ""
        workflow = ""
        for line in lines:
            if line.startswith("#P"):
                parts = line.split()
                workflow = parts[1]
                functional_basis_set = parts[2].split('\\')
                functional = functional_basis_set[0]
                if len(functional_basis_set) > 1:
                    basis_set = functional_basis_set[1]
                else:
                    basis_set = ""

        if not basis_set or not functional or not workflow:
            raise ValueError("Invalid calculation type string")

        return cls(basis_set=basis_set, functional=functional, workflow=workflow)


@dataclasses.dataclass
class GJFFile:
    """
    Represents a Gaussian Job File (GJF) with properties for allocation, calculation type,
    molecular structure, title, charge, multiplicity, and fragment.
    """
    allocation: AllocationProperty = dataclasses.field(default_factory=AllocationProperty)
    calculation_type: CalculationType = dataclasses.field(default_factory=CalculationType)
    molecule: xyz.XyzMolecule = dataclasses.field(default_factory=lambda: xyz.XyzMolecule(name="unnamed", index=0, children=[]))
    title: str = "GJF file"
    charge: int = 0
    multiplicity: int = 1
    fragment: int = 1

