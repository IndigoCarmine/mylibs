import dataclasses
import mole
import mole.xyz


@dataclasses.dataclass
class AllocationProperty:
    memory: int = 0  # GB
    processors: int = 0

    def __str__(self):
        return f"%mem={self.memory}GB\n%NProcShared={self.processors}\n"

    @classmethod
    def from_string(cls, string: str):
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
    basis_set: str = "6-31G*"
    functional: str = "B3LYP"
    workflow: str = "opt"

    def __str__(self):
        return (
            "#P " + self.workflow + " " + self.functional + "\\" + self.basis_set + "\n"
        )

    @classmethod
    def from_string(cls, string: str):
        raise NotImplementedError("from_string method is not implemented")
        lines = string.splitlines()
        basis_set = ""
        functional = ""
        workflow = ""
        for line in lines:
            if line.startswith("#P"):
                parts = line.split()
                workflow = parts[1]
                functional = parts[2]
                basis_set = parts[3]

        if not basis_set or not functional or not workflow:
            raise ValueError("Invalid calculation type string")

        return cls(basis_set=basis_set, functional=functional, workflow=workflow)


@dataclasses.dataclass
class GJFFile:
    allocation: AllocationProperty = AllocationProperty()
    calculation_type: CalculationType = CalculationType()
    molecule: mole.xyz.XyzMolecule = mole.xyz.XyzMolecule()
    title: str = "GJF file"
    charge: int = 0
    multiplicity: int = 1
    fragment: int = 1

