from typing import Callable
import copy

"""
The file contains mdp parser and mdp default parameters.
"""


class _Parameter:
    def __init__(self, key: str, checker: Callable[[str], bool]):
        self.key = key
        self.checker = checker


class MDParameters:
    def __init__(self, data: dict[str, str] = {}, ignore_deepcopy=False):

        if ignore_deepcopy:
            self.data = data
        else:
            self.data = copy.deepcopy(data)

    def export(self) -> str:
        """
        Export the MDP parameters to a string.
        """
        lines = []
        max_length = max(len(key) for key in self.data.keys())
        key: str
        for key, value in self.data.items():
            lines.append(f"{key.ljust(max_length)} = {str(value)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        text = """; GROMACS MDP parameters file preview"""
        text += "\n"
        text += self.export()
        text += "\n"
        text += """; End of MDP parameters file preview"""

        return text

    def _print_python_dict(self):
        print(self.data)

    @classmethod
    def from_text(cls, text: str) -> "MDParameters":
        """
        Create a new MDParameters instance from a string.
        """
        mdp = cls()
        for line in text.split("\n"):
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            key, value = line.split("=")
            key = key.strip()
            value = value.strip()
            mdp.data[key] = value
        return mdp

    @classmethod
    def from_file(cls, path: str) -> "MDParameters":
        """
        Create a new MDParameters instance from a file.
        """
        with open(path, "r") as f:
            return cls.from_text(f.read())

    def check(self, key: str) -> bool:
        """
        Check if a key exists in the MDP parameters.
        """
        return key in self.data

    def add_or_update(
        self,
        key: str,
        value: str,
    ) -> "MDParameters":
        """
        Add a new key-value pair or update an existing one.
        """
        self.data[key] = value

        return self

    def remove(self, key: str) -> "MDParameters":
        """
        Remove a key from the MDP parameters.
        """
        self.data.pop(key, None)

        return self

    def get(self, key: str) -> str | None:
        """
        Get the value of a key.
        """
        return self.data.get(key, None)


EM_MDP = {
    "integrator": "steep",
    "nsteps": "100000000",
    "emtol": "100",
    "emstep": "0.1",
    "ns_type": "grid",
    "rlist": "1",
    "rcoulomb": "1",
    "rvdw": "1",
    "pbc": "xyz",
}
V_RESCALE_C_RESCALE_MDP = {
    "cpp": "/usr/bin/cpp",
    "integrator": "md",
    "dt": "0.002",
    "nsteps": "nsteps",
    "nstxout": "nstxout",
    "nstvout": "nstvout",
    "nstfout": "nstfout",
    "nstenergy": "nstenergy",
    "cutoff-scheme": "verlet",
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "nstlist": "10",
    "ns_type": "grid",
    "tcoupl": "v-rescale",
    "tc_grps": "system",
    "tau_t": "0.2",
    "ref_t": "ref_t",
    "rlist": "1.4",
    "coulombtype": "PME",
    "rcoulomb": "1.4",
    "fourierspacing": "0.30",
    "pme_order": "4",
    "vdwtype": "Cut-off",
    "rvdw": "1.4",
    "Pcoupl": "c-rescale",
    "tau_p": "2.0",
    "ref_p": "1",
    "compressibility": "4.5e-05",
    "gen_vel": "gen_vel",
    "gen_temp": "gen_temp",
    "pbc": "xyz",
}


if __name__ == "__main__":
    # move to this file directory
    import os

    os.chdir(os.path.dirname(__file__))

    file = "DefaultFiles/v_rescale_c_rescale.mdp"

    mdp = MDParameters.from_file(file)
    mdp._print_python_dict()

    mdp.add_or_update("ptype", "semiisotropic")
    print(mdp.export())
