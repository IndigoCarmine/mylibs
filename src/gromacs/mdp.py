"""
This module provides a class for parsing, manipulating, and exporting GROMACS MDP (Molecular Dynamics Parameters) files.
It also defines default MDP parameters for common simulation types like energy minimization (EM) and molecular dynamics (MD).
"""

from typing import Callable
import copy


class _Parameter:
    """
    Internal class to represent an MDP parameter with a key and an optional checker function.
    """

    def __init__(self, key: str, checker: Callable[[str], bool]):
        self.key = key
        self.checker = checker


class MDParameters:
    """
    A class to manage GROMACS MDP (Molecular Dynamics Parameters) settings.
    It allows loading, modifying, and exporting MDP parameters.
    """

    def __init__(self, data: dict[str, str|int|float] = {}, ignore_deepcopy=False):
        """
        Initializes the MDParameters object.
        Args:
            data (dict[str, str]): Initial MDP parameters as a dictionary.
            ignore_deepcopy (bool): If True, the input data dictionary is used directly without deep copying.
        """
        if ignore_deepcopy:
            self.data = data
        else:
            self.data = copy.deepcopy(data)

    def export(self) -> str:
        """
        Exports the MDP parameters to a string in GROMACS MDP file format.
        Returns:
            str: The formatted MDP parameters string.
        """
        lines = []
        max_length = max(len(key) for key in self.data.keys())
        key: str
        for key, value in self.data.items():
            lines.append(f"{key.ljust(max_length)} = {str(value)}")
        return "\n".join(lines)

    def __str__(self) -> str:
        """
        Returns a string representation of the MDP parameters, suitable for preview.
        """
        text = """; GROMACS MDP parameters file preview"""
        text += "\n"
        text += self.export()
        text += "\n"
        text += """; End of MDP parameters file preview"""

        return text

    def _print_python_dict(self):
        """
        Prints the internal dictionary representation of the MDP parameters.
        (For debugging purposes)
        """
        print(self.data)

    @classmethod
    def from_text(cls, text: str) -> "MDParameters":
        """
        Creates a new MDParameters instance by parsing a string containing MDP parameters.
        Args:
            text (str): A string containing MDP parameters.
        Returns:
            MDParameters: A new MDParameters object.
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
        Creates a new MDParameters instance by reading MDP parameters from a file.
        Args:
            path (str): The path to the MDP file.
        Returns:
            MDParameters: A new MDParameters object.
        """
        with open(path, "r") as f:
            return cls.from_text(f.read())

    def check(self, key: str) -> bool:
        """
        Checks if a specific key exists in the MDP parameters.
        Args:
            key (str):
            The key to check.
        Returns:
            bool: True if the key exists, False otherwise.
        """
        return key in self.data

    def add_or_update(
        self,
        key: str,
        value: str|int|float,
    ) -> "MDParameters":
        """
        Adds a new key-value pair or updates an existing one in the MDP parameters.
        Args:
            key (str): The parameter key.
            value (str|int|float): The parameter value.
        Returns:
            MDParameters: The current MDParameters object (for chaining).
        """
        self.data[key] = value

        return self

    def remove(self, key: str) -> "MDParameters":
        """
        Removes a key-value pair from the MDP parameters.
        Args:
            key (str): The key to remove.
        Returns:
            MDParameters: The current MDParameters object (for chaining).
        """
        self.data.pop(key, None)

        return self

    def get(self, key: str) -> str | int | float | None:
        """
        Gets the value associated with a specific key.
        Args:
            key (str): The key to retrieve the value for.
        Returns:
            str | int | float | None: The value if the key exists, otherwise None.
        """
        return self.data.get(key, None)


EM_MDP:dict[str, str|int|float] = {
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
V_RESCALE_C_RESCALE_MDP:dict[str, str|int|float] = {
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

V_RESCALE_ONLY_NVT_MDP:dict[str, str|int|float] = {
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
    "Pcoupl": "no",
    "gen_vel": "gen_vel",
    "gen_temp": "gen_temp",
    "pbc": "xyz",
}


NOSE_HOOVER_PARINELLO_RAHMAN_MDP:dict[str, str|int|float] = {
    "integrator": "md",
    "dt": "0.002",
    "nsteps": "10000",
    "nstxout": "5000",
    "nstvout": "1000",
    "nstfout": "1000",
    "nstenergy": "1000",
    "cutoff-scheme": "verlet",
    "continuation": "yes",
    "constraints": "h-bonds",
    "constraint_algorithm": "LINCS",
    "nstlist": "10",
    "ns_type": "grid",
    "tcoupl": "nose-hoover",
    "tc_grps": "system",
    "tau_t": "1",
    "ref_t": "300",
    "rlist": "1.4",
    "coulombtype": "PME",
    "rcoulomb": "1.4",
    "fourierspacing": "0.30",
    "pme_order": "4",
    "vdwtype": "Cut-off",
    "rvdw": "1.4",
    "Pcoupl": "Parrinello-Rahman",
    "tau_p": "5.0",
    "ref_p": "1",
    "compressibility": "2.0e-05",
    "gen_vel": "no",
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
