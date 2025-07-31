"""
This module provides functions for modifying and loading GROMACS topology files (.top).
It focuses on manipulating the [ molecules ] section of the topology file.
"""
def modify_top(path: str, molecules: dict[str, int]):
    """
    Modifies the [ molecules ] section of a GROMACS topology file.
    This function is currently not implemented.
    Args:
        path (str): The path to the topology file.
        molecules (dict[str, int]): A dictionary where keys are molecule names and values are their counts.
    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError
    lines = []
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    is_molecules_section = False
    for i, line in enumerate(lines):
        if line.rstrip() == "[molecules]":
            is_molecules_section = True
            continue

        if is_molecules_section:
            if line.startswith(";"):
                continue
            if line.rstrip().startswith("["):
                is_molecules_section = False
                break

            l = line.split()
            if len(l) == 2:
                mol, num = l
                if mol in molecules:
                    lines[i] = f"{mol} {molecules[mol]}\n"


def load_molecules_from_file(path: str) -> dict[str, int]:
    """
    Loads molecule counts from the [ molecules ] section of a GROMACS topology file.
    This function is currently not implemented.
    Args:
        path (str): The path to the topology file.
    Returns:
        dict[str, int]: A dictionary where keys are molecule names and values are their counts.
    Raises:
        NotImplementedError: This function is not yet implemented.
    """
    raise NotImplementedError
    with open(path, "r") as f:
        lines = f.readlines()

    molecules = {}
    is_molecules_section = False
    for line in lines:
        if line.rstrip() == "[molecules]":
            is_molecules_section = True
            continue

        if is_molecules_section:
            if line.startswith(";"):
                continue
            if line.rstrip().startswith("["):
                is_molecules_section = False

            l = line.split()
            if len(l) == 2:
                mol, num = l
                molecules[mol] = int(num)
    return molecules
