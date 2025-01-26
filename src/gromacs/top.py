def modify_top(path: str, molecules: dict[str, int]):
    """
    This Function has not yet been coded.
    Modify the molecules section on topology file.
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
    This Function has not yet been coded.
    Load molecules from a file.
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
