"""PLUMED (well-tempered) metadynamics calculation files, in the mylibs
Calculation style.

`MetaD` generates a standard NPT `setting.mdp`, a `mdrun.sh` that adds
``-plumed plumed.dat``, the usual `grommp.sh`, and writes the supplied
`plumed.dat`. `build_twist_plumed()` writes a PLUMED input that biases the
inter-rosette rotation (twist) between two adjacent rosettes via a TORSION of
CENTER virtual atoms, with a well-tempered METAD bias.

GROMACS must be PLUMED-patched (`gmx mdrun -plumed`); check with
`gmx --version | grep -i plumed`.
"""

from __future__ import annotations

import dataclasses
from pydantic.dataclasses import dataclass
from typing import override

from . import mdp
from .calculation import Calculation, default_file_content, _gmx_alias


def _mdrun_plumed_sh(plumed_file: str = "plumed.dat") -> str:
    return (
        "#!/bin/bash\n"
        + _gmx_alias
        + "if [ -f \"output.cpt\" ]; then\n"
        + f"    inner_gmx mdrun -deffnm output -plumed {plumed_file} -v "
        "-cpi output.cpt | tee run.out\n"
        + "else\n"
        + f"    inner_gmx mdrun -deffnm output -plumed {plumed_file} -v "
        "| tee run.out\n"
        + "fi\n"
    )


@dataclass(kw_only=True)
class MetaD(Calculation):
    """Well-tempered metadynamics (PLUMED) production run.

    plumed_content : the full text of plumed.dat (e.g. from build_twist_plumed).
    """

    calculation_name: str = "metad"
    plumed_content: str
    nsteps: int = 25_000_000          # 50 ns @ 2 fs (adjust when settings fixed)
    nstout: int = 25_000              # 50 ps/frame
    gen_vel: str = "no"
    temperature: float = 300.0
    maxwarn: int = 0
    useRestraint: bool = False
    index_file: str | None = None
    additional_mdp_parameters: dict[str, str | int | float] = dataclasses.field(
        default_factory=dict)

    @property
    def name(self) -> str:
        return self.calculation_name

    @override
    def generate(self) -> dict[str, str]:
        m = (
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
        for k, v in self.additional_mdp_parameters.items():
            m.add_or_update(k, str(v))
        options = f" -maxwarn {self.maxwarn}"
        if self.useRestraint:
            options += " -r input.gro"
            m.add_or_update("refcoord_scaling", "all")
        if self.index_file:
            options += f" -n {self.index_file}"
        return {
            "setting.mdp": m.export(),
            "grommp.sh": default_file_content("grommp.sh").format(options=options),
            "mdrun.sh": _mdrun_plumed_sh(),
            "plumed.dat": self.plumed_content,
            "generate_xtc.sh": default_file_content("generate_xtc.sh"),
        }


def _atomlist(indices_1based) -> str:
    """PLUMED atom selection string, compressed to a,b,c or ranges a-b."""
    idx = sorted(int(i) for i in indices_1based)
    parts, i = [], 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and idx[j + 1] == idx[j] + 1:
            j += 1
        parts.append(f"{idx[i]}-{idx[j]}" if j > i else f"{idx[i]}")
        i = j + 1
    return ",".join(parts)


def build_twist_plumed(
    core_atoms_lower: list[int],
    core_atoms_upper: list[int],
    spoke_atoms_lower: list[int],
    spoke_atoms_upper: list[int],
    *,
    height: float = 1.0,           # kJ/mol
    pace: int = 500,               # steps between hills
    sigma: float = 0.10,           # rad (~5.7 deg)
    biasfactor: float = 10.0,
    temp: float = 300.0,
    stride: int = 500,
) -> str:
    """PLUMED input biasing the inter-rosette twist (TORSION of CENTERs).

    lower/upper = the two adjacent rosettes; spoke_* = the core atoms of ONE
    reference molecule of each rosette (defines the in-plane reference vector).
    TORSION(spokeLo, cenLo, cenUp, spokeUp) about the cenLo-cenUp stacking axis
    is the twist; it is periodic, and the 6-fold symmetry gives period 60 deg.
    """
    return f"""# PLUMED: well-tempered metadynamics on the inter-rosette twist
# GROMACS is PLUMED-patched; run: gmx mdrun ... -plumed plumed.dat
UNITS LENGTH=nm TIME=ps ENERGY=kj/mol

cenLo:   CENTER ATOMS={_atomlist(core_atoms_lower)}
cenUp:   CENTER ATOMS={_atomlist(core_atoms_upper)}
spokeLo: CENTER ATOMS={_atomlist(spoke_atoms_lower)}
spokeUp: CENTER ATOMS={_atomlist(spoke_atoms_upper)}

# inter-rosette twist (rad, periodic); 6-fold symmetry -> physical period 60 deg
twist: TORSION ATOMS=spokeLo,cenLo,cenUp,spokeUp

METAD ...
  LABEL=mtd
  ARG=twist
  PACE={pace}
  HEIGHT={height}
  SIGMA={sigma}
  BIASFACTOR={biasfactor}
  TEMP={temp}
  GRID_MIN=-pi
  GRID_MAX=pi
  GRID_BIN=360
  CALC_RCT
  FILE=HILLS
... METAD

PRINT ARG=twist,mtd.bias,mtd.rbias STRIDE={stride} FILE=COLVAR
"""
