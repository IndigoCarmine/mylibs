"""PLUMED (well-tempered) metadynamics calculation files, in the mylibs
Calculation style, plus PLUMED post-processing helpers.

Calculation steps:
  * `MetaD` — a from-scratch NPT production step: standard `setting.mdp`, a
    `mdrun.sh` that adds ``-plumed plumed.dat``, the usual `grommp.sh`, and the
    supplied `plumed.dat`.
  * `MDWithPlumed` — the thin alternative: an ordinary `gromacs.calculation.MD`
    step (so it inherits every ensemble / define / semiisotropic option) with a
    `plumed.dat` (and any extra files) dropped alongside. Use this when the run
    is "a normal MD, but biased".

`build_twist_plumed()` writes a PLUMED input that biases the inter-rosette
rotation (twist) between two adjacent rosettes via a TORSION of CENTER virtual
atoms, with a well-tempered METAD bias.

Analysis helpers (`load_fes`, `fold_fes_to_period`) read a finished run's FES
into a free-energy curve F(CV), optionally folded into one period of a symmetric
CV by Boltzmann-averaging the equivalent branches.

GROMACS must be PLUMED-patched (`gmx mdrun -plumed`); check with
`gmx --version | grep -i plumed`.
"""

from __future__ import annotations

import dataclasses
import os

import numpy as np
from pydantic.dataclasses import dataclass
from typing import override

from . import mdp
from .calculation import MD, Calculation, default_file_content, _gmx_alias

#: Boltzmann constant in kJ/mol/K (GROMACS/PLUMED energy units).
KB_KJ_PER_MOL_K = 0.0083144626181532


def _mdrun_plumed_sh(plumed_file: str = "plumed.dat") -> str:
    """mdrun.sh for a PLUMED run: run ``gmx mdrun`` with ``-plumed``, resuming
    from ``output.cpt`` if one is present."""
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


@dataclass(kw_only=True)
class MDWithPlumed(MD):
    """An ordinary ``MD`` step run under PLUMED (metadynamics / steered MD / any
    bias), reusing every ``MD`` option (ensemble ``type``, ``defines``,
    ``useSemiisotropic``, restraints, extra mdp parameters, ...).

    It reuses ``MD.generate()`` and only swaps in a ``mdrun.sh`` that adds
    ``-plumed plumed.dat``, then drops the PLUMED input (and any extra files)
    beside the mdp.

    Supply the bias exactly one of two ways:
      * ``plumed_content`` — the full text of plumed.dat (e.g. from
        :func:`build_twist_plumed`), or
      * ``plumed_file`` — a path to read it from at generate() time.
    ``additional_files`` are copied in verbatim (basename preserved), for e.g. a
    reference structure an RMSD/RESTRAINT CV needs.
    """

    plumed_content: str | None = None
    plumed_file: str | None = None
    additional_files: list[str] = dataclasses.field(default_factory=list)

    @override
    def generate(self) -> dict[str, str]:
        if (self.plumed_content is None) == (self.plumed_file is None):
            raise ValueError(
                "Provide exactly one of plumed_content or plumed_file")
        files = super().generate()
        files["mdrun.sh"] = _mdrun_plumed_sh()
        if self.plumed_content is not None:
            files["plumed.dat"] = self.plumed_content
        else:
            assert self.plumed_file is not None
            with open(self.plumed_file, "r") as f:
                files["plumed.dat"] = f.read()
        for path in self.additional_files:
            with open(path, "r") as f:
                files[os.path.basename(path)] = f.read()
        return files


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


# ===========================================================================
# Post-processing: FES -> folded free-energy curve
# ===========================================================================

def load_fes(path: str | os.PathLike) -> tuple[np.ndarray, np.ndarray]:
    """Load a PLUMED 1-D FES file; return ``(cv, free_energy)`` as two arrays
    (columns 0 and 1, comment lines skipped)."""
    d = np.loadtxt(path, comments="#")
    return d[:, 0], d[:, 1]


def fold_fes_to_period(
    cv: np.ndarray,
    fes: np.ndarray,
    period: float,
    *,
    nbins: int = 60,
    temperature: float = 300.0,
    cv_in_radians: bool = True,
    fold_in_degrees: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Fold a free-energy curve of a symmetric CV into a single period.

    The ``k`` physically equivalent branches of a periodic CV are combined by a
    Boltzmann average, ``F_fold = -kT ln <exp(-F/kT)>_branches``, then shifted so
    the minimum is zero. Typical use: an inter-rosette twist with 6-fold symmetry
    → ``period=60`` degrees.

    cv : the CV values from :func:`load_fes` (radians if ``cv_in_radians``, else
        already in the folding unit). ``period`` and the returned bin centers are
        in degrees when ``fold_in_degrees`` (the default), otherwise in the CV's
        own unit. Returns ``(bin_centers, folded_free_energy)``; empty bins are
        NaN.
    """
    kt = KB_KJ_PER_MOL_K * temperature
    x = np.degrees(cv) if (cv_in_radians and fold_in_degrees) else np.asarray(cv)
    x = np.mod(x, period)
    edges = np.linspace(0.0, period, nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    weights = np.exp(-fes / kt)
    idx = np.clip(np.digitize(x, edges) - 1, 0, nbins - 1)
    wsum = np.zeros(nbins)
    np.add.at(wsum, idx, weights)
    wsum[wsum == 0] = np.nan
    f_fold = -kt * np.log(wsum)
    f_fold -= np.nanmin(f_fold)
    return centers, f_fold
