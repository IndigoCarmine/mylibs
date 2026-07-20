"""Generic Martini coarse-graining helpers.

Molecule-agnostic building blocks for bottom-up CG parametrization:

* ``map_aa_to_cg`` — map an all-atom structure (or trajectory frame) onto CG
  beads, given a mapping that assigns AA atom names to beads. Bead positions are
  the center of geometry (Martini 3 standard) or center of mass of their atoms.
* ``bonded_distributions`` / ``mean_lengths`` — mapped pairwise bead-distance and
  angle samples from an AA universe, used as bottom-up bonded targets.
* ``bead_mass`` / ``density_from_gro`` — small numeric utilities.

Nothing here is specific to a particular molecule: a ``mapping`` is always
``{bead_name: [aa_atom_name, ...]}`` where the atom-name lists include *every*
atom assigned to the bead (heavy atoms and their hydrogens). Molecule-specific
mappings, hydrogen bookkeeping and ``.itp`` writers live with the project that
uses them (e.g. ``cgmch.mch_model``), not in this library.
"""

from __future__ import annotations

import os
import sys
from itertools import combinations
from typing import Iterable

import numpy as np

# The library's ``mole`` package imports itself as ``src.mole.*`` (see mole/gro.py),
# so the mylibs project root (parent of ``src/``) must be importable. gromacs
# modules use relative imports and don't need this, but ``mole`` does — make it
# work regardless of how martini.py is imported.
_MYLIBS_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _MYLIBS_ROOT not in sys.path:
    sys.path.insert(0, _MYLIBS_ROOT)

import mole.gro as gro

# Atomic masses (g/mol) keyed by element symbol. Extend as needed.
ATOMIC_MASS: dict[str, float] = {
    "H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
    "F": 18.998, "P": 30.974, "S": 32.06, "Cl": 35.45,
}


def _element(atom_name: str) -> str:
    """Best-effort element symbol from an atom name (e.g. ``C7`` -> ``C``).

    Recognizes two-letter elements present in ``ATOMIC_MASS`` (e.g. ``Cl``);
    otherwise falls back to the leading alphabetic character.
    """
    letters = "".join(c for c in atom_name if c.isalpha())
    if len(letters) >= 2 and letters[:2].capitalize() in ATOMIC_MASS:
        return letters[:2].capitalize()
    return letters[:1].upper()


def _normalize_bead(spec) -> list[tuple[str, float]]:
    """Normalize a bead's atom spec to ``[(atom_name, weight), ...]``.

    Accepts a plain list of names (each weight 1.0), a list of ``(name, weight)``
    pairs, or a ``{name: weight}`` dict. Weights < 1 let an atom be **shared**
    between beads (e.g. ring carbons split half-and-half for a symmetric map);
    a shared atom contributes its weighted fraction to each bead's center and mass.
    """
    if isinstance(spec, dict):
        return [(k, float(v)) for k, v in spec.items()]
    out: list[tuple[str, float]] = []
    for item in spec:
        if isinstance(item, (tuple, list)):
            out.append((item[0], float(item[1])))
        else:
            out.append((item, 1.0))
    return out


def bead_mass(atom_spec, atomic_mass: dict[str, float] = ATOMIC_MASS) -> float:
    """Total mass (g/mol) of a bead's constituent atoms, from their names.

    Supports weighted specs (see ``_normalize_bead``): a shared atom with weight
    0.5 contributes half its element mass, so mass is conserved across beads.
    """
    return sum(atomic_mass.get(_element(n), 0.0) * w for n, w in _normalize_bead(atom_spec))


def _atom_mass(atom: gro.GroAtom) -> float:
    return ATOMIC_MASS.get(atom.symbol, 0.0)


def map_aa_to_cg(
    aa_gro: gro.GroFile,
    mapping: dict[str, list[str]],
    residue_name: str = "MOL",
    bead_residue: str | None = None,
    map_mode: str = "cog",
) -> gro.GroFile:
    """Map an all-atom ``GroFile`` (one or many molecules) to CG beads.

    Each molecule (identified by ``residue_number``) becomes ``len(mapping)`` beads
    placed at the center of its mapped atoms. Box vectors are preserved. Returns a
    new ``GroFile`` of beads.

    ``mapping`` assigns each bead an explicit list of AA atom names (heavy atoms
    *and* their hydrogens). ``bead_residue`` defaults to ``residue_name``.

    ``map_mode`` selects the bead position convention:

    * ``"cog"`` — center of **geometry** (Martini 3 standard; improves molecular
      volume/SASA and bulk density vs COM).
    * ``"com"`` — mass-weighted center of mass (legacy).
    """
    if bead_residue is None:
        bead_residue = residue_name

    # group atoms by residue number, keyed by atom_name for fast lookup
    residues: dict[int, dict[str, gro.GroAtom]] = {}
    for atom in aa_gro.atoms:
        if atom.residue_name != residue_name:
            continue
        residues.setdefault(atom.residue_number, {})[atom.atom_name] = atom

    beads: list[gro.GroAtom] = []
    bead_index = 1
    for res_i, (resnum, atoms_by_name) in enumerate(sorted(residues.items()), start=1):
        for bead_name, atom_names in mapping.items():
            coord = _center(atoms_by_name, atom_names, map_mode)
            beads.append(
                gro.GroAtom(
                    atom_number=bead_index,
                    atom_name=bead_name,
                    residue_name=bead_residue,
                    residue_number=res_i,
                    coordinate=coord,
                )
            )
            bead_index += 1

    return gro.GroFile(
        title=f"CG {residue_name} ({len(residues)} molecules, {len(mapping)} beads each)",
        atoms=beads,
        box_x=aa_gro.box_x,
        box_y=aa_gro.box_y,
        box_z=aa_gro.box_z,
    )


def scale_about_centroid(cg: gro.GroFile, factor: float) -> gro.GroFile:
    """Scale bead positions about each molecule's geometric center.

    The Martini way to realise a *distance-adjusted* coarse model: first build the
    raw mapped coarse model (``map_aa_to_cg``), then scale it so inter-bead
    distances match the tuned bonded lengths **while preserving the relative
    geometry and orientation**. Every bead of a molecule is moved along the ray
    from that molecule's geometric center (mean of its bead coordinates) by
    ``factor``, so all pairwise distances scale by exactly ``factor`` and the
    shape/orientation is unchanged. Operates per molecule (grouped by
    ``residue_number``); box vectors are preserved. Returns a new ``GroFile``.
    """
    from copy import deepcopy

    out = deepcopy(cg)
    groups: dict[int, list[gro.GroAtom]] = {}
    for atom in out.atoms:
        groups.setdefault(atom.residue_number, []).append(atom)
    for atoms in groups.values():
        centroid = np.mean([np.asarray(a.coordinate, dtype=float) for a in atoms], axis=0)
        for a in atoms:
            a.coordinate = centroid + factor * (np.asarray(a.coordinate, dtype=float) - centroid)
    return out


def _center(atoms_by_name: dict[str, gro.GroAtom], spec,
            mode: str = "cog") -> np.ndarray:
    """Weighted center of a bead's atoms: geometry (``cog``) or mass (``com``).

    ``spec`` is any form accepted by ``_normalize_bead``. For ``cog`` the bead
    position is the weight-average of atom coordinates; for ``com`` it is the
    (weight × mass)-average. Plain unweighted specs reproduce the old behaviour.
    """
    coords = []
    weights = []
    masses = []
    for name, w in _normalize_bead(spec):
        atom = atoms_by_name.get(name)
        if atom is None:
            continue
        coords.append(np.asarray(atom.coordinate, dtype=float))
        weights.append(w)
        masses.append(_atom_mass(atom))
    if not coords:
        raise ValueError(f"no atoms found for spec {spec!r}")
    coords_arr = np.array(coords)
    if mode == "com":
        wm = np.array(weights) * np.array(masses)
        return (coords_arr * wm[:, None]).sum(axis=0) / wm.sum()
    elif mode == "cog":
        wg = np.array(weights)
        return (coords_arr * wg[:, None]).sum(axis=0) / wg.sum()
    raise ValueError(f"unknown map_mode {mode!r} (use 'cog' or 'com')")


# --- Analysis (bottom-up bonded targets) -------------------------------------


def bonded_distributions(
    aa_gro_path: str,
    aa_xtc_path: str | None = None,
    mapping: dict[str, list[str]] | None = None,
    residue_name: str = "MOL",
    map_mode: str = "cog",
) -> dict[str, np.ndarray]:
    """Compute mapped bead-distance and angle samples from an AA trajectory.

    Requires MDAnalysis. For every molecule (residue ``residue_name``) in every
    frame, the AA atoms are mapped to bead centers and every pairwise bead
    distance is recorded, keyed ``"<beadA>-<beadB>"`` (nm) in mapping order. When
    the mapping has exactly three beads, the angle at the middle bead is also
    recorded under ``"angle"`` (degrees).

    ``mapping`` values are explicit AA atom-name lists (see ``map_aa_to_cg``). If
    ``aa_xtc_path`` is None only the single structure in ``aa_gro_path`` is used.
    """
    import MDAnalysis as mda

    if mapping is None:
        raise ValueError("mapping is required")

    u = mda.Universe(aa_gro_path, aa_xtc_path) if aa_xtc_path else mda.Universe(aa_gro_path)

    bead_names = list(mapping.keys())
    pairs = list(combinations(range(len(bead_names)), 2))
    pair_keys = [f"{bead_names[i]}-{bead_names[j]}" for i, j in pairs]

    # per residue: list of (AtomGroup, weight-array) per bead (supports shared atoms)
    bead_specs = [_normalize_bead(v) for v in mapping.values()]
    bead_atomgroups: list[list] = []
    residues = [r for r in u.residues if r.resname == residue_name]
    for res in residues:
        groups = []
        for spec in bead_specs:
            names = [n for n, _ in spec]
            wmap = {n: w for n, w in spec}
            ag = res.atoms[np.isin(res.atoms.names, names)]
            w = np.array([wmap[nm] for nm in ag.names], dtype=float)
            groups.append((ag, w))
        bead_atomgroups.append(groups)

    def _pos(ag, w):
        base = ag.masses * w if map_mode == "com" else w
        return (ag.positions * base[:, None]).sum(axis=0) / base.sum()

    samples: dict[str, list[float]] = {k: [] for k in pair_keys}
    ang: list[float] = []
    for _ in u.trajectory:
        for groups in bead_atomgroups:
            p = np.array([_pos(ag, w) for ag, w in groups]) / 10.0  # A -> nm
            for (i, j), key in zip(pairs, pair_keys):
                samples[key].append(float(np.linalg.norm(p[i] - p[j])))
            if len(bead_names) == 3:
                v0 = p[0] - p[1]
                v2 = p[2] - p[1]
                cos = np.dot(v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2))
                ang.append(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    out: dict[str, np.ndarray] = {k: np.array(v) for k, v in samples.items()}
    if ang:
        out["angle"] = np.array(ang)
    return out


def mean_lengths(dists: dict[str, np.ndarray], keys: Iterable[str] | None = None) -> dict[str, float]:
    """Mean bead distances (nm) from a ``bonded_distributions`` result.

    ``keys`` defaults to every distance key (all keys except ``"angle"``).
    """
    if keys is None:
        keys = [k for k in dists if k != "angle"]
    return {k: float(np.mean(dists[k])) for k in keys}


def density_from_gro(gro_path: str, total_mass_amu: float) -> float:
    """Bulk density (kg/m^3) from a .gro box, given the system's total mass.

    ``total_mass_amu`` = n_molecules * molar_mass. Density = mass / box_volume.
    """
    g = gro.GroFile.from_gro_file(gro_path)
    vol_nm3 = g.box_x * g.box_y * g.box_z
    # amu/nm^3 -> kg/m^3 : 1 amu = 1.66053906660e-27 kg, 1 nm^3 = 1e-27 m^3
    return total_mass_amu * 1.66053906660 / vol_nm3


# --- Hydrogen bookkeeping (build heavy+H mappings from an AA topology) --------


def _iter_itp_sections(itp_path: str):
    """Yield (section_name, [token_lists]) for each ``[ section ]`` in an .itp/.top.

    Comment (``;``) and preprocessor (``#``) lines are skipped; each data line is
    returned as its whitespace-split tokens.
    """
    section = None
    rows: list[list[str]] = []
    with open(itp_path, "r", encoding="utf-8") as handle:
        for raw in handle:
            line = raw.split(";", 1)[0].strip()
            if not line:
                continue
            if line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                if section is not None:
                    yield section, rows
                section = line.strip("[] ").strip()
                rows = []
                continue
            if section is not None:
                rows.append(line.split())
    if section is not None:
        yield section, rows


def hydrogens_from_itp(itp_path: str) -> dict[str, list[str]]:
    """Map each heavy atom name -> list of hydrogen names bonded to it.

    Parses the AA ``[atoms]`` (name<->index, element via :func:`_element`) and
    ``[bonds]`` (atom-index pairs) of a GROMACS topology and attaches every
    hydrogen to the heavy atom it is bonded to. Replaces hand-written
    ``_H_OF_CARBON``-style tables. Atoms whose element is not ``H`` get an entry
    (possibly empty); hydrogens are not keys.
    """
    name_by_index: dict[int, str] = {}
    elem_by_index: dict[int, str] = {}
    bonds: list[tuple[int, int]] = []
    for section, rows in _iter_itp_sections(itp_path):
        if section == "atoms":
            for tok in rows:
                if not tok or not tok[0].isdigit():
                    continue
                idx = int(tok[0])
                name = tok[4] if len(tok) > 4 else tok[1]
                name_by_index[idx] = name
                elem_by_index[idx] = _element(name)
        elif section == "bonds":
            for tok in rows:
                if len(tok) >= 2 and tok[0].isdigit() and tok[1].isdigit():
                    bonds.append((int(tok[0]), int(tok[1])))

    h_of_heavy: dict[str, list[str]] = {
        name_by_index[i]: [] for i, e in elem_by_index.items() if e != "H"
    }
    for a, b in bonds:
        ea, eb = elem_by_index.get(a), elem_by_index.get(b)
        if ea == "H" and eb and eb != "H":
            h_of_heavy.setdefault(name_by_index[b], []).append(name_by_index[a])
        elif eb == "H" and ea and ea != "H":
            h_of_heavy.setdefault(name_by_index[a], []).append(name_by_index[b])
    return h_of_heavy


def bead_bonds_from_itp(
    itp_path: str, mapping: dict[str, list[str]]
) -> list[tuple[int, int]]:
    """Derive CG bead-bead connectivity from an AA topology's ``[bonds]``.

    Any AA bond whose two atoms fall in *different* beads implies a CG bond
    between those beads. Returns a sorted list of unique 1-based bead-index pairs
    (indices into the ``mapping`` order). Bonds fully inside one bead are dropped.
    Useful to build the CG backbone topology automatically instead of by hand.
    """
    bead_of_atom: dict[str, int] = {}
    for idx, (bead, names) in enumerate(mapping.items(), start=1):
        for name in names:
            bead_of_atom[name] = idx

    name_by_index: dict[int, str] = {}
    aa_bonds: list[tuple[int, int]] = []
    for section, rows in _iter_itp_sections(itp_path):
        if section == "atoms":
            for tok in rows:
                if tok and tok[0].isdigit():
                    name_by_index[int(tok[0])] = tok[4] if len(tok) > 4 else tok[1]
        elif section == "bonds":
            for tok in rows:
                if len(tok) >= 2 and tok[0].isdigit() and tok[1].isdigit():
                    aa_bonds.append((int(tok[0]), int(tok[1])))

    pairs: set[tuple[int, int]] = set()
    for a, b in aa_bonds:
        ba = bead_of_atom.get(name_by_index.get(a, ""))
        bb = bead_of_atom.get(name_by_index.get(b, ""))
        if ba and bb and ba != bb:
            pairs.add((min(ba, bb), max(ba, bb)))
    return sorted(pairs)


def expand_heavy_mapping(
    heavy_mapping: dict[str, list[str]], h_of_heavy: dict[str, list[str]]
) -> dict[str, list[str]]:
    """Expand a heavy-atom-only bead mapping to include bonded hydrogens.

    ``heavy_mapping`` = ``{bead: [heavy_atom_name, ...]}``; ``h_of_heavy`` from
    :func:`hydrogens_from_itp`. Returns ``{bead: [heavy + its H, ...]}`` ready for
    :func:`map_aa_to_cg` / :func:`mapped_internal_distributions`.
    """
    out: dict[str, list[str]] = {}
    for bead, heavies in heavy_mapping.items():
        names: list[str] = []
        for heavy in heavies:
            names.append(heavy)
            names.extend(h_of_heavy.get(heavy, []))
        out[bead] = names
    return out


# --- Generic N-bead Martini .itp writer --------------------------------------


def write_cg_itp(
    mol_name: str,
    mapping: dict[str, list[str]],
    bead_types: dict[str, str],
    bonds: list[tuple] | None = None,
    constraints: list[tuple] | None = None,
    angles: list[tuple] | None = None,
    dihedrals: list[tuple] | None = None,
    charges: dict[str, float] | None = None,
    flexible_k: float = 5000.0,
    nrexcl: int = 1,
    header: str | None = None,
) -> str:
    """Render a generic N-bead Martini ``[moleculetype]`` as .itp text.

    ``mapping`` (bead order defines the 1-based bead index and masses via
    :func:`bead_mass`) and ``bead_types`` (bead -> Martini type) are required.
    Bonded terms use 1-based bead indices:

    * ``bonds``       — ``(i, j, length, k)`` (harmonic bonds, funct 1).
    * ``constraints`` — ``(i, j, length)`` rigid constraints. They are wrapped so
      that under ``-DFLEXIBLE`` they become stiff bonds (``flexible_k``) for
      energy minimization, and are true ``[constraints]`` otherwise (mirrors the
      MCH model).
    * ``angles``      — ``(i, j, k, theta_deg, k)`` (funct 2, G96 angle).
    * ``dihedrals``   — ``(i, j, k, l, funct, phi_deg, k[, mult])``.

    ``charges`` maps bead -> partial charge (default 0.0).
    """
    names = list(mapping.keys())
    charges = charges or {}
    masses = {n: bead_mass(mapping[n]) for n in names}

    lines: list[str] = []
    if header:
        lines.append(f"; {header}")
    else:
        lines.append(f"; Martini 3 coarse-grained model of {mol_name}")
    lines.append("; generated by gromacs.martini.write_cg_itp")
    lines.append("")
    lines.append("[ moleculetype ]")
    lines.append("; name   nrexcl")
    lines.append(f"  {mol_name}     {nrexcl}")
    lines.append("")
    lines.append("[ atoms ]")
    lines.append(";  nr  type  resnr  residue  atom  cgnr  charge   mass")
    for i, n in enumerate(names, start=1):
        lines.append(
            f"   {i:<4d} {bead_types[n]:<6s} 1     {mol_name:<5s} {n:<5s} {i:<4d} "
            f"{charges.get(n, 0.0):6.3f}  {masses[n]:7.3f}"
        )
    lines.append("")

    if bonds:
        lines.append("[ bonds ]")
        lines.append(";  i    j  funct   length   force_const")
        for b in bonds:
            i, j, length, k = b[0], b[1], b[2], b[3]
            lines.append(f"   {i:<4d} {j:<4d}  1     {length:.4f}   {k:.1f}")
        lines.append("")

    if constraints:
        lines.append("#ifdef FLEXIBLE")
        lines.append("; rigid constraints as stiff bonds for energy minimization (grompp -DFLEXIBLE)")
        lines.append("[ bonds ]")
        lines.append(";  i    j  funct   length   force_const")
        for c in constraints:
            i, j, length = c[0], c[1], c[2]
            lines.append(f"   {i:<4d} {j:<4d}  1     {length:.4f}   {flexible_k:.1f}")
        lines.append("#else")
        lines.append("[ constraints ]")
        lines.append(";  i    j  funct   length")
        for c in constraints:
            i, j, length = c[0], c[1], c[2]
            lines.append(f"   {i:<4d} {j:<4d}  1     {length:.4f}")
        lines.append("#endif")
        lines.append("")

    if angles:
        lines.append("[ angles ]")
        lines.append(";  i    j    k  funct   theta   force_const")
        for a in angles:
            i, j, k, theta, fc = a[0], a[1], a[2], a[3], a[4]
            lines.append(f"   {i:<4d} {j:<4d} {k:<4d}  2     {theta:.2f}   {fc:.2f}")
        lines.append("")

    if dihedrals:
        lines.append("[ dihedrals ]")
        lines.append(";  i    j    k    l  funct   phi   force_const  mult")
        for d in dihedrals:
            i, j, k, l, funct, phi, fc = d[0], d[1], d[2], d[3], d[4], d[5], d[6]
            mult = f"   {int(d[7])}" if len(d) > 7 else ""
            lines.append(
                f"   {i:<4d} {j:<4d} {k:<4d} {l:<4d}  {funct}   {phi:.2f}   {fc:.2f}{mult}"
            )
        lines.append("")

    return "\n".join(lines)


# --- Analysis for arbitrary bead index tuples --------------------------------


def mapped_internal_distributions(
    aa_gro_path: str,
    aa_xtc_path: str | None = None,
    mapping: dict[str, list[str]] | None = None,
    bonds: list[tuple[int, int]] | None = None,
    angles: list[tuple[int, int, int]] | None = None,
    dihedrals: list[tuple[int, int, int, int]] | None = None,
    residue_name: str = "MOL",
    map_mode: str = "cog",
) -> dict[str, np.ndarray]:
    """Sample explicit intramolecular bead bonds/angles/dihedrals from an AA run.

    Generalizes :func:`bonded_distributions` to arbitrary molecules: instead of
    all-pairs, the caller passes 1-based bead-index tuples (indices into the
    ``mapping`` order, per molecule). For every molecule (residue ``residue_name``)
    in every frame the AA atoms are mapped to bead centers and:

    * each ``bonds`` pair -> distance (nm), keyed ``"b<i>-<j>"``;
    * each ``angles`` triple -> angle (deg), keyed ``"a<i>-<j>-<k>"``;
    * each ``dihedrals`` quad -> signed dihedral (deg, -180..180),
      keyed ``"d<i>-<j>-<k>-<l>"``.

    Returns ``{key: np.ndarray of samples}`` pooled over all molecules and frames.
    """
    import MDAnalysis as mda

    if mapping is None:
        raise ValueError("mapping is required")
    bonds = bonds or []
    angles = angles or []
    dihedrals = dihedrals or []

    u = mda.Universe(aa_gro_path, aa_xtc_path) if aa_xtc_path else mda.Universe(aa_gro_path)
    bead_names = list(mapping.keys())

    residues = [r for r in u.residues if r.resname == residue_name]
    per_res_groups: list[list] = []
    for res in residues:
        groups = [res.atoms[np.isin(res.atoms.names, names)] for names in mapping.values()]
        per_res_groups.append(groups)

    _pos = (lambda g: g.center_of_geometry()) if map_mode == "cog" else (lambda g: g.center_of_mass())

    def _keyset(prefix, tuples):
        return {f"{prefix}{'-'.join(str(x) for x in t)}": [] for t in tuples}

    samples: dict[str, list[float]] = {}
    samples.update(_keyset("b", bonds))
    samples.update(_keyset("a", angles))
    samples.update(_keyset("d", dihedrals))

    for _ in u.trajectory:
        for groups in per_res_groups:
            p = np.array([_pos(g) for g in groups]) / 10.0  # A -> nm
            for (i, j) in bonds:
                samples[f"b{i}-{j}"].append(float(np.linalg.norm(p[i - 1] - p[j - 1])))
            for (i, j, k) in angles:
                samples[f"a{i}-{j}-{k}"].append(_angle(p[i - 1], p[j - 1], p[k - 1]))
            for (i, j, k, l) in dihedrals:
                samples[f"d{i}-{j}-{k}-{l}"].append(_dihedral(p[i - 1], p[j - 1], p[k - 1], p[l - 1]))

    return {key: np.array(vals) for key, vals in samples.items()}


def _angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle at ``b`` (degrees) for points a-b-c."""
    v0, v2 = a - b, c - b
    cos = np.dot(v0, v2) / (np.linalg.norm(v0) * np.linalg.norm(v2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def _dihedral(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> float:
    """Signed dihedral (degrees, -180..180) for points a-b-c-d."""
    b0, b1, b2 = a - b, c - b, d - c
    b1n = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1n) * b1n
    w = b2 - np.dot(b2, b1n) * b1n
    x = np.dot(v, w)
    y = np.dot(np.cross(b1n, v), w)
    return float(np.degrees(np.arctan2(y, x)))


def circular_mean(degrees: np.ndarray) -> float:
    """Circular mean of angles in degrees (-180..180], robust for dihedrals."""
    rad = np.deg2rad(np.asarray(degrees, dtype=float))
    ang = np.arctan2(np.sin(rad).mean(), np.cos(rad).mean())
    return float(np.degrees(ang))
