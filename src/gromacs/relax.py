"""Relax a structure with a soft-core LJ potential.

On a structures built by geometric placement, neighbouring
monomers can start with atom pairs close enough that plain 12-6 LJ gives a
practically infinite force and minimization fails. Here the LJ part of the
force field is replaced by a soft-core form

    4*eps*((sig^6/(r^6+delta^6))^2 - sig^6/(r^6+delta^6))

whose energy stays finite at r = 0 while delta > 0. Minimization is run once
per stage in STAGES, ending at delta = 0 where the expression is exactly the
original LJ, so the final structure is relaxed on the real potential.

The inter-fiber hydrogen bonds that grompp would pull in via ``-DINTER`` are
honoured too, but not through GromacsTopFile: OpenMM has no handler for
GROMACS' [ intermolecular_interactions ], and would silently attribute those
bonds to the last [ moleculetype ] instead. Instead, the atom pairs are read
straight from the #included .itp and pinned to their starting positions with
a stiff CustomExternalForce, which sidesteps the same problem without needing
OpenMM to understand the bond section at all.

This final structures are not guaranteed to be same as what GROMACS would produce,
but it is a good starting point for a subsequent GROMACS minimization.
"""

import os
import re

from openmm import (
    CustomNonbondedForce,
    CustomExternalForce,
    HarmonicBondForce,
    NonbondedForce,
    VerletIntegrator,
    unit,
)
from openmm.app import GromacsGroFile, GromacsTopFile, Simulation, CutoffPeriodic

nanometer = unit.nano * unit.meter

# Minimization stages as (soft-core radius / nm, charge scaling). Coulomb has no
# soft core here, so it is switched off while the worst overlaps are still being
# pushed apart and faded back in as delta closes. The last stage must be
# (0, 1) so the structure ends up minimised on the unmodified force field.
STAGES = [(0.3, 0.0), (0.2, 0.0), (0.02, 0.75), (0.0, 1.0)]

NONBONDED_CUTOFF = 1.0 * nanometer
MAX_ITERATIONS = 2000
TOLERANCE = 10 * (unit.kilojoule_per_mole / nanometer)


def add_softcore_lj(system) -> CustomNonbondedForce:
    """Move the LJ part of the NonbondedForce into a soft-core CustomNonbondedForce.

    The 1-4 exception terms stay in the original NonbondedForce; only the plain
    (non-excluded) LJ interactions are softened, which is what blows up on a
    freshly packed bundle. Coulomb also diverges on an overlap, so every charge
    is additionally put behind a ``lambda_q`` parameter offset. Both ``delta``
    and ``lambda_q`` are global parameters, so one System can be re-minimised at
    each stage.
    """
    nb = next(f for f in system.getForces() if isinstance(f, NonbondedForce))

    softcore = CustomNonbondedForce(
        "4*epsilon*((sigma^6/(r^6+delta^6))^2-sigma^6/(r^6+delta^6));"
        "sigma=0.5*(sigma1+sigma2);"
        "epsilon=sqrt(epsilon1*epsilon2)"
    )
    softcore.addGlobalParameter("delta", STAGES[0][0])
    softcore.addPerParticleParameter("sigma")
    softcore.addPerParticleParameter("epsilon")
    softcore.setNonbondedMethod(CustomNonbondedForce.CutoffPeriodic)
    softcore.setCutoffDistance(nb.getCutoffDistance())
    softcore.setUseSwitchingFunction(False)
    # The soft-core form is not a pure r^-6 tail, so the analytic LJ tail
    # correction does not apply; drop it on both forces to keep them consistent.
    softcore.setUseLongRangeCorrection(False)
    nb.setUseDispersionCorrection(False)

    nb.addGlobalParameter("lambda_q", STAGES[0][1])
    for index in range(nb.getNumParticles()):
        charge, sigma, epsilon = nb.getParticleParameters(index)
        softcore.addParticle([sigma, epsilon])
        nb.setParticleParameters(index, 0.0, sigma, 0.0)
        if charge.value_in_unit(unit.elementary_charge) != 0.0:
            nb.addParticleParameterOffset("lambda_q", index, charge, 0.0, 0.0)

    for index in range(nb.getNumExceptions()):
        p1, p2, chargeprod, sigma, epsilon = nb.getExceptionParameters(index)
        softcore.addExclusion(p1, p2)
        # Exception charges are handled directly rather than via the softened
        # force, so they need the same fade-in as the particle charges.
        if chargeprod.value_in_unit(unit.elementary_charge**2) != 0.0:
            nb.setExceptionParameters(index, p1, p2, 0.0, sigma, epsilon)
            nb.addExceptionParameterOffset("lambda_q", index, chargeprod, 0.0, 0.0)

    system.addForce(softcore)
    return softcore


def find_inter_itp(top_path: str) -> str | None:
    """Return the .itp that ``top_path`` #includes inside its #ifdef INTER block.

    This is the file grompp -DINTER would pull in; None if the .top has no such
    block (the generated single-monomer tops don't).
    """
    with open(top_path, "r", encoding="utf-8") as file:
        text = file.read()

    block = re.search(r"^#ifdef\s+INTER\b(.*?)^#endif", text, re.S | re.M)
    if block is None:
        return None

    include = re.search(r'^#include\s+"([^"]+)"', block.group(1), re.M)
    if include is None:
        return None

    return os.path.join(os.path.dirname(os.path.abspath(top_path)), include.group(1))


def add_intermolecular_bonds(system, itp_path: str) -> int:
    """Add the [ intermolecular_interactions ] bonds of ``itp_path`` to ``system``.

    Only function type 6 is accepted: that is GROMACS' plain harmonic bond that
    generates no exclusions, which maps exactly onto a HarmonicBondForce and
    leaves the nonbonded setup untouched. Atom indices in this section are
    1-based and global over the whole system, matching OpenMM's ordering.
    """
    force = HarmonicBondForce()
    force.setName("IntermolecularHBonds")

    category = None
    count = 0
    with open(itp_path, "r", encoding="utf-8") as file:
        for raw in file:
            line = raw.split(";")[0].strip()
            if not line:
                continue
            if line.startswith("["):
                category = line.strip("[] ").lower()
                continue
            if category != "bonds":
                continue

            fields = line.split()
            i, j, funct = int(fields[0]), int(fields[1]), int(fields[2])
            if funct != 6:
                raise ValueError(
                    f"{itp_path}: unsupported intermolecular bond type {funct} "
                    f"for atoms {i}-{j} (only type 6 is handled)"
                )
            length, k = float(fields[3]), float(fields[4])
            force.addBond(i - 1, j - 1, length * nanometer, k * unit.kilojoule_per_mole / nanometer**2)
            count += 1

    if count == 0:
        raise ValueError(f"{itp_path}: no [ bonds ] found under intermolecular interactions")

    for index in range(force.getNumBonds()):
        p1, p2, _, _ = force.getBondParameters(index)
        if p1 >= system.getNumParticles() or p2 >= system.getNumParticles():
            raise ValueError(
                f"{itp_path}: bond references atom {max(p1, p2) + 1}, but the "
                f"system has only {system.getNumParticles()} atoms"
            )

    system.addForce(force)
    return count


def add_fixed_atoms(system, positions, fixed_indices: list[int]) -> int:
    """Add a CustomExternalForce pinning the given atoms to their positions.

    The restraint is harmonic with a very large spring constant, so the atoms
    stay effectively in place. Reference coordinates must be passed in: a System
    carries no positions, they only exist once a Context is built.

    Indices are deduplicated, since an atom appearing in two hydrogen bonds
    would otherwise be pinned twice and feel double the spring constant.
    """
    if not fixed_indices:
        return 0

    k_fixed = 1e6 * unit.kilojoule_per_mole / nanometer**2  # Large spring constant
    fixed_force = CustomExternalForce("0.5 * k * ((x - x0)^2 + (y - y0)^2 + (z - z0)^2)")
    fixed_force.setName("FixedAtoms")
    fixed_force.addPerParticleParameter("x0")
    fixed_force.addPerParticleParameter("y0")
    fixed_force.addPerParticleParameter("z0")
    fixed_force.addGlobalParameter("k", k_fixed)

    coords = positions.value_in_unit(nanometer)
    for index in sorted(set(fixed_indices)):
        if index >= system.getNumParticles():
            raise ValueError(f"fixed atom {index + 1} is outside the system " f"({system.getNumParticles()} atoms)")
        x, y, z = coords[index]
        fixed_force.addParticle(index, [x, y, z])

    system.addForce(fixed_force)
    return fixed_force.getNumParticles()


def write_gro(topology, positions, box_vectors, out_path: str, title: str):
    """Write a GROMACS .gro. OpenMM ships a reader but no writer for this format."""
    nm = unit.nanometer
    coords = positions.value_in_unit(nm)
    atoms = list(topology.atoms())

    with open(out_path, "w", encoding="utf-8", newline="\n") as file:
        file.write(f"{title}\n")
        file.write(f"{len(atoms):5d}\n")
        for atom in atoms:
            x, y, z = coords[atom.index]
            residue = atom.residue
            file.write(
                "%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n"
                % (
                    (int(residue.id) % 100000),
                    residue.name[:5],
                    atom.name[:5],
                    ((atom.index + 1) % 100000),
                    x,
                    y,
                    z,
                )
            )
        (ax, _, _), (_, by, _), (_, _, cz) = box_vectors.value_in_unit(nm)
        file.write("%10.5f%10.5f%10.5f\n" % (ax, by, cz))


def relax(gro_path: str, top_path: str, out_dir: str, include_dir: str | None = None) -> str:
    """Relax ``gro_path``/``top_path`` and write ``<stem>_relaxed.gro`` into ``out_dir``.

    ``include_dir`` is forwarded to GromacsTopFile for resolving ``#include``s
    that live outside the .top file's own directory (e.g. a shared
    force-field directory); the .top file's own directory is always searched
    in addition, so pass None when every ``#include`` sits next to it.
    """
    gro = GromacsGroFile(gro_path)
    box = gro.getPeriodicBoxVectors()
    top = GromacsTopFile(
        top_path,
        periodicBoxVectors=box,
        includeDir=include_dir,
    )
    system = top.createSystem(
        nonbondedMethod=CutoffPeriodic,
        nonbondedCutoff=NONBONDED_CUTOFF,
    )
    softcore = add_softcore_lj(system)

    # Pin the C=O/NH atoms listed in the #ifdef INTER itp to their initial
    # positions instead of restraining them with a HarmonicBondForce: see the
    # module docstring for why the bonds can't go through GromacsTopFile.
    fixed_indices = []
    inter_itp = find_inter_itp(top_path)
    if inter_itp is None:
        print("    no #ifdef INTER block; relaxing without inter-fiber restraints")
    else:
        with open(inter_itp, "r", encoding="utf-8") as file:
            for line in file:
                line = line.split(";")[0].strip()
                if not line or line.startswith("["):
                    continue
                fields = line.split()
                fixed_indices.append(int(fields[0]) - 1)  # Convert to 0-based index
                fixed_indices.append(int(fields[1]) - 1)  # Convert to 0-based index

        count = add_fixed_atoms(system, gro.positions, fixed_indices)
        print(f"    fixed {count} atoms from {os.path.basename(inter_itp)}")

    simulation = Simulation(top.topology, system, VerletIntegrator(0.001 * unit.picosecond))
    simulation.context.setPositions(gro.positions)

    for delta, lambda_q in STAGES:
        simulation.context.setParameter("delta", delta)
        simulation.context.setParameter("lambda_q", lambda_q)
        before = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        simulation.minimizeEnergy(tolerance=TOLERANCE, maxIterations=MAX_ITERATIONS)
        after = simulation.context.getState(getEnergy=True).getPotentialEnergy()
        print(
            f"    delta={delta:4.2f} nm q={lambda_q:4.2f}: {before} -> {after}",
            flush=True,
        )

    assert STAGES[-1] == (0.0, 1.0), "final stage must run on the unmodified force field"
    del softcore

    state = simulation.context.getState(getPositions=True)
    stem = os.path.basename(gro_path)[: -len(".gro")]
    out_path = os.path.join(out_dir, f"{stem}_relaxed.gro")
    write_gro(
        top.topology,
        state.getPositions(),
        state.getPeriodicBoxVectors(),
        out_path,
        f"{stem} relaxed with soft-core LJ",
    )
    return out_path
