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

def relax(gro_path: str, top_path: str, out_dir: str, include_dir: str | None = None) -> str:
    """Relax ``gro_path``/``top_path`` and write ``<stem>_relaxed.gro`` into ``out_dir``.

    ``include_dir`` is forwarded to GromacsTopFile for resolving ``#include``s
    that live outside the .top file's own directory (e.g. a shared
    force-field directory); the .top file's own directory is always searched
    in addition, so pass None when every ``#include`` sits next to it.
    """
