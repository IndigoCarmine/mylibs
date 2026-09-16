"""
Apply a solvation step's edits to a GROMACS topology file (``topo.top``).

Two edits are made in a single pass:

* ``#include "<solvent>.itp"`` is inserted just before ``[ system ]``
  (one ``--include`` flag per itp file; omit it to skip this edit).
* The ``<solvent> <count>`` line that ``gmx solvate -p dummy.top`` appends to
  the scratch topology is inserted at the *end of the* ``[ molecules ]``
  *section* -- not at the end of the file, which is where GROMACS itself puts
  it.  That distinction matters as soon as an ``[ intermolecular_interactions ]``
  section follows ``[ molecules ]``::

      [ system ]
      SYSTEMNAME

      [ molecules ]
      ; Compound       mols
      MOL             100
      MCH             432    <- inserted here, not after the section below

      [ intermolecular_interactions ]

The script is **idempotent**: running it twice, or resuming an interrupted
calculation, produces exactly the same ``topo.top`` as running it once.  That
is achieved by never editing ``topo.top`` in place.  The first run snapshots
the untouched topology as ``topo.top.orig`` and every run rebuilds ``topo.top``
from that snapshot, then swaps it in with ``os.replace`` so the file is either
fully old or fully new even if the job is killed mid-write.

``topo.top.orig`` deliberately does not match the ``*.top`` glob that the
generated ``copy.sh`` uses, so the snapshot stays in the step that made it and
each step snapshots its own input topology.

Two situations make the run abort rather than guess, both of them what a
directory damaged by the older, non-idempotent version of this script looks
like: ``topo.top`` having diverged from the snapshot by more than this script's
own edits, and the same itp being included twice.  Restore ``topo.top`` from
``topo.top.orig`` -- or regenerate the directory -- and run again.
"""

# for python 3.6.8 (it is server version)

import argparse
import os
import re
import shutil
import sys
from typing import List, Optional

MOLECULES_HEADER = re.compile(r"^\s*\[\s*molecules\s*\]")
SYSTEM_HEADER = re.compile(r"^\s*\[\s*system\s*\]")
ANY_HEADER = re.compile(r"^\s*\[")

# "MOL 100" -- a molecule name followed by a count.  The name is deliberately
# matched as \S+ : names like "MOL1" or "PEG_A" are legal, and testing them
# with str.isalpha() (as this script used to) mistakes them for the end of the
# section and inserts the solvent line too early, which reorders the topology
# with respect to the coordinates.
COUNT_LINE = re.compile(r"^\s*(\S+)\s+(\d+)\s*$")


class TopologyError(Exception):
    """The topology does not have the shape this script needs."""


def strip_comment(line: str) -> str:
    return line.split(";")[0]


def read_lines(path: str) -> List[str]:
    # newline="" keeps CRLF intact so the file round-trips byte for byte.
    with open(path, "r", newline="") as f:
        return f.readlines()


def write_atomically(path: str, lines: List[str]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        f.writelines(lines)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def last_count_line(path: str) -> Optional[str]:
    """
    Return the molecule-count line to add, taken from the scratch topology.

    ``gmx solvate -p`` *appends* to that file, so a re-run leaves several
    lines behind; the last one belongs to the most recent solvate call.
    """
    try:
        lines = read_lines(path)
    except (IOError, OSError):
        return None

    found = None  # type: Optional[str]
    for line in lines:
        match = COUNT_LINE.match(strip_comment(line))
        if match:
            found = "{:<16}{}\n".format(match.group(1), match.group(2))
    return found


def include_line(itp: str) -> str:
    return '#include "{}"\n'.format(itp)


def count_include(lines: List[str], itp: str) -> int:
    wanted = include_line(itp).split()
    return sum(1 for line in lines if line.split() == wanted)


def has_include(lines: List[str], itp: str) -> bool:
    return count_include(lines, itp) > 0


def is_end_of_molecules(line: str) -> bool:
    """Is *line* the first line that no longer belongs to ``[ molecules ]``?"""
    if not strip_comment(line).strip():
        # Blank and comment-only lines stay inside the section.
        return False
    if ANY_HEADER.match(line):
        return True
    if line.lstrip().startswith("#"):
        # A preprocessor directive (#include, #ifdef, ...) ends the listing.
        return True
    return COUNT_LINE.match(strip_comment(line)) is None


def find_header(lines: List[str], header: "re.Pattern") -> Optional[int]:
    for index, line in enumerate(lines):
        if header.match(line):
            return index
    return None


def build(lines: List[str], count_line: str, includes: List[str]) -> List[str]:
    """
    Rebuild the topology from *lines*, which must be the pristine snapshot.

    An include already present in *lines* is not added again -- that is the
    legitimate "second solvation step in one chain" case.  The molecule-count
    line, by contrast, is always added: a second solvation step really does
    need a second ``<solvent> <count>`` line.
    """
    out = list(lines)

    start = find_header(out, MOLECULES_HEADER)
    if start is None:
        raise TopologyError("no '[ molecules ]' section found")

    # Put the solvent right after the last molecule already listed, so trailing
    # blank lines and comments keep sitting at the end of the section.
    index = start + 1
    insert_at = None  # type: Optional[int]
    while index < len(out) and not is_end_of_molecules(out[index]):
        if COUNT_LINE.match(strip_comment(out[index])):
            insert_at = index + 1
        index += 1
    if insert_at is None:
        # Nothing is listed yet; the whole section is comments or empty.
        insert_at = index
    out.insert(insert_at, count_line)

    pending = [itp for itp in includes if not has_include(out, itp)]
    if pending:
        system = find_header(out, SYSTEM_HEADER)
        if system is None:
            raise TopologyError(
                "no '[ system ]' section found, cannot add: " + ", ".join(pending)
            )
        out[system:system] = [include_line(itp) for itp in pending]
    return out


def snapshot(top: str, pristine: str) -> bool:
    """Save *top* as *pristine* once.  Returns True if it already existed."""
    if os.path.exists(pristine):
        return True
    tmp = pristine + ".tmp"
    shutil.copyfile(top, tmp)
    os.replace(tmp, pristine)
    return False


def is_subsequence_within(pristine: List[str], current: List[str], slack: int) -> bool:
    """Is *current* the snapshot plus at most *slack* inserted lines?"""
    if len(current) < len(pristine) or len(current) - len(pristine) > slack:
        return False
    i = 0
    for line in current:
        if i < len(pristine) and line == pristine[i]:
            i += 1
    return i == len(pristine)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Insert the solvent's molecule count (and itp include) into a "
        "GROMACS topology, idempotently."
    )
    parser.add_argument("--top", default="topo.top", help="topology to edit")
    parser.add_argument(
        "--dummy",
        default="dummy.top",
        help="scratch topology that 'gmx solvate -p' appended the count to",
    )
    parser.add_argument(
        "--pristine", default=None, help="snapshot path (default: <top>.orig)"
    )
    parser.add_argument(
        "--include",
        action="append",
        metavar="ITP",
        help='add #include "ITP" before [ system ]; may be repeated',
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="rebuild from the snapshot even if topo.top was edited by hand",
    )
    args = parser.parse_args(argv)

    includes = list(args.include or [])
    pristine_path = args.pristine or args.top + ".orig"

    if not os.path.exists(args.top):
        print("topology file '{}' not found.".format(args.top))
        return 1

    # Read the count before touching anything: a missing count used to leave
    # the directory with no topo.top at all, which makes a resume impossible.
    count_line = last_count_line(args.dummy)
    if count_line is None:
        print(
            "no molecule count found in '{}'. "
            "'gmx solvate -p {}' should have written one.".format(
                args.dummy, args.dummy
            )
        )
        return 1

    current_lines = read_lines(args.top)

    # A topology never includes the same itp twice -- grompp rejects the
    # redefined moleculetype.  Seeing it means this directory was already
    # rewritten twice by the older, non-idempotent version of this script.
    # Refuse before taking a snapshot of the damage.
    doubled = [itp for itp in includes if count_include(current_lines, itp) > 1]
    if doubled and not args.force:
        print(
            "'{}' includes {} more than once, so it was rewritten twice. That is "
            "what the old version of this script did when a calculation was "
            "resumed.\nRemove the duplicated '#include' and solvent count lines "
            "(or regenerate this directory) and run again.".format(
                args.top, " and ".join("'" + itp + "'" for itp in doubled)
            )
        )
        return 1

    already_snapshotted = snapshot(args.top, pristine_path)
    pristine_lines = read_lines(pristine_path)

    if already_snapshotted and not args.force:
        # Everything this script adds is at most one count line plus the
        # includes, so anything beyond that is somebody else's edit.
        if not is_subsequence_within(pristine_lines, current_lines, 1 + len(includes)):
            print(
                "'{top}' has diverged from the snapshot '{orig}' by more than this "
                "script's own edits, so rebuilding it would lose those changes.\n"
                "This is also what a directory corrupted by the old, "
                "non-idempotent version of this script looks like.\n"
                "Fix it in one of these ways:\n"
                "  - edit '{orig}' instead and re-run,\n"
                "  - restore '{top}' from '{orig}' and re-run,\n"
                "  - delete '{orig}' to take a fresh snapshot of '{top}',\n"
                "  - or re-run with --force to rebuild from '{orig}' anyway.".format(
                    top=args.top, orig=pristine_path
                )
            )
            return 1

    try:
        new_lines = build(pristine_lines, count_line, includes)
    except TopologyError as error:
        print("cannot edit '{}': {}".format(pristine_path, error))
        return 1

    if new_lines == current_lines:
        print("'{}' is already up to date.".format(args.top))
        return 0

    write_atomically(args.top, new_lines)
    print("added '{}' to '{}'.".format(count_line.strip(), args.top))
    for itp in includes:
        print("'{}' is included by '{}'.".format(itp, args.top))
    return 0


if __name__ == "__main__":
    sys.exit(main())
