"""Generic SLURM submission helpers.

Submit a mylibs calculation tree to SLURM, poll it, and detect completion via a
``DONE``/``EXITCODE`` sentinel the batch script drops into the job directory (so a
filesystem watcher or recursive callback can react to it without polling SLURM).

Nothing here is site-specific: the target cluster (``-M`` for SLURM's
multi-cluster interface), partition, and any shared-filesystem constraint are all
passed in by the caller. Projects that always target one cluster keep those
defaults in their own config (e.g. ``cgmch.config``) and pass them through.

Typical use::

    from gromacs import calculation as calc, slurm

    calc.launch([...], input_gro="mol.gro", working_dir="/shared/runs/iter0")
    jobid = slurm.submit("/shared/runs/iter0", job_name="iter0",
                         cluster="yagai", partition="Gromacs",
                         shared_root="/shared")   # enforce visibility on compute node
    state = slurm.wait(jobid, cluster="yagai")   # blocks until terminal state
    # or, for the recursive-callback design, poll without blocking:
    #   state = slurm.poll(jobid, cluster="yagai")
"""

from __future__ import annotations

import os
import re
import subprocess
import time

TERMINAL_STATES = {
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
    "PREEMPTED",
}

# The submit script runs the mylibs-generated ``run.sh`` in the (shared) submit
# directory and always drops a DONE sentinel + EXITCODE, even on failure, so the
# controller-side poll / watcher can detect completion.
SBATCH_TEMPLATE = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=sbatch_%x.log
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem={mem}
{extra_directives}
export OMP_NUM_THREADS={cpus}

cd "${{SLURM_SUBMIT_DIR}}"

finish() {{
    ec=$?
    echo "$ec" > EXITCODE
    date -u +%Y-%m-%dT%H:%M:%SZ > DONE
    echo "[slurm] job finished with exit code $ec" >> DONE
}}
trap finish EXIT

set -o pipefail
bash {script} 2>&1 | tee run_all.log
"""


def render_sbatch(
    job_name: str,
    script: str = "run.sh",
    cpus: int = 32,
    mem: str = "64GB",
    partition: str | None = None,
    extra_directives: str = "",
) -> str:
    """Render a SLURM batch script for a mylibs calculation tree.

    Args:
        job_name: SLURM job name.
        script: the driver script to run inside the job dir (mylibs writes ``run.sh``).
        cpus: cores requested (also ``OMP_NUM_THREADS``).
        mem: memory request string (e.g. ``"64GB"``).
        partition: SLURM partition; omitted from the script when ``None``.
        extra_directives: extra ``#SBATCH`` lines (e.g. mail), newline-separated.
    """
    directives: list[str] = []
    if partition:
        directives.append(f"#SBATCH --partition={partition}")
    if extra_directives:
        directives.append(extra_directives)
    return SBATCH_TEMPLATE.format(
        job_name=job_name,
        script=script,
        cpus=cpus,
        mem=mem,
        extra_directives="\n".join(directives),
    )


def write_sbatch(job_dir: str, job_name: str, filename: str = "Gromacs.sbatch", **kwargs) -> str:
    """Render and write the batch script into ``job_dir``; return its path."""
    text = render_sbatch(job_name, **kwargs)
    path = os.path.join(job_dir, filename)
    with open(path, "w", newline="\n") as f:
        f.write(text)
    return path


def submit(
    job_dir: str,
    job_name: str | None = None,
    sbatch_name: str = "Gromacs.sbatch",
    cluster: str | None = None,
    shared_root: str | None = None,
    **render_kwargs,
) -> int:
    """Submit ``job_dir`` to SLURM and return the job id.

    If ``<job_dir>/<sbatch_name>`` does not exist it is rendered from the template.
    ``cluster`` (optional) targets SLURM's multi-cluster interface (``-M``). If
    ``shared_root`` is given, ``job_dir`` must live under it — use this when the
    compute node only sees a shared mount (e.g. a CIFS export) and not local disk.
    """
    job_dir = os.path.abspath(job_dir)
    if shared_root and not job_dir.startswith(os.path.abspath(shared_root)):
        raise ValueError(
            f"job_dir must be under {shared_root} (shared with the compute node); got {job_dir}"
        )
    if job_name is None:
        job_name = os.path.basename(job_dir.rstrip("/"))

    sbatch_path = os.path.join(job_dir, sbatch_name)
    if not os.path.exists(sbatch_path):
        write_sbatch(job_dir, job_name, filename=sbatch_name, **render_kwargs)

    # remove a stale sentinel from a previous run so wait()/watchers don't fire early
    for stale in ("DONE", "EXITCODE"):
        p = os.path.join(job_dir, stale)
        if os.path.exists(p):
            os.remove(p)

    cmd = ["sbatch"] + (["-M", cluster] if cluster else []) + [sbatch_name]
    result = subprocess.run(
        cmd,
        cwd=job_dir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"sbatch failed: {result.stderr.strip() or result.stdout.strip()}")

    # "Submitted batch job 123 on cluster yagai"
    m = re.search(r"Submitted batch job (\d+)", result.stdout)
    if not m:
        raise RuntimeError(f"could not parse job id from: {result.stdout!r}")
    jobid = int(m.group(1))
    with open(os.path.join(job_dir, "JOBID"), "w") as f:
        f.write(str(jobid) + "\n")
    return jobid


def poll(jobid: int, cluster: str | None = None) -> str:
    """Return the SLURM state of ``jobid`` (e.g. RUNNING, COMPLETED, FAILED).

    Uses ``sacct`` (works for finished jobs too). Returns ``"UNKNOWN"`` if the job
    is not yet visible in accounting. ``cluster`` targets ``-M`` when given.
    """
    cmd = ["sacct"] + (["-M", cluster] if cluster else []) + \
        ["-j", str(jobid), "-n", "-P", "-o", "State"]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        state = line.strip().split()[0] if line.strip() else ""
        # the top .batch/.extern steps repeat; the first non-empty line is the job
        if state:
            return state
    return "UNKNOWN"


def is_terminal(state: str) -> bool:
    """True if ``state`` is a finished SLURM state (base word, ignoring reasons)."""
    return state.split()[0] in TERMINAL_STATES if state else False


def wait(jobid: int, cluster: str | None = None, interval: float = 30.0, timeout: float | None = None) -> str:
    """Block until ``jobid`` reaches a terminal state; return that state.

    Prefer this for simple synchronous scripts. For the non-blocking recursive
    callback design, use ``poll`` from a background loop instead.
    """
    start = time.time()
    while True:
        state = poll(jobid, cluster)
        if is_terminal(state):
            return state
        if timeout is not None and (time.time() - start) > timeout:
            raise TimeoutError(f"job {jobid} did not finish within {timeout}s (last state {state})")
        time.sleep(interval)


def exit_code(job_dir: str) -> int | None:
    """Read the EXITCODE sentinel written by the batch script, or None if absent."""
    path = os.path.join(job_dir, "EXITCODE")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        txt = f.read().strip()
    return int(txt) if txt.isdigit() else None
