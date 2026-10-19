#!/bin/bash

if [ -f "freeze" ]; then
    echo "File \"freeze\" exists"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!! it is stopped for protection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit
fi

# for supporting all gmx (gmx_d, gmx_mpi, gmx) commands
# Enable alias expansion
shopt -s expand_aliases
if command -v gmx_d &> /dev/null
then
    alias inner_gmx=gmx_d
elif command -v gmx_mpi &> /dev/null
then
    alias inner_gmx=gmx_mpi
elif command -v gmx &> /dev/null
then
    alias inner_gmx=gmx
else
    echo "No gromacs installation found."
    exit 1
fi
# end of alias support

# ===========================================================================
# Run on the GPU by default; fall back to the CPU only when a problem occurs.
#
# The most GPU-offloaded command is tried first, then we step down one level at
# a time until one succeeds:
#       full GPU (nb+pme+bonded+update)  ->  GPU without -update  ->  CPU
#
#   * -update gpu is used only for md / md-vv integrators. It is skipped for
#     energy minimisation (steep / cg) and whenever a plumed.dat is present
#     (PLUMED must modify forces on the CPU each step, so the GPU cannot own the
#     update). Ensembles the GPU update path does not support (e.g. Nose-Hoover
#     / Parrinello-Rahman) simply fail the first attempt and drop to the next.
#   * if an offloaded run dies mid-way it is retried from its checkpoint (-cpi)
#     one level down, so no sampling is wasted.
#   * a plumed.dat in the run directory is picked up automatically (-plumed).
#
# Overrides (no code change needed):
#   * a file named "cpu" in the run dir  -> force CPU
#   * export GMX_MDRUN_CPU=1             -> force CPU
#   * export GMX_MDRUN_NTMPI="-ntmpi 4"  -> rank count (default "-ntmpi 1";
#                                           set to "" to let GROMACS choose)
#   * export GMX_MDRUN_EXTRA="-gpu_id 0 -pinoffset 8 -pinstride 1 -pin on"
#                                        -> extra flags added to every attempt
#                                           (e.g. to share one GPU between runs)
# ===========================================================================
set -o pipefail

NTMPI="${GMX_MDRUN_NTMPI--ntmpi 1}"
EXTRA="${GMX_MDRUN_EXTRA:-}"

PLUMED=""
if [ -f "plumed.dat" ]; then
    PLUMED="-plumed plumed.dat"
fi

_has_gpu() {
    command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"
}

# integrator decides whether the GPU may own the update step
integrator=$(grep -iE '^[[:space:]]*integrator' setting.mdp 2>/dev/null \
    | tail -n 1 | sed 's/;.*//' | cut -d= -f2 | tr -d '[:space:]' | tr '[:upper:]' '[:lower:]')

GPU_BASE="-nb gpu -pme gpu -bonded gpu"

declare -a ATTEMPTS
if [ -f "cpu" ] || [ "${GMX_MDRUN_CPU:-0}" = "1" ]; then
    echo "[mdrun] CPU forced (cpu sentinel / GMX_MDRUN_CPU)"
    ATTEMPTS=("")
elif _has_gpu; then
    if { [ "$integrator" = "md" ] || [ "$integrator" = "md-vv" ]; } && [ -z "$PLUMED" ]; then
        ATTEMPTS=("$GPU_BASE -update gpu" "$GPU_BASE" "")
    else
        ATTEMPTS=("$GPU_BASE" "")
    fi
else
    echo "[mdrun] no GPU detected -> CPU"
    ATTEMPTS=("")
fi

run_one() {
    # $1 = offload flag string ("" for pure CPU)
    if [ -f "output.cpt" ]; then
        inner_gmx mdrun -deffnm output -v $NTMPI $1 $PLUMED $EXTRA -cpi output.cpt | tee run.out
    else
        inner_gmx mdrun -deffnm output -v $NTMPI $1 $PLUMED $EXTRA | tee run.out
    fi
}

ok=1
n=${#ATTEMPTS[@]}
for i in "${!ATTEMPTS[@]}"; do
    flags="${ATTEMPTS[$i]}"
    label="${flags:-CPU}"
    echo "[mdrun] attempt $((i + 1))/$n: mdrun $NTMPI $flags $PLUMED $EXTRA"
    if run_one "$flags"; then
        echo "[mdrun] succeeded ($label)"
        ok=0
        break
    fi
    echo "[mdrun] FAILED ($label)"
    if [ $((i + 1)) -lt "$n" ]; then
        echo "[mdrun] falling back to a less GPU-offloaded command ..."
    fi
done

if [ "$ok" -ne 0 ]; then
    echo "[mdrun] all execution modes failed"
    exit 1
fi
