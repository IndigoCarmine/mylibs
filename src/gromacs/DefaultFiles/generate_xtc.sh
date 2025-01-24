#!/bin/bash
if [ "$1" == "-h" ]; then
    echo "Usage: source generate_xtc.sh"
    echo "This script generates xtc file from trr file."
    echo "If -no is set, then it does not open the files with ovito."
    return
fi

# if -no is set, then do not open the files with ovito
open_with_ovito=true
if [ "$1" == "-no" ]; then
    echo "Not opening files with ovito."
    open_with_ovito=false
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

echo 0 | inner_gmx trjconv -f output.trr -s output.tpr -o output.xtc -pbc mol

if [ ! -f output.xtc ]; then
    echo "Failed to generate xtc file."
    return
fi

if not $open_with_ovito; then
    return
fi

read -p "Do you want to open them with ovito? (y/n)" response

if [ "${response}" = "y" ] || [ "${response}" = "Y" ]; then
    echo "Opening files with ovito..."
    ovito output.xtc input.gro
else
    echo "Not opening files with ovito."
fi
