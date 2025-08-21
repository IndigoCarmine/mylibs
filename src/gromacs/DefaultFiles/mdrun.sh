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


if [ -f "output.cpt" ]; then
    inner_gmx mdrun -deffnm output -v -cpi output.cpt | tee run.out 
else
    inner_gmx mdrun -deffnm output -v | tee run.out 
fi
