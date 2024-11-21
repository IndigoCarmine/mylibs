#!/bin/bash

# Enable alias expansion
shopt -s expand_aliases
# if alias gmx_d is not defined, define it
if ! command -v gmx_d &> /dev/null
then
    alias gmx_d="gmx"
fi

gmx_d trjconv -f output.trr -s output.tpr -o output.xtc -pbc mol

read -p "Do you want to open them with ovito? (y/n)" response

if [ "${response}" = "y" ] || [ "${response}" = "Y" ]; then
    echo "Opening files with ovito..."
    ovito output.xtc input.gro
else
    echo "Not opening files with ovito."
fi
