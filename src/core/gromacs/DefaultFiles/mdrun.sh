#!/bin/bash

# Enable alias expansion
shopt -s expand_aliases

if [ -f "freeze" ]; then
    echo "File \"freeze\" exists"
    echo "!!!!!!!!!!!!!!!!!!!!!!!!!!!!! it is stopped for protection!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
    exit
fi

# if alias gmx_d is not defined, define it
if ! command -v gmx_d &> /dev/null
then
    alias gmx_d='gmx'
fi


gmx_d mdrun -deffnm output -v  |tee run.out 