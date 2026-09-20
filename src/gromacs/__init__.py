"""
This package provides a collection of utilities for GROMACS simulations.

Import the submodules directly -- ``from gromacs import calculation``, or
``import gromacs.itp``. They are deliberately not imported here: ``analyzing``
needs MDAnalysis, which only ships in the optional ``analysis`` extra, so
importing it eagerly would break a default install.
"""
