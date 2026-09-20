"""
Odds and ends shared by the other packages.

Import the submodules directly -- ``from base_utils import plotlib``, or
``import base_utils.cui_utils``. They are deliberately not imported here, so
that reaching for one of them does not drag matplotlib and pandas (and
plotlib's global rcParams tweaks) into every caller.
"""
