"""
PathSim-Batt: Battery Simulation Blocks for PathSim

A toolbox providing battery simulation blocks for the PathSim framework,
using PyBaMM as the electrochemical backend.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .thermal import LumpedThermal

__all__ = ["__version__", "LumpedThermal"]

# Cell blocks rely on pybamm (+ its casadi backend), which can't load in
# Pyodide. Re-export them only when the import succeeds; on a normal pip
# install both submodules load eagerly.
try:
    from .cells import (
        CellCoSimElectrical,
        CellCoSimElectrothermal,
        CellElectrical,
        CellElectrothermal,
    )

    __all__ += [
        "CellElectrical",
        "CellElectrothermal",
        "CellCoSimElectrical",
        "CellCoSimElectrothermal",
    ]
except ImportError:
    pass
