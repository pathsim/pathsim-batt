#########################################################################################
##
##                          LUMPED THERMAL MODEL
##                           (thermal/lumped.py)
##
##              Single-node thermal model for battery cell temperature
##
#########################################################################################

# IMPORTS ==============================================================================

import numpy as np
from pathsim.blocks import DynamicalSystem

# BLOCKS ===============================================================================


class LumpedThermal(DynamicalSystem):
    """Single-node lumped thermal model.

    .. math::

        m C_p \\frac{dT}{dt} = \\dot{Q} - U A (T - T_{\\mathrm{amb}})

    Parameters
    ----------
    mass : float
        Thermal mass [kg].
    Cp : float
        Specific heat capacity [J kg⁻¹ K⁻¹].
    UA : float
        Overall heat transfer conductance [W K⁻¹].
    T0 : float
        Initial temperature [K].  Default 298.15 K.

    Inputs
    ------
    Q_dot (0) : heat generation rate [W]
    T_amb (1) : ambient temperature [K]

    Outputs
    -------
    T (0) : cell temperature [K]
    """

    input_port_labels = {"Q_dot": 0, "T_amb": 1}
    output_port_labels = {"T": 0}

    def __init__(self, mass=0.065, Cp=750.0, UA=0.5, T0=298.15):
        # input validation
        if mass <= 0:
            raise ValueError(f"'mass' must be positive but is {mass}")
        if Cp <= 0:
            raise ValueError(f"'Cp' must be positive but is {Cp}")
        if UA < 0:
            raise ValueError(f"'UA' must be non-negative but is {UA}")

        # store parameters
        self.mass = float(mass)
        self.Cp = float(Cp)
        self.UA = float(UA)

        def _fn_d(x, u, t):
            (T,) = x
            Q_dot, T_amb = u
            return np.array([(Q_dot - self.UA * (T - T_amb)) / (self.mass * self.Cp)])

        def _fn_a(x, u, t):
            return x.copy()

        super().__init__(
            func_dyn=_fn_d,
            func_alg=_fn_a,
            initial_value=np.array([float(T0)]),
        )
