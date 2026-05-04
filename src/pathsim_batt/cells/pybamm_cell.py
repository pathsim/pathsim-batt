#########################################################################################
##
##                          PyBaMM CELL BLOCKS
##                          (cells/pybamm_cell.py)
##
##              Battery cell blocks wrapping PyBaMM models for PathSim
##
#########################################################################################

# IMPORTS ==============================================================================

import casadi
import numpy as np
import pybamm
from pathsim.blocks import DynamicalSystem

# BLOCKS ===============================================================================


class _CellBase(DynamicalSystem):
    """Shared base for PyBaMM cell blocks.

    Discretises the PyBaMM model at construction time and exposes its ODE
    right-hand side to PathSim's numerical integrator via the ``DynamicalSystem``
    interface.  The differential state vector of the discretised model becomes
    the PathSim state; PathSim's chosen solver advances it in time.

    Only PyBaMM models that produce a pure ODE after discretisation are
    supported (i.e. models with no algebraic variables, such as SPMe and SPM).
    Models that result in a DAE system (e.g. DFN) are not supported and will
    raise ``NotImplementedError`` at construction time.

    Because the SPMe/SPM family of models is stiff, users should prefer an
    implicit solver (e.g. ``ESDIRK43``, ``BDF``) when constructing the
    PathSim ``Simulation``.

    Subclasses set ``_thermal_option`` and ``_pybamm_output_vars`` to select the
    thermal sub-model and define which PyBaMM variables map to the block's
    output ports (SOC is always appended last).
    """

    _thermal_option: str = ""
    _pybamm_output_vars: list[str] = []

    def __init__(
        self,
        model,
        parameter_values,
        initial_soc,
        pybamm_solver,
    ):
        self._initial_soc = float(initial_soc)

        if model is None:
            model = pybamm.lithium_ion.SPMe(options={"thermal": self._thermal_option})

        if parameter_values is None:
            parameter_values = pybamm.ParameterValues("Chen2020")
        parameter_values = parameter_values.copy()
        parameter_values["Current function [A]"] = "[input]"
        parameter_values["Ambient temperature [K]"] = "[input]"
        self._parameter_values = parameter_values

        pybamm_solver = pybamm_solver or pybamm.CasadiSolver(mode="safe")

        _build_inputs = {
            "Current function [A]": 0.0,
            "Ambient temperature [K]": 298.15,
        }
        sim = pybamm.Simulation(
            model,
            parameter_values=parameter_values,
            solver=pybamm_solver,
        )
        sim.build(initial_soc=self._initial_soc, inputs=_build_inputs)

        all_out_vars = self._pybamm_output_vars + ["Discharge capacity [A.h]"]
        input_order = ["Current function [A]", "Ambient temperature [K]"]
        casadi_objs = sim.built_model.export_casadi_objects(
            all_out_vars, input_parameter_order=input_order
        )

        t_sym = casadi_objs["t"]
        x_sym = casadi_objs["x"]
        z_sym = casadi_objs["z"]
        p_sym = casadi_objs["inputs"]

        if z_sym.numel() > 0:
            raise NotImplementedError(
                f"{type(self).__name__}: the supplied PyBaMM model has "
                f"{z_sym.numel()} algebraic variable(s) after discretisation "
                "(DAE system). Only pure ODE models are supported. "
                "Use SPMe or SPM instead of DFN."
            )

        rhs_fn = casadi.Function("rhs", [t_sym, x_sym, p_sym], [casadi_objs["rhs"]])
        jac_fn = casadi.Function(
            "jac_rhs", [t_sym, x_sym, p_sym], [casadi_objs["jac_rhs"]]
        )

        out_var_fns = {}
        for idx, var_name in enumerate(all_out_vars):
            var_expr = casadi_objs["variables"][var_name]
            out_var_fns[var_name] = casadi.Function(
                f"outvar_{idx}", [t_sym, x_sym, p_sym], [var_expr]
            )

        self._casadi_rhs = rhs_fn
        self._jac_rhs_eval = jac_fn
        self._out_var_fcns = out_var_fns
        self._q_nominal = float(parameter_values["Nominal cell capacity [A.h]"])

        q_nominal = self._q_nominal
        initial_soc_val = float(initial_soc)
        pybamm_output_vars = list(self._pybamm_output_vars)

        def _pack(u):
            return casadi.DM([float(u[0]), float(u[1])])

        def func_dyn(x, u, t):
            xv = casadi.DM(x.reshape(-1, 1))
            p = _pack(u)
            return np.array(rhs_fn(t, xv, p)).flatten()

        def jac_dyn(x, u, t):
            xv = casadi.DM(x.reshape(-1, 1))
            p = _pack(u)
            return np.array(jac_fn(t, xv, p))

        def func_alg(x, u, t):
            xv = casadi.DM(x.reshape(-1, 1))
            p = _pack(u)
            outputs = [float(out_var_fns[n](t, xv, p)) for n in pybamm_output_vars]
            q_dis = float(out_var_fns["Discharge capacity [A.h]"](t, xv, p))
            soc = max(0.0, min(1.0, initial_soc_val - q_dis / q_nominal))
            outputs.append(soc)
            return np.array(outputs)

        x0_fn = casadi.Function("x0", [p_sym], [casadi_objs["x0"]])

        default_inputs = casadi.DM([0.0, 298.15])
        y0 = np.array(x0_fn(default_inputs)).flatten()

        super().__init__(
            func_dyn=func_dyn,
            func_alg=func_alg,
            initial_value=y0,
            jac_dyn=jac_dyn,
        )

    def __len__(self):
        return len(self._pybamm_output_vars) + 1

    def reset(self):
        super().reset()


class CellElectrical(_CellBase):
    """Cell block — electrical outputs only, external thermal coupling.

    PathSim integrates both the electrochemical state (via the discretised
    PyBaMM ODE) and the cell temperature ODE.  Wire ``Q_heat`` to a
    ``LumpedThermal`` (or similar) block and feed its temperature output
    back to ``T_cell``.

    .. note::
        The SPMe/SPM ODE is stiff.  Use an implicit solver (e.g.
        ``ESDIRK43``, ``BDF``) when constructing the PathSim
        ``Simulation`` to avoid prohibitively small step sizes.

    Parameters
    ----------
    model : pybamm.BaseBatteryModel or None
        PyBaMM lithium-ion model.  Defaults to ``SPMe(thermal="isothermal")``.
    parameter_values : pybamm.ParameterValues or None
        PyBaMM parameter set.  Defaults to ``Chen2020``.
    initial_soc : float
        Initial state of charge (0–1).  Default 1.0.
    pybamm_solver : pybamm.BaseSolver or None
        PyBaMM solver used only during model build / discretisation.
        Defaults to ``CasadiSolver(mode="safe")``.

    Inputs
    ------
    I (0) : current [A], positive = discharge
    T_cell (1) : cell temperature [K] from external PathSim thermal block

    Outputs
    -------
    V (0) : terminal voltage [V]
    Q_heat (1) : X-averaged volumetric heat generation [W m⁻³]
    SOC (2) : state of charge (0–1)
    """

    _thermal_option = "isothermal"
    _pybamm_output_vars = [
        "Terminal voltage [V]",
        "X-averaged total heating [W.m-3]",
    ]

    input_port_labels = {"I": 0, "T_cell": 1}
    output_port_labels = {"V": 0, "Q_heat": 1, "SOC": 2}

    def __init__(
        self,
        model=None,
        parameter_values=None,
        initial_soc=1.0,
        pybamm_solver=None,
    ):
        super().__init__(model, parameter_values, initial_soc, pybamm_solver)


class CellElectrothermal(_CellBase):
    """Cell block — coupled electrical and thermal model.

    PathSim integrates the full electrochemical + thermal state (via the
    discretised PyBaMM ODE).  The cell temperature is part of the PyBaMM
    state vector and is read back as output port ``T``.  Supply a
    time-varying ambient / coolant temperature via ``T_amb`` to couple to a
    pack-level thermal model.

    .. note::
        The SPMe/SPM ODE is stiff.  Use an implicit solver (e.g.
        ``ESDIRK43``, ``BDF``) when constructing the PathSim
        ``Simulation`` to avoid prohibitively small step sizes.

    Parameters
    ----------
    model : pybamm.BaseBatteryModel or None
        PyBaMM lithium-ion model.  Defaults to ``SPMe(thermal="lumped")``.
    parameter_values : pybamm.ParameterValues or None
        PyBaMM parameter set.  Defaults to ``Chen2020``.
    initial_soc : float
        Initial state of charge (0–1).  Default 1.0.
    pybamm_solver : pybamm.BaseSolver or None
        PyBaMM solver used only during model build / discretisation.
        Defaults to ``CasadiSolver(mode="safe")``.

    Inputs
    ------
    I (0) : current [A], positive = discharge
    T_amb (1) : ambient / coolant temperature [K]

    Outputs
    -------
    V (0) : terminal voltage [V]
    T (1) : cell temperature [K] (part of PyBaMM state)
    Q_heat (2) : X-averaged volumetric heat generation [W m⁻³]
    SOC (3) : state of charge (0–1)
    """

    _thermal_option = "lumped"
    _pybamm_output_vars = [
        "Terminal voltage [V]",
        "X-averaged cell temperature [K]",
        "X-averaged total heating [W.m-3]",
    ]

    input_port_labels = {"I": 0, "T_amb": 1}
    output_port_labels = {"V": 0, "T": 1, "Q_heat": 2, "SOC": 3}

    def __init__(
        self,
        model=None,
        parameter_values=None,
        initial_soc=1.0,
        pybamm_solver=None,
    ):
        super().__init__(model, parameter_values, initial_soc, pybamm_solver)


Cell = CellElectrothermal
