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
import numpy.typing as npt
import pybamm
from pathsim.blocks import DynamicalSystem, Wrapper

# HELPERS =============================================================================

_DEFAULT_INPUTS = {
    "Current function [A]": 0.0,
    "Ambient temperature [K]": 298.15,
}


def _prepare_parameter_values(
    parameter_values: pybamm.ParameterValues | None,
) -> pybamm.ParameterValues:
    """Copy *parameter_values* (defaulting to Chen2020) and mark both
    driving inputs as PyBaMM ``"[input]"`` placeholders."""
    if parameter_values is None:
        parameter_values = pybamm.ParameterValues("Chen2020")
    parameter_values = parameter_values.copy()
    parameter_values["Current function [A]"] = "[input]"
    parameter_values["Ambient temperature [K]"] = "[input]"
    return parameter_values


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
        model: pybamm.BaseBatteryModel | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_soc: float = 1.0,
        pybamm_solver: pybamm.BaseSolver | None = None,
    ) -> None:
        self._initial_soc = float(initial_soc)

        if model is None:
            model = pybamm.lithium_ion.SPMe(options={"thermal": self._thermal_option})

        self._parameter_values = _prepare_parameter_values(parameter_values)

        pybamm_solver = pybamm_solver or pybamm.CasadiSolver(mode="safe")

        sim = pybamm.Simulation(
            model,
            parameter_values=self._parameter_values,
            solver=pybamm_solver,
        )
        sim.build(initial_soc=self._initial_soc, inputs=_DEFAULT_INPUTS)

        all_out_vars = self._pybamm_output_vars + ["Discharge capacity [A.h]"]
        casadi_objs = sim.built_model.export_casadi_objects(
            all_out_vars,
            input_parameter_order=list(_DEFAULT_INPUTS.keys()),
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
        self._q_nominal = float(self._parameter_values["Nominal cell capacity [A.h]"])

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

        y0 = np.array(x0_fn(casadi.DM(list(_DEFAULT_INPUTS.values())))).flatten()

        super().__init__(
            func_dyn=func_dyn,
            func_alg=func_alg,
            initial_value=y0,
            jac_dyn=jac_dyn,
        )

    def __len__(self) -> int:
        return len(self._pybamm_output_vars) + 1

    def reset(self) -> None:
        super().reset()


class _CoSimCellBase(Wrapper):
    """Shared base for co-simulation PyBaMM cell blocks.

    Wraps ``pybamm.Simulation.step()`` in a periodic ``Wrapper`` event, so
    PyBaMM advances on discrete macro-steps while PathSim sees a zero-order-held
    output signal between events. This allows using PyBaMM models that produce
    DAE systems after discretisation (e.g. DFN), because PyBaMM owns the
    differential-algebraic solve internally.

    Subclasses set ``_thermal_option``, ``_pybamm_output_vars`` and port labels.
    """

    _thermal_option: str = ""
    _pybamm_output_vars: list[str] = []

    def __init__(
        self,
        model: pybamm.BaseBatteryModel | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_soc: float = 1.0,
        pybamm_solver: pybamm.BaseSolver | None = None,
        dt: float = 1.0,
    ) -> None:
        self._initial_soc = float(initial_soc)
        self._dt = float(dt)
        if self._dt <= 0.0:
            raise ValueError("dt must be positive")

        if model is None:
            model = pybamm.lithium_ion.SPMe(options={"thermal": self._thermal_option})

        self._model = model
        self._parameter_values = _prepare_parameter_values(parameter_values)
        self._pybamm_solver = pybamm_solver or pybamm.IDAKLUSolver()
        self._q_nominal = float(self._parameter_values["Nominal cell capacity [A.h]"])

        self._sim = self._build_sim()

        n_outputs = len(self._pybamm_output_vars) + 1
        self._last_outputs: npt.NDArray[np.float64] = np.zeros(n_outputs)
        self._last_outputs[-1] = self._initial_soc

        super().__init__(func=self._discrete_step, T=self._dt, tau=self._dt)

        # ensure outputs are valid before first scheduled sample
        self.outputs.update_from_array(self._last_outputs)

    def _build_sim(self) -> pybamm.Simulation:
        """Create and build a fresh ``pybamm.Simulation`` with default inputs."""
        sim = pybamm.Simulation(
            self._model,
            parameter_values=self._parameter_values,
            solver=self._pybamm_solver,
        )
        sim.build(initial_soc=self._initial_soc, inputs=_DEFAULT_INPUTS)
        return sim

    def _discrete_step(self, current: float, t_amb: float) -> npt.NDArray[np.float64]:
        inputs = {
            "Current function [A]": float(current),
            "Ambient temperature [K]": float(t_amb),
        }
        self._sim.step(dt=self._dt, inputs=inputs)

        sol = self._sim.solution
        outputs = [float(sol[n].entries[-1]) for n in self._pybamm_output_vars]
        q_dis = float(sol["Discharge capacity [A.h]"].entries[-1])
        soc = max(0.0, min(1.0, self._initial_soc - q_dis / self._q_nominal))
        outputs.append(soc)

        self._last_outputs = np.array(outputs, dtype=np.float64)
        return self._last_outputs

    def update(self, t: float) -> None:
        self.outputs.update_from_array(self._last_outputs)

    def __len__(self) -> int:
        return len(self._pybamm_output_vars) + 1

    def reset(self) -> None:
        super().reset()
        self._sim = self._build_sim()
        self._last_outputs = np.zeros(len(self._pybamm_output_vars) + 1)
        self._last_outputs[-1] = self._initial_soc
        self.outputs.update_from_array(self._last_outputs)


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


class CellCoSimElectrical(_CoSimCellBase):
    """Cell block (co-simulation) — electrical outputs only, external thermal coupling.

    PyBaMM advances internally on discrete macro-steps of ``dt`` via
    ``pybamm.Simulation.step()``. PathSim receives zero-order-held outputs
    between macro-steps.

    This mode supports PyBaMM models that result in DAE systems (e.g. DFN).

    Parameters
    ----------
    model : pybamm.BaseBatteryModel or None
        PyBaMM lithium-ion model. Defaults to ``SPMe(thermal="isothermal")``.
    parameter_values : pybamm.ParameterValues or None
        PyBaMM parameter set. Defaults to ``Chen2020``.
    initial_soc : float
        Initial state of charge (0–1). Default 1.0.
    pybamm_solver : pybamm.BaseSolver or None
        Solver used by PyBaMM for the internal time stepping.
        Defaults to ``IDAKLUSolver()``.
    dt : float
        Co-simulation macro-step size [s]. Must be > 0.
    """

    _thermal_option = "isothermal"
    _pybamm_output_vars = [
        "Terminal voltage [V]",
        "X-averaged total heating [W.m-3]",
    ]

    input_port_labels = {"I": 0, "T_cell": 1}
    output_port_labels = {"V": 0, "Q_heat": 1, "SOC": 2}


class CellCoSimElectrothermal(_CoSimCellBase):
    """Cell block (co-simulation) — coupled electrical and thermal model.

    PyBaMM advances internally on discrete macro-steps of ``dt`` via
    ``pybamm.Simulation.step()``. PathSim receives zero-order-held outputs
    between macro-steps.

    This mode supports PyBaMM models that result in DAE systems (e.g. DFN).

    Parameters
    ----------
    model : pybamm.BaseBatteryModel or None
        PyBaMM lithium-ion model. Defaults to ``SPMe(thermal="lumped")``.
    parameter_values : pybamm.ParameterValues or None
        PyBaMM parameter set. Defaults to ``Chen2020``.
    initial_soc : float
        Initial state of charge (0–1). Default 1.0.
    pybamm_solver : pybamm.BaseSolver or None
        Solver used by PyBaMM for the internal time stepping.
        Defaults to ``IDAKLUSolver()``.
    dt : float
        Co-simulation macro-step size [s]. Must be > 0.
    """

    _thermal_option = "lumped"
    _pybamm_output_vars = [
        "Terminal voltage [V]",
        "X-averaged cell temperature [K]",
        "X-averaged total heating [W.m-3]",
    ]

    input_port_labels = {"I": 0, "T_amb": 1}
    output_port_labels = {"V": 0, "T": 1, "Q_heat": 2, "SOC": 3}
