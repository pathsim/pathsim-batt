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
from pathsim.exceptions import StopSimulation

# HELPERS =============================================================================

_DEFAULT_INPUTS = {
    "Current function [A]": 0.0,
    "Ambient temperature [K]": 298.15,
}

# Canonical PyBaMM variable name for terminal voltage (lithium-ion / lead-acid).
# Equivalent-circuit and sodium-ion BasicDFN models export it under a different
# name; ``_VOLTAGE_VAR_CANDIDATES`` lists all known aliases in priority order.
_TERMINAL_VOLTAGE_VAR = "Terminal voltage [V]"

_VOLTAGE_VAR_CANDIDATES: list[str] = [
    "Terminal voltage [V]",
    "Voltage [V]",
    "Battery voltage [V]",
]
_HEATING_VAR_CANDIDATES: list[str] = [
    "Total heating [W]",
    "Total heat generation [W]",
]
_TEMP_VAR_CANDIDATES: list[str] = [
    "X-averaged cell temperature [K]",
    "Cell temperature [K]",
]
_SOC_CAPACITY_VAR = "Discharge capacity [A.h]"
_SOC_DIRECT_CANDIDATES: list[str] = ["State of charge", "State of Charge", "SoC"]

# Map from the canonical names used in ``_pybamm_output_vars`` to their
# per-model-family fallback lists.
_VAR_ALIAS_MAP: dict[str, list[str]] = {
    "Terminal voltage [V]": _VOLTAGE_VAR_CANDIDATES,
    "Total heating [W]": _HEATING_VAR_CANDIDATES,
    "X-averaged cell temperature [K]": _TEMP_VAR_CANDIDATES,
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


def _pick_var(variables: dict, candidates: list[str], description: str) -> str:
    """Return the first name in *candidates* present in *variables*.

    Raises ``ValueError`` listing all tried names if none match.
    """
    for name in candidates:
        if name in variables:
            return name
    raise ValueError(
        f"No {description} variable found in PyBaMM model. Tried: {candidates}"
    )


def _resolve_output_vars(
    requested: list[str],
    available: dict,
) -> tuple[list[str], str | None, str | None]:
    """Map *requested* variable names to names actually present in *available*.

    Returns ``(resolved, soc_cap_var, soc_direct_var)``:

    * ``resolved`` — same length as *requested*; each entry is the actual
      PyBaMM variable name to use (canonical or alias).
    * ``soc_cap_var`` — ``"Discharge capacity [A.h]"`` when available (standard
      electrochemical models), otherwise ``None``.
    * ``soc_direct_var`` — a direct SoC variable name used when *soc_cap_var*
      is ``None`` (ECM, some basic models).
    """
    resolved = [
        _pick_var(available, _VAR_ALIAS_MAP[name], name)
        if (name not in available and name in _VAR_ALIAS_MAP)
        else name
        for name in requested
    ]

    if _SOC_CAPACITY_VAR in available:
        return resolved, _SOC_CAPACITY_VAR, None
    return (
        resolved,
        None,
        _pick_var(available, _SOC_DIRECT_CANDIDATES, "state of charge"),
    )


def _detect_soc_direct_scale(
    sim: pybamm.Simulation,
    soc_direct_var: str | None,
) -> float:
    """Return 1/100 if *soc_direct_var* is in percentage form, else 1.0.

    Some models (e.g. lead-acid) export ``"State of Charge"`` in the range
    0–100 rather than 0–1.  Evaluating the variable at the initial state with
    zero current lets us detect this once at construction time.
    """
    if soc_direct_var is None:
        return 1.0
    objs = sim.built_model.export_casadi_objects(
        [soc_direct_var],
        input_parameter_order=list(_DEFAULT_INPUTS.keys()),
    )
    p0 = casadi.DM(list(_DEFAULT_INPUTS.values()))
    x0 = casadi.Function("x0", [objs["inputs"]], [objs["x0"]])(p0)
    z0 = casadi.Function("z0", [objs["inputs"]], [objs["z0"]])(p0)
    soc_fn = casadi.Function(
        "soc",
        [objs["t"], objs["x"], objs["z"], objs["inputs"]],
        [objs["variables"][soc_direct_var]],
    )
    raw = float(soc_fn(0.0, x0, z0, p0))
    return 1.0 / 100.0 if raw > 1.0 else 1.0


def _build_simulation(
    sim: pybamm.Simulation,
    model: pybamm.BaseBatteryModel,
    initial_soc: float,
) -> None:
    """Build *sim*, handling model-family-specific initialisation quirks.

    * **ECM** (``Thevenin``): initial SoC is set via ``set_initial_state`` on
      the parameter values before build; ``initial_soc`` cannot be passed to
      ``sim.build()`` directly.
    * **Lead-acid** models: the eSOH solver used by the standard build path is
      lithium-ion–specific and fails for lead-acid parameter sets; build
      without ``initial_soc``.
    * **All other models**: standard ``sim.build(initial_soc=...)``.
    """
    if isinstance(model, pybamm.equivalent_circuit.Thevenin):
        from pybamm.models.full_battery_models.equivalent_circuit import (
            set_initial_state,
        )

        set_initial_state(initial_soc, sim.parameter_values)
        sim.build(inputs=_DEFAULT_INPUTS)
    elif isinstance(model, pybamm.lead_acid.BaseModel):
        if initial_soc != 1.0:
            raise ValueError(
                "initial_soc is not supported for lead-acid models: PyBaMM's "
                "lead-acid parameter sets do not include the electrode OCP data "
                "required to map a target SoC to initial stoichiometries.  "
                "The initial state is always determined by the parameter values."
            )
        sim.build(inputs=_DEFAULT_INPUTS)
    else:
        sim.build(initial_soc=initial_soc, inputs=_DEFAULT_INPUTS)


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
    output ports (SOC is always appended last).  ``_pybamm_output_vars`` must
    contain a terminal-voltage entry — either the canonical
    ``"Terminal voltage [V]"`` or any alias listed in
    ``_VOLTAGE_VAR_CANDIDATES``; the actual name exported by the built model is
    resolved automatically at construction time.
    """

    _thermal_option: str = ""
    _thermal_extra_options: dict[str, str] = {}
    _pybamm_output_vars: list[str] = []

    def __init__(
        self,
        model: pybamm.BaseBatteryModel | None = None,
        parameter_values: pybamm.ParameterValues | None = None,
        initial_soc: float = 1.0,
        pybamm_solver: pybamm.BaseSolver | None = None,
    ) -> None:
        self._initial_soc = float(initial_soc)

        if not any(v in self._pybamm_output_vars for v in _VOLTAGE_VAR_CANDIDATES):
            raise TypeError(
                f"{type(self).__name__}._pybamm_output_vars must contain one of "
                f"{_VOLTAGE_VAR_CANDIDATES}."
            )

        if model is None:
            model = pybamm.lithium_ion.SPMe(
                options={"thermal": self._thermal_option, **self._thermal_extra_options}
            )

        self._parameter_values = _prepare_parameter_values(parameter_values)
        try:
            self._v_lower = float(self._parameter_values["Lower voltage cut-off [V]"])
            self._v_upper = float(self._parameter_values["Upper voltage cut-off [V]"])
        except KeyError as exc:
            raise ValueError(
                f"parameter_values is missing a voltage cut-off entry: {exc}. "
                "Ensure your parameter set defines both 'Lower voltage cut-off [V]' "
                "and 'Upper voltage cut-off [V]'."
            ) from exc

        pybamm_solver = pybamm_solver or pybamm.CasadiSolver(mode="safe")

        sim = pybamm.Simulation(
            model,
            parameter_values=self._parameter_values,
            solver=pybamm_solver,
        )
        _build_simulation(sim, model, self._initial_soc)

        available = sim.built_model.variables

        # Early DAE check: probe with just the voltage variable so that the
        # NotImplementedError is raised before variable-resolution, giving a
        # cleaner error message for models that are DAE *and* also lack other
        # required output variables (e.g. sodium_ion.BasicDFN).
        _vol_var = _pick_var(available, _VOLTAGE_VAR_CANDIDATES, "terminal voltage")
        _probe = sim.built_model.export_casadi_objects(
            [_vol_var], input_parameter_order=list(_DEFAULT_INPUTS.keys())
        )
        if _probe["z"].numel() > 0:
            raise NotImplementedError(
                f"{type(self).__name__}: the supplied PyBaMM model has "
                f"{_probe['z'].numel()} algebraic variable(s) after discretisation "
                "(DAE system). Only pure ODE models are supported by this block. "
                "Use a CellCoSim* block for DAE models."
            )

        resolved_output_vars, soc_cap_var, soc_direct_var = _resolve_output_vars(
            self._pybamm_output_vars, available
        )
        self._v_idx = resolved_output_vars.index(
            next(v for v in _VOLTAGE_VAR_CANDIDATES if v in resolved_output_vars)
        )

        extra_soc_var = soc_cap_var if soc_cap_var is not None else soc_direct_var
        all_out_vars = resolved_output_vars + [extra_soc_var]
        casadi_objs = sim.built_model.export_casadi_objects(
            all_out_vars,
            input_parameter_order=list(_DEFAULT_INPUTS.keys()),
        )

        t_sym = casadi_objs["t"]
        x_sym = casadi_objs["x"]
        p_sym = casadi_objs["inputs"]

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
        soc_direct_scale = _detect_soc_direct_scale(sim, soc_direct_var)

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

        v_lower = self._v_lower
        v_upper = self._v_upper
        v_idx = self._v_idx

        def func_alg(x, u, t):
            xv = casadi.DM(x.reshape(-1, 1))
            p = _pack(u)
            outputs = [float(out_var_fns[n](t, xv, p)) for n in resolved_output_vars]
            if soc_cap_var is not None:
                q_dis = float(out_var_fns[soc_cap_var](t, xv, p))
                soc = max(0.0, min(1.0, initial_soc_val - q_dis / q_nominal))
            else:
                raw = float(out_var_fns[soc_direct_var](t, xv, p))
                soc = max(0.0, min(1.0, raw * soc_direct_scale))
            outputs.append(soc)
            V = outputs[v_idx]
            if V <= v_lower:
                raise StopSimulation(f"undervoltage: V={V:.4f} V <= {v_lower} V")
            if V >= v_upper:
                raise StopSimulation(f"overvoltage: V={V:.4f} V >= {v_upper} V")
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
    ``_pybamm_output_vars`` must contain a terminal-voltage entry — either the
    canonical ``"Terminal voltage [V]"`` or any alias in
    ``_VOLTAGE_VAR_CANDIDATES``; the actual name is resolved against the built
    model at construction time.
    """

    _thermal_option: str = ""
    _thermal_extra_options: dict[str, str] = {}
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

        if not any(v in self._pybamm_output_vars for v in _VOLTAGE_VAR_CANDIDATES):
            raise TypeError(
                f"{type(self).__name__}._pybamm_output_vars must contain one of "
                f"{_VOLTAGE_VAR_CANDIDATES}."
            )

        self._dt = float(dt)
        if self._dt <= 0.0:
            raise ValueError("dt must be positive")

        if model is None:
            model = pybamm.lithium_ion.SPMe(
                options={"thermal": self._thermal_option, **self._thermal_extra_options}
            )

        self._model = model
        self._parameter_values = _prepare_parameter_values(parameter_values)
        try:
            self._v_lower = float(self._parameter_values["Lower voltage cut-off [V]"])
            self._v_upper = float(self._parameter_values["Upper voltage cut-off [V]"])
        except KeyError as exc:
            raise ValueError(
                f"parameter_values is missing a voltage cut-off entry: {exc}. "
                "Ensure your parameter set defines both 'Lower voltage cut-off [V]' "
                "and 'Upper voltage cut-off [V]'."
            ) from exc
        self._pybamm_solver = pybamm_solver or pybamm.IDAKLUSolver()
        self._q_nominal = float(self._parameter_values["Nominal cell capacity [A.h]"])

        self._sim = self._build_sim()

        available = self._sim.built_model.variables
        self._resolved_output_vars, self._soc_cap_var, self._soc_direct_var = (
            _resolve_output_vars(self._pybamm_output_vars, available)
        )
        self._v_idx = self._resolved_output_vars.index(
            next(v for v in _VOLTAGE_VAR_CANDIDATES if v in self._resolved_output_vars)
        )
        # Scale factor: 1.0 if SOC direct variable is in [0,1] fraction form,
        # 1/100 if it is in percentage form (e.g. lead_acid "State of Charge").
        self._soc_direct_scale = _detect_soc_direct_scale(
            self._sim, self._soc_direct_var
        )

        self._last_outputs: npt.NDArray[np.float64] = self._initial_outputs()

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
        _build_simulation(sim, self._model, self._initial_soc)
        return sim

    def _initial_outputs(self) -> npt.NDArray[np.float64]:
        """Compute outputs at t=0 from the built PyBaMM model using default inputs.

        Uses the same CasADi export approach as ``_CellBase`` to evaluate each
        output variable at the initial state vector.  The evaluation uses
        ``_DEFAULT_INPUTS`` (0 A current, 298.15 K ambient temperature) because
        the wired input ports are not yet available at construction time.  The
        resulting open-circuit voltage is therefore physically meaningful but does
        not account for a non-default initial temperature or a non-zero current at
        t=0.
        """
        extra_soc_var = (
            self._soc_cap_var if self._soc_cap_var is not None else self._soc_direct_var
        )
        all_out_vars = self._resolved_output_vars + [extra_soc_var]
        casadi_objs = self._sim.built_model.export_casadi_objects(
            all_out_vars,
            input_parameter_order=list(_DEFAULT_INPUTS.keys()),
        )
        t_sym = casadi_objs["t"]
        x_sym = casadi_objs["x"]
        z_sym = casadi_objs["z"]
        p_sym = casadi_objs["inputs"]
        p0 = casadi.DM(list(_DEFAULT_INPUTS.values()))
        x0 = casadi.Function("x0", [p_sym], [casadi_objs["x0"]])(p0)
        # Algebraic initial conditions (empty for ODE models such as SPMe;
        # non-empty for DAE models such as DFN).
        z0 = casadi.Function("z0", [p_sym], [casadi_objs["z0"]])(p0)

        outputs: list[float] = []
        for name in self._resolved_output_vars:
            fn = casadi.Function(
                "v", [t_sym, x_sym, z_sym, p_sym], [casadi_objs["variables"][name]]
            )
            outputs.append(float(fn(0.0, x0, z0, p0)))

        extra_fn = casadi.Function(
            "e",
            [t_sym, x_sym, z_sym, p_sym],
            [casadi_objs["variables"][extra_soc_var]],
        )
        extra_val = float(extra_fn(0.0, x0, z0, p0))
        if self._soc_cap_var is not None:
            soc = max(0.0, min(1.0, self._initial_soc - extra_val / self._q_nominal))
        else:
            soc = max(0.0, min(1.0, extra_val * self._soc_direct_scale))
        outputs.append(soc)

        return np.array(outputs, dtype=np.float64)

    def _discrete_step(self, current: float, t_amb: float) -> npt.NDArray[np.float64]:
        inputs = {
            "Current function [A]": float(current),
            "Ambient temperature [K]": float(t_amb),
        }
        self._sim.step(dt=self._dt, inputs=inputs, save=False)

        sol = self._sim.solution
        outputs = [float(sol[n].entries[-1]) for n in self._resolved_output_vars]
        if self._soc_cap_var is not None:
            q_dis = float(sol[self._soc_cap_var].entries[-1])
            soc = max(0.0, min(1.0, self._initial_soc - q_dis / self._q_nominal))
        else:
            raw = float(sol[self._soc_direct_var].entries[-1])
            soc = max(0.0, min(1.0, raw * self._soc_direct_scale))
        outputs.append(soc)

        self._last_outputs = np.array(outputs, dtype=np.float64)
        self.outputs.update_from_array(self._last_outputs)
        V = outputs[self._v_idx]
        if V <= self._v_lower:
            raise StopSimulation(f"undervoltage: V={V:.4f} V <= {self._v_lower} V")
        if V >= self._v_upper:
            raise StopSimulation(f"overvoltage: V={V:.4f} V >= {self._v_upper} V")
        return self._last_outputs

    def __len__(self) -> int:
        return len(self._pybamm_output_vars) + 1

    def reset(self) -> None:
        super().reset()
        self._sim = self._build_sim()
        self._last_outputs = self._initial_outputs()
        self.outputs.update_from_array(self._last_outputs)


class CellElectrical(_CellBase):
    """Cell block — electrical outputs only, external thermal coupling.

    PathSim integrates the electrochemical state via the discretised PyBaMM
    ODE.  Temperature dynamics live outside this block: wire ``Q_dot`` to a
    ``LumpedThermal`` (or similar) block and feed its temperature output back
    to ``T_cell``.

    .. note::
        The SPMe/SPM ODE is stiff.  Use an implicit solver (e.g.
        ``ESDIRK43``, ``BDF``) when constructing the PathSim
        ``Simulation`` to avoid prohibitively small step sizes.

    Parameters
    ----------
    model : pybamm.BaseBatteryModel or None
        PyBaMM lithium-ion model.  Defaults to isothermal SPMe with heat
        source calculation enabled.
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
    Q_dot (1) : total heat generation [W]
    SOC (2) : state of charge (0–1)
    """

    _thermal_option = "isothermal"
    _thermal_extra_options = {"calculate heat source for isothermal models": "true"}
    _pybamm_output_vars = [
        "Terminal voltage [V]",
        "Total heating [W]",
    ]

    input_port_labels = {"I": 0, "T_cell": 1}
    output_port_labels = {"V": 0, "Q_dot": 1, "SOC": 2}


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
    Q_dot (2) : total heat generation [W]
    SOC (3) : state of charge (0–1)
    """

    _thermal_option = "lumped"
    _pybamm_output_vars = [
        "Terminal voltage [V]",
        "X-averaged cell temperature [K]",
        "Total heating [W]",
    ]

    input_port_labels = {"I": 0, "T_amb": 1}
    output_port_labels = {"V": 0, "T": 1, "Q_dot": 2, "SOC": 3}


class CellCoSimElectrical(_CoSimCellBase):
    """Cell block (co-simulation) — electrical outputs only, external thermal coupling.

    PyBaMM advances internally on discrete macro-steps of ``dt`` via
    ``pybamm.Simulation.step()``. PathSim receives zero-order-held outputs
    between macro-steps.

    This mode supports PyBaMM models that result in DAE systems (e.g. DFN).

    Parameters
    ----------
    model : pybamm.BaseBatteryModel or None
        PyBaMM lithium-ion model.  Defaults to isothermal SPMe with heat
        source calculation enabled.
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
    _thermal_extra_options = {"calculate heat source for isothermal models": "true"}
    _pybamm_output_vars = [
        "Terminal voltage [V]",
        "Total heating [W]",
    ]

    input_port_labels = {"I": 0, "T_cell": 1}
    output_port_labels = {"V": 0, "Q_dot": 1, "SOC": 2}


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
        "Total heating [W]",
    ]

    input_port_labels = {"I": 0, "T_amb": 1}
    output_port_labels = {"V": 0, "T": 1, "Q_dot": 2, "SOC": 3}
