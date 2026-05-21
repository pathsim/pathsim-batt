"""Shared runner helpers and assertion helpers for cell test modules.

Import these plain functions directly in test files; they are not pytest
fixtures.  Every runner accepts an optional ``pybamm_solver`` keyword so
individual tests can override the PyBaMM solver (e.g. for ODE models that
require CasadiSolver instead of the default IDAKLUSolver).
"""

from pathsim import Connection, Simulation
from pathsim.blocks import Constant
from pathsim.solvers import ESDIRK43

from pathsim_batt.cells import (
    CellCoSimElectrical,
    CellCoSimElectrothermal,
    CellElectrical,
    CellElectrothermal,
)

# ---------------------------------------------------------------------------
# Runner helpers
# ---------------------------------------------------------------------------


def run_electrical(
    model,
    pv,
    current=1.0,
    t_cell=298.15,
    duration=2,
    dt=1.0,
    initial_soc=1.0,
    pybamm_solver=None,
):
    kwargs = {}
    if pybamm_solver is not None:
        kwargs["pybamm_solver"] = pybamm_solver
    cell = CellElectrical(
        model=model, parameter_values=pv, initial_soc=initial_soc, **kwargs
    )
    I_src = Constant(current)
    T_src = Constant(t_cell)
    sim = Simulation(
        blocks=[I_src, T_src, cell],
        connections=[
            Connection(I_src, cell["I"]),
            Connection(T_src, cell["T_cell"]),
        ],
        dt=dt,
        Solver=ESDIRK43,
    )
    sim.run(duration)
    return cell


def run_electrothermal(
    model,
    pv,
    current=1.0,
    t_amb=298.15,
    duration=2,
    dt=1.0,
    initial_soc=1.0,
    pybamm_solver=None,
):
    kwargs = {}
    if pybamm_solver is not None:
        kwargs["pybamm_solver"] = pybamm_solver
    cell = CellElectrothermal(
        model=model, parameter_values=pv, initial_soc=initial_soc, **kwargs
    )
    I_src = Constant(current)
    T_src = Constant(t_amb)
    sim = Simulation(
        blocks=[I_src, T_src, cell],
        connections=[
            Connection(I_src, cell["I"]),
            Connection(T_src, cell["T_amb"]),
        ],
        dt=dt,
        Solver=ESDIRK43,
    )
    sim.run(duration)
    return cell


def run_cosim_electrical(
    model,
    pv,
    current=1.0,
    t_cell=298.15,
    duration=2,
    cosim_dt=1.0,
    initial_soc=1.0,
    pybamm_solver=None,
):
    kwargs = {}
    if pybamm_solver is not None:
        kwargs["pybamm_solver"] = pybamm_solver
    cell = CellCoSimElectrical(
        model=model,
        parameter_values=pv,
        dt=cosim_dt,
        initial_soc=initial_soc,
        **kwargs,
    )
    I_src = Constant(current)
    T_src = Constant(t_cell)
    sim = Simulation(
        blocks=[I_src, T_src, cell],
        connections=[
            Connection(I_src, cell["I"]),
            Connection(T_src, cell["T_cell"]),
        ],
        dt=cosim_dt / 2,
        Solver=ESDIRK43,
    )
    sim.run(duration)
    return cell


def run_cosim_electrothermal(
    model,
    pv,
    current=1.0,
    t_amb=298.15,
    duration=2,
    cosim_dt=1.0,
    initial_soc=1.0,
    pybamm_solver=None,
):
    kwargs = {}
    if pybamm_solver is not None:
        kwargs["pybamm_solver"] = pybamm_solver
    cell = CellCoSimElectrothermal(
        model=model,
        parameter_values=pv,
        dt=cosim_dt,
        initial_soc=initial_soc,
        **kwargs,
    )
    I_src = Constant(current)
    T_src = Constant(t_amb)
    sim = Simulation(
        blocks=[I_src, T_src, cell],
        connections=[
            Connection(I_src, cell["I"]),
            Connection(T_src, cell["T_amb"]),
        ],
        dt=cosim_dt / 2,
        Solver=ESDIRK43,
    )
    sim.run(duration)
    return cell


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def assert_electrical_outputs(test, cell, v_lo, v_hi, check_q_dot_nonneg=True):
    """Assert physically plausible outputs for a CellElectrical run.

    Checks:
    - V in (v_lo, v_hi)
    - Q_dot >= 0 (optional)
    - SOC in (0, 1]
    """
    V = float(cell.outputs[0])
    Q = float(cell.outputs[1])
    soc = float(cell.outputs[2])
    test.assertGreater(V, v_lo, f"V={V:.3f} below lower cutoff {v_lo}")
    test.assertLess(V, v_hi, f"V={V:.3f} above upper cutoff {v_hi}")
    if check_q_dot_nonneg:
        test.assertGreaterEqual(Q, 0.0, f"Q_dot={Q:.4f} is negative")
    test.assertGreater(soc, 0.0)
    test.assertLessEqual(soc, 1.0)


def assert_electrothermal_outputs(test, cell, v_lo, v_hi, check_q_dot_nonneg=True):
    """Assert physically plausible outputs for a CellElectrothermal run.

    Checks:
    - V in (v_lo, v_hi)
    - T in (250, 400) K
    - Q_dot >= 0 (optional)
    - SOC in (0, 1]
    """
    V = float(cell.outputs[0])
    T = float(cell.outputs[1])
    Q = float(cell.outputs[2])
    soc = float(cell.outputs[3])
    test.assertGreater(V, v_lo, f"V={V:.3f} below lower cutoff {v_lo}")
    test.assertLess(V, v_hi, f"V={V:.3f} above upper cutoff {v_hi}")
    test.assertGreater(T, 250.0, f"T={T:.1f} K unreasonably cold")
    test.assertLess(T, 400.0, f"T={T:.1f} K unreasonably hot")
    if check_q_dot_nonneg:
        test.assertGreaterEqual(Q, 0.0, f"Q_dot={Q:.4f} is negative")
    test.assertGreater(soc, 0.0)
    test.assertLessEqual(soc, 1.0)
