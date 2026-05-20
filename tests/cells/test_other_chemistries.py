"""Smoke tests for lead_acid, sodium_ion and ECM model families.

Each test verifies that the corresponding block can be constructed, runs a
short simulation without error, and produces physically plausible outputs.

Block / model matrix covered
-----------------------------
sodium_ion.BasicDFN  — only model available; it is a DAE *and* exports no
                       heating or temperature variables, so it is incompatible
                       with all four existing block classes.  Tests document
                       the expected exceptions.

lead_acid.LOQS       — ODE → all 4 blocks
lead_acid.Full       — DAE → CoSim blocks only

ecm.Thevenin         — ODE → all 4 blocks
                       Note: ECM's reversible heat term can be negative, so
                       Q_dot is not sign-constrained in ECM smoke tests.
"""

import unittest

import pybamm
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
# Shared helpers
# ---------------------------------------------------------------------------


def _run_electrical(
    model, pv, current=1.0, t_cell=298.15, duration=2, dt=1.0, initial_soc=1.0
):
    cell = CellElectrical(model=model, parameter_values=pv, initial_soc=initial_soc)
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


def _run_electrothermal(
    model, pv, current=1.0, t_amb=298.15, duration=2, dt=1.0, initial_soc=1.0
):
    cell = CellElectrothermal(model=model, parameter_values=pv, initial_soc=initial_soc)
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


def _run_cosim_electrical(
    model, pv, current=1.0, t_cell=298.15, duration=2, cosim_dt=1.0, initial_soc=1.0
):
    cell = CellCoSimElectrical(
        model=model, parameter_values=pv, dt=cosim_dt, initial_soc=initial_soc
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


def _run_cosim_electrothermal(
    model, pv, current=1.0, t_amb=298.15, duration=2, cosim_dt=1.0, initial_soc=1.0
):
    cell = CellCoSimElectrothermal(
        model=model, parameter_values=pv, dt=cosim_dt, initial_soc=initial_soc
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


def _assert_electrical_outputs(test, cell, v_lo, v_hi, check_q_dot_nonneg=True):
    V = float(cell.outputs[0])
    Q = float(cell.outputs[1])
    soc = float(cell.outputs[2])
    test.assertGreater(V, v_lo, f"V={V:.3f} below lower cutoff {v_lo}")
    test.assertLess(V, v_hi, f"V={V:.3f} above upper cutoff {v_hi}")
    if check_q_dot_nonneg:
        test.assertGreaterEqual(Q, 0.0, f"Q_dot={Q:.4f} is negative")
    test.assertGreater(soc, 0.0)
    test.assertLessEqual(soc, 1.0)


def _assert_electrothermal_outputs(test, cell, v_lo, v_hi, check_q_dot_nonneg=True):
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


# ---------------------------------------------------------------------------
# sodium_ion.BasicDFN
# ---------------------------------------------------------------------------


class TestSodiumIon(unittest.TestCase):
    """sodium_ion.BasicDFN is incompatible with all four existing block classes.

    BasicDFN is a DAE model, so monolithic blocks raise ``NotImplementedError``.
    Co-simulation blocks fail with ``ValueError`` because BasicDFN exports
    neither heating nor temperature variables, which the block classes require.
    Tests document these boundaries.
    """

    def setUp(self):
        self.pv = pybamm.ParameterValues("Chen2020")

    def test_monolithic_electrical_raises_not_implemented(self):
        """BasicDFN is a DAE — CellElectrical must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            CellElectrical(model=pybamm.sodium_ion.BasicDFN(), parameter_values=self.pv)

    def test_monolithic_electrothermal_raises_not_implemented(self):
        """BasicDFN is a DAE — CellElectrothermal must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            CellElectrothermal(
                model=pybamm.sodium_ion.BasicDFN(), parameter_values=self.pv
            )

    def test_cosim_electrical_raises_missing_heating_var(self):
        """BasicDFN has no heating variable — CellCoSimElectrical must raise."""
        with self.assertRaises(ValueError):
            CellCoSimElectrical(
                model=pybamm.sodium_ion.BasicDFN(), parameter_values=self.pv, dt=1.0
            )

    def test_cosim_electrothermal_raises_missing_temp_var(self):
        """BasicDFN has no temperature variable — CellCoSimElectrothermal must raise."""
        with self.assertRaises(ValueError):
            CellCoSimElectrothermal(
                model=pybamm.sodium_ion.BasicDFN(), parameter_values=self.pv, dt=1.0
            )


# ---------------------------------------------------------------------------
# lead_acid.LOQS  (ODE — all 4 blocks)
# ---------------------------------------------------------------------------


class TestLeadAcidLOQS(unittest.TestCase):
    """lead_acid.LOQS with Sulzer2019 parameters (ODE model — all blocks).

    Sulzer2019 cutoffs: lower 1.75 V, upper 2.42 V, nominal capacity 17 A·h.
    """

    def setUp(self):
        self.pv = pybamm.ParameterValues("Sulzer2019")
        self.v_lo = float(self.pv["Lower voltage cut-off [V]"])
        self.v_hi = float(self.pv["Upper voltage cut-off [V]"])

    def _model(self, thermal="isothermal"):
        return pybamm.lead_acid.LOQS(options={"thermal": thermal})

    def test_electrical_smoke(self):
        cell = _run_electrical(self._model(), self.pv, current=17.0)
        _assert_electrical_outputs(self, cell, self.v_lo, self.v_hi)

    def test_electrothermal_smoke(self):
        cell = _run_electrothermal(self._model("lumped"), self.pv, current=17.0)
        _assert_electrothermal_outputs(self, cell, self.v_lo, self.v_hi)

    def test_cosim_electrical_smoke(self):
        # LOQS is an ODE; IDAKLUSolver (co-sim default) requires a Jacobian
        # for ODE models and errors, so use CasadiSolver explicitly.
        solver = pybamm.CasadiSolver(mode="safe")
        cell = CellCoSimElectrical(
            model=self._model(),
            parameter_values=self.pv,
            pybamm_solver=solver,
            dt=1.0,
        )
        I_src = Constant(17.0)
        T_src = Constant(298.15)
        sim = Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_cell"]),
            ],
            dt=0.5,
            Solver=ESDIRK43,
        )
        sim.run(2)
        _assert_electrical_outputs(self, cell, self.v_lo, self.v_hi)

    def test_cosim_electrothermal_smoke(self):
        solver = pybamm.CasadiSolver(mode="safe")
        cell = CellCoSimElectrothermal(
            model=self._model("lumped"),
            parameter_values=self.pv,
            pybamm_solver=solver,
            dt=1.0,
        )
        I_src = Constant(17.0)
        T_src = Constant(298.15)
        sim = Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_amb"]),
            ],
            dt=0.5,
            Solver=ESDIRK43,
        )
        sim.run(2)
        _assert_electrothermal_outputs(self, cell, self.v_lo, self.v_hi)

    def test_electrical_soc_decreases(self):
        """SOC must decrease under discharge current."""
        cell = _run_electrical(self._model(), self.pv, current=17.0, duration=60)
        self.assertLess(float(cell.outputs[2]), 1.0)

    def test_cutoff_values_match_parameter_set(self):
        cell = CellElectrical(model=self._model(), parameter_values=self.pv)
        self.assertAlmostEqual(cell._v_lower, self.v_lo)
        self.assertAlmostEqual(cell._v_upper, self.v_hi)


# ---------------------------------------------------------------------------
# lead_acid.Full  (DAE — co-simulation only)
# ---------------------------------------------------------------------------


class TestLeadAcidFull(unittest.TestCase):
    """lead_acid.Full with Sulzer2019 parameters (DAE model — co-sim only)."""

    def setUp(self):
        self.pv = pybamm.ParameterValues("Sulzer2019")
        self.v_lo = float(self.pv["Lower voltage cut-off [V]"])
        self.v_hi = float(self.pv["Upper voltage cut-off [V]"])

    def _model(self, thermal="isothermal"):
        return pybamm.lead_acid.Full(options={"thermal": thermal})

    def test_monolithic_electrical_raises(self):
        """Full is a DAE — CellElectrical must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            CellElectrical(model=self._model(), parameter_values=self.pv)

    def test_monolithic_electrothermal_raises(self):
        """Full is a DAE — CellElectrothermal must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            CellElectrothermal(model=self._model("lumped"), parameter_values=self.pv)

    def test_cosim_electrical_smoke(self):
        cell = _run_cosim_electrical(self._model(), self.pv, current=17.0)
        _assert_electrical_outputs(self, cell, self.v_lo, self.v_hi)

    def test_cosim_electrothermal_smoke(self):
        cell = _run_cosim_electrothermal(self._model("lumped"), self.pv, current=17.0)
        _assert_electrothermal_outputs(self, cell, self.v_lo, self.v_hi)


# ---------------------------------------------------------------------------
# equivalent_circuit.Thevenin  (ODE — all 4 blocks)
# ---------------------------------------------------------------------------


class TestECM(unittest.TestCase):
    """equivalent_circuit.Thevenin with ECM_Example parameters (ODE — all blocks).

    ECM_Example cutoffs: lower 3.2 V, upper 4.2 V, nominal capacity 100 A·h.

    ECM's reversible heat term can be negative (entropy effect), so Q_dot is
    not constrained to be non-negative in these tests.

    Co-simulation blocks are started at SOC=0.9 to avoid PyBaMM's internal
    "Maximum SoC" event firing immediately at the upper boundary.
    """

    def setUp(self):
        self.pv = pybamm.ParameterValues("ECM_Example")
        self.v_lo = float(self.pv["Lower voltage cut-off [V]"])
        self.v_hi = float(self.pv["Upper voltage cut-off [V]"])

    def _model(self):
        return pybamm.equivalent_circuit.Thevenin()

    def test_electrical_smoke(self):
        cell = _run_electrical(self._model(), self.pv, current=10.0)
        _assert_electrical_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_electrothermal_smoke(self):
        cell = _run_electrothermal(self._model(), self.pv, current=10.0)
        _assert_electrothermal_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_cosim_electrical_smoke(self):
        # Start at 0.9: PyBaMM's Maximum-SoC event fires if initial == upper boundary.
        cell = _run_cosim_electrical(
            self._model(), self.pv, current=10.0, initial_soc=0.9
        )
        _assert_electrical_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_cosim_electrothermal_smoke(self):
        cell = _run_cosim_electrothermal(
            self._model(), self.pv, current=10.0, initial_soc=0.9
        )
        _assert_electrothermal_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_electrical_soc_decreases(self):
        cell = _run_electrical(self._model(), self.pv, current=10.0, duration=60)
        self.assertLess(float(cell.outputs[2]), 1.0)

    def test_cutoff_values_match_parameter_set(self):
        cell = CellElectrical(model=self._model(), parameter_values=self.pv)
        self.assertAlmostEqual(cell._v_lower, self.v_lo)
        self.assertAlmostEqual(cell._v_upper, self.v_hi)


if __name__ == "__main__":
    unittest.main()
