"""Tests for lead_acid model families: LOQS (ODE) and Full (DAE).

Block / model matrix covered
-----------------------------
lead_acid.LOQS  — ODE → all 4 blocks
lead_acid.Full  — DAE → CoSim blocks only
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

from ._helpers import (
    assert_electrical_outputs,
    assert_electrothermal_outputs,
    run_cosim_electrical,
    run_cosim_electrothermal,
    run_electrical,
    run_electrothermal,
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

    def _model(self):
        return pybamm.lead_acid.LOQS()

    def test_electrical_smoke(self):
        cell = run_electrical(self._model(), self.pv, current=17.0)
        assert_electrical_outputs(self, cell, self.v_lo, self.v_hi)

    def test_electrothermal_smoke(self):
        cell = run_electrothermal(self._model(), self.pv, current=17.0)
        assert_electrothermal_outputs(self, cell, self.v_lo, self.v_hi)

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
        assert_electrical_outputs(self, cell, self.v_lo, self.v_hi)

    def test_cosim_electrothermal_smoke(self):
        solver = pybamm.CasadiSolver(mode="safe")
        cell = CellCoSimElectrothermal(
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
                Connection(T_src, cell["T_amb"]),
            ],
            dt=0.5,
            Solver=ESDIRK43,
        )
        sim.run(2)
        assert_electrothermal_outputs(self, cell, self.v_lo, self.v_hi)

    def test_electrical_soc_decreases(self):
        """SOC must decrease under discharge current."""
        cell = run_electrical(self._model(), self.pv, current=17.0, duration=60)
        self.assertLess(float(cell.outputs[2]), 1.0)

    def test_cutoff_values_match_parameter_set(self):
        cell = CellElectrical(model=self._model(), parameter_values=self.pv)
        self.assertAlmostEqual(cell._v_lower, self.v_lo)
        self.assertAlmostEqual(cell._v_upper, self.v_hi)

    def test_q_dot_nonzero_during_discharge(self):
        """Q_dot must be strictly positive during discharge (isothermal LOQS).

        The block automatically injects the heat-source calculation flag into
        isothermal models that lack it, so no manual option is needed.
        """
        cell = run_electrical(self._model(), self.pv, current=17.0, duration=60)
        self.assertGreater(
            float(cell.outputs[1]),
            0.0,
            "Q_dot is zero — thermal model may not compute heat sources",
        )

    def test_tamb_affects_temperature(self):
        """A warmer ambient temperature must yield a higher output cell temperature."""
        solver = pybamm.CasadiSolver(mode="safe")
        cell_cold = run_electrothermal(
            self._model(),
            self.pv,
            current=17.0,
            t_amb=278.15,
            duration=120,
            pybamm_solver=solver,
        )
        cell_warm = run_electrothermal(
            self._model(),
            self.pv,
            current=17.0,
            t_amb=318.15,
            duration=120,
            pybamm_solver=solver,
        )
        T_cold = float(cell_cold.outputs[1])
        T_warm = float(cell_warm.outputs[1])
        self.assertLess(
            T_cold,
            T_warm,
            msg=(
                f"Warmer ambient (318.15 K) did not yield higher cell temperature: "
                f"T_cold={T_cold:.2f} K, T_warm={T_warm:.2f} K"
            ),
        )

    def test_soc_scale_factor(self):
        """SOC must be well below 1.0 after sustained discharge.

        Guards against PyBaMM's percentage-form 'State of Charge' being
        misidentified as a fraction, which would clamp all values to 1.0.
        """
        cell = run_electrical(self._model(), self.pv, current=17.0, duration=360)
        self.assertLess(
            float(cell.outputs[2]),
            0.95,
            "SOC did not decrease — possible percentage/fraction scale error",
        )


# ---------------------------------------------------------------------------
# lead_acid.Full  (DAE — co-simulation only)
# ---------------------------------------------------------------------------


class TestLeadAcidFull(unittest.TestCase):
    """lead_acid.Full with Sulzer2019 parameters (DAE model — co-sim only)."""

    def setUp(self):
        self.pv = pybamm.ParameterValues("Sulzer2019")
        self.v_lo = float(self.pv["Lower voltage cut-off [V]"])
        self.v_hi = float(self.pv["Upper voltage cut-off [V]"])

    def _model(self):
        return pybamm.lead_acid.Full()

    def test_monolithic_electrical_raises(self):
        """Full is a DAE — CellElectrical must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            CellElectrical(model=self._model(), parameter_values=self.pv)

    def test_monolithic_electrothermal_raises(self):
        """Full is a DAE — CellElectrothermal must raise NotImplementedError."""
        with self.assertRaises(NotImplementedError):
            CellElectrothermal(model=self._model(), parameter_values=self.pv)

    def test_cosim_electrical_smoke(self):
        cell = run_cosim_electrical(self._model(), self.pv, current=17.0)
        assert_electrical_outputs(self, cell, self.v_lo, self.v_hi)

    def test_cosim_electrothermal_smoke(self):
        cell = run_cosim_electrothermal(self._model(), self.pv, current=17.0)
        assert_electrothermal_outputs(self, cell, self.v_lo, self.v_hi)


if __name__ == "__main__":
    unittest.main()
