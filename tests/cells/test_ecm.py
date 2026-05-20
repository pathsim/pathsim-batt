"""Tests for equivalent_circuit.Thevenin (ECM) model.

ECM_Example cutoffs: lower 3.2 V, upper 4.2 V, nominal capacity 100 A·h.

ECM's reversible heat term can be negative (entropy effect), so Q_dot is
not constrained to be non-negative in the smoke tests.

Co-simulation blocks are started at SOC=0.9 to avoid PyBaMM's internal
"Maximum SoC" event firing immediately at the upper boundary.
"""

import unittest

import pybamm

from pathsim_batt.cells import CellElectrical

from ._helpers import (
    assert_electrical_outputs,
    assert_electrothermal_outputs,
    run_cosim_electrical,
    run_cosim_electrothermal,
    run_electrical,
    run_electrothermal,
)


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
        cell = run_electrical(self._model(), self.pv, current=10.0)
        assert_electrical_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_electrothermal_smoke(self):
        cell = run_electrothermal(self._model(), self.pv, current=10.0)
        assert_electrothermal_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_cosim_electrical_smoke(self):
        # Start at 0.9: PyBaMM's Maximum-SoC event fires if initial == upper boundary.
        cell = run_cosim_electrical(
            self._model(), self.pv, current=10.0, initial_soc=0.9
        )
        assert_electrical_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_cosim_electrothermal_smoke(self):
        cell = run_cosim_electrothermal(
            self._model(), self.pv, current=10.0, initial_soc=0.9
        )
        assert_electrothermal_outputs(
            self, cell, self.v_lo, self.v_hi, check_q_dot_nonneg=False
        )

    def test_electrical_soc_decreases(self):
        cell = run_electrical(self._model(), self.pv, current=10.0, duration=60)
        self.assertLess(float(cell.outputs[2]), 1.0)

    def test_cutoff_values_match_parameter_set(self):
        cell = CellElectrical(model=self._model(), parameter_values=self.pv)
        self.assertAlmostEqual(cell._v_lower, self.v_lo)
        self.assertAlmostEqual(cell._v_upper, self.v_hi)

    def test_initial_soc_reflected_in_output(self):
        """Output SOC must match initial_soc after zero-current run of 1 s.

        Verifies that set_initial_state is correctly wired to the CasADi output.
        """
        cell = run_electrical(
            self._model(),
            self.pv,
            current=0.0,
            initial_soc=0.8,
            duration=1,
        )
        self.assertAlmostEqual(
            float(cell.outputs[2]),
            0.8,
            delta=0.01,
            msg="Output SOC does not reflect initial_soc=0.8",
        )

    def test_soc_decrease_magnitude(self):
        """Actual ΔSOC must match Coulombic prediction within 5 %.

        ECM's SoC state is purely Coulombic, so this should be tight.
        """
        current = 10.0
        duration = 360
        pv = self.pv
        q_nominal = float(pv["Nominal cell capacity [A.h]"])
        expected_delta = current * duration / 3600.0 / q_nominal
        cell = run_electrical(
            self._model(),
            pv,
            current=current,
            initial_soc=1.0,
            duration=duration,
        )
        actual_delta = 1.0 - float(cell.outputs[2])
        self.assertAlmostEqual(
            actual_delta,
            expected_delta,
            delta=expected_delta * 0.05,
            msg=(
                f"ΔSOC={actual_delta:.5f} deviates from Coulombic "
                f"prediction {expected_delta:.5f} by more than 5 %"
            ),
        )

    def test_q_dot_nonzero_during_discharge(self):
        """abs(Q_dot) must be non-zero while current flows.

        ECM Q_dot can be negative (reversible heat) but must not be zero
        during current flow.
        """
        cell = run_electrical(self._model(), self.pv, current=10.0, duration=10)
        self.assertGreater(
            abs(float(cell.outputs[1])),
            1e-6,
            "Q_dot is zero during discharge — heat generation not wired",
        )

    def test_tamb_affects_temperature(self):
        """A warmer ambient temperature must yield a higher output cell temperature."""
        cell_cold = run_electrothermal(
            self._model(), self.pv, current=10.0, t_amb=278.15, duration=120
        )
        cell_warm = run_electrothermal(
            self._model(), self.pv, current=10.0, t_amb=318.15, duration=120
        )
        T_cold = float(cell_cold.outputs[1])
        T_warm = float(cell_warm.outputs[1])
        self.assertLess(
            T_cold,
            T_warm,
            msg=(
                f"Warmer ambient did not yield higher cell temperature: "
                f"T_cold={T_cold:.2f} K, T_warm={T_warm:.2f} K"
            ),
        )


if __name__ == "__main__":
    unittest.main()
