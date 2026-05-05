import unittest

import numpy as np
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


class TestPorts(unittest.TestCase):
    def test_electrical_input_labels(self):
        self.assertEqual(CellElectrical.input_port_labels["I"], 0)
        self.assertEqual(CellElectrical.input_port_labels["T_cell"], 1)

    def test_electrical_output_labels(self):
        self.assertEqual(CellElectrical.output_port_labels["V"], 0)
        self.assertEqual(CellElectrical.output_port_labels["Q_heat"], 1)
        self.assertEqual(CellElectrical.output_port_labels["SOC"], 2)

    def test_electrothermal_input_labels(self):
        self.assertEqual(CellElectrothermal.input_port_labels["I"], 0)
        self.assertEqual(CellElectrothermal.input_port_labels["T_amb"], 1)

    def test_electrothermal_output_labels(self):
        self.assertEqual(CellElectrothermal.output_port_labels["V"], 0)
        self.assertEqual(CellElectrothermal.output_port_labels["T"], 1)
        self.assertEqual(CellElectrothermal.output_port_labels["Q_heat"], 2)
        self.assertEqual(CellElectrothermal.output_port_labels["SOC"], 3)

    def test_is_dynamic(self):
        self.assertTrue(hasattr(CellElectrical(), "initial_value"))
        self.assertTrue(hasattr(CellElectrothermal(), "initial_value"))

    def test_cosim_len_zero(self):
        cell_e = CellCoSimElectrical(dt=1.0)
        self.assertEqual(len(cell_e), 3)  # V, Q_heat, SOC
        cell_et = CellCoSimElectrothermal(dt=1.0)
        self.assertEqual(len(cell_et), 4)  # V, T, Q_heat, SOC

    def test_len_zero(self):
        cell_e = CellElectrical()
        cell_e.set_solver(ESDIRK43, None)
        self.assertEqual(len(cell_e), 3)  # V, Q_heat, SOC
        cell_et = CellElectrothermal()
        cell_et.set_solver(ESDIRK43, None)
        self.assertEqual(len(cell_et), 4)  # V, T, Q_heat, SOC

    def test_current_always_input(self):
        pv = pybamm.ParameterValues("Chen2020")
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls(parameter_values=pv)
            self.assertIsInstance(
                cell._parameter_values["Current function [A]"],
                pybamm.InputParameter,
            )

    def test_custom_soc(self):
        self.assertAlmostEqual(CellElectrical(initial_soc=0.5)._initial_soc, 0.5)
        self.assertAlmostEqual(CellElectrothermal(initial_soc=0.8)._initial_soc, 0.8)

    def test_initial_value_is_numpy_array(self):
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            self.assertIsInstance(cell.initial_value, np.ndarray)
            self.assertGreater(len(cell.initial_value), 1)

    def test_has_casadi_rhs(self):
        """CasADi RHS is compiled and callable at construction time."""
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            self.assertIsNotNone(cell._casadi_rhs)

    def test_state_size_equals_differential_states_only(self):
        """State must contain only differential (x) variables, not algebraic (z)."""
        import pybamm as pb

        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            # Rebuild the same model to get the expected x size from PyBaMM
            model = pb.lithium_ion.SPMe(options={"thermal": cell._thermal_option})
            pv = cell._parameter_values.copy()
            sim = pb.Simulation(
                model,
                parameter_values=pv,
                solver=pb.CasadiSolver(mode="safe"),
            )
            sim.build(
                initial_soc=cell._initial_soc,
                inputs={
                    "Current function [A]": 0.0,
                    "Ambient temperature [K]": 298.15,
                },
            )
            objs = sim.built_model.export_casadi_objects(
                ["Terminal voltage [V]"],
                input_parameter_order=[
                    "Current function [A]",
                    "Ambient temperature [K]",
                ],
            )
            expected_x_size = objs["x"].numel()
            self.assertEqual(len(cell.initial_value), expected_x_size)

    def test_jac_dyn_is_square(self):
        """jac_dyn must return a square (n×n) matrix where n is the state size."""
        for cls in (CellElectrical, CellElectrothermal):
            cell = cls()
            n = len(cell.initial_value)
            x = cell.initial_value
            u = np.array([0.0, 298.15])
            J = cell.jac_dyn(x, u, 0.0)
            self.assertEqual(J.shape, (n, n))

    def test_dfn_model_raises(self):
        """DFN models (DAE after discretisation) must raise NotImplementedError."""
        dfn = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
        with self.assertRaises(NotImplementedError):
            CellElectrical(model=dfn)

    def test_dfn_lumped_raises(self):
        """DFN with lumped thermal also has algebraic variables and must raise."""
        dfn = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
        with self.assertRaises(NotImplementedError):
            CellElectrothermal(model=dfn)

    def test_dfn_cosim_supported(self):
        """DFN is supported by co-simulation blocks."""
        dfn = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
        cell = CellCoSimElectrical(model=dfn, dt=1.0)
        self.assertEqual(len(cell), 3)

    def test_dfn_cosim_electrothermal_supported(self):
        """DFN with lumped thermal is supported by the electrothermal co-sim block."""
        dfn = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
        cell = CellCoSimElectrothermal(model=dfn, dt=1.0)
        self.assertEqual(len(cell), 4)


class TestElectrical(unittest.TestCase):
    """Integration tests for CellElectrical — PathSim integrates the PyBaMM ODE."""

    def _make_simulation(self, cell, current, T_cell):
        """Create a Simulation with the cell and constant inputs."""
        I_src = Constant(current)
        T_src = Constant(T_cell)
        return Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_cell"]),
            ],
            dt=1.0,
            Solver=ESDIRK43,
        )

    def setUp(self):
        self.cell = CellElectrical(initial_soc=1.0)
        self.sim = self._make_simulation(self.cell, 1.0, 298.15)

    def test_outputs_in_range(self):
        self.sim.run(1)
        self.assertGreater(self.cell.outputs[0], 3.0)  # V
        self.assertLess(self.cell.outputs[0], 4.3)
        self.assertGreaterEqual(self.cell.outputs[1], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[2], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[2], 1.0)

    def test_step_returns_success(self):
        self.sim.run(1)
        # Simulation completed without error
        self.assertIsNotNone(self.cell.outputs)

    def test_soc_decreases_on_discharge(self):
        self.sim.run(1)
        soc_0 = self.cell.outputs[2]
        self.sim.run(60)
        self.assertLess(self.cell.outputs[2], soc_0)

    def test_pathsim_state_advances(self):
        """The PathSim engine state changes after a step (not a stub)."""
        self.sim.run(1)
        state_before = self.cell.engine.state.copy()
        self.sim.run(2)
        self.assertFalse(np.allclose(self.cell.engine.state, state_before))

    def test_q_heat_nonzero_during_discharge(self):
        """Q_heat must be strictly positive when a discharge current flows.

        With thermal='isothermal' PyBaMM does not compute heat source terms,
        so Q_heat would be identically zero — this test guards against that.
        """
        cell = CellElectrical(initial_soc=1.0)
        I_src = Constant(5.0)  # 1C-ish discharge
        T_src = Constant(298.15)
        sim = Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_cell"]),
            ],
            dt=10.0,
            Solver=ESDIRK43,
        )
        sim.run(60)
        self.assertGreater(
            cell.outputs[1],
            0.0,
            "Q_heat is zero — thermal model may not compute heat sources",
        )

    def test_temperature_input_affects_voltage(self):
        """T_cell must actually influence the electrochemistry.

        Butler-Volmer kinetics are temperature-dependent, so discharging at
        a significantly higher temperature must yield a measurably different
        terminal voltage after the same duration.
        """

        def _run_and_get_voltage(T_cell):
            cell = CellElectrical(initial_soc=1.0)
            I_src = Constant(5.0)
            T_src = Constant(T_cell)
            sim = Simulation(
                blocks=[I_src, T_src, cell],
                connections=[
                    Connection(I_src, cell["I"]),
                    Connection(T_src, cell["T_cell"]),
                ],
                dt=10.0,
                Solver=ESDIRK43,
            )
            sim.run(300)
            return cell.outputs[0]  # terminal voltage [V]

        V_cold = _run_and_get_voltage(278.15)  # 5 °C
        V_hot = _run_and_get_voltage(318.15)  # 45 °C
        self.assertNotAlmostEqual(
            V_cold,
            V_hot,
            places=3,
            msg="T_cell input has no effect on terminal voltage",
        )


class TestElectrothermal(unittest.TestCase):
    """Integration tests for CellElectrothermal — PathSim integrates the PyBaMM ODE."""

    def _make_simulation(self, cell, current, T_amb):
        """Create a Simulation with the cell and constant inputs."""
        I_src = Constant(current)
        T_src = Constant(T_amb)
        return Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_amb"]),
            ],
            dt=1.0,
            Solver=ESDIRK43,
        )

    def setUp(self):
        self.cell = CellElectrothermal(initial_soc=1.0)
        self.sim = self._make_simulation(self.cell, 1.0, 298.15)

    def test_outputs_in_range(self):
        self.sim.run(1)
        self.assertGreater(self.cell.outputs[0], 3.0)  # V
        self.assertLess(self.cell.outputs[0], 4.3)
        self.assertGreater(self.cell.outputs[1], 250.0)  # T
        self.assertLess(self.cell.outputs[1], 400.0)
        self.assertGreaterEqual(self.cell.outputs[2], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[3], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[3], 1.0)

    def test_step_returns_success(self):
        self.sim.run(1)
        # Simulation completed without error
        self.assertIsNotNone(self.cell.outputs)

    def test_soc_decreases_on_discharge(self):
        self.sim.run(1)
        soc_0 = self.cell.outputs[3]
        self.sim.run(60)
        self.assertLess(self.cell.outputs[3], soc_0)

    def test_pathsim_state_advances(self):
        """The PathSim engine state changes after a step (not a stub)."""
        self.sim.run(1)
        state_before = self.cell.engine.state.copy()
        self.sim.run(2)
        self.assertFalse(np.allclose(self.cell.engine.state, state_before))

    def test_q_heat_nonzero_during_discharge(self):
        """Q_heat must be strictly positive when a discharge current flows."""
        cell = CellElectrothermal(initial_soc=1.0)
        I_src = Constant(5.0)
        T_src = Constant(298.15)
        sim = Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_amb"]),
            ],
            dt=10.0,
            Solver=ESDIRK43,
        )
        sim.run(60)
        self.assertGreater(
            cell.outputs[2],
            0.0,
            "Q_heat is zero — thermal model may not compute heat sources",
        )

    def test_tamb_input_affects_cell_temperature(self):
        """T_amb must influence the output cell temperature.

        With a lower ambient temperature the cell should run cooler after
        the same discharge duration.
        """

        def _run_and_get_T_cell(T_amb):
            cell = CellElectrothermal(initial_soc=1.0)
            I_src = Constant(5.0)
            T_src = Constant(T_amb)
            sim = Simulation(
                blocks=[I_src, T_src, cell],
                connections=[
                    Connection(I_src, cell["I"]),
                    Connection(T_src, cell["T_amb"]),
                ],
                dt=10.0,
                Solver=ESDIRK43,
            )
            sim.run(300)
            return cell.outputs[1]  # cell temperature [K]

        T_cell_cold_amb = _run_and_get_T_cell(278.15)  # 5 °C ambient
        T_cell_hot_amb = _run_and_get_T_cell(318.15)  # 45 °C ambient
        self.assertLess(
            T_cell_cold_amb,
            T_cell_hot_amb,
            msg="T_amb input has no effect on output cell temperature",
        )


class TestCoSimulationElectrical(unittest.TestCase):
    """Integration tests for CellCoSimElectrical — PyBaMM performs the stepping."""

    def _make_simulation(self, cell, current, T_cell):
        I_src = Constant(current)
        T_src = Constant(T_cell)
        return Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_cell"]),
            ],
            dt=0.5,
            Solver=ESDIRK43,
        )

    def setUp(self):
        self.cell = CellCoSimElectrical(initial_soc=1.0, dt=1.0)
        self.sim = self._make_simulation(self.cell, 1.0, 298.15)

    def test_outputs_in_range(self):
        self.sim.run(2)
        self.assertGreater(self.cell.outputs[0], 2.0)  # V
        self.assertLess(self.cell.outputs[0], 5.0)
        self.assertGreaterEqual(self.cell.outputs[1], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[2], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[2], 1.0)

    def test_soc_decreases_on_discharge(self):
        self.sim.run(2)
        soc_0 = self.cell.outputs[2]
        self.sim.run(60)
        self.assertLess(self.cell.outputs[2], soc_0)

    def test_discrete_step_fires_and_voltage_physical(self):
        """_discrete_step must be called and produce a physical terminal voltage."""
        self.sim.run(2)
        # After at least one macro-step the voltage must be in a physical range.
        self.assertGreater(self.cell.outputs[0], 3.0)
        self.assertLess(self.cell.outputs[0], 4.3)

    def test_dfn_step_outputs_physical(self):
        """DFN-backed co-sim cell must produce physical outputs after stepping."""
        dfn = pybamm.lithium_ion.DFN(options={"thermal": "isothermal"})
        cell = CellCoSimElectrical(model=dfn, dt=1.0)
        sim = self._make_simulation(cell, 1.0, 298.15)
        sim.run(2)
        self.assertGreater(cell.outputs[0], 3.0)  # V
        self.assertLess(cell.outputs[0], 4.3)
        self.assertGreater(cell.outputs[2], 0.0)  # SOC
        self.assertLessEqual(cell.outputs[2], 1.0)

    def test_q_heat_nonzero_during_discharge(self):
        """Q_heat must be strictly positive when a discharge current flows."""
        cell = CellCoSimElectrical(initial_soc=1.0, dt=10.0)
        I_src = Constant(5.0)
        T_src = Constant(298.15)
        sim = Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_cell"]),
            ],
            dt=5.0,
            Solver=ESDIRK43,
        )
        sim.run(60)
        self.assertGreater(
            cell.outputs[1],
            0.0,
            "Q_heat is zero — thermal model may not compute heat sources",
        )

    def test_temperature_input_affects_voltage(self):
        """T_cell must actually influence the electrochemistry."""

        def _run_and_get_voltage(T_cell):
            cell = CellCoSimElectrical(initial_soc=1.0, dt=10.0)
            I_src = Constant(5.0)
            T_src = Constant(T_cell)
            sim = Simulation(
                blocks=[I_src, T_src, cell],
                connections=[
                    Connection(I_src, cell["I"]),
                    Connection(T_src, cell["T_cell"]),
                ],
                dt=5.0,
                Solver=ESDIRK43,
            )
            sim.run(300)
            return cell.outputs[0]  # terminal voltage [V]

        V_cold = _run_and_get_voltage(278.15)  # 5 °C
        V_hot = _run_and_get_voltage(318.15)  # 45 °C
        self.assertNotAlmostEqual(
            float(V_cold),
            float(V_hot),
            places=3,
            msg="T_cell input has no effect on terminal voltage",
        )


class TestCoSimulationElectrothermal(unittest.TestCase):
    """Integration tests for CellCoSimElectrothermal — PyBaMM performs the stepping."""

    def _make_simulation(self, cell, current, T_amb):
        I_src = Constant(current)
        T_src = Constant(T_amb)
        return Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_amb"]),
            ],
            dt=0.5,
            Solver=ESDIRK43,
        )

    def setUp(self):
        self.cell = CellCoSimElectrothermal(initial_soc=1.0, dt=1.0)
        self.sim = self._make_simulation(self.cell, 1.0, 298.15)

    def test_outputs_in_range(self):
        self.sim.run(2)
        self.assertGreater(self.cell.outputs[0], 2.0)  # V
        self.assertLess(self.cell.outputs[0], 5.0)
        self.assertGreater(self.cell.outputs[1], 250.0)  # T
        self.assertLess(self.cell.outputs[1], 400.0)
        self.assertGreaterEqual(self.cell.outputs[2], 0.0)  # Q_heat
        self.assertGreater(self.cell.outputs[3], 0.0)  # SOC
        self.assertLessEqual(self.cell.outputs[3], 1.0)

    def test_soc_decreases_on_discharge(self):
        self.sim.run(2)
        soc_0 = self.cell.outputs[3]
        self.sim.run(60)
        self.assertLess(self.cell.outputs[3], soc_0)

    def test_discrete_step_fires_and_voltage_physical(self):
        """_discrete_step must be called and produce a physical terminal voltage."""
        self.sim.run(2)
        self.assertGreater(self.cell.outputs[0], 3.0)
        self.assertLess(self.cell.outputs[0], 4.3)

    def test_dfn_step_outputs_physical(self):
        """DFN-backed electrothermal co-sim cell must produce physical outputs."""
        dfn = pybamm.lithium_ion.DFN(options={"thermal": "lumped"})
        cell = CellCoSimElectrothermal(model=dfn, dt=1.0)
        sim = self._make_simulation(cell, 1.0, 298.15)
        sim.run(2)
        self.assertGreater(cell.outputs[0], 3.0)  # V
        self.assertLess(cell.outputs[0], 4.3)
        self.assertGreater(cell.outputs[1], 250.0)  # T
        self.assertLess(cell.outputs[1], 400.0)
        self.assertGreater(cell.outputs[3], 0.0)  # SOC
        self.assertLessEqual(cell.outputs[3], 1.0)

    def test_q_heat_nonzero_during_discharge(self):
        """Q_heat must be strictly positive when a discharge current flows."""
        cell = CellCoSimElectrothermal(initial_soc=1.0, dt=10.0)
        I_src = Constant(5.0)
        T_src = Constant(298.15)
        sim = Simulation(
            blocks=[I_src, T_src, cell],
            connections=[
                Connection(I_src, cell["I"]),
                Connection(T_src, cell["T_amb"]),
            ],
            dt=5.0,
            Solver=ESDIRK43,
        )
        sim.run(60)
        self.assertGreater(
            cell.outputs[2],
            0.0,
            "Q_heat is zero — thermal model may not compute heat sources",
        )

    def test_tamb_input_affects_cell_temperature(self):
        """T_amb must influence the output cell temperature."""

        def _run_and_get_T_cell(T_amb):
            cell = CellCoSimElectrothermal(initial_soc=1.0, dt=10.0)
            I_src = Constant(5.0)
            T_src = Constant(T_amb)
            sim = Simulation(
                blocks=[I_src, T_src, cell],
                connections=[
                    Connection(I_src, cell["I"]),
                    Connection(T_src, cell["T_amb"]),
                ],
                dt=5.0,
                Solver=ESDIRK43,
            )
            sim.run(300)
            return cell.outputs[1]  # cell temperature [K]

        T_cold = _run_and_get_T_cell(278.15)
        T_hot = _run_and_get_T_cell(318.15)
        self.assertLess(
            T_cold, T_hot, msg="T_amb input has no effect on output cell temperature"
        )


if __name__ == "__main__":
    unittest.main()
