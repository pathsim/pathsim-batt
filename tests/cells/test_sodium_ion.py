"""Smoke tests for sodium_ion.BasicDFN.

sodium_ion.BasicDFN is incompatible with all four existing block classes.
BasicDFN is a DAE model, so monolithic blocks raise ``NotImplementedError``.
Co-simulation blocks fail with ``ValueError`` because BasicDFN exports
neither heating nor temperature variables, which the block classes require.
Tests document these boundaries.
"""

import unittest

import pybamm

from pathsim_batt.cells import (
    CellCoSimElectrical,
    CellCoSimElectrothermal,
    CellElectrical,
    CellElectrothermal,
)


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


if __name__ == "__main__":
    unittest.main()
