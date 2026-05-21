
<p align="center">
  <img src="https://raw.githubusercontent.com/pathsim/pathsim-batt/master/docs/source/logos/batt_logo.png" width="300" alt="PathSim-Batt Logo" />
</p>

<p align="center">
  <strong>Battery simulation blocks for PathSim</strong>
</p>

<p align="center">
  <a href="https://pypi.org/project/pathsim-batt/"><img src="https://img.shields.io/pypi/v/pathsim-batt" alt="PyPI"></a>
  <img src="https://img.shields.io/github/license/pathsim/pathsim-batt" alt="License">
</p>

<p align="center">
  <a href="https://docs.pathsim.org/batt">Documentation</a> &bull;
  <a href="https://pathsim.org">PathSim Homepage</a> &bull;
  <a href="https://github.com/pathsim/pathsim-batt">GitHub</a>
</p>

---

PathSim-Batt extends the [PathSim](https://github.com/pathsim/pathsim) simulation framework with battery cell blocks backed by [PyBaMM](https://pybamm.org). All blocks follow the standard PathSim interface and can be wired into any simulation diagram.

## Install

```bash
pip install pathsim-batt
```

## Quick start

```python
import pybamm
from pathsim import Connection, Simulation
from pathsim.blocks import Constant
from pathsim.solvers import ESDIRK43
from pathsim_batt import CellElectrothermal

cell = CellElectrothermal(initial_soc=1.0)   # defaults: SPMe + Chen2020
I_src = Constant(5.0)                         # 5 A discharge
T_src = Constant(298.15)                      # 25 °C ambient

sim = Simulation(
    blocks=[I_src, T_src, cell],
    connections=[Connection(I_src, cell["I"]), Connection(T_src, cell["T_amb"])],
    dt=1.0,
    Solver=ESDIRK43,
)
sim.run(3600)
print(f"V = {cell.outputs[0]:.3f} V  T = {cell.outputs[1]:.1f} K  SOC = {cell.outputs[3]:.3f}")
```

## Choosing a block

Two decisions determine the right block: **thermal ownership** and **integration strategy**.

| Block | Thermal | Strategy | Use when |
|---|---|---|---|
| `CellElectrothermal` | PyBaMM (internal) | Monolithic ODE | Single cell, coupled electro-thermal, ODE model |
| `CellElectrical` + `LumpedThermal` | PathSim (external) | Monolithic ODE | Pack-level, custom cooling, ODE model |
| `CellCoSimElectrothermal` | PyBaMM (internal) | Co-simulation | DAE models (DFN, lead_acid.Full), mixed solvers |
| `CellCoSimElectrical` + `LumpedThermal` | PathSim (external) | Co-simulation | DAE models with external thermal network |

`LumpedThermal` is a single-node thermal block (`mass`, `Cp`, `UA`, `T0`) that receives `Q_dot` from a `CellElectrical` block and feeds back cell temperature.

## PyBaMM model compatibility

Thermal sub-model and heat-source options are injected automatically — pass the bare model class with no `options=`.

| PyBaMM model | Default parameter set | `CellElectrical` | `CellElectrothermal` | `CellCoSimElectrical` | `CellCoSimElectrothermal` |
|---|---|:---:|:---:|:---:|:---:|
| `lithium_ion.SPM` | `Chen2020` | ✅ | ✅ | ✅ | ✅ |
| `lithium_ion.SPMe` | `Chen2020` | ✅ | ✅ | ✅ | ✅ |
| `lithium_ion.DFN` | `Chen2020` | ❌ DAE | ❌ DAE | ✅ | ✅ |
| `lead_acid.LOQS` | `Sulzer2019` | ✅ | ✅ | ✅ ¹ | ✅ ¹ |
| `lead_acid.Full` | `Sulzer2019` | ❌ DAE | ❌ DAE | ✅ | ✅ |
| `equivalent_circuit.Thevenin` | `ECM_Example` | ✅ | ✅ | ✅ ² | ✅ ² |

¹ Pass `pybamm_solver=pybamm.CasadiSolver(mode="safe")` — the default `IDAKLUSolver` requires a Jacobian that ODE models do not provide.

² `initial_soc=1.0` fails because PyBaMM requires event values to be strictly positive at `t=0`; the "Maximum SoC" event is zero exactly at full charge. Any value below 1.0 (e.g. `initial_soc=0.99`) works.

```python
import pybamm
from pathsim_batt import CellElectrothermal, CellCoSimElectrical

# Custom chemistry / parameter set
cell = CellElectrothermal(
    model=pybamm.lithium_ion.SPMe(),
    parameter_values=pybamm.ParameterValues("Mohtat2020"),
)

# Lead-acid via co-simulation (DAE model)
cell = CellCoSimElectrical(
    model=pybamm.lead_acid.Full(),
    parameter_values=pybamm.ParameterValues("Sulzer2019"),
    dt=1.0,
)

# Equivalent circuit model
cell = CellElectrical(
    model=pybamm.equivalent_circuit.Thevenin(),
    parameter_values=pybamm.ParameterValues("ECM_Example"),
    initial_soc=0.9,
)
```

## License

MIT
