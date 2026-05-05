
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

PathSim-Batt extends the [PathSim](https://github.com/pathsim/pathsim) simulation framework with battery cell blocks using [PyBaMM](https://pybamm.org) as the electrochemical backend. All blocks follow the standard PathSim block interface and can be connected into simulation diagrams.

## Install

```bash
pip install pathsim-batt
```

## Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| `CellElectrothermal` | Coupled electrical + thermal cell (PathSim integrates PyBaMM ODE incl. temperature) | `model`, `parameter_values`, `initial_soc` |
| `CellElectrical` | Electrical only, isothermal; wire to `LumpedThermal` for external thermal coupling | `model`, `parameter_values`, `initial_soc` |
| `CellCoSimElectrothermal` | Coupled electrical + thermal co-simulation cell (PyBaMM steps internally) | `model`, `parameter_values`, `initial_soc`, `dt` |
| `CellCoSimElectrical` | Electrical co-simulation cell for external thermal coupling | `model`, `parameter_values`, `initial_soc`, `dt` |
| `LumpedThermal` | Single-node thermal model for external thermal coupling | `mass`, `Cp`, `UA`, `T0` |

## PyBaMM integration

The cell blocks wrap [PyBaMM](https://pybamm.org) models behind the PathSim block interface.

- `CellElectrothermal` / `CellElectrical` use PathSim monolithic integration (`DynamicalSystem`) and exported CasADi ODE right-hand sides.
- `CellCoSimElectrothermal` / `CellCoSimElectrical` use periodic co-simulation (`Wrapper`) and call `pybamm.Simulation.step()` internally.

Only models that yield a **pure ODE** after discretisation are supported by the monolithic blocks (`CellElectrothermal`, `CellElectrical`) — currently SPMe and SPM. Models such as DFN that produce a DAE system (algebraic variables) will raise `NotImplementedError` there.

For DAE models (e.g. DFN), use the co-simulation blocks (`CellCoSimElectrothermal`, `CellCoSimElectrical`).

- **ODE-type PyBaMM models** (SPMe, SPM) can be injected via the `model` parameter
- **Any parameter set** can be used via `parameter_values` (defaults to `Chen2020`)
- **Immediate initialisation** — the PyBaMM model is discretised during block construction

```python
import pybamm

model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
params = pybamm.ParameterValues("Mohtat2020")
cell   = CellElectrothermal(model=model, parameter_values=params)

# DAE example (DFN): use co-simulation mode
dfn_cell = CellCoSimElectrothermal(
  model=pybamm.lithium_ion.DFN(options={"thermal": "lumped"}),
  parameter_values=params,
  dt=0.1,
)
```

## Thermal coupling modes

| Mode | Block | Owns cell temperature | Use when |
|---|---|---|---|
| Internal | `CellElectrothermal` | PyBaMM | Single-cell simulations, quick setup |
| External | `CellElectrical` + `LumpedThermal` | PathSim | Multi-cell packs, custom cooling models |
| Co-sim internal | `CellCoSimElectrothermal` | PyBaMM | DAE models (e.g. DFN), mixed-solver workflows |
| Co-sim external | `CellCoSimElectrical` + `LumpedThermal` | PathSim | DAE models with external thermal network |

## License

MIT
