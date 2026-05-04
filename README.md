
<p align="center">
  <strong>Battery simulation blocks for PathSim</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/github/license/pathsim/pathsim-batt" alt="License">
</p>

<p align="center">
  <a href="https://docs.pathsim.org/batt">Documentation</a> &bull;
  <a href="https://pathsim.org">PathSim Homepage</a> &bull;
  <a href="https://github.com/pathsim/pathsim-batt">GitHub</a>
</p>

---

PathSim-Batt extends the [PathSim](https://github.com/pathsim/pathsim) simulation framework with battery cell blocks using [PyBaMM](https://pybamm.org) as the electrochemical backend. All blocks follow the standard PathSim block interface and can be connected into simulation diagrams.

## Blocks

| Block | Description | Key Parameters |
|-------|-------------|----------------|
| `CellElectrothermal` | Coupled electrical + thermal cell (PathSim integrates PyBaMM ODE incl. temperature) | `model`, `parameter_values`, `initial_soc` |
| `CellElectrical` | Electrical only, isothermal; wire to `LumpedThermal` for external thermal coupling | `model`, `parameter_values`, `initial_soc` |
| `LumpedThermal` | Single-node thermal model for external thermal coupling | `mass`, `Cp`, `UA`, `T0` |

`Cell` is an alias for `CellElectrothermal`.

## PyBaMM integration

The cell blocks wrap [PyBaMM](https://pybamm.org) models behind the PathSim block interface. PyBaMM discretises the electrochemistry equations at construction time, then PathSim's numerical integrator advances the state vector using the exported ODE right-hand side.

Only models that yield a **pure ODE** after discretisation are supported — currently SPMe and SPM. Models such as DFN that produce a DAE system (algebraic variables) will raise `NotImplementedError` at construction time.

- **ODE-type PyBaMM models** (SPMe, SPM) can be injected via the `model` parameter
- **Any parameter set** can be used via `parameter_values` (defaults to `Chen2020`)
- **Immediate initialisation** — the PyBaMM model is discretised during block construction

```python
import pybamm

model  = pybamm.lithium_ion.SPMe(options={"thermal": "lumped"})
params = pybamm.ParameterValues("Mohtat2020")
cell   = CellElectrothermal(model=model, parameter_values=params)
```

## Thermal coupling modes

| Mode | Block | Owns cell temperature | Use when |
|---|---|---|---|
| Internal | `CellElectrothermal` | PyBaMM | Single-cell simulations, quick setup |
| External | `CellElectrical` + `LumpedThermal` | PathSim | Multi-cell packs, custom cooling models |

## Install

```bash
pip install pathsim-batt
```

## License

MIT
