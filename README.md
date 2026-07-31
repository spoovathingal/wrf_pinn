# WRF PINN

`wrf-pinn` is a PyTorch physics-informed neural network for reconstructing a
four-dimensional atmospheric flow field from:

- dense flow-field data,
- sparse velocity sensor data,
- wall-boundary locations, and
- PDE collocation points sampled inside a scaled Cartesian domain.

The current model maps

```text
(x, y, z, t) -> (u, v, w, rho)
```

All coordinates and state values must be scaled before they enter this package.
The code does not require a particular numerical range such as `[-1, 1]`; it
requires every data source and the collocation domain to use the same scaling
convention.

## Current physics

The implemented residual system is a reduced, zero-forcing Cartesian transport
model. It contains mass continuity and three primitive-variable momentum
residuals:

```text
mass:
rho_t + (rho u)_x + (rho v)_y + (rho w)_z

x momentum:
u_t + u u_x + v u_y + w u_z

y momentum:
v_t + u v_x + v v_y + w v_z

z momentum:
w_t + u w_x + v w_y + w w_z
```

The current equations do not include pressure gradients, gravity, Coriolis
terms, temperature, moisture, viscosity, or turbulence closure. They are a
minimal PINN scaffold rather than a complete WRF equation set.

When scaled coordinates and outputs are used, these residuals operate in that
scaled coordinate system. Physical-equation residuals require a consistent
nondimensionalization or derivative chain-rule factors from the preprocessing
scales.

## Training objective

`train_pinn` can combine four loss groups:

1. PDE residual loss at sampled collocation points.
2. No-slip boundary residuals at supplied wall locations.
3. Sparse sensor-data error for `u`, `v`, and `w`.
4. Dense flow-field error for `u`, `v`, `w`, and `rho`.

Each group can be activated and weighted through `ConditionsConfig`. The
default configuration enables PDE and flow-field losses and disables boundary
and sensor losses.

## Source layout

```text
src/wrf_pinn/
    config/
        boundary_data.py
        conditions.py
        domain.py
        flow_field_data.py
        model.py
        physics.py
        sampling.py
        sensors_data.py
        training.py
        validation.py

    data/
        boundary.py
        flow_field.py
        sensors.py

    models/
        mlp.py

    physics/
        residuals_boundary.py
        residuals_pde.py

    sampling/
        collocation.py

    training/
        losses.py
        train_pinn.py
```

### Configuration

- `config/domain.py` describes the scaled Cartesian bounds used to sample PDE
  collocation points.
- `config/model.py` configures the coordinate-to-state MLP.
- `config/physics.py` defines the active reduced state and documents supported
  physics assumptions.
- `config/conditions.py` activates and weights the four loss groups.
- `config/sampling.py` configures collocation and dataset sampling.
- `config/training.py` configures epochs, optimizer, device, logging, and
  gradient clipping.
- The data configuration modules define CSV paths and column order.

### Runtime modules

- `data/` reads scaled CSV data into NumPy-backed data objects.
- `sampling/collocation.py` generates random-uniform, Latin-hypercube, or grid
  collocation coordinates.
- `models/mlp.py` implements the PyTorch MLP.
- `physics/` evaluates PDE and no-slip residuals.
- `training/losses.py` assembles weighted mean-squared losses.
- `training/train_pinn.py` performs optimization and records loss history.
- `evaluation/` writes result files and streams a live training monitor
  (see [Live training monitor](#live-training-monitor)).

## Data formats

### Dense flow field

The default dense flow-field CSV schema is:

```text
x,y,z,t,u,v,w,rho
```

All eight columns are required. Rows are treated as independent coordinate-state
samples; the reader does not require a particular flattened-grid ordering.

### Sensor data

The default sparse sensor CSV schema is:

```text
x,y,z,t,u,v,w
```

### Wall-boundary locations

The wall CSV contains scaled coordinates:

```text
x,y,z,t
```

The current boundary residual targets numerical zero for `u`, `v`, and `w`.
If physical zero velocity does not map to numerical zero under the chosen
output scaling, the boundary residual must be adjusted to use the correctly
scaled target.

## Uniform-flow smoke test

The included entry point trains on:

```text
../synthetic_data_testing/uniform_flow_outputs/normalized_train.csv
```

It:

- reads the scaled dense flow field,
- infers collocation bounds from its coordinate columns,
- samples 2,048 Latin-hypercube PDE points,
- enables PDE and flow-field losses,
- trains the default MLP for 100 epochs, and
- prints component losses during training.

Run it from this directory:

```bash
cd /Users/saviopoovathingal/Desktop/PINN/AtmosphereModel/wrf_pinn
PYTHONPATH=src ../.venv/bin/python train_uniform_flow.py
```

The script prints progress in this form:

```text
epoch=1/100 | total=... | pde=... | boundary=... |
sensor_data=... | flow_field_data=...
```

It also prints the initial and final total loss after training.

## Programmatic use

The main training interface is:

```python
history = train_pinn(
    model=model,
    domain=domain,
    flow_field_data=flow_field_data,
    sensor_data=sensor_data,
    boundary_points=boundary_points,
    conditions=conditions,
    sampling=sampling,
    training=training,
)
```

Only inputs corresponding to active conditions are required. A domain is
required whenever PDE loss is active.

`train_pinn` mutates the supplied model and returns a `TrainingHistory`
containing total and component losses for every epoch.

## Live training monitor

`LiveTrainingMonitor` (in `wrf_pinn.evaluation`) streams training progress to
disk *as the run proceeds*, so long runs can be watched without waiting for
completion. It is fully opt-in: `train_pinn` takes an optional `monitor`
argument that defaults to `None`, so existing scripts are unaffected.

On the logging cadence (`training.log_every`) it writes, into its `output_dir`:

```text
loss_history.csv   appended each interval (epoch, total, <components>)
loss_curves.png    re-rendered each interval (open in any viewer; it refreshes)
loss_curves.gif    time-lapse rebuilt from per-interval frames in frames/
```

It plots the total plus whichever component losses are active (`pde`,
`boundary`, `sensor_data`, `flow_field_data`); components that stay zero for the
whole run are dropped from the plot. All rendering is exception-isolated, so a
plotting failure logs a warning and never interrupts training.

### Enabling it

The monitor is created in the driver script and passed to `train_pinn`:

```python
from wrf_pinn.evaluation import LiveTrainingMonitor
from wrf_pinn.training.train_pinn import COMPONENT_LOSS_NAMES, train_pinn

monitor = LiveTrainingMonitor(
    output_dir=results_dir / "live",
    component_names=COMPONENT_LOSS_NAMES,
    total_epochs=training.epochs,
)
history = train_pinn(model=model, ..., monitor=monitor)
```

Without the `monitor=` argument, training runs exactly as before with no live
output. `plt`/`matplotlib` is required only when a monitor is used.

## Results output

After a run, `write_training_results` (also in `wrf_pinn.evaluation`) saves an
organized set of files a separate plotting script can read:

```text
loss_history.csv   per-epoch total and component losses
predictions.csv    coordinates with predicted and true state per point
run_metadata.json  config, dataset shape, final losses, and a file manifest
```

`predict_flow_field` evaluates a trained model to plain NumPy arrays for the
predictions file.

## Installation and dependencies

The package uses a `src/` layout and is described by `pyproject.toml`. Runtime
code requires:

```text
Python 3.10+
NumPy
PyTorch
```

The project metadata should declare NumPy and PyTorch before relying on a clean
`pip install -e .` installation. Until then, the command above uses the existing
project virtual environment and sets `PYTHONPATH=src` explicitly.

## Testing

A pytest suite under `tests/` verifies the data-reading path, the loss-mode
conditions, and the end-to-end read-to-model pipeline. Tests are self-contained
(each writes its own input data under a temporary directory that pytest removes
after the run, so nothing persists) and use distinguishable fixture data
(coordinates are asymmetric and each value column is a distinct function of the
coordinates: `u=x, v=y, w=z, rho=t`) so a shuffled row, dropped column, or
swapped axis is detected rather than passing silently.

Install the test dependency and run:

```bash
pip install -e ".[test]"     # or: pip install pytest
pytest -v                    # per-test pass/fail
pytest -v -s                 # also print the actual verified values
```

`pytest -v` gives the standard pass/fail list. Adding `-s` reveals the values
each test confirmed (the read-back coordinate and target arrays), which is hidden
by default to keep output clean; on failure, NumPy's `assert_allclose` prints the
exact expected-vs-actual arrays automatically.

What is covered:

```text
test_flow_field.py   read_flow_field: shape, column metadata, per-point values,
                     and rejection of missing columns, empty files, NaN, and inf
test_sensors.py      read_sensor_data: same, plus the sensor-only time_index and
                     unique_times helpers (targets are u,v,w with no density)
test_conditions.py   ConditionSpec/ConditionsConfig activation and weight rules,
                     and that on/off state and weights flow through to the
                     assembled loss (inactive -> zero, weight scales the term)
test_pipeline.py     end-to-end read -> tensors -> model: the tensors fed to the
                     model hold correct values/shape/dtype, and the MLP consumes
                     them and returns a finite (n, 4) state
```

## Current limitations

- The PDE is a reduced pressureless, inviscid transport system, not full WRF
  physics.
- Full no-slip is not physically consistent with an inviscid momentum model;
  no-penetration or a viscous/turbulence term is needed for a consistent wall
  model.
- Dense flow, sensor, and boundary tensors are evaluated in full each epoch.
  Dataset mini-batching and the dataset sampling configuration are not yet
  connected to the trainer.
- Model output columns are currently interpreted positionally as
  `u, v, w, rho`.
- Positive-density, NaN/Inf, validation, checkpoint, and early-stopping
  configurations are not integrated into the training loop.
- The model and PDE code do not store preprocessing or inverse-scaling
  metadata.
- The uniform-flow case is a pipeline smoke test. Because a constant state has
  zero derivatives, it does not strongly validate the PDE implementation.
