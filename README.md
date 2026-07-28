# WRF PINN

The goal is to generate a conditional PINN that uses WRF equations, sensor and
LES data to generate predictions of atmospheric winds, which would then be transferred
to another site. HRRR data will be used to define the initial atmospheric
conditions, and the PINN should predict the corresponding evolution of the flow
field within the domain based on the input HRRR state. We will train it based on data from
the DOE ARM SGP site first, and then transfer it to Lexington to assess if it works. 

## Overall Plan

This project is being built as a clean WRF-focused PINN package rather than an
extension of the earlier Euler-equation prototype. The core idea is to train a
neural network that maps continuous space-time coordinates, and eventually
site/condition parameters, to atmospheric state variables. The model will be
constrained by reduced WRF-style governing equations, HRRR-derived initial
conditions, sparse sensor measurements, and high-fidelity LES data when
available.

The first version uses a local Cartesian approximation. This keeps the coordinate
system simple while the project structure is being built. The initial physics
state is intentionally reduced to velocity and density:

```text
inputs:  x, y, z, t
outputs: u, v, w, rho
```

The first residual system contains mass continuity and three momentum residuals
under simplifying assumptions:

```text
local Cartesian coordinates
zero forcing
no temperature equation
no moisture equation
no pressure-gradient term yet
no Coriolis term yet
no turbulence closure yet
```

This is not the final atmospheric model. It is the first stable scaffold that
lets us build the PINN infrastructure carefully before adding more WRF physics.

## Code Architecture

The package is organized around a separation between configuration, physics,
data handling, models, training, and evaluation.

```text
src/wrf_pinn/
    config/
    data/
    physics/
    models/
    conditions/
    training/
    evaluation/
```

Each subpackage has a specific role.

## Configuration

The `config/` package defines the structure of an experiment. These files should
describe choices and assumptions, but should not perform expensive computation,
read large files, train models, or evaluate losses.

### `config/domain.py`

Defines the physical space-time domain.

This file is geometry-only. It describes coordinate bounds, optional grid
metadata, grid spacing, time step, and coordinate normalization helpers.

It should answer questions like:

```text
What are the x, y, z, and t limits?
What is the physical size of the domain?
If a grid is known, what are dx, dy, dz, and dt?
How do we normalize coordinates for the neural network?
```

It should not define boundary conditions, WRF file structure, HRRR metadata, or
residual equations.

### `config/boundaries.py`

Defines boundary-condition configuration.

This file names the boundaries of the space-time domain and describes what kind
of condition, if any, is applied to each one.

Examples:

```text
west, east, south, north
bottom, top
initial, final
```

Possible condition types include:

```text
none
periodic
dirichlet
neumann
no_penetration
open
sponge
data_forced
```

The actual enforcement of boundary conditions should happen later in the
`conditions/` or `training/` packages.

### `config/state.py`

Defines the state-variable contract.

This file describes the ordering and names of coordinates, prognostic variables,
and diagnostic variables.

For the current reduced model:

```text
coordinates: x, y, z, t
active outputs: u, v, w, rho
```

A broader WRF state may later include:

```text
theta
pressure
moisture species
geopotential
perturbation variables
```

The state config prevents ambiguity when model outputs are sliced inside
residual and loss code.

### `config/physics.py`

Defines the active physics assumptions.

For the current reduced system, this file states that the model uses:

```text
local Cartesian coordinates
zero forcing
velocity and density only
mass residual
x, y, z momentum residuals
```

It also stores physical constants such as gravity, gas constants, reference
pressure, Earth rotation rate, and Earth radius. These constants are available
for future physics extensions, even if the current residual system does not use
them yet.

This file does not compute residuals.

### `config/model.py`

Defines neural-network architecture settings.

The first model is a simple MLP:

```text
input dimension: 4
output dimension: 4
hidden width: 128
hidden layers: 6
activation: tanh
```

This config should remain lightweight until the research direction requires a
more specific architecture such as Fourier features, SIREN, DeepONet, attention,
or multiple output heads.

### `config/data.py`

Defines data-source configuration.

This file describes WRF, HRRR, LES, synthetic, or sensor data sources and maps
their external variable names to the internal PINN state.

It should answer:

```text
Where is the data?
What type of data source is it?
Which external variable corresponds to u, v, w, rho?
What coordinate system does the data use?
What interpolation method should be used?
```

It should not read NetCDF, GRIB, Zarr, CSV, or sensor files directly.

### `config/scaling.py`

Defines input, output, and residual scaling choices.

This file stores reference scales for variables such as:

```text
x, y, z, t
u, v, w, rho
mass residual
momentum residuals
```

The actual scaling operations should live in a future implementation module,
likely `data/scaling.py` or `transforms/scaling.py`.

This distinction matters because PINN residuals require care when coordinates
are scaled. Derivatives with respect to normalized coordinates need chain-rule
corrections before they represent physical derivatives.

### `config/sampling.py`

Defines how training points should be sampled.

It configures point counts and sampling methods for:

```text
interior collocation points
initial-condition points
boundary-condition points
measured-data points
validation points
```

This file does not generate samples. Actual sampling logic should live in
`data/samplers.py` or `training/samplers.py`.

### `config/conditions.py`

Defines which training conditions are active and how much they are weighted.

Current condition groups are:

```text
PDE residual
initial condition
boundary condition
measured data
regularization
```

This file should not compute losses. It only tells the training code which loss
components to include and how strongly to weight them.

### `config/training.py`

Defines training-loop settings.

Examples include:

```text
number of epochs
optimizer type
learning rate
scheduler
logging interval
checkpoint interval
gradient clipping
device selection
```

The actual training loop should live in the `training/` package.

### `config/evaluation.py`

Defines what to evaluate after or during training.

It describes:

```text
fields to evaluate
metrics to compute
residual diagnostics to report
data sources to compare against
output directory and formats
slice/profile evaluation options
```

The actual metric computation and plotting should live in the `evaluation/`
package.

### `config/validation.py`

Defines run health checks and acceptance criteria.

This includes:

```text
NaN/Inf detection
maximum loss thresholds
gradient norm thresholds
positive density checks
maximum wind speed checks
residual tolerances
data-fit tolerances
early stopping
```

This is different from evaluation. Evaluation computes metrics; validation
decides whether those metrics are acceptable.

## Physics Package

The `physics/` package contains equation-level implementation.

### `physics/residuals.py`

Computes the current reduced Cartesian residuals.

For coordinates:

```text
x, y, z, t
```

and state:

```text
u, v, w, rho
```

the current residuals are:

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

This module should grow carefully. Future versions may add pressure-gradient
terms, gravity, Coriolis terms, temperature, moisture, turbulence closures, and
WRF-specific vertical-coordinate transformations.

## Data Package

The `data/` package will contain code that reads and prepares external
information.

Future modules may include:

```text
data/wrf_reader.py
    Read WRF NetCDF output.

data/hrrr_reader.py
    Read HRRR GRIB/Zarr fields.

data/les_reader.py
    Read LES data for high-resolution supervision.

data/sensor_reader.py
    Read sparse tower, lidar, radar, or other sensor observations.

data/interpolation.py
    Interpolate discrete data onto continuous PINN points.

data/scaling.py
    Apply the scaling described in config/scaling.py.

data/samplers.py
    Generate collocation, boundary, initial, data, and validation points.
```

This package is where discrete atmospheric datasets are converted into the
continuous coordinate/state format needed by the PINN.

## Models Package

The `models/` package will contain neural-network implementations.

The first likely model is:

```text
models/mlp.py
```

It will map:

```text
(x, y, z, t) -> (u, v, w, rho)
```

Later models may include conditional inputs so that one trained model can
represent different atmospheric sites, terrain classes, inflow regimes, stability
conditions, or forcing scenarios.

A future conditional model may look like:

```text
(x, y, z, t, condition_vector) -> (u, v, w, rho)
```

where `condition_vector` could encode site metadata, inflow state, LES case
identity, or WRF/HRRR-derived large-scale conditions.

## Conditions Package

The `conditions/` package will contain evaluators for constraints that are not
the PDE residual itself.

Future modules may include:

```text
conditions/initial.py
    Compare predictions to initial atmospheric state.

conditions/boundary.py
    Enforce selected boundary conditions.

conditions/data.py
    Compare predictions to WRF, HRRR, LES, or sensor measurements.

conditions/regularization.py
    Apply optional smoothness or physical sanity constraints.
```

These modules should produce error tensors. They should not assemble the total
loss themselves.

## Training Package

The `training/` package coordinates optimization.

Current module:

```text
training/losses.py
```

This file assembles weighted loss terms from already-computed errors:

```text
PDE residuals
measured data errors
initial-condition errors
boundary-condition errors
regularization errors
```

Future modules may include:

```text
training/trainer.py
    Full training loop.

training/optimizers.py
    Optimizer and scheduler construction.

training/checkpoints.py
    Save and load model states.

training/logging.py
    Store training metrics.
```

The training loop should call the model, evaluate residuals, evaluate data and
boundary conditions, assemble the loss, backpropagate, and save progress.

## Evaluation Package

The `evaluation/` package will assess model quality after or during training.

Future modules may include:

```text
evaluation/metrics.py
    RMSE, MAE, bias, relative L2, correlation.

evaluation/diagnostics.py
    Residual norms and physical sanity diagnostics.

evaluation/plots.py
    Field slices, vertical profiles, time histories, wind-vector comparisons.
```

This package should make it clear whether the PINN is producing useful dense
fields and whether those fields remain physically reasonable.

## Near-Term Development Order

The current scaffold is intentionally modular. The next useful implementation
steps are:

```text
1. Implement models/mlp.py.
2. Implement data/samplers.py for continuous collocation points.
3. Add data/scaling.py so coordinates and state variables are transformed
   consistently.
4. Add a small synthetic training example before using WRF, HRRR, sensor, or LES
   data.
5. Build training/trainer.py around the model, residuals, samplers, and loss
   assembly.
6. Add an integrator that combines everything to start the training.
6. Add real WRF, HRRR, sensor, and LES readers once the training loop works on
   synthetic data.
7. Add conditional inputs for site transfer once the base model is stable.
```

The guiding principle is to keep configuration, physics, data handling, and
training separate. That makes it easier to replace a simplified assumption with
a more realistic WRF component without rewriting the whole project.
