# Physics residual scaling

The neural network is trained on normalized coordinates and normalized output
variables, but the PDE residuals should represent physical equations. This
directory therefore treats scaling as part of residual evaluation, not as part
of the model.

The model sees normalized inputs and returns normalized outputs:

```text
(x_hat, y_hat, z_hat, t_hat) -> (u_hat, v_hat, w_hat, rho_hat)
```

The residual code maps these quantities back to physical variables before
assembling the PDE terms.

## Scaling convention

Every coordinate and state variable uses the same affine convention:

```text
physical_value = offset + scale * normalized_value
```

For the current reduced WRF PINN state, this means:

```text
x   = x_offset   + x_scale   * x_hat
y   = y_offset   + y_scale   * y_hat
z   = z_offset   + z_scale   * z_hat
t   = t_offset   + t_scale   * t_hat

u   = u_offset   + u_scale   * u_hat
v   = v_offset   + v_scale   * v_hat
w   = w_offset   + w_scale   * w_hat
rho = rho_offset + rho_scale * rho_hat
```

The scale factors are stored in `ResidualScalingConfig`:

```python
from wrf_pinn.config.scaling import ResidualScalingConfig, VariableScale

scaling = ResidualScalingConfig(
    x=VariableScale(offset=0.0, scale=1000.0),
    y=VariableScale(offset=0.0, scale=1000.0),
    z=VariableScale(offset=0.0, scale=100.0),
    t=VariableScale(offset=0.0, scale=60.0),
    u=VariableScale(offset=0.0, scale=10.0),
    v=VariableScale(offset=0.0, scale=10.0),
    w=VariableScale(offset=0.0, scale=1.0),
    rho=VariableScale(offset=0.0, scale=1.225),
)
```

The default scaling is identity for every variable. With identity scaling, the
residuals behave as they did before the scaling support was added.

## Coordinate derivative scaling

PyTorch autograd differentiates with respect to the tensor passed to the model.
In this codebase, that tensor may contain normalized coordinates:

```text
(x_hat, y_hat, z_hat, t_hat)
```

The physical residuals need derivatives with respect to physical coordinates:

```text
(x, y, z, t)
```

Using

```text
x = x_offset + x_scale * x_hat
```

the chain rule gives:

```text
dq/dx = (dq/dx_hat) / x_scale
```

Similarly:

```text
dq/dy = (dq/dy_hat) / y_scale
dq/dz = (dq/dz_hat) / z_scale
dq/dt = (dq/dt_hat) / t_scale
```

The helper `_physical_gradient(...)` applies these chain-rule factors. It
returns derivatives in the order:

```text
dq/dx, dq/dy, dq/dz, dq/dt
```

## State-variable scaling

The PDE residuals should use physical `u`, `v`, `w`, and `rho`, not normalized
network outputs. The helper `_to_physical_state(...)` applies:

```text
u   = u_offset   + u_scale   * u_hat
v   = v_offset   + v_scale   * v_hat
w   = w_offset   + w_scale   * w_hat
rho = rho_offset + rho_scale * rho_hat
```

After this conversion, the residual equations are assembled using physical
velocity and density values.

## Current PDE residuals

The current implementation in `residuals_pde.py` is a reduced local Cartesian,
zero-forcing system with active state:

```text
u, v, w, rho
```

It does not yet include pressure gradients, gravity, Coriolis terms,
temperature, moisture, viscosity, or turbulence closure.

The mass residual is:

```text
R_mass = rho_t + (rho u)_x + (rho v)_y + (rho w)_z
```

The primitive-variable momentum residuals are:

```text
R_u = u_t + u u_x + v u_y + w u_z
R_v = v_t + u v_x + v v_y + w v_z
R_w = w_t + u w_x + v w_y + w w_z
```

All derivatives above are physical derivatives after chain-rule scaling.

## Why products are differentiated after physical scaling

For the mass equation, the code forms physical products first:

```text
rho * u
rho * v
rho * w
```

and then differentiates them:

```text
d(rho u)/dx
d(rho v)/dy
d(rho w)/dz
```

This keeps the equation readable and lets autograd handle product-rule terms.
The coordinate chain rule is then applied by dividing by the physical coordinate
scale factor.

## Boundary residual caveat

The PDE residuals now understand physical scaling. The boundary residuals are
still written in normalized output space. For example, no-slip currently drives:

```text
u_hat = 0
v_hat = 0
w_hat = 0
```

That is physically correct only when the velocity normalization preserves
physical zero, such as:

```text
u_hat = u / U_ref
```

If min-max scaling is used, physical zero may not map to normalized zero. In
that case, boundary residuals should later be updated to use the same scaling
metadata and enforce the correctly normalized value corresponding to physical
zero.

## Metadata file reader

Scaling metadata can be read from a simple text file using:

```python
from wrf_pinn.data.scaling import read_residual_scaling_txt

scaling = read_residual_scaling_txt("residual_scaling.txt")
```

The text file format is:

```text
# name offset scale
x   0.0   1000.0
y   0.0   1000.0
z   0.0   100.0
t   0.0   60.0
u   0.0   10.0
v   0.0   10.0
w   0.0   1.0
rho 0.0   1.225
```

The required variables are:

```text
x, y, z, t, u, v, w, rho
```

