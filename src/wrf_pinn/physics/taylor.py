"""Coordinate derivatives by forward Taylor propagation, instead of nested autograd.

The PDE residuals need, at every collocation point, the network outputs' first
derivatives in (x, y, z, t) and their second derivatives in (x, y, z), then
derivatives of pointwise expressions of those (densities, SGS stresses, fluxes).
Taking each with ``torch.autograd.grad(create_graph=True)`` costs a backward pass
through the network per field, and training then differentiates through all of
them again. With only 4 coordinate inputs it is far cheaper to carry the
derivatives forward:

- ``taylor_forward`` pushes value, 4 first derivatives and 6 spatial second
  derivatives through a tanh MLP in one pass (one matmul per layer over 11x the
  rows), exactly, by the chain rule.
- ``Jet`` is a value with its first derivatives; arithmetic on jets applies the
  product, quotient and chain rules, so pointwise physics written on jets yields
  its coordinate gradients without any backward pass. ``Jet.deriv(i)`` turns a
  second-order network output into the first-order jet of its i-th derivative,
  which is what the stress and flux divergences need.

All derivatives are with respect to physical coordinates: the input seed of
coordinate k is 1 / scale_k, so the chain rule through the normalization is built in.
Training backpropagates through all of this with ordinary autograd.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache

import torch
from torch import nn

from wrf_pinn.models.conditional import (
    ConditionalModelConfig,
    TKE_INDEX,
    first_layer,
    transform_state_output,
)

#: spatial second-derivative channels, in order (xx, yy, zz, xy, xz, yz)
PAIRS: tuple[tuple[int, int], ...] = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))
#: (i, j) and (j, i) -> channel of d2/(di dj)
_PAIR_INDEX: dict[tuple[int, int], int] = {}
for _k, (_i, _j) in enumerate(PAIRS):
    _PAIR_INDEX[(_i, _j)] = _PAIR_INDEX[(_j, _i)] = _k

Scalar = float | int


@dataclass
class Jet:
    """A field and its coordinate derivatives at n points.

    ``v`` is (n, c). ``d`` is (k, n, c): first derivatives in (x, y, z, t) (k = 4) or
    in (x, y, z) only (k = 3). ``h`` is (6, n, c) spatial second derivatives in
    ``PAIRS`` order, present only on network outputs; arithmetic drops it, since
    the residuals only ever differentiate pointwise expressions once."""

    v: torch.Tensor
    d: torch.Tensor
    h: "torch.Tensor | None" = None

    # -- structure -------------------------------------------------------------
    def column(self, i: int) -> "Jet":
        """Channel i as an (n, 1) jet."""
        sl = slice(i, i + 1)
        return Jet(self.v[:, sl], self.d[..., sl], None if self.h is None else self.h[..., sl])

    def deriv(self, i: int) -> "Jet":
        """First-order spatial jet of d(self)/d(coordinate i), i in (x=0, y=1, z=2)."""
        if self.h is None:
            raise ValueError("deriv needs second derivatives (a network-output jet).")
        return Jet(self.d[i], torch.stack([self.h[_PAIR_INDEX[(i, j)]] for j in range(3)]))

    def _k(self) -> int:
        return self.d.shape[0]

    # -- tensor-like surface used by the residual helpers (dtype, comparisons) --
    @property
    def dtype(self) -> torch.dtype:
        return self.v.dtype

    @property
    def device(self) -> torch.device:
        return self.v.device

    @staticmethod
    def _value(other):
        return other.v if isinstance(other, Jet) else other

    def __gt__(self, other):
        return self.v > Jet._value(other)

    def __ge__(self, other):
        return self.v >= Jet._value(other)

    def __lt__(self, other):
        return self.v < Jet._value(other)

    def __le__(self, other):
        return self.v <= Jet._value(other)

    # -- arithmetic (first order; h is dropped) --------------------------------
    @staticmethod
    def _align(a: "Jet", b: "Jet") -> tuple[torch.Tensor, torch.Tensor]:
        k = min(a._k(), b._k())
        return a.d[:k], b.d[:k]

    def __add__(self, other):
        if isinstance(other, Jet):
            da, db = Jet._align(self, other)
            return Jet(self.v + other.v, da + db)
        return Jet(self.v + other, self.d)

    __radd__ = __add__

    def __neg__(self):
        return Jet(-self.v, -self.d)

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __mul__(self, other):
        if isinstance(other, Jet):
            da, db = Jet._align(self, other)
            return Jet(self.v * other.v, self.v * db + other.v * da)
        return Jet(self.v * other, self.d * other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, Jet):
            da, db = Jet._align(self, other)
            q = self.v / other.v
            return Jet(q, (da - q * db) / other.v)
        return Jet(self.v / other, self.d / other)

    def __rtruediv__(self, other):
        # other / self, other a constant
        q = other / self.v
        return Jet(q, -q / self.v * self.d)

    def pow(self, p: Scalar) -> "Jet":
        return Jet(self.v.pow(p), p * self.v.pow(p - 1) * self.d)

    __pow__ = pow

    def sqrt(self) -> "Jet":
        r = torch.sqrt(self.v)
        return Jet(r, self.d / (2.0 * r))

    def clamp_min(self, low: Scalar) -> "Jet":
        keep = self.v >= low  # torch.clamp's gradient mask is inclusive too
        return Jet(self.v.clamp_min(low), self.d * keep)

    def clamp_max(self, high: Scalar) -> "Jet":
        keep = self.v <= high
        return Jet(self.v.clamp_max(high), self.d * keep)


def where(cond: torch.Tensor, a, b):
    """``torch.where`` for tensors or jets; non-jet operands (tensors or Python
    numbers) are constants with zero derivative.

    Written as plain Python on ``.v``/``.d`` (not through ``__torch_function__``) so
    torch.compile traces it without graph breaks; the same holds for minimum/maximum."""
    ja, jb = (x if isinstance(x, Jet) else None for x in (a, b))
    if ja is None and jb is None:
        return torch.where(cond, a, b)
    k = min(j._k() for j in (ja, jb) if j is not None)
    va = ja.v if ja is not None else a
    vb = jb.v if jb is not None else b
    da = ja.d[:k] if ja is not None else 0.0
    db = jb.d[:k] if jb is not None else 0.0
    return Jet(torch.where(cond, va, vb), torch.where(cond, da, db))


def minimum(a, b):
    """``torch.minimum`` for tensors or jets; for jets, ties take ``a``."""
    if not isinstance(a, Jet) and not isinstance(b, Jet):
        return torch.minimum(a, b)
    return where(Jet._value(a) <= Jet._value(b), a, b)


def maximum(a, b):
    """``torch.maximum`` for tensors or jets; for jets, ties take ``a``."""
    if not isinstance(a, Jet) and not isinstance(b, Jet):
        return torch.maximum(a, b)
    return where(Jet._value(a) >= Jet._value(b), a, b)


def coordinate_jet(values: torch.Tensor, axis: int, n_coord: int = 4) -> Jet:
    """Jet of a physical coordinate (n, 1): derivative 1 along ``axis``, 0 elsewhere."""
    d = torch.zeros((n_coord, 1, 1), dtype=values.dtype, device=values.device)
    d[axis] = 1.0
    return Jet(values, d.expand(n_coord, values.shape[0], 1))


@lru_cache(maxsize=32)
def _scale_column(scales: tuple[float, ...], device: torch.device,
                  dtype: torch.dtype) -> torch.Tensor:
    """Coordinate scales as a (n_coord, 1) tensor on ``device``, built once: a
    per-call host-to-device copy makes the CPU wait for the GPU."""
    return torch.tensor(scales, dtype=dtype, device=device)[:, None]

def transform_state_jet(
    raw: Jet,
    output_config: ConditionalModelConfig | None,
) -> Jet:
    """Apply the model's output transform to values and coordinate derivatives."""
    if output_config is None or output_config.tke_output_mode == "linear":
        return raw

    values = transform_state_output(raw.v, output_config)
    sl = slice(TKE_INDEX, TKE_INDEX + 1)
    a = raw.v[:, sl].to(values.dtype)
    derivatives = raw.d.to(values.dtype)

    # Match F.softplus(beta=1.0, threshold=20.0) in conditional.py,
    # including its linear branch for inputs greater than 20.
    below_threshold = a < 20.0

    # At exactly 20, PyTorch retains the nonlinear first-derivative value
    # but uses zero for its second derivative. Keep that value constant
    # on this boundary so parameter differentiation agrees as well.
    sigmoid_argument = torch.where(
        below_threshold, a, torch.full_like(a, 20.0)
    )
    sigmoid = torch.sigmoid(sigmoid_argument)

    ratio = output_config.tke_softplus_scale / output_config.tke_scale
    first = ratio * torch.where(
        a > 20.0, torch.ones_like(a), sigmoid
    )
    second = ratio * torch.where(
        below_threshold, sigmoid * (1.0 - sigmoid), torch.zeros_like(a)
    )

    # First derivatives: d[f(a)]/dx_i = f'(a) * da/dx_i.
    tke_d = first * derivatives[..., sl]
    transformed_d = torch.cat(
        (
            derivatives[..., :TKE_INDEX],
            tke_d,
            derivatives[..., TKE_INDEX + 1:],
        ),
        dim=-1,
    )

    # Spatial second derivatives:
    # d2[f(a)]/dx_i dx_j = f'(a) * a_ij + f''(a) * a_i * a_j.
    transformed_h = None
    if raw.h is not None:
        hessian = raw.h.to(values.dtype)
        products = torch.stack(
            [
                derivatives[i, :, sl] * derivatives[j, :, sl]
                for i, j in PAIRS
            ],
            dim=0,
        )
        tke_h = first * hessian[..., sl] + second * products
        transformed_h = torch.cat(
            (
                hessian[..., :TKE_INDEX],
                tke_h,
                hessian[..., TKE_INDEX + 1:],
            ),
            dim=-1,
        )

    return Jet(values, transformed_d, transformed_h)

def taylor_forward(net: nn.Sequential, coordinates: torch.Tensor, z_rows: torch.Tensor,
                   coordinate_scales: Sequence[float], second: bool = True,
                   sizes: "torch.Tensor | int | None" = None,
                   *,
                   output_config: ConditionalModelConfig | None = None) -> Jet:
    """Network outputs with first (x, y, z, t) and spatial second derivatives.

    ``net`` is a Sequential of Linear layers with Tanh between them, fed
    ``[coordinates | z_rows]`` (the ConditionalModel layout). Coordinates are
    normalized, physical = offset + scale * normalized, so derivatives come out
    with respect to physical coordinates. ``second=False`` carries first
    derivatives only (5 channels instead of 11; the result has ``h=None``).
    With ``sizes``, ``z_rows`` is one latent row per case (see
    ``ConditionalModel.forward``) and is multiplied through the first layer once per
    case; it carries no coordinate derivatives either way."""
    layers = list(net)
    if not (isinstance(layers[0], nn.Linear) and isinstance(layers[-1], nn.Linear)
            and all(isinstance(m, nn.Tanh) for m in layers[1::2])
            and all(isinstance(m, nn.Linear) for m in layers[0::2])):
        raise TypeError("taylor_forward supports Linear/Tanh MLPs only.")
    n_coord = coordinates.shape[1]
    first = layers[0]
    scales = _scale_column(tuple(coordinate_scales), coordinates.device, coordinates.dtype)
    if not z_rows.shape[1]:
        v = first(coordinates)
    elif sizes is None and z_rows.shape[0] == coordinates.shape[0]:
        v = first(torch.cat([coordinates, z_rows], dim=1))
    else:
        v = first_layer(first, coordinates, z_rows, sizes)
    # d(W x + b)/d(coord k) = W[:, k] / scale_k: the same for every point.
    d = (first.weight[:, :n_coord].T / scales)[:, None, :]               # (4, 1, width)
    h = None                                                             # zero so far

    for i in range(1, len(layers), 2):
        a = torch.tanh(v)
        s = 1.0 - a * a                     # tanh'
        d_next = s * d                      # broadcasts (4, 1, w) -> (4, n, w)
        n, w = a.shape
        parts = [a[None], d_next]
        if second:
            s2 = -2.0 * a * s               # tanh''
            hs = [s2 * d[p] * d[q] for p, q in PAIRS]
            parts.append(torch.stack(hs) if h is None else s * h + torch.stack(hs))
        lin = layers[i + 1]
        stacked = torch.cat(parts).reshape(-1, w)          # (11 n, w), or (5 n, w) first-order
        out = (stacked @ lin.weight.T).reshape(-1, n, lin.out_features)   # (channels, n, out)
        v = out[0] + lin.bias
        d = out[1:1 + n_coord]
        h = out[1 + n_coord:] if second else None
    return transform_state_jet(Jet(v, d, h), output_config)
