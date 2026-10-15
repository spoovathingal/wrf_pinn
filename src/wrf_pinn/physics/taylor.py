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

import torch
from torch import nn

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

    # -- tensor protocol, so the residual helpers run unchanged on jets --------
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

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if func is torch.sqrt:
            return args[0].sqrt()
        if func is torch.minimum:
            return minimum(*args, **kwargs)
        if func is torch.maximum:
            return maximum(*args, **kwargs)
        if func is torch.where:
            return where(*args, **kwargs)
        return NotImplemented

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


def where(cond: torch.Tensor, a: "Jet | torch.Tensor | Scalar",
          b: "Jet | torch.Tensor | Scalar") -> Jet:
    """Elementwise select between jets (or constants, which have zero derivative)."""
    ja, jb = (x if isinstance(x, Jet) else None for x in (a, b))
    ref = ja if ja is not None else jb
    va = ja.v if ja is not None else torch.as_tensor(a, dtype=ref.v.dtype, device=ref.v.device)
    vb = jb.v if jb is not None else torch.as_tensor(b, dtype=ref.v.dtype, device=ref.v.device)
    k = min(j._k() for j in (ja, jb) if j is not None)
    da = ja.d[:k] if ja is not None else torch.zeros((), dtype=ref.v.dtype, device=ref.v.device)
    db = jb.d[:k] if jb is not None else torch.zeros((), dtype=ref.v.dtype, device=ref.v.device)
    return Jet(torch.where(cond, va, vb), torch.where(cond, da, db))


def minimum(a: Jet, b: "Jet | torch.Tensor | Scalar") -> Jet:
    vb = b.v if isinstance(b, Jet) else b
    return where(a.v <= vb, a, b)


def maximum(a: Jet, b: "Jet | torch.Tensor | Scalar") -> Jet:
    vb = b.v if isinstance(b, Jet) else b
    return where(a.v >= vb, a, b)


def coordinate_jet(values: torch.Tensor, axis: int, n_coord: int = 4) -> Jet:
    """Jet of a physical coordinate (n, 1): derivative 1 along ``axis``, 0 elsewhere."""
    d = torch.zeros((n_coord, 1, 1), dtype=values.dtype, device=values.device)
    d[axis] = 1.0
    return Jet(values, d.expand(n_coord, values.shape[0], 1))


def taylor_forward(net: nn.Sequential, coordinates: torch.Tensor, z_rows: torch.Tensor,
                   coordinate_scales: Sequence[float]) -> Jet:
    """Network outputs with first (x, y, z, t) and spatial second derivatives.

    ``net`` is a Sequential of Linear layers with Tanh between them, fed
    ``[coordinates | z_rows]`` (the ConditionalModel layout). Coordinates are
    normalized, physical = offset + scale * normalized, so derivatives come out
    with respect to physical coordinates."""
    layers = list(net)
    if not (isinstance(layers[0], nn.Linear) and isinstance(layers[-1], nn.Linear)
            and all(isinstance(m, nn.Tanh) for m in layers[1::2])
            and all(isinstance(m, nn.Linear) for m in layers[0::2])):
        raise TypeError("taylor_forward supports Linear/Tanh MLPs only.")
    n_coord = coordinates.shape[1]
    x = torch.cat([coordinates, z_rows], dim=1) if z_rows.shape[1] else coordinates

    first = layers[0]
    seeds = 1.0 / torch.as_tensor(coordinate_scales, dtype=x.dtype, device=x.device)
    v = first(x)
    # d(W x + b)/d(coord k) = W[:, k] / scale_k: the same for every point.
    d = (first.weight[:, :n_coord].T * seeds[:, None])[:, None, :]       # (4, 1, width)
    h = None                                                             # zero so far

    for i in range(1, len(layers), 2):
        a = torch.tanh(v)
        s = 1.0 - a * a                     # tanh'
        s2 = -2.0 * a * s                   # tanh''
        d_next = s * d                      # broadcasts (4, 1, w) -> (4, n, w)
        hs = [s2 * d[p] * d[q] for p, q in PAIRS]
        h_next = torch.stack(hs) if h is None else s * h + torch.stack(hs)
        lin = layers[i + 1]
        n, w = a.shape
        stacked = torch.cat([a[None], d_next, h_next]).reshape(-1, w)      # (11 n, w)
        out = (stacked @ lin.weight.T).reshape(1 + n_coord + len(PAIRS), n, -1)
        v = out[0] + lin.bias
        d, h = out[1:1 + n_coord], out[1 + n_coord:]
    return Jet(v, d, h)
