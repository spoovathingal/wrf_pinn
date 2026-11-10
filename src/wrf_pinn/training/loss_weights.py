"""Automatic weights for the loss terms, one method per class."""

import torch


class Fixed:
    def __init__(self, weights: dict[str, float]):
        self.weights = {n: w for n, w in weights.items() if w != 0.0}  # off stays off

    def total(self, terms: dict[str, torch.Tensor]) -> torch.Tensor:
        return sum(self.weights[n] * terms[n] for n in self.weights)

    def update(self, losses: dict[str, float]) -> None:
        pass

    def state(self) -> dict:
        return {k: v for k, v in vars(self).items() if not k.startswith("_")}

    def load(self, state: dict) -> None:
        vars(self).update(state)


class GradBalance(Fixed):
    """Wang, Teng and Perdikaris 2021: each weight scales its term's mean gradient to the
    largest data gradient, smoothed over updates."""

    def __init__(self, weights, every: int = 100, smoothing: float = 0.9):
        super().__init__(weights)
        self.every, self.smoothing = every, smoothing

    def rebalance(self, terms: dict[str, torch.Tensor], params: list[torch.nn.Parameter]) -> None:
        def grads(term):
            g = torch.autograd.grad(term, params, retain_graph=True, allow_unused=True)
            return torch.cat([x.abs().flatten() for x in g if x is not None])
        peak = grads(terms["data"]).max()
        for n in self.weights:
            if n != "data" and terms[n].requires_grad:
                target = float(peak / (grads(terms[n]).mean() + 1e-12))
                self.weights[n] = self.smoothing * self.weights[n] + (1 - self.smoothing) * target


class ReLoBRaLo(Fixed):
    """Bischof and Kraus 2021: softmax of each term's progress against its last value and its
    first value, with random lookback to the first."""

    def __init__(self, weights, alpha: float = 0.999, temperature: float = 0.1, rho: float = 0.999):
        super().__init__(weights)
        self.alpha, self.temperature, self.rho = alpha, temperature, rho
        self.first = self.last = None

    def _balance(self, losses, ref):
        names = list(self.weights)
        ratio = torch.tensor([losses[n] / (self.temperature * ref[n] + 1e-12) for n in names])
        return dict(zip(names, (len(names) * torch.softmax(ratio, 0)).tolist()))

    def update(self, losses):
        if self.first is None:
            self.first = self.last = dict(losses)
            return
        to_first, to_last = self._balance(losses, self.first), self._balance(losses, self.last)
        keep = float(torch.rand(()) < self.rho)
        for n in self.weights:
            history = keep * self.weights[n] + (1 - keep) * to_first[n]
            self.weights[n] = self.alpha * history + (1 - self.alpha) * to_last[n]
        self.last = dict(losses)


class SoftAdapt(Fixed):
    """Heydari, Thompson and Mehmood 2019, loss-weighted: softmax of each term's recent rate of
    change, times its size, so slow and large terms get more weight."""

    def __init__(self, weights, beta: float = 0.1):
        super().__init__(weights)
        self.beta, self.last = beta, None

    def update(self, losses):
        if self.last is not None:
            names = list(self.weights)
            rate = torch.tensor([(losses[n] - self.last[n]) / (self.last[n] + 1e-12) for n in names])
            w = torch.softmax(self.beta * rate, 0) * torch.tensor([losses[n] for n in names])
            self.weights = dict(zip(names, (len(names) * w / w.sum()).tolist()))
        self.last = dict(losses)


class Uncertainty(Fixed, torch.nn.Module):
    """Kendall, Gal and Cipolla 2018: total = sum exp(-s) L + s, with one learned s per term."""

    def __init__(self, weights):
        torch.nn.Module.__init__(self)
        Fixed.__init__(self, weights)
        self.log_var = torch.nn.Parameter(torch.zeros(len(weights)))

    def total(self, terms):
        s = self.log_var
        return sum(torch.exp(-s[i]) * terms[n] + s[i] for i, n in enumerate(self.weights))

    def update(self, losses):
        self.weights = dict(zip(self.weights, torch.exp(-self.log_var.detach()).tolist()))

    def state(self):
        return {"weights": self.weights, "log_var": self.log_var.detach().cpu()}

    def load(self, state):
        self.weights = state["weights"]
        self.log_var.data.copy_(state["log_var"])


METHODS = {"fixed": Fixed, "grad-balance": GradBalance, "relobralo": ReLoBRaLo,
           "softadapt": SoftAdapt, "uncertainty": Uncertainty}
