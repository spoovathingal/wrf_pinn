"""Baseline fixed weights and W1 gradient-balance weights for conditional training."""

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
        if set(state["weights"]) != set(self.weights):
            raise ValueError("Checkpoint active loss groups differ; use a new checkpoint path.")
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


METHODS = {"fixed": Fixed, "grad-balance": GradBalance}
