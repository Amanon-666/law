"""Existing ReFi historical GP and identity-recomposition prior.

Compatibility extraction only: full spatial covariance and historical
availability branches are preserved. Structural learning is not enabled.
"""
import torch
from torch import nn


def rbf(x, y, amplitude, length):
    return amplitude ** 2 * torch.exp(-0.5 * ((x[:, None] - y[None]) ** 2).sum(-1) / length ** 2)


class Transfer(nn.Module):
    def __init__(self, kind):
        super().__init__()
        self.kind = kind
        if kind != "identity":
            self.a = nn.Parameter(torch.tensor(1.0))
            self.b = nn.Parameter(torch.tensor(0.0))
        if kind == "linear_shape":
            self.stencil = nn.Parameter(torch.zeros(4))
        if kind in ("pointwise", "shape"):
            self.network = nn.Sequential(nn.Linear(1 if kind == "pointwise" else 5, 16),
                                         nn.Tanh(), nn.Linear(16, 1, bias=False))
            nn.init.zeros_(self.network[-1].weight)

    def scale(self):
        return 1.0 if self.kind == "identity" else self.a

    def forward(self, center, differences=None):
        if self.kind == "identity":
            return center
        output = -75 + self.a * (center + 75) + 20 * self.b
        if self.kind == "linear_shape":
            output = output + differences @ self.stencil
        if self.kind in ("pointwise", "shape"):
            inputs = ((center + 75) / 20).unsqueeze(-1)
            if self.kind == "shape":
                inputs = torch.cat((inputs, differences / 20), dim=-1)
            output = output + 20 * self.network(inputs).squeeze(-1)
        return output


class History:
    def __init__(self, positions, posterior, base):
        self.z, self.base = positions, base
        self.m, self.c = posterior["mean"].detach(), posterior["cov"].detach()
        j = len(positions)
        eye = torch.eye(j, device=positions.device, dtype=positions.dtype)
        self.chol = torch.linalg.cholesky(self.kernel(positions, positions) + 1e-6 * eye)
        # Numerically symmetrized covariance; no extra observation noise here.
        root = torch.linalg.cholesky((self.c + self.c.T) / 2 + 1e-6 * eye)
        offsets = j ** 0.5 * root.T
        self.samples = torch.cat((self.m[None] + offsets, self.m[None] - offsets))

    def kernel(self, x, y):
        return rbf(x, y, self.base["amplitude"], self.base["length"])

    def projection(self, x):
        return torch.cholesky_solve(self.kernel(self.z, x), self.chol).T

    def sample_mean(self, x):
        p = self.projection(x)
        return self.base["mean"] + p @ (self.samples - self.base["mean"]).T

    def shape(self, x):
        center = self.sample_mean(x)
        radius = self.base["length"] / 2
        offsets = x.new_tensor([[radius, 0], [-radius, 0], [0, radius], [0, -radius]])
        neighbor = self.sample_mean((x[:, None] + offsets).reshape(-1, 2))
        neighbor = neighbor.reshape(len(x), 4, len(self.samples)).permute(0, 2, 1)
        return center, neighbor - center.unsqueeze(-1)


class Prior:
    def __init__(self, model, history, base, change=None):
        self.model, self.history, self.base = model, history, base
        self.change = change

    def features(self, x):
        """Return mean and centered cubature functions [positions, 2*H]."""
        if self.history is None:
            return x.new_full((len(x),), self.base["mean"]), None
        if self.model.kind in ("shape", "linear_shape"):
            center, differences = self.history.shape(x)
        else:
            center, differences = self.history.sample_mean(x), None
        g = self.model(center, differences)
        mean = g.mean(dim=1)
        return mean, (g - mean[:, None]) / g.shape[1] ** 0.5

    def covariance(self, x, y, fx, fy):
        base = self.base
        if self.history is None:
            return rbf(x, y, base["amplitude"], base["length"])
        residual = self.history.kernel(x, y) - self.history.projection(x) @ self.history.kernel(self.history.z, y)
        change = (rbf(x, y, base["drift"], base["length"]) if self.change is None else
                  self.change.kernel(x, y, self.history, base))
        return (change +
                self.model.scale() ** 2 * residual + fx @ fy.T)

    def diagonal(self, x, fx):
        if self.history is None:
            return x.new_full((len(x),), self.base["amplitude"] ** 2)
        residual = self.base["amplitude"] ** 2 - (self.history.projection(x) * self.history.kernel(x, self.history.z)).sum(-1)
        return self.base["drift"] ** 2 + self.model.scale() ** 2 * residual + (fx * fx).sum(-1)


class BankPrior:
    def __init__(self, histories, weights, base):
        self.base = base
        self.weights = weights
        self.sources = [Prior(Transfer('identity'), h, dict(base, drift=0.)) for h in histories]

    def features(self, x):
        features = [p.features(x) for p in self.sources]
        if not features:
            return x.new_full((len(x),), self.base['mean']), []
        return sum(w * m for w, (m, _) in zip(self.weights, features)), [f for _, f in features]

    def covariance(self, x, y, fx, fy):
        if not self.sources:
            return rbf(x, y, self.base['amplitude'], self.base['length'])
        return rbf(x, y, self.base['drift'], self.base['length']) + sum(
            w ** 2 * p.covariance(x, y, a, b)
            for w, p, a, b in zip(self.weights, self.sources, fx, fy))

    def diagonal(self, x, fx):
        if not self.sources:
            return x.new_full((len(x),), self.base['amplitude'] ** 2)
        return self.base['drift'] ** 2 + sum(
            w ** 2 * p.diagonal(x, f) for w, p, f in zip(self.weights, self.sources, fx))


class InterceptBankPrior(BankPrior):
    def __init__(self, histories, weights, base, intercept_std):
        present = any(h is not None for h in histories)
        super().__init__(histories if present else [], weights if present else [], base)
        self.intercept_std = intercept_std

    def covariance(self, x, y, fx, fy):
        return super().covariance(x, y, fx, fy) + self.intercept_std ** 2

    def diagonal(self, x, fx):
        return super().diagonal(x, fx) + self.intercept_std ** 2
