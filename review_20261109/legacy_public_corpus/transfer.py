"""Continuous reference-conditioned priors with positive Gaussian cubature.

Transfer means and optional change correlations are separate learned objects.
Historical uncertainty remains a full kernel; another query cannot change S.
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


class ChangeCorrelation(nn.Module):
    """Unit-diagonal change kernels, averaged over the SAME historical samples.

    product: the old learned-feature-RBF times spatial-RBF family.
    linear/gram: unit feature Gram with a spatial component. These are mature
    conditional-kernel controls, not an algorithmic novelty claim.
    """
    def __init__(self, kind):
        super().__init__()
        if kind not in ("product", "linear", "gram"):
            raise ValueError(kind)
        self.kind = kind
        self.logit_local = nn.Parameter(torch.tensor(0.0))
        self.encoder = (nn.Linear(5, 4) if kind == "linear" else
                        nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 4)))

    def embedding(self, history, x):
        center, differences = history.shape(x)
        inputs = torch.cat((((center + 75) / 20).unsqueeze(-1), differences / 20), dim=-1)
        raw = self.encoder(inputs)
        if self.kind == "product":
            return raw
        augmented = torch.cat((torch.ones_like(raw[..., :1]), raw), dim=-1)
        return augmented / torch.linalg.vector_norm(augmented, dim=-1, keepdim=True)

    def kernel(self, x, y, history, base):
        left, right = self.embedding(history, x), self.embedding(history, y)
        spatial = rbf(x, y, 1.0, base["length"])
        if self.kind == "product":
            distance = (left[:, None] - right[None]).square().sum(-1)
            relation = spatial * torch.exp(-0.5 * distance).mean(-1)
        else:
            relation = torch.einsum("nsd,msd->nm", left, right) / left.shape[1]
        local = torch.sigmoid(self.logit_local)
        return base["drift"] ** 2 * (local * spatial + (1 - local) * relation)


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
