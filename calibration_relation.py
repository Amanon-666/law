"""Matched source-learned controls for historical calibration information.

All kernels have the same pre-S marginal variance. Shared local length is
learned independently of the historical-field length in EVERY method.
These are controlled function families, not new kernel constructions.
"""
import math

import torch
from torch import nn

from transfer import rbf


class CalibrationRelation(nn.Module):
    def __init__(self, kind, initial_length):
        super().__init__()
        if kind not in ("uniform", "strength_rbf", "pointwise", "linear", "shape"):
            raise ValueError(kind)
        self.kind = kind
        self.logit_local = nn.Parameter(torch.tensor(0.))
        self.log_length = nn.Parameter(torch.tensor(math.log(initial_length)))
        if kind == "strength_rbf":
            self.log_strength_length = nn.Parameter(torch.tensor(math.log(20.)))
        if kind in ("pointwise", "linear", "shape"):
            self.encoder = (nn.Linear(5, 4) if kind == "linear" else
                            nn.Sequential(nn.Linear(5, 16), nn.Tanh(), nn.Linear(16, 4)))

    def embedding(self, history, x):
        if self.kind == "pointwise":
            center = history.sample_mean(x)
            differences = center.new_zeros((*center.shape, 4))
        else:
            center, differences = history.shape(x)
        inputs = torch.cat((((center + 75) / 20).unsqueeze(-1), differences / 20), -1)
        raw = self.encoder(inputs)
        augmented = torch.cat((torch.ones_like(raw[..., :1]), raw), -1)
        return augmented / torch.linalg.vector_norm(augmented, dim=-1, keepdim=True)

    def kernel(self, x, y, history, base):
        spatial = rbf(x, y, 1., self.log_length.exp())
        if self.kind == "uniform":
            relation = torch.ones_like(spatial)
        elif self.kind == "strength_rbf":
            left, right = history.sample_mean(x), history.sample_mean(y)
            differences = (left[:, None] - right[None]) / self.log_strength_length.exp()
            relation = torch.exp(-.5 * differences.square()).mean(-1)
        else:
            left, right = self.embedding(history, x), self.embedding(history, y)
            relation = torch.einsum("nsd,msd->nm", left, right) / left.shape[1]
        local = torch.sigmoid(self.logit_local)
        return base["drift"] ** 2 * (local * spatial + (1 - local) * relation)
