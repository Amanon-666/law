"""History-profile operator prior for multi-source LAW (2026-10-26).

f_a(x) = sum_j w_j h_ja(x) + g(z_a(x)) + delta_a(x),
Cov[delta] = drift^2 k_l + s(z(x)) s(z(x')) k_l.
z reads only H and the relative floor offset dz = source - target.
net=None recovers BankPrior exactly; the convex mean can then exceed max_j h_j.
"""
import math

import torch
from torch import nn
from torch.nn import functional as F

from history_bank import BankPrior
from inference import GH_NODES, GH_WEIGHTS, _detection_logs
from transfer import rbf

TOKEN = 6
MISSING_PEAK, MISSING_DISTANCE = -105., 100.


def history_moments(history, x, base):
    """Marginal mean/variance of one historical GP posterior at x (same as Prior with drift 0)."""
    if history is None:
        return x.new_full((len(x),), base['mean']), x.new_full((len(x),), base['amplitude'] ** 2)
    p = history.projection(x)
    mean = base['mean'] + p @ (history.m - base['mean'])
    delta = history.c - history.kernel(history.z, history.z)
    var = base['amplitude'] ** 2 + ((p @ delta) * p).sum(-1)
    return mean, var.clamp_min(1e-6)


def source_tokens(history, dz, x, base):
    mean, var = history_moments(history, x, base)
    n = len(x)
    if history is None:
        peak, distance, present = x.new_full((n,), MISSING_PEAK), x.new_full((n,), MISSING_DISTANCE), x.new_zeros(n)
    else:
        i = history.m.argmax()
        peak = history.m[i].expand(n)
        distance = (x - history.z[i]).norm(dim=-1)
        present = x.new_ones(n)
    return torch.stack([(mean + 75) / 20, var.sqrt() / 20, x.new_full((n,), dz / 3.),
                        (peak + 75) / 20, distance / 30, present], -1)


class OperatorNet(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.phi = nn.Sequential(nn.Linear(TOKEN, width), nn.Tanh(), nn.Linear(width, width), nn.Tanh())
        self.rho = nn.Sequential(nn.Linear(width + 1, width), nn.Tanh(), nn.Linear(width, 2))
        nn.init.zeros_(self.rho[-1].weight)
        with torch.no_grad():
            self.rho[-1].bias.copy_(torch.tensor([0., -4.]))

    def forward(self, tokens):
        """tokens [J, n, TOKEN] -> shift g [n] (dB), extra scale s [n] (dB)."""
        pooled = self.phi(tokens).mean(0)
        count = pooled.new_full((len(pooled), 1), math.log1p(tokens.shape[0]))
        out = self.rho(torch.cat([pooled, count], -1))
        return 20 * out[:, 0], 10 * F.softplus(out[:, 1])


class OperatorBankPrior(BankPrior):
    def __init__(self, histories, weights, base, net=None, dz=None):
        super().__init__(histories, weights, base)
        self.histories, self.net, self.dz = histories, net, dz
        self.active = net is not None and any(h is not None for h in histories)
        self._cache = {}

    def operator(self, x):
        key = (x.data_ptr(), tuple(x.shape))
        if key not in self._cache:
            tokens = torch.stack([source_tokens(h, d, x, self.base) for h, d in zip(self.histories, self.dz)])
            self._cache[key] = self.net(tokens)
        return self._cache[key]

    def features(self, x):
        mean, f = super().features(x)
        if self.active:
            mean = mean + self.operator(x)[0]
        return mean, f

    def covariance(self, x, y, fx, fy):
        c = super().covariance(x, y, fx, fy)
        if self.active:
            c = c + self.operator(x)[1][:, None] * self.operator(y)[1][None] * rbf(x, y, 1., self.base['length'])
        return c

    def diagonal(self, x, fx):
        d = super().diagonal(x, fx)
        if self.active:
            d = d + self.operator(x)[1].square()
        return d


def marginal_loglik(mean, var, n, k, s1, s2, obs):
    """Exact log p(all scans at a position) for f ~ N(mean, var), positions independent.

    RSS factors integrate in closed form; detection factors use the original GH rule
    on the resulting RSS posterior. Equals observation_log_prob for a single scan.
    """
    noise2 = float(obs['noise_std']) ** 2
    kk = k.clamp_min(1)
    rbar = s1 / kk
    spread = (s2 - s1.square() / kk).clamp_min(0)
    total = var + noise2 / kk
    logz = (-.5 * (k - 1) * math.log(2 * math.pi * noise2) - .5 * kk.log() - spread / (2 * noise2)
            - .5 * torch.log(2 * math.pi * total) - (rbar - mean).square() / (2 * total))
    logz = torch.where(k > 0, logz, torch.zeros_like(logz))
    post_var = 1 / (1 / var + k / noise2)
    post_mean = post_var * (mean / var + k * rbar / noise2)
    signal = post_mean[..., None] + post_var.sqrt()[..., None] * GH_NODES
    detected, missing = _detection_logs(signal, obs)
    logd = torch.logsumexp(GH_WEIGHTS.log() + k[..., None] * detected + (n - k)[..., None] * missing, -1)
    return logz + logd
