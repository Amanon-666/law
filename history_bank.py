"""Mature linear multi-source GP prior; LAW observation model is unchanged.

The first study fits ONE source coefficient shared by all APs from target S.
It is empirical-Bayes adaptation, not a newly trained cross-task network.
Independent source fields are an explicit modeling assumption. Identical
acquisitions must not be supplied twice as independent sources.
"""
import math

import numpy as np
import torch
from scipy.optimize import minimize

from inference import GH_NODES, GH_WEIGHTS, JITTER, _detection_logs
from transfer import Prior, Transfer, rbf


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


def gaussian_batch(mean, cov, stats, noise_std):
    """Exact detected-RSS update, including likelihood constants, for AP batches.

    Precision form handles zero detections without an artificial RSS value.
    Axes are [AP, position] and [AP, position, position].
    """
    n = mean.shape[-1]
    eye = torch.eye(n, dtype=mean.dtype)
    root = torch.linalg.cholesky(cov + JITTER * eye)
    k, s1, s2 = (stats[key] for key in ('k', 's1', 's2'))
    noise2 = noise_std ** 2
    precision = k / noise2
    a = eye + root.transpose(-1, -2) @ (precision[..., None] * root)
    achol = torch.linalg.cholesky(a)
    v = (root.transpose(-1, -2) @ ((s1 - k * mean) / noise2)[..., None])
    solved = torch.cholesky_solve(v, achol)
    center = mean + (root @ solved).squeeze(-1)
    covariance = root @ torch.cholesky_solve(root.transpose(-1, -2), achol)
    constant = (k * math.log(2 * math.pi * noise2) +
                (s2 - 2 * mean * s1 + k * mean.square()) / noise2).sum(-1)
    loss = .5 * (constant + 2 * achol.diagonal(dim1=-2, dim2=-1).log().sum(-1)
                 - (v * solved).sum((-1, -2)))
    return center, covariance, loss


def fit_two_source_weight(means, covariances, residual, stats, obs):
    """Joint full-Gaussian VI and one bounded source weight, using S only.

    For each AP: f = w*h0 + (1-w)*h1 + delta. Three fixed initializations
    compare the SAME support ELBO, never held-out query errors. We use the
    original RSS and detection factors, including nondetections. The final
    location calculation still uses the established M-PMC posterior solver.
    """
    aps, _, k = means.shape
    rows, cols = torch.tril_indices(k, k)
    diagonal = rows == cols
    width = k + len(rows)
    initial = np.zeros(1 + aps * width)
    packed = initial[1:].reshape(aps, width)
    packed[:, k + np.flatnonzero(diagonal.numpy())] = 1.
    bounds = [(0., 1.)] + [(None, None)] * (aps * width)
    for ap in range(aps):
        for j in np.flatnonzero(diagonal.numpy()):
            bounds[1 + ap * width + k + j] = (1e-8, None)

    def objective(value):
        variable = torch.from_numpy(value).requires_grad_()
        w = variable[0]
        mean = w * means[:, 0] + (1 - w) * means[:, 1]
        covariance = w.square() * covariances[:, 0] + (1 - w).square() * covariances[:, 1] + residual
        center, covariance, rss_loss = gaussian_batch(mean, covariance, stats, obs['noise_std'])
        root = torch.linalg.cholesky(covariance + JITTER * torch.eye(k))
        phi = variable[1:].reshape(aps, width)
        eta, values = phi[:, :k], phi[:, k:]
        b = variable.new_zeros((aps, k, k))
        b[:, rows, cols] = values
        pm = center + (root @ eta[..., None]).squeeze(-1)
        pv = (root @ b).square().sum(-1)
        signal = pm[..., None] + pv.sqrt()[..., None] * GH_NODES
        detected, missing = _detection_logs(signal, obs)
        expected = ((stats['k'][..., None] * detected +
                     (stats['n'] - stats['k'])[..., None] * missing) * GH_WEIGHTS).sum((-1, -2))
        kl = .5 * (eta.square().sum(-1) + b.square().sum((-1, -2)) - k
                   - 2 * values[:, diagonal].log().sum(-1))
        loss = (rss_loss + kl - expected).mean()
        gradient, = torch.autograd.grad(loss, variable)
        return loss.item(), gradient.detach().numpy()

    trials = []
    for start in (.25, .5, .75):
        initial[0] = start
        result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=800, maxls=40, ftol=1e-12, gtol=1e-6))
        projected = result.jac.copy()
        if result.x[0] == 0 and projected[0] > 0 or result.x[0] == 1 and projected[0] < 0:
            projected[0] = 0
        record = dict(start=start, weight=float(result.x[0]), objective=float(result.fun),
                      iterations=int(result.nit), success=bool(result.success), message=str(result.message),
                      projected_gradient=float(np.abs(projected).max()))
        trials.append(record)
        if not result.success or not np.isfinite(result.fun) or record['projected_gradient'] > 1e-3:
            raise RuntimeError(f'Unfinished support-weight fit: {record}')
    best = min(trials, key=lambda x: x['objective'])
    return [best['weight'], 1 - best['weight']], trials
