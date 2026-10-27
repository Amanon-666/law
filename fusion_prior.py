"""Frozen affine source maps with optional local precision-inspired mixing.

This is a mature linear-GP construction, not a learned gating mechanism or a
product of experts. Sources are assumed independent conditional on their H.
Target discrepancy is added ONCE after mixing. All sources absent uses the
existing no-history prior. Callers must not duplicate an acquisition as an
independent source. Source transforms and transfer variance are source-fitted;
only the shared source weights are adapted to target S.
"""
import numpy as np
import torch
from scipy.optimize import minimize

from history_bank import gaussian_batch
from inference import GH_NODES, GH_WEIGHTS, JITTER, _detection_logs
from multi_source_adaptation import fit_source_weights
from transfer import Prior, Transfer, rbf


def mixing_weights(weights, variances, local, transfer_variance):
    """variances [..., source, point]; source transform already propagated."""
    w = torch.as_tensor(weights, dtype=variances.dtype, device=variances.device)
    if not local:
        return torch.broadcast_to(w[:, None], variances.shape)
    precision = w[:, None] / (variances + transfer_variance)
    return precision / precision.sum(-2, keepdim=True)


def combine(means, covariances, residual, weights, local, transfer_variance):
    a = mixing_weights(weights, covariances.diagonal(dim1=-2, dim2=-1),
                       local, transfer_variance)
    return ((a * means).sum(-2),
            (a[..., :, :, None] * covariances * a[..., :, None, :]).sum(-3) + residual)


class FusionPrior:
    def __init__(self, histories, weights, base, transforms, *, local=False,
                 transfer_variance=None, intercept_std=0.):
        self.base, self.weights = base, weights
        self.local, self.transfer_variance = local, transfer_variance
        self.intercept_std = intercept_std
        self.source_count = len(histories)
        self.sources = []
        if any(h is not None for h in histories) and float(torch.as_tensor(weights).sum()) > 0:
            for history, (rho, beta) in zip(histories, transforms):
                model = Transfer('affine' if history is not None else 'identity').double()
                if history is not None:
                    with torch.no_grad():
                        model.a.fill_(rho)
                        model.b.fill_(beta / 20)
                model.requires_grad_(False)
                self.sources.append(Prior(model, history, dict(base, drift=0.)))

    def residual(self, x, y):
        scale = self.base['drift'] if self.sources else self.base['amplitude']
        return rbf(x, y, scale, self.base['length']) + self.intercept_std ** 2

    def source_moments(self, x):
        """S-only matrices for the fitter; no query access or refitting here."""
        if not self.sources:
            return (x.new_full((self.source_count, len(x)), self.base['mean']),
                    x.new_zeros((self.source_count, len(x), len(x))), self.residual(x, x))
        features = [p.features(x) for p in self.sources]
        return (torch.stack([m for m, _ in features]),
                torch.stack([p.covariance(x, x, f, f) for p, (_, f) in zip(self.sources, features)]),
                self.residual(x, x))

    def features(self, x):
        if not self.sources:
            return x.new_full((len(x),), self.base['mean']), None
        features = [p.features(x) for p in self.sources]
        variance = torch.stack([p.diagonal(x, f) for p, (_, f) in zip(self.sources, features)])
        a = mixing_weights(self.weights, variance, self.local, self.transfer_variance)
        return (a * torch.stack([m for m, _ in features])).sum(0), (a, [f for _, f in features])

    def covariance(self, x, y, fx, fy):
        result = self.residual(x, y)
        if self.sources:
            ax, xf = fx
            ay, yf = fy
            for j, p in enumerate(self.sources):
                result = result + ax[j, :, None] * p.covariance(x, y, xf[j], yf[j]) * ay[j, None, :]
        return result

    def diagonal(self, x, fx):
        scale = self.base['drift'] if self.sources else self.base['amplitude']
        result = x.new_full((len(x),), scale ** 2 + self.intercept_std ** 2)
        if self.sources:
            a, features = fx
            for j, p in enumerate(self.sources):
                result = result + a[j].square() * p.diagonal(x, features[j])
        return result


def fit_weights(means, covariances, residual, stats, obs, *, local, transfer_variance):
    """Versioned extension of the existing joint Gaussian-VI S-only fitter.

Same starts, budget and stationarity checks; local weights now depend on S
position, so the old global sum(w**2 C) objective cannot be reused unchanged.
No new hyperparameters are selected here. Frozen old files stay untouched.
"""
    aps, sources, k = means.shape
    if sources == 0:
        return [], []
    if k == 0:
        return [1. / sources] * sources, []
    if not local:
        return fit_source_weights(means, covariances, residual, stats, obs)
    if sources == 1:
        return [1.], []
    rows, cols = torch.tril_indices(k, k)
    diagonal = rows == cols
    width, prefix = k + len(rows), sources - 1
    initial = np.zeros(prefix + aps * width)
    initial[prefix:].reshape(aps, width)[:, k + np.flatnonzero(diagonal.numpy())] = 1.
    bounds = [(None, None)] * len(initial)
    diagonal_indexes = [a * width + k + j for a in range(aps) for j in np.flatnonzero(diagonal.numpy())]
    for index in diagonal_indexes:
        bounds[prefix + index] = (1e-8, None)

    def loss(weights, packed):
        mean, covariance = combine(means, covariances, residual, weights, True, transfer_variance)
        center, cov, rss_loss = gaussian_batch(mean, covariance, stats, obs['noise_std'])
        root = torch.linalg.cholesky(cov + JITTER * torch.eye(k))
        phi = packed.reshape(aps, width)
        eta, values = phi[:, :k], phi[:, k:]
        lower = packed.new_zeros((aps, k, k))
        lower[:, rows, cols] = values
        pm = center + (root @ eta[..., None]).squeeze(-1)
        pv = (root @ lower).square().sum(-1)
        signal = pm[..., None] + pv.sqrt()[..., None] * GH_NODES
        detected, missing = _detection_logs(signal, obs)
        expected = ((stats['k'][..., None] * detected +
                    (stats['n']-stats['k'])[..., None] * missing) * GH_WEIGHTS).sum((-1, -2))
        kl = .5 * (eta.square().sum(-1) + lower.square().sum((-1, -2)) - k -
                   2 * values[:, diagonal].log().sum(-1))
        return (rss_loss + kl - expected).mean()

    def objective(value):
        variable = torch.from_numpy(value).requires_grad_()
        weights = torch.softmax(torch.cat((variable[:prefix], variable.new_zeros(1))), 0)
        value = loss(weights, variable[prefix:])
        gradient, = torch.autograd.grad(value, variable)
        return value.item(), gradient.detach().numpy()

    starts = [np.full(sources, 1 / sources)]
    for source in range(sources):
        w = np.full(sources, .2 / (sources-1))
        w[source] = .8
        starts.append(w)
    trials = []
    for start in starts:
        initial[:prefix] = np.log(start[:-1] / start[-1])
        result = minimize(objective, initial, jac=True, method='L-BFGS-B', bounds=bounds,
                          options=dict(maxiter=800, maxls=40, ftol=1e-12, gtol=1e-6))
        variable = torch.from_numpy(result.x)
        weights = torch.softmax(torch.cat((variable[:prefix], variable.new_zeros(1))), 0).requires_grad_()
        gradient, = torch.autograd.grad(loss(weights, variable[prefix:]), weights)
        reduced = gradient - (weights.detach() * gradient).sum()
        violation = torch.where(weights.detach() > 1e-6, reduced.abs(), (-reduced).clamp_min(0)).max().item()
        projected = result.jac[prefix:].copy()
        for index in diagonal_indexes:
            if result.x[prefix+index] <= 1e-8 and projected[index] > 0:
                projected[index] = 0.
        good = np.isfinite(result.fun) and violation < 1e-3 and np.abs(projected).max() < 1e-3
        trials.append(dict(start=start.tolist(), weights=weights.detach().tolist(), objective=float(result.fun),
                           optimizer_success=bool(result.success), iterations=int(result.nit), message=str(result.message),
                           simplex_kkt_residual=violation, variational_gradient=float(np.abs(projected).max()),
                           stationary=bool(good)))
    valid = [row for row in trials if row['stationary']]
    if not valid:
        raise RuntimeError(f'No stationary local-source fit: {trials}')
    return min(valid, key=lambda row: row['objective'])['weights'], trials
