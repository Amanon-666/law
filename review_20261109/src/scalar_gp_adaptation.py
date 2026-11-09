"""K1 support-only GP evidence profile over continuous w and existing sigma grid.

Same Gaussian/censored likelihood and jitter as ScalarSupportIntegral. This is
an empirical-Bayes reference, not a learned transfer mechanism. A dense profile
plus bracket polishing is not a mathematical global-optimality certificate.
"""
import math
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import log_ndtr, logsumexp

from observations import JITTER


def support_log_evidence(mean, variance, stats, obs, order):
    """One entry per AP; repeats and missing RSS factors are not dropped."""
    k, n, s1, s2 = (stats[x] for x in ('k', 'n', 's1', 's2'))
    count = np.maximum(k, 1.)
    noise2 = obs['noise_std']**2
    average = s1/count
    total = variance+noise2/count+JITTER
    gain = variance/total
    mu = np.where(k > 0, mean+gain*(average-mean), mean)
    pv = np.where(k > 0, variance-gain*variance, variance)
    constant = .5*((count-1)*math.log(2*math.pi*noise2)+np.log(count)+(s2-s1*s1/count)/noise2)
    rss_nll = np.where(k > 0, .5*(math.log(2*math.pi)+np.log(total)+(average-mean)**2/total)+constant, 0.)
    x, w = np.polynomial.hermite.hermgauss(order)
    signal = mu[:, None]+np.sqrt(pv[:, None]+JITTER)*x[None]*np.sqrt(2.)
    z = (signal-obs['beta'])/obs['detect_scale']
    ld = math.log(obs['alpha'])+log_ndtr(z)
    lm = np.logaddexp(math.log1p(-obs['alpha']), math.log(obs['alpha'])+log_ndtr(-z))
    logd = logsumexp(np.log(w/np.sqrt(np.pi))[None]+k[:, None]*ld+(n-k)[:, None]*lm, axis=1)
    return logd-rss_nll


def evidence_at(weight, sigma, means, variances, residual, stats, obs, order):
    mean = weight*means[:, 0]+(1-weight)*means[:, 1]
    variance = weight**2*variances[:, 0]+(1-weight)**2*variances[:, 1]+residual+sigma**2
    return support_log_evidence(mean, variance, stats, obs, order)


def fit_profile(means, variances, residual, stats, obs, sigmas, order):
    grid = np.linspace(0., 1., 101)
    profiles = []; candidates = []
    for sigma in sigmas:
        def objective(weight):
            return -float(evidence_at(weight, sigma, means, variances, residual, stats, obs, order).sum())
        values = [objective(w) for w in grid]
        profiles.append(dict(sigma=float(sigma), weights=grid.tolist(), nll=values))
        for i in (0, len(grid)-1):
            candidates.append(dict(sigma=float(sigma), weight=float(grid[i]), nll=values[i],
                                   success=True, kind='boundary'))
        for i in range(1, len(grid)-1):
            if values[i] <= values[i-1] and values[i] <= values[i+1]:
                result = minimize_scalar(objective, bounds=(grid[i-1], grid[i+1]), method='bounded',
                                         options=dict(xatol=1e-10, maxiter=100))
                candidates.append(dict(sigma=float(sigma), weight=float(result.x), nll=float(result.fun),
                                       success=bool(result.success), kind='polished', iterations=int(result.nit)))
    selected = min(candidates, key=lambda c: c['nll'])
    w = selected['weight']; h = 1e-5; lo, hi = max(0., w-h), min(1., w+h)
    def objective(weight):
        return -float(evidence_at(weight, selected['sigma'], means, variances, residual, stats, obs, order).sum())
    derivative = (objective(hi)-objective(lo))/(hi-lo)
    projected = 0. if (w == 0. and derivative > 0.) or (w == 1. and derivative < 0.) else abs(derivative)
    return dict(order=order, selected=dict(selected, derivative=derivative, projected_derivative=projected),
                candidates=candidates, profiles=profiles)
