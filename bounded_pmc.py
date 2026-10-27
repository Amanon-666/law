"""Bounded deterministic proposal starts for the existing M-PMC solver.

setup is copied from reference_pmc.setup; only its K<=10 gate and exhaustive
2**missing initialization loop are replaced. The target density, BFGS and its
stationarity threshold, positive-curvature check, mode merging and Laplace
weights are unchanged. No Support positions are dropped. This changes the
numerical proposal search, not the observation model or posterior density.

setup retains the original five-item return API. sample_reference and
log_proposal are the original functions, directly re-exported. Callers keep
defensive_fraction=.1, six 4096-draw adaptations and 32768 final draws.
"""
import math

import numpy as np
from pypmc.density.gauss import Gauss
from pypmc.density.mixture import MixtureDensity
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp

from inference import _chol, gaussian_update
from reference_pmc import sample_reference, log_proposal


def _initial_points(center, missing, beta, scale):
    """At most 2+2m complete vectors, with exact duplicate removal."""
    lowered = center.copy()
    lowered[missing] = np.minimum(lowered[missing], beta-2*scale)
    candidates = [center.copy(),lowered]
    for i in missing:
        point = center.copy();point[i] = lowered[i]
        candidates.append(point)
    for i in missing:
        point = lowered.copy();point[i] = center[i]
        candidates.append(point)
    unique = []
    for point in candidates:
        if not any(np.array_equal(point,other) for other in unique):
            unique.append(point)
    return unique


def setup(mean, covariance, counts, obs):
    rss = gaussian_update(mean, covariance, counts, obs)
    center, root = rss['mean'].numpy(), _chol(rss['cov']).numpy()
    n, k = counts['n'].numpy(), counts['k'].numpy()
    alpha,beta,scale = obs['alpha'],obs['beta'],obs['detect_scale']
    d = len(mean)

    def terms(u):
        f = center+u@root.T
        z = (f-beta)/scale
        ld = math.log(alpha)+log_ndtr(z)
        lm = np.logaddexp(math.log1p(-alpha),math.log(alpha)+log_ndtr(-z))
        return z,ld,lm

    def log_target(u):
        _,ld,lm = terms(u)
        return -.5*(d*math.log(2*math.pi)+(u*u).sum(-1))+(k*ld+(n-k)*lm).sum(-1)

    def objective(u):
        z,ld,lm = terms(u)
        logphi = -.5*(math.log(2*math.pi)+z*z)
        h = np.exp(math.log(alpha)+logphi-ld)
        b = np.exp(math.log(alpha)+logphi-lm)
        gradient = u-root.T@((k*h-(n-k)*b)/scale)
        return -float(log_target(u)),gradient

    def curvature(u):
        z,ld,lm = terms(u)
        logphi = -.5*(math.log(2*math.pi)+z*z)
        h = np.exp(math.log(alpha)+logphi-ld)
        b = np.exp(math.log(alpha)+logphi-lm)
        second = (-k*h*(z+h)+(n-k)*b*(z-b))/scale**2
        return np.eye(d)-root.T@(second[:,None]*root)

    missing = np.flatnonzero(n>k)
    modes,components,log_masses = [],[],[]
    for initial in _initial_points(center,missing,beta,scale):
        start = np.linalg.solve(root,initial-center)
        optimum = minimize(objective,start,jac=True,method='BFGS',options=dict(gtol=1e-8,maxiter=300))
        if np.max(np.abs(optimum.jac))>1e-5:
            raise RuntimeError(f'Nonstationary initialization: {np.max(np.abs(optimum.jac))}')
        if any(np.linalg.norm(optimum.x-mode)<1e-3 for mode in modes):
            continue
        precision = curvature(optimum.x)
        np.linalg.cholesky(precision)
        covariance_mode = np.linalg.inv(precision)
        modes.append(optimum.x)
        components.append(Gauss(optimum.x,covariance_mode))
        log_masses.append(log_target(optimum.x)+.5*np.linalg.slogdet(covariance_mode)[1])
    weights = np.exp(log_masses-logsumexp(log_masses))
    return log_target,MixtureDensity(components,weights),center,root,np.array(modes)
