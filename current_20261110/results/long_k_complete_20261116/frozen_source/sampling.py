"""Existing bounded M-PMC primitives, with no experiment imports.

The legacy low-ESS QMC policy is not yet connected. This file alone is not
claimed to reproduce the full frozen evaluation backend.
"""
import math
import numpy as np
import torch
from pypmc.density.gauss import Gauss
from pypmc.density.mixture import MixtureDensity
from pypmc.mix_adapt.pmc import gaussian_pmc
from scipy.optimize import minimize
from scipy.special import log_ndtr, logsumexp
from observations import _chol, gaussian_update


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


def _draw_mixture(proposal, count, rng, trace=False):
    """pypmc 1.2.6 MixtureDensity.propose does not forward rng to components.

    Keep its multinomial mixture sampling, explicitly passing the same local
    generator to each Gaussian. Otherwise the declared seed is incomplete.
    """
    counts = rng.multinomial(count, proposal.weights)
    # Gauss.propose draws and transforms each row separately. These are the
    # same Gaussian draws in the same RNG order, evaluated as a matrix batch.
    # BLAS accumulation can differ by floating-point roundoff, not in density.
    nodes = np.concatenate([rng.normal(size=(int(n), len(component.mu))) @
                            np.linalg.cholesky(component.sigma).T + component.mu
                            for component, n in zip(proposal.components, counts) if n])
    if trace:
        return nodes, np.repeat(np.arange(len(counts)), counts)
    order = np.arange(len(nodes))
    rng.shuffle(order)
    return nodes[order]


def sample_reference(log_target, initial, center, root, seed, defensive_fraction=0.):
    rng = np.random.RandomState(seed)
    proposal = MixtureDensity([Gauss(c.mu.copy(), c.sigma.copy()) for c in initial.components], initial.weights.copy())
    adaptation = []
    for iteration in range(6):
        nodes, labels = _draw_mixture(proposal, 4096, rng, trace=True)
        logw = log_target(nodes) - proposal.multi_evaluate(nodes)
        weights = np.exp(logw - logsumexp(logw))
        adaptation.append(dict(iteration=iteration, ess=float(1 / np.sum(weights ** 2)),
                               component_counts=np.bincount(labels, minlength=len(proposal.components)).tolist(),
                               proposal_weights=proposal.weights.tolist()))
        # The documented pypmc safeguard prevents fitting a covariance from
        # an effectively empty component. This modifies the proposal only;
        # the original posterior density still determines importance weights.
        proposal = gaussian_pmc(nodes, proposal, weights=weights, latent=labels, mincount=20, rb=True)
        # pypmc 1.2.6 can return a one-component weight 1+2e-15; NumPy's
        # multinomial correctly rejects >1. Use the library's normalization.
        proposal.normalize()
        proposal.prune(0.)
    if defensive_fraction:
        # In whitened coordinates target = N(0,I) * detection likelihood <= N(0,I).
        # A fixed prior component therefore bounds unnormalized weights by 1/epsilon.
        proposal = MixtureDensity(list(proposal.components) + [Gauss(np.zeros(len(center)),np.eye(len(center)))],
            np.r_[(1-defensive_fraction)*proposal.weights, defensive_fraction])
    # Fresh final draws: never reuse adaptation samples with the final density.
    nodes = _draw_mixture(proposal, 32768, rng)
    logw = log_target(nodes) - proposal.multi_evaluate(nodes)
    log_weights = logw - logsumexp(logw)
    samples = torch.tensor(center + nodes @ root.T)
    return samples, torch.tensor(log_weights), dict(
        seed=seed, defensive_fraction=defensive_fraction, adaptation=adaptation, log_detection_evidence=float(logsumexp(logw) - math.log(len(logw))),
        effective_samples=float(1 / np.sum(np.exp(log_weights) ** 2)),
        components=len(proposal.components), mixture_weights=proposal.weights.tolist(),
        mixture_means=[c.mu.tolist() for c in proposal.components],
        mixture_covariances=[c.sigma.tolist() for c in proposal.components])


def log_proposal(samples, center, root, info):
    """Frozen proposal log density in physical RSS coordinates."""
    components = [Gauss(np.array(m), np.array(c)) for m, c in
                  zip(info["mixture_means"], info["mixture_covariances"])]
    mixture = MixtureDensity(components, np.array(info["mixture_weights"]))
    whitened = np.linalg.solve(root, (samples.numpy() - center).T).T
    return torch.tensor(mixture.multi_evaluate(whitened) - np.log(np.diag(root)).sum())
