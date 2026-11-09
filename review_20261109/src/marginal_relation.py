"""Observed three-state history relations with exact GP marginal backfill.

The RSS cut is the GP latent mean at each candidate. Bins describe observed
events, not ranks hidden in the non-detection atom. Within-bin dependence is
deliberately discarded. IPF and forest factorization are mature mechanisms.
"""
import itertools
import math

import numpy as np
from scipy.special import ndtr, owens_t

IPF_TOL = 1e-8
IPF_STEPS = 500


def gp_bins(mean, variance, obs):
    """P(missing), P(detected,R<=mean), P(detected,R>mean).

R and detection share F. Meyer arXiv:0912.2816 (3.14) evaluates the
bivariate normal probability at the standardized RSS cut zero.
"""
    v = np.asarray(variance)
    z = (mean-obs['beta'])/np.sqrt(v+obs['detect_scale']**2)
    rho = v/np.sqrt((v+obs['noise_std']**2)*(v+obs['detect_scale']**2))
    t = owens_t(z, rho/np.sqrt(1-rho*rho))
    half = .5*ndtr(z)
    p = np.stack((1-obs['alpha']*2*half, obs['alpha']*(half-t),
                  obs['alpha']*(half+t)), axis=-1)
    assert np.isfinite(p).all() and (p > 0).all(), 'GP bin probability outside positive floating-point range'
    assert np.max(np.abs(p.sum(-1)-1)) <= 1e-12
    return p


def history_tables(proxy, mean, edges):
    """Integrate the same empirical mixed kernel used in section 149.

Eight candidate positions per chunk bound memory. Direct row sums retain
pairing; no sampling, independent imputation or location pruning is used.
"""
    rss = proxy['rss']
    detected = (rss != 100).astype(float)
    missing = 1-detected
    nodes = np.empty((*mean.shape, 3))
    pairs = np.empty((len(mean), len(edges), 3, 3))
    a, b = edges.T
    for start in range(0, len(mean), 8):
        sl = slice(start, start+8)
        weights = proxy['spatial'][sl, proxy['group']]
        low = ndtr((mean[sl, None, :]-rss[None, :, :])/proxy['bandwidth'])*detected
        kernel = (np.broadcast_to(missing, low.shape), low, detected-low)
        for i in range(3):
            nodes[sl, :, i] = np.einsum('gn,gna->ga', weights, kernel[i])
        for i in range(3):
            left = kernel[i][:, :, a]
            for j in range(3):
                pairs[sl, :, i, j] = np.einsum('gn,gne,gne->ge', weights, left,
                                               kernel[j][:, :, b], optimize=True)
    assert np.max(np.abs(nodes.sum(-1)-1)) <= 1e-10
    assert np.max(np.abs(pairs.sum(-1)-nodes[:, a])) <= 1e-10
    assert np.max(np.abs(pairs.sum(-2)-nodes[:, b])) <= 1e-10
    return nodes, pairs


def project_relations(history_pairs, probabilities, edges):
    """Positive-background IPF: preserve GP marginals and table odds ratios."""
    a, b = edges.T
    row, column = probabilities[:, a], probabilities[:, b]
    independent = row[..., :, None]*column[..., None, :]
    table = .5*(independent+history_pairs)
    assert (table > 0).all() and np.isfinite(table).all()
    for iteration in range(1, IPF_STEPS+1):
        table *= (row/table.sum(-1))[..., :, None]
        table *= (column/table.sum(-2))[..., None, :]
        error = max(float(np.max(np.abs(table.sum(-1)/row-1))),
                    float(np.max(np.abs(table.sum(-2)/column-1))))
        if error <= IPF_TOL:
            break
    else:
        raise RuntimeError(f'IPF did not meet frozen marginal tolerance: {error}')
    ratio = np.log(table)-np.log(row)[..., :, None]-np.log(column)[..., None, :]
    assert np.isfinite(ratio).all()
    return ratio, dict(iterations=iteration, maximum_relative_marginal_error=error,
                       minimum_GP_bin_probability=float(probabilities.min()))


def correction(query, field, edges, log_ratio):
    """Full continuous GP likelihood is multiplied by the selected cell ratio."""
    classes = np.where(query[None, :] == 100, 0,
                       np.where(query[None, :] <= field['mean'], 1, 2))
    a, b = edges.T
    return log_ratio[np.arange(len(field['grid']))[:, None], np.arange(len(edges))[None, :],
                     classes[:, a], classes[:, b]].sum(1)


def finite_check():
    """One finite enumeration verifies the actual scaling/backfill factors."""
    p = np.array([[[.2, .3, .5], [.4, .35, .25], [.1, .7, .2]]])
    edges = np.array([[0, 1], [1, 2]])
    h = np.array([[[[1., 2., 3.], [2., 6., 1.], [4., 2., 2.]],
                   [[5., 1., 2.], [1., 7., 1.], [2., 2., 3.]]]])
    h /= h.sum(axis=(-2, -1), keepdims=True)
    ratio, diagnostics = project_relations(h, p, edges)
    states = np.array(list(itertools.product(range(3), repeat=3)))
    mass = np.array([np.prod(p[0, np.arange(3), row])
        *np.exp(sum(ratio[0, e, row[a], row[b]] for e, (a, b) in enumerate(edges)))
        for row in states])
    error = abs(mass.sum()-1)
    for a in range(3):
        for state in range(3):
            error = max(error, abs(mass[states[:, a] == state].sum()-p[0, a, state]))
    independent = p[:, edges[:, 0], :, None]*p[:, edges[:, 1], None, :]
    identity, _ = project_relations(independent, p, edges)
    error = max(error, float(np.abs(identity).max()))
    obs = dict(alpha=.8, beta=-85., detect_scale=5., noise_std=3.)
    m, v = np.array([-90., -85.]), np.array([0., 25.])
    bins = gp_bins(m, v, obs)
    rho = v[1]/math.sqrt((v[1]+25.)*(v[1]+9.))
    expected_low = .8*(.25-math.asin(rho)/(2*math.pi))
    error = max(error, abs(bins[0, 1]-.4*ndtr(-1)), abs(bins[1, 1]-expected_low))
    assert error <= 1e-8, error
    return dict(passed=True, enumerated_states=27, maximum_error=float(error), IPF=diagnostics)
