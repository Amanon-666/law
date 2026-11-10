"""Fixed predictive log pool on the unchanged Hurdle candidate entries.

GP full-scan grid log scores are interpolated in PROBABILITY space, using
logsumexp for numerical stability. This is a piecewise-bilinear approximation
to the GP position surface, not exact GP evaluation at continuous coordinates.
Uniform 2m cell area and per-query normalization only add constants that cancel
when normalizing on candidate entries. The original GP coordinates remain a
separate, unchanged comparator; candidate restriction is explicitly reported.

Every Hurdle candidate row remains an atom, including coincident H/S entries
with different Hurdle masses. No deduplication, area reweighting, resampling,
probability floor, extrapolation, refitting or parameter selection is performed.
Hurdle/GP share H/S and Query, so this is predictive fusion, not independent
Bayesian evidence multiplication.

Counterexample: a uniform GP surface gives normalized sqrt(q_H), NOT q_H.
Fixed half/half pooling therefore softens Hurdle and does not simply ignore a
flat expert. If both distributions agree on the SAME candidate entries, their
half/half pool preserves that distribution. Neither fact guarantees improved
localization or tails. All readouts are posterior means, not mode selection.
"""
import numpy as np
from scipy.special import logsumexp


def interpolate_gp_log_density(grid, gp_scores, candidates):
    """Return [Nquery,Ncandidate] log values proportional to GP density.

    grid[G,2] is the complete rectangular 2m grid in the candidates' absolute
    metric frame; gp_scores[Nquery,G] has that exact column binding. Grid row
    order may vary, but duplicate/missing grid nodes are invalid. At least two
    nodes per axis are needed. Candidates outside the grid are rejected.
    Zero geometric weights have log weight -inf; no weights are clamped.
    The returned values are unnormalized, as are the supplied GP log scores.
    """
    grid = np.asarray(grid, dtype=np.float64)
    scores = np.asarray(gp_scores, dtype=np.float64)
    candidates = np.asarray(candidates, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError('Expected grid[G,2]')
    if candidates.ndim != 2 or candidates.shape[1] != 2 or not len(candidates):
        raise ValueError('Expected nonempty candidates[N,2]')
    if scores.ndim != 2 or scores.shape[1] != len(grid):
        raise ValueError('GP scores must be bound to grid columns')
    if not all(np.isfinite(a).all() for a in (grid, scores, candidates)):
        raise ValueError('Nonfinite grid, GP scores or candidate cache')
    x, y = np.unique(grid[:, 0]), np.unique(grid[:, 1])
    if (len(x) < 2 or len(y) < 2 or len(grid) != len(x)*len(y)
            or not np.allclose(np.diff(x), 2., rtol=0, atol=1e-8)
            or not np.allclose(np.diff(y), 2., rtol=0, atol=1e-8)):
        raise ValueError('Expected a complete rectangular 2m grid')
    ix, iy = np.searchsorted(x, grid[:, 0]), np.searchsorted(y, grid[:, 1])
    flat = iy*len(x) + ix
    if len(np.unique(flat)) != len(grid):
        raise ValueError('Duplicate grid nodes imply missing rectangle nodes')
    lookup = np.empty((len(y), len(x)), dtype=np.int64)
    lookup[iy, ix] = np.arange(len(grid))
    if ((candidates < [x[0], y[0]]).any()
            or (candidates > [x[-1], y[-1]]).any()):
        raise ValueError('Candidate outside cached GP grid; no extrapolation')
    # A point on the upper edge belongs to the final cell, with weight one.
    hi_x = np.minimum(np.searchsorted(x, candidates[:, 0], side='right'), len(x)-1)
    hi_y = np.minimum(np.searchsorted(y, candidates[:, 1], side='right'), len(y)-1)
    lo_x, lo_y = hi_x-1, hi_y-1
    tx = (candidates[:, 0]-x[lo_x])/(x[hi_x]-x[lo_x])
    ty = (candidates[:, 1]-y[lo_y])/(y[hi_y]-y[lo_y])
    corners = np.column_stack((lookup[lo_y, lo_x], lookup[lo_y, hi_x],
                               lookup[hi_y, lo_x], lookup[hi_y, hi_x]))
    weights = np.column_stack(((1-tx)*(1-ty), tx*(1-ty), (1-tx)*ty, tx*ty))
    log_weights = np.full_like(weights, -np.inf)
    np.log(weights, out=log_weights, where=weights != 0)
    return logsumexp(scores[:, corners] + log_weights[None], axis=-1)


def fuse_hurdle_gp(candidates, log_hurdle, grid, gp_scores, gp_native, bounds):
    """Five fixed readouts, with original Hurdle row masses and native GP.

    log_hurdle[Nquery,Ncandidate] must already be normalized as exported by
    the original Hurdle; it is neither resampled nor renormalized here.
    gp_native[Nquery,2] is its original projected GP output and is returned
    unchanged. bounds[2,2] are the declared H/S bounds, containing candidates.
    """
    candidates = np.asarray(candidates, dtype=np.float64)
    log_hurdle = np.asarray(log_hurdle, dtype=np.float64)
    gp_native = np.asarray(gp_native, dtype=np.float64)
    bounds = np.asarray(bounds, dtype=np.float64)
    interpolated = interpolate_gp_log_density(grid, gp_scores, candidates)
    if log_hurdle.shape != interpolated.shape or gp_native.shape != (len(interpolated), 2):
        raise ValueError('Hurdle and native GP cache rows/shapes must match')
    if not np.isfinite(log_hurdle).all() or not np.isfinite(gp_native).all():
        raise ValueError('Nonfinite Hurdle or native GP cache')
    if not np.allclose(logsumexp(log_hurdle, axis=1), 0., rtol=0, atol=1e-6):
        raise ValueError('Expected original normalized Hurdle log masses')
    if (bounds.shape != (2, 2) or not np.isfinite(bounds).all()
            or (bounds[0] > bounds[1]).any()
            or (candidates < bounds[0]).any() or (candidates > bounds[1]).any()):
        raise ValueError('H/S bounds must contain every candidate')
    log_gp = interpolated - logsumexp(interpolated, axis=1, keepdims=True)
    pooled = .5*log_hurdle + .5*interpolated
    log_pool = pooled - logsumexp(pooled, axis=1, keepdims=True)
    hurdle = np.clip(np.exp(log_hurdle) @ candidates, bounds[0], bounds[1])
    gp_candidates = np.clip(np.exp(log_gp) @ candidates, bounds[0], bounds[1])
    probability_half = np.clip(np.exp(log_pool) @ candidates, bounds[0], bounds[1])
    return dict(predictions=dict(hurdle=hurdle, GP=gp_native.copy(),
        GP_candidates=gp_candidates, coordinate_half=(hurdle+gp_native)/2,
        probability_half=probability_half),
        log_probability_half=log_pool, log_gp_candidates=log_gp)
