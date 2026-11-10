"""Literature GP fingerprinting baselines for law_bench (array level; no project-specific assets).

gp_S   SIMPLIFIED GP fingerprinting (not a faithful Ferris 2006 reproduction): per AP a GP over position fitted to
       the target calibration S only (constant prior mean = S mean of that AP); whole-scan Gaussian likelihood on a
       2 m grid raised to the power 1/n_used (the tempering of Ferris et al. eq. 14, gamma = 1/#APs used),
       MAP position (the tempered posterior mean degenerates to the grid centroid). Not-detected handled by -105 encoding, not by a censoring model.
gp_HS  same GP and reader, but the prior mean is the history reference field (nearest lower/upper history floor
       at the same planar position, as in translator.history_features 'ref'), and the GP models the residual of S
       around it. This is the standard 'history as prior mean + few-shot Bayesian correction' recipe
       (ScaML-GP / AcMu-style use of old maps) without any learned component - the natural non-learned rival of ours.

Encoding: detected dBm, not detected -> -105 (encoded), as in most fingerprint GP baselines.
Hyperparameters (ell, sf2, sn2) are shared by all APs and chosen once on SOURCE buildings by the summed GP
marginal likelihood of real target-floor scans (never on the task building).
"""
import numpy as np
from scipy.special import logsumexp

ND = -105.0


def enc(rss):
    return np.where(rss == 100, ND, rss).astype(float)


def rbf(a, b, ell):
    d2 = ((a[:, None] - b[None]) ** 2).sum(-1)
    return np.exp(-0.5 * d2 / ell ** 2)


def gp_posterior(Xs, R, Xg, ell, sf2, sn2):
    """Xs [K,2] calibration positions, R [K,A] residual targets (already minus prior mean), Xg [G,2]
    -> mean residual [G,A], predictive variance [G] (shared by APs: same kernel, same inputs)"""
    Kss = sf2 * rbf(Xs, Xs, ell) + sn2 * np.eye(len(Xs))
    Kgs = sf2 * rbf(Xg, Xs, ell)
    L = np.linalg.cholesky(Kss + 1e-9 * np.eye(len(Xs)))
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, R))
    v = np.linalg.solve(L, Kgs.T)
    return Kgs @ alpha, sf2 - (v ** 2).sum(0) + sn2


def grid_of(xy, step=2.0):
    lo = np.floor(xy.min(0) / step) * step; hi = np.ceil(xy.max(0) / step) * step
    gx, gy = np.meshgrid(np.arange(lo[0], hi[0] + 1e-9, step), np.arange(lo[1], hi[1] + 1e-9, step))
    return np.c_[gx.ravel(), gy.ravel()]


def locate(q_enc, mu, var, grid, aps_mask=None, readout='map'):
    """Gaussian whole-scan likelihood on the grid, tempered by 1/n_used (Ferris eq. 14); MAP readout.
    (2026-11-11: the posterior MEAN of the 1/n-tempered posterior is nearly flat and collapses to the grid centroid
    - observed ~36 m on the UJI B0 fold - so the readout is the maximum, which tempering does not change.)"""
    out = np.empty((len(q_enc), 2))
    for i, q in enumerate(q_enc):
        use = (q > ND) | (mu.max(0) > ND + 5) if aps_mask is None else aps_mask
        ll = -0.5 * (((q[use][None] - mu[:, use]) ** 2) / var[:, None] + np.log(var)[:, None]).sum(1) / max(use.sum(), 1)
        if readout == 'map':
            out[i] = grid[np.argmax(ll)]
        else:                                  # original posterior-mean readout (kept as its own version)
            w = np.exp(ll - logsumexp(ll)); out[i] = w @ grid
    return out


def s_positions(s_rss, s_xy):
    P, inv = np.unique(s_xy, axis=0, return_inverse=True); inv = inv.ravel()
    Y = np.zeros((len(P), s_rss.shape[1])); c = np.bincount(inv, minlength=len(P))[:, None]
    np.add.at(Y, inv, enc(s_rss)); return P, Y / c


def gp_S(s_rss, s_xy, q_rss, bounds_xy, hyper, readout='map'):
    """S enters scan by scan (<= K x R rows, repeated positions kept; sn2 is the per-scan noise)"""
    ell, sf2, sn2 = hyper
    P, Y = s_xy, enc(s_rss)
    m0 = Y.mean(0)
    grid = grid_of(bounds_xy)
    mu, var = gp_posterior(P, Y - m0, grid, ell, sf2, sn2)
    return locate(enc(q_rss), mu + m0, var, grid, readout=readout)


def gp_HS(s_rss, s_xy, q_rss, bounds_xy, ref_fn, hyper, readout='map'):
    """ref_fn(X [n,2]) -> history reference field [n,A] (encoded dBm) at arbitrary positions"""
    ell, sf2, sn2 = hyper
    P, Y = s_xy, enc(s_rss)                                         # scan level, as in gp_S
    grid = grid_of(bounds_xy)
    mu, var = gp_posterior(P, Y - ref_fn(P), grid, ell, sf2, sn2)
    return locate(enc(q_rss), mu + ref_fn(grid), var, grid, readout=readout)


def fit_hyper(floors, grid=((4, 100, 16), (8, 100, 16), (16, 100, 16), (8, 50, 16), (8, 200, 16),
                            (8, 100, 9), (8, 100, 36), (16, 200, 25), (4, 50, 25))):
    """floors: list of (xy [n,2], Y [n,A] encoded RAW SCANS) from SOURCE buildings; pick the shared (ell, sf2, sn2)
    maximising the summed GP log marginal likelihood (constant mean per AP) on 150 real scans per floor, the same
    scan level at which S enters at deployment; the subsample is fixed once."""
    rng = np.random.default_rng(0); best, bestv = None, -np.inf
    sub = [rng.choice(len(xy), min(150, len(xy)), replace=False) for xy, _ in floors]   # fixed once for all candidates
    for ell, sf2, sn2 in grid:
        tot = 0.0
        for (xy, Y), idx in zip(floors, sub):
            X, R = xy[idx], Y[idx]
            keep = (R > ND).any(0); R = R[:, keep] - R[:, keep].mean(0)
            Kx = sf2 * rbf(X, X, ell) + sn2 * np.eye(len(X)); L = np.linalg.cholesky(Kx + 1e-9 * np.eye(len(X)))
            a = np.linalg.solve(L.T, np.linalg.solve(L, R))
            tot += float(-0.5 * (R * a).sum() - R.shape[1] * np.log(np.diag(L)).sum())
        if tot > bestv:
            best, bestv = (float(ell), float(sf2), float(sn2)), tot
    return best
