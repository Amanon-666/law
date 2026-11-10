"""Learned cross-floor field translator (numpy + sklearn only).

For a target floor t of a building, the translator predicts, for every candidate position x (all
history positions) and every AP a, the expected encoded RSS on floor t from:
  * history floors only: the AP's value at the same xy on each other floor (slots by floor offset),
    neighbourhood-smoothed values, detection rates and the AP's per-floor max/mean profile;
  * the few calibration scans S: per-AP observed-minus-history differences at the S positions, pooled
    statistics across APs (low-dimensional shared change), nearest-S values and distance.
It is trained on OTHER buildings, where every floor in turn plays the target (labels = that floor's
real position-level mean map). Deployment reads only H and S of the new building.
Encoded RSS: detected -> dBm, not detected -> -105; position-level means over scans.
"""
import numpy as np

ND = -105.0
SLOTS = (-4, -3, -2, -1, 1, 2, 3, 4)
R_SAME, R_SMOOTH, R_LABEL = 5.0, 10.0, 4.0


def encode(rss):
    return np.where(rss == 100, ND, rss)


def position_map(rss, xy):
    """unique positions, mean encoded RSS [n, A], detection rate [n, A]"""
    P, inv = np.unique(xy, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = np.bincount(inv, minlength=len(P)).astype(float)[:, None]
    M = np.zeros((len(P), rss.shape[1])); R = np.zeros_like(M)
    np.add.at(M, inv, encode(rss)); np.add.at(R, inv, (rss != 100).astype(float))
    return P, M / n, R / n


def history_features(X, t, maps):
    """X [n,2] positions, t target floor number, maps {floor: (P, M, R)} of HISTORY floors only.
    Returns F [n, A, nf] and the naive reference ref [n, A] (nearest floors at same xy)."""
    n = len(X); A = next(iter(maps.values()))[1].shape[1]
    feats = []
    near = {}
    for d in SLOTS:
        h = t + d
        if h not in maps:
            feats += [np.full((n, A), np.nan)] * 5
            continue
        P, M, R = maps[h]
        D = np.linalg.norm(X[:, None] - P[None], axis=2)
        j = D.argmin(1); ok = D.min(1) <= R_SAME
        v = np.where(ok[:, None], M[j], np.nan)
        W = (D <= R_SMOOTH).astype(float); cnt = W.sum(1, keepdims=True)
        sm = np.where(cnt > 0, W @ M / np.maximum(cnt, 1), np.nan)
        det = np.where(ok[:, None], R[j], np.nan)
        fmax = np.broadcast_to(M.max(0), (n, A)); fmean = np.broadcast_to(M.mean(0), (n, A))
        feats += [v, sm, det, fmax, fmean]
        near[d] = (v, sm)
    below = [d for d in SLOTS if d < 0 and d in near]; above = [d for d in SLOTS if d > 0 and d in near]
    parts = []
    for side in (below[-1:] , above[:1]):
        if side:
            v, sm = near[side[0]]
            parts.append(np.where(np.isnan(v), sm, v))
    stack = np.stack(parts) if parts else np.full((1, n, A), np.nan)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        ref = np.nanmean(stack, 0)
    ref = np.where(np.isnan(ref), ND, ref)
    gap_dn = np.full((n, A), float(-below[-1]) if below else np.nan)
    gap_up = np.full((n, A), float(above[0]) if above else np.nan)
    feats += [ref, gap_dn, gap_up]
    return np.stack(feats, 2), ref


def support_features(X, ref, s_xy, s_rss, s_ref, hist_F):
    """S: scans s_rss [m, A] at s_xy [m,2]; s_ref [m, A] naive history reference at the S scans.
    Returns [n, A, nf] features describing how S differs from history."""
    n, A = ref.shape
    if len(s_xy) == 0:
        return np.full((n, A, 12), np.nan)
    P, inv = np.unique(s_xy, axis=0, return_inverse=True); inv = inv.ravel()
    K = len(P)
    y = np.zeros((K, A)); r = np.zeros((K, A)); c = np.bincount(inv, minlength=K)[:, None]
    np.add.at(y, inv, encode(s_rss)); np.add.at(r, inv, s_ref)
    y /= c; r /= c
    delta = y - r                                           # [K, A] observed target minus history
    seen = (y > ND) | (r > ND)                              # AP carries information at that S position
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        d_mean = np.where(seen.any(0), np.nanmean(np.where(seen, delta, np.nan), 0), np.nan)
    y_max = y.max(0); y_mean = y.mean(0)
    pooled = np.nanmean(d_mean) if np.isfinite(d_mean).any() else np.nan
    # pooled by AP "home slot": which history floor offset shows the AP strongest
    fmax = hist_F[0, :, 3::5][:, :len(SLOTS)]               # [A, slots] per-floor max profile
    home = np.where(np.isnan(fmax).all(1), -1, np.nanargmax(np.where(np.isnan(fmax), -1e9, fmax), 1))
    grp = np.full(A, np.nan)
    for g in np.unique(home):
        m = (home == g) & np.isfinite(d_mean)
        if m.any():
            grp[home == g] = d_mean[m].mean()
    # pooled by history strength bin at S (strong / weak / absent in history)
    rmax = r.max(0); bins = np.digitize(rmax, [-104.9, -85, -70])
    sgrp = np.full(A, np.nan)
    for g in np.unique(bins):
        m = (bins == g) & np.isfinite(d_mean)
        if m.any():
            sgrp[bins == g] = d_mean[m].mean()
    D = np.linalg.norm(X[:, None] - P[None], axis=2); j = D.argmin(1)
    bc = lambda v: np.broadcast_to(v, (n, A))
    out = [bc(d_mean), bc(y_max), bc(y_mean), bc(np.float64(pooled)), bc(grp), bc(sgrp),
           delta[j], y[j], r[j], np.broadcast_to(D.min(1)[:, None], (n, A)),
           bc(np.float64(K)), bc(seen.mean(0))]
    return np.stack(out, 2)


def naive_ref_at(xy, t, maps):
    _, ref = history_features(xy, t, maps)
    return ref
