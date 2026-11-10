"""Heteroscedastic translator + full-scan likelihood reader.

The translator (vt_lib.Model) predicts the target-floor mean mu[c, a]. A second regressor predicts the
squared translation error v[c, a] from the same features plus mu; it is trained on CROSS-FITTED residuals
(target floors of the training buildings split into two groups, each group predicted by a mean model trained
on the other group), so v reflects out-of-sample error, not training fit.

The reader scores every candidate c (translated positions plus the real S scans) against a query scan q by a
Gaussian log-likelihood over all event APs,
    l_c = -1/2 * sum_a [ (q_a - mu_ca)^2 / (v_ca + s0^2) + log(v_ca + s0^2) ],
tempers it by beta / A (A = number of APs), and returns the posterior-mean position (MMSE). beta and s0 are
fixed on the development fold only. This carries the GP path's strengths (full-scan likelihood, per-cell
uncertainty, posterior mean) into the learned-translation pipeline.
"""
import numpy as np
import core as C, translator as T, vt_lib as V


def pairs_of(buildings):
    return [(b, t) for b in buildings for t in V.FLOORS[b]]


def training_set_pairs(pairs, rng, draws=8, Ks=(1, 3, 5), max_rows=6000):
    """same as vt_lib.training_set (mixed-acquisition S, with support) restricted to explicit (building, floor) targets"""
    Xh, Xs, y = [], [], []
    for b, t in pairs:
        Pt, Mt, _ = V.MAPS[(b, t)]
        Xc, F, ref = V.hist_feats(b, t)
        D = np.linalg.norm(Xc[:, None] - Pt[None], axis=2); j = D.argmin(1); ok = D.min(1) <= T.R_LABEL
        hdet = C.DET[np.flatnonzero((C.BLD == b) & (C.FLR != t))].any(0)
        for K in Ks:
            for _ in range(draws):
                s_rows = V.sample_support(b, t, K, rng, False)
                aps = np.flatnonzero(hdet | C.DET[s_rows].any(0))
                Fe = F[ok][:, aps]; lab = Mt[j[ok]][:, aps]
                F2 = Fe.reshape(-1, Fe.shape[2]); l2 = lab.ravel()
                S2 = V.supp_feats(b, t, s_rows)[ok][:, aps].reshape(len(l2), -1)
                info = (np.nan_to_num(F2[:, 0:40:5], nan=T.ND).max(1) > T.ND) | (l2 > T.ND)
                info |= np.nan_to_num(S2[:, 1], nan=T.ND) > T.ND
                idx = np.flatnonzero(info | (rng.random(len(l2)) < 0.05))
                if len(idx) > max_rows:
                    idx = rng.choice(idx, max_rows, replace=False)
                Xh.append(F2[idx]); Xs.append(S2[idx]); y.append(l2[idx])
    return np.hstack([np.vstack(Xh), np.vstack(Xs)]), np.concatenate(y)


class HeteroTranslator:
    """mean model on all training targets + variance model on cross-fitted squared residuals"""
    def __init__(self, buildings, seed, draws=8):
        pairs = pairs_of(buildings)
        groups = [pairs[0::2], pairs[1::2]]                 # alternate floors: both groups span every building
        parts = []
        for g in groups:
            X, y = training_set_pairs(g, np.random.default_rng([seed, len(parts)]), draws)
            parts.append((X, y))
        Xall = np.vstack([p[0] for p in parts]); yall = np.concatenate([p[1] for p in parts])
        self.mean = V.Model(Xall, yall)
        del Xall, yall
        Xv, rv = [], []
        for i in (0, 1):
            m = V.Model(*parts[1 - i])
            Xi, yi = parts[i]
            mu = np.clip(m.predict(Xi), T.ND, 0)
            Xv.append(np.hstack([Xi, mu[:, None].astype(np.float32)])); rv.append((yi - mu) ** 2)
            del m
        parts.clear()
        Xv = np.vstack(Xv); rv = np.log1p(np.concatenate(rv))
        self.var = V.Model(Xv, rv)

    def translate(self, b, t, s_rows, aps):
        Xc, F, ref = V.hist_feats(b, t)
        Fe = F[:, aps]; n, A, _ = Fe.shape
        X = np.hstack([Fe.reshape(n * A, -1), V.supp_feats(b, t, s_rows)[:, aps].reshape(n * A, -1)])
        mu = np.clip(self.mean.predict(X), T.ND, 0)
        v = np.expm1(np.clip(self.var.predict(np.hstack([X, mu[:, None]])), 0, 12))
        return Xc, mu.reshape(n, A), v.reshape(n, A)


def read_likelihood(map_xy, mu, v, s_rows, q_rows, aps, bounds, beta=1.0, s0=4.0, s_var=9.0):
    """posterior-mean position over translated candidates + real S scans"""
    s = T.encode(C.RSS[s_rows][:, aps]).astype(float)
    M = np.vstack([mu, s]); Vv = np.vstack([v, np.full_like(s, s_var)]) + s0 ** 2
    xy = np.vstack([map_xy, C.XY[s_rows]])
    q = T.encode(C.RSS[q_rows][:, aps]).astype(float)
    A = len(aps)
    out = np.empty((len(q), 2))
    logdet = np.log(Vv).sum(1)
    inv = 1.0 / Vv
    for i, row in enumerate(q):
        l = -0.5 * ((((row[None] - M) ** 2) * inv).sum(1) + logdet)
        l = l * beta / A
        w = np.exp(l - l.max()); w /= w.sum()
        out[i] = w @ xy
    return np.clip(out, bounds[0], bounds[1])
