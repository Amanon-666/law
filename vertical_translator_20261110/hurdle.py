"""Observation-distribution field operator (hurdle form) on the vertical-translator features.

For every candidate position x and AP a of the target floor the operator predicts the distribution of ONE
real scan value:
    P(detected)           p[x, a]      HGB classifier, scan-level labels (weights = detected / missed scans)
    RSS | detected        N(m, s^2)    m: HGB regressor on detected scans; s^2: HGB on cross-fitted
                                       E[(v - m_hat)^2] = within-position variance + (mean_det - m_hat)^2
Same 55 history/support features as the mean translator (translator.py); position-level sufficient
statistics make the losses identical to scan-level Bernoulli / Gaussian losses without materialising scans.

Reader (no coordinate regression): for a query scan q and candidate c
    l_c = sum_a log[(1-eps) * P_c,a(q_a) + eps * U(q_a)]           robust to single wrong APs
with P(missing) = 1-p, P(v) = p * N(v - delta; m, s^2) and the query device offset delta marginalised over a
small grid (uniform prior), tempered l_c * beta / A, posterior mean over candidates (translated positions +
real S positions, S with their empirical 3-scan statistics). beta, eps chosen on the B0 development fold.
"""
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import core as C, translator as T, vt_lib as V

S0 = 3.0                        # floor on the scale (dB)
U_DET = 1.0 / 60.0              # uniform density over a 60 dB detected range
U_MISS = 0.5
OFFSETS = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])


def det_stats(rss, xy):
    """position-level stats of real scans: positions, n, detection count, mean and variance of detected values"""
    P, inv = np.unique(xy, axis=0, return_inverse=True); inv = inv.ravel()
    n = np.bincount(inv, minlength=len(P)).astype(float)
    det = (rss != 100).astype(float); val = np.where(rss != 100, rss, 0.0)
    nd = np.zeros((len(P), rss.shape[1])); s1 = np.zeros_like(nd); s2 = np.zeros_like(nd)
    np.add.at(nd, inv, det); np.add.at(s1, inv, val); np.add.at(s2, inv, val ** 2)
    mean = np.where(nd > 0, s1 / np.maximum(nd, 1), np.nan)
    var = np.where(nd > 1, s2 / np.maximum(nd, 1) - mean ** 2, np.nan)
    return P, n, nd, mean, np.maximum(var, 0)


STATS = {(b, f): det_stats(C.RSS[(C.BLD == b) & (C.FLR == f)], C.XY[(C.BLD == b) & (C.FLR == f)])
         for b in V.FLOORS for f in V.FLOORS[b]}


def cell_table(pairs, rng, draws=4, Ks=(1, 3, 5), max_rows=6000):
    """features + sufficient statistics of the target floor's real scans at each training cell"""
    X, n, nd, mu, var = [], [], [], [], []
    for b, t in pairs:
        Pt, nt, ndt, mt, vt = STATS[(b, t)]
        Xc, F, ref = V.hist_feats(b, t)
        D = np.linalg.norm(Xc[:, None] - Pt[None], axis=2); j = D.argmin(1); ok = D.min(1) <= T.R_LABEL
        hdet = C.DET[np.flatnonzero((C.BLD == b) & (C.FLR != t))].any(0)
        for K in Ks:
            for _ in range(draws):
                s_rows = V.sample_support(b, t, K, rng, False)
                aps = np.flatnonzero(hdet | C.DET[s_rows].any(0))
                Fe = F[ok][:, aps]; nf = Fe.shape[2]
                F2 = Fe.reshape(-1, nf)
                S2 = V.supp_feats(b, t, s_rows)[ok][:, aps].reshape(len(F2), -1)
                jj = j[ok]
                cn = np.repeat(nt[jj], len(aps)); cnd = ndt[jj][:, aps].ravel()
                cm = mt[jj][:, aps].ravel(); cv = vt[jj][:, aps].ravel()
                info = (np.nan_to_num(F2[:, 0:40:5], nan=T.ND).max(1) > T.ND) | (cnd > 0) | \
                       (np.nan_to_num(S2[:, 1], nan=T.ND) > T.ND)
                idx = np.flatnonzero(info | (rng.random(len(F2)) < 0.05))
                if len(idx) > max_rows:
                    idx = rng.choice(idx, max_rows, replace=False)
                X.append(np.hstack([F2[idx], S2[idx]])); n.append(cn[idx]); nd.append(cnd[idx])
                mu.append(cm[idx]); var.append(cv[idx])
    return np.vstack(X).astype(np.float32), *(np.concatenate(a) for a in (n, nd, mu, var))


class _Fit:
    """HGB wrapper that drops columns with no observed value in ITS training rows (sklearn binning fails on them)"""
    def __init__(self, est, X, y, w):
        self.cols = np.flatnonzero(~np.isnan(X).all(0)); self.est = est.fit(X[:, self.cols], y, sample_weight=w)

    def predict(self, X):
        return self.est.predict(X[:, self.cols])

    def predict_proba(self, X):
        return self.est.predict_proba(X[:, self.cols])


def _hgb_reg(seed):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=seed)


class HurdleField:
    def __init__(self, buildings, seed, draws=4):
        pairs = V.FLOORS and [(b, t) for b in buildings for t in V.FLOORS[b]]
        groups = [pairs[0::2], pairs[1::2]]
        parts = [cell_table(g, np.random.default_rng([seed, i]), draws) for i, g in enumerate(groups)]
        X = np.vstack([p[0] for p in parts]); n, nd, mu, var = (np.concatenate([p[k] for p in parts]) for k in (1, 2, 3, 4))
        self.cols = np.flatnonzero(~np.isnan(X).all(0))
        # detection: Bernoulli with scan counts as weights
        Xd = np.vstack([X, X])[:, self.cols]; yd = np.r_[np.ones(len(X)), np.zeros(len(X))]
        wd = np.r_[nd, n - nd]; keep = wd > 0
        self.det = _Fit(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed), Xd[keep], yd[keep], wd[keep])
        del Xd
        # intensity given detected
        d = nd > 0
        self.mean = _Fit(_hgb_reg(seed), X[d][:, self.cols], mu[d], nd[d])
        # scale: cross-fitted (half -> other half) residual + within-position variance
        Xs, ys, ws = [], [], []
        for i in (0, 1):
            Xi, ni, ndi, mi, vi = parts[i]; Xo, no, ndo, mo, vo = parts[1 - i]
            do = ndo > 0
            m_other = _Fit(_hgb_reg(seed), Xo[do][:, self.cols], mo[do], ndo[do])
            di = ndi > 0
            mh = m_other.predict(Xi[di][:, self.cols])
            Xs.append(np.hstack([Xi[di][:, self.cols], mh[:, None].astype(np.float32)]))
            ys.append(np.log(np.nan_to_num(vi[di], nan=0.0) + (mi[di] - mh) ** 2 + S0 ** 2)); ws.append(ndi[di])
        self.scale = _Fit(_hgb_reg(seed), np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))

    def predict(self, b, t, s_rows, aps):
        Xc, F, ref = V.hist_feats(b, t)
        Fe = F[:, aps]; nc, A, _ = Fe.shape
        X = np.hstack([Fe.reshape(nc * A, -1), V.supp_feats(b, t, s_rows)[:, aps].reshape(nc * A, -1)])[:, self.cols]
        p = self.det.predict_proba(X)[:, 1]
        m = self.mean.predict(X)
        s2 = np.exp(self.scale.predict(np.hstack([X, m[:, None]])))
        return Xc, p.reshape(nc, A), m.reshape(nc, A), s2.reshape(nc, A)


def support_candidates(s_rows, aps):
    P, n, nd, m, v = det_stats(C.RSS[s_rows][:, aps], C.XY[s_rows])
    p = (nd + 0.5) / (n[:, None] + 1.0)
    m = np.where(np.isnan(m), -95.0, m)
    s2 = np.nan_to_num(v, nan=0.0) + S0 ** 2 + 4.0 ** 2
    return P, p, m, s2


def read_hurdle(cand_xy, p, m, s2, s_rows, q_rows, aps, bounds, beta=4.0, eps=0.05, offsets=True):
    Ps, ps, ms, s2s = support_candidates(s_rows, aps)
    xy = np.vstack([cand_xy, Ps]); p = np.clip(np.vstack([p, ps]), 1e-4, 1 - 1e-4)
    m = np.vstack([m, ms]); s2 = np.vstack([s2, s2s])
    lmiss = np.log((1 - eps) * (1 - p) + eps * U_MISS)                 # [N, A]
    q = C.RSS[q_rows][:, aps].astype(float); A = len(aps)
    offs = OFFSETS if offsets else np.array([0.0])
    out = np.empty((len(q), 2))
    logp = np.log(p); lnorm = -0.5 * np.log(2 * np.pi * s2)
    for i, row in enumerate(q):
        dmask = row != 100
        lm = lmiss[:, ~dmask].sum(1)
        v = row[dmask][None]
        per_off = []
        for d in offs:
            ld = logp[:, dmask] + lnorm[:, dmask] - 0.5 * (v - d - m[:, dmask]) ** 2 / s2[:, dmask]
            ld = np.logaddexp(np.log(1 - eps) + ld, np.log(eps * U_DET))
            per_off.append(ld.sum(1))
        l = logsumexp(np.stack(per_off), axis=0) - np.log(len(offs)) + lm
        l = l * (beta / A if beta is not None else 1.0)
        w = np.exp(l - l.max()); w /= w.sum()
        out[i] = w @ xy
    return np.clip(out, bounds[0], bounds[1])
