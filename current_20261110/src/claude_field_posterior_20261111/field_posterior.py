"""Learned-prior field + Bayesian conditioning on the target calibration S (one field, one reader).

Why (mechanism, 2026-11-11, Claude):
  The adopted 8.78 m system multiplies two position posteriors: the learned observation-distribution operator
  (Hurdle) and the project GP.  Both read the SAME H, S and query, so the product is a predictive pool, not two
  independent evidences.  The GP's distinct ability is Bayesian conditioning of a field on S with a spatial
  covariance: near an S position the field is corrected AND becomes more certain; far away it reverts to the prior.
  The Hurdle operator sees S only through hand features (per-AP mean difference, nearest-S value / distance) and
  its variance never shrinks because S was observed.  This module puts the GP mechanism INSIDE the learned field:

      history H --(learned operator, trained across buildings)--> prior field  (p0, m0, s2_0) per (x, AP)
      S          --(Gaussian conditioning, covariance learned on source)--> posterior field (p, m, s2)
      query      --(unchanged frozen Hurdle reader)--> position posterior

  Information flow is clean in the main variant ('H'): the operator does not see S, so S is used exactly once
  (conditioning) plus as empirical S candidates in the reader, as in the adopted system.  Variant 'HS' keeps the
  adopted S features AND conditions (double use), reported only as a control.

Residual model for one AP a (all scalars below are per AP, positions x, x'):
  intensity: m_true(x) = m0(x) + e(x),  Cov[e(x), e(x')] = c * sqrt(s2_0(x) s2_0(x')) * rho(|x-x'|)
             observed S mean ybar_j = m_true(S_j) + noise,  Var = (1-c) * s2_0(S_j) / nd_j
             (the operator's s2_0 is trained on cross-fitted residual + within-position variance, so c splits it
              into translation error (spatially correlated) and scan noise (independent).)
  detection: logit p_true(x) = logit p0(x) + z(x),  Cov[z(x), z(x')] = g * rho(|x-x'|)
             observed logit((nd+.5)/(n+1)) with variance 1 / ((n+1) ph (1-ph))
  rho(d) = exp(-d / ell).   theta = (ell, c, g): three scalars, selected on source building B0 only by the
  predictive log-likelihood of held-out real target-floor scans (not by localization, never on B1/B2).

Degenerate limits (falsifiability): ell -> 0 or c, g -> 0 returns the prior field unchanged; a fitted ell far above
the floor size means 'no spatial structure, only a global offset' (already handled by the reader's offset
integration).  Either outcome refutes the hypothesis that GP's increment is spatial S-conditioning.
"""
import os, sys
_VT = os.environ.get('VT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'vertical_translator_20261110'))
sys.path.insert(0, os.path.abspath(_VT))
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
import core as C, vt_lib as V, hurdle as Hd

NH = 43          # history feature columns (8 floor-offset slots x 5 + ref + 2 gaps); support adds 12
R_PRIOR = 4.0    # S position -> nearest candidate for the prior value (same radius as training labels)


class HurdleFieldX:
    """Hurdle operator with the same recipe as hurdle.HurdleField; use_support=False drops the 12 S columns."""
    def __init__(self, buildings, seed, draws=4, use_support=True):
        self.use_support = use_support
        pairs = [(b, t) for b in buildings for t in V.FLOORS[b]]
        groups = [pairs[0::2], pairs[1::2]]
        parts = []
        for i, g in enumerate(groups):
            X, n, nd, mu, var = Hd.cell_table(g, np.random.default_rng([seed, i]), draws)
            parts.append((X if use_support else X[:, :NH], n, nd, mu, var))
        X = np.vstack([p[0] for p in parts]); n, nd, mu, var = (np.concatenate([p[k] for p in parts]) for k in (1, 2, 3, 4))
        Xd = np.vstack([X, X]); yd = np.r_[np.ones(len(X)), np.zeros(len(X))]
        wd = np.r_[nd, n - nd]; keep = wd > 0
        self.det = Hd._Fit(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed), Xd[keep], yd[keep], wd[keep])
        del Xd
        d = nd > 0
        self.mean = Hd._Fit(Hd._hgb_reg(seed), X[d], mu[d], nd[d])
        Xs, ys, ws = [], [], []
        for i in (0, 1):
            Xi, ni, ndi, mi, vi = parts[i]; Xo, no, ndo, mo, vo = parts[1 - i]
            do = ndo > 0; di = ndi > 0
            m_other = Hd._Fit(Hd._hgb_reg(seed), Xo[do], mo[do], ndo[do])
            mh = m_other.predict(Xi[di])
            Xs.append(np.hstack([Xi[di], mh[:, None].astype(np.float32)]))
            ys.append(np.log(np.nan_to_num(vi[di], nan=0.0) + (mi[di] - mh) ** 2 + Hd.S0 ** 2)); ws.append(ndi[di])
        self.scale = Hd._Fit(Hd._hgb_reg(seed), np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))

    def predict(self, b, t, s_rows, aps):
        Xc, F, ref = V.hist_feats(b, t)
        Fe = F[:, aps]; nc, A, _ = Fe.shape
        X = Fe.reshape(nc * A, -1)
        if self.use_support:
            X = np.hstack([X, V.supp_feats(b, t, s_rows)[:, aps].reshape(nc * A, -1)])
        p = self.det.predict_proba(X)[:, 1]
        m = self.mean.predict(X)
        s2 = np.exp(self.scale.predict(np.hstack([X, m[:, None].astype(np.float32)])))
        return Xc, p.reshape(nc, A), m.reshape(nc, A), s2.reshape(nc, A)


def ensemble(preds):
    """average a list of (p, m, s2) like run_nine_floor_hurdle: s2 = mean s2 + between-model variance of m"""
    ps, ms, vs = (np.stack([x[k] for x in preds]) for k in range(3))
    return ps.mean(0), ms.mean(0), vs.mean(0) + ms.var(0)


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p) - np.log1p(-p)


def condition(Xc, p0, m0, s20, s_rows, aps, theta):
    """Gaussian conditioning of the prior field on the S scans. Returns (p, m, s2) on the same candidates."""
    ell, c, g = theta
    if len(s_rows) == 0 or ell <= 0 or (c <= 0 and g <= 0):
        return p0, m0, s20
    Ps, n, nd, ybar, _ = Hd.det_stats(C.RSS[s_rows][:, aps], C.XY[s_rows])     # [K], [K], [K,A], [K,A]
    D = np.linalg.norm(Ps[:, None] - Xc[None], axis=2)                            # [K, N]
    j = D.argmin(1)                                                                # prior at S = nearest candidate
    far = D.min(1) > R_PRIOR
    rho_SS = np.exp(-np.linalg.norm(Ps[:, None] - Ps[None], axis=2) / ell)        # [K, K]
    rho_xS = np.exp(-np.linalg.norm(Xc[:, None] - Ps[None], axis=2) / ell)        # [N, K]
    K_, A = nd.shape
    big = 1e6
    out_m, out_s2, out_p = m0.copy(), s20.copy(), p0.copy()
    if c > 0:
        tau_x = np.sqrt(c * s20)                                                   # [N, A]
        tau_S = tau_x[j]                                                           # [K, A]
        det = nd > 0
        r = np.where(det, np.nan_to_num(ybar) - m0[j], 0.0)                        # [K, A]
        noise = np.where(det & ~far[:, None], (1 - c) * s20[j] / np.maximum(nd, 1), big)
        KSS = tau_S.T[:, :, None] * tau_S.T[:, None, :] * rho_SS[None] \
            + np.eye(K_)[None] * noise.T[:, :, None]                               # [A, K, K]
        W = np.linalg.inv(KSS)                                                     # K <= 5
        kxS = tau_x[:, :, None] * tau_S.T[None] * rho_xS[:, None, :]              # [N, A, K]
        alpha = np.einsum('akl,la->ak', W, r)                                      # [A, K]
        out_m = m0 + np.einsum('nak,ak->na', kxS, alpha)
        red = np.einsum('nak,akl,nal->na', kxS, W, kxS)
        out_s2 = (1 - c) * s20 + np.maximum(c * s20 - red, 0.0)
    if g > 0:
        ph = (nd + 0.5) / (n[:, None] + 1.0)
        zr = np.where(far[:, None], 0.0, _logit(ph) - _logit(p0[j]))               # [K, A]
        nz = np.where(far[:, None], big, 1.0 / ((n[:, None] + 1.0) * ph * (1 - ph)))
        KSS = g * rho_SS[None] + np.eye(K_)[None] * nz.T[:, :, None]               # [A, K, K]
        alpha = np.linalg.solve(KSS, zr.T[:, :, None])[..., 0]                     # [A, K]
        out_p = 1 / (1 + np.exp(-(_logit(p0) + g * rho_xS @ alpha.T)))             # [N, A]
    return out_p, out_m, out_s2


def scan_loglik(p, m, s2, rss, eps=0.02):
    """per-scan, per-AP robust hurdle log-likelihood of real scans rss [M, A] at fields p,m,s2 [M, A] (no offset)"""
    p = np.clip(p, 1e-4, 1 - 1e-4); d = rss != 100
    lmiss = np.log((1 - eps) * (1 - p) + eps * Hd.U_MISS)
    ld = np.log(p) - 0.5 * np.log(2 * np.pi * s2) - 0.5 * (np.where(d, rss, 0) - m) ** 2 / s2
    ld = np.logaddexp(np.log(1 - eps) + ld, np.log(eps * Hd.U_DET))
    return np.where(d, ld, lmiss)


def residual_correlogram(Xc, m, b, t, aps, bins=(0, 3, 6, 12, 24, 48)):
    """D2: spatial correlation of (true detected mean - predicted m) for the same AP, plus cross-AP same-position
    correlation. Candidates are matched to the target floor's real positions within R_PRIOR."""
    Pt, nt, ndt, mt, vt = Hd.STATS[(b, t)]
    D = np.linalg.norm(Xc[:, None] - Pt[None], axis=2); j = D.argmin(1); ok = D.min(1) <= R_PRIOR
    R = np.where(ndt[j][:, aps] > 0, mt[j][:, aps] - m, np.nan)[ok]               # [n_ok, A]
    X = Xc[ok]
    dd = np.linalg.norm(X[:, None] - X[None], axis=2)
    iu = np.triu_indices(len(X), 1)
    edges = np.asarray(bins, float)
    num = np.zeros(len(edges)); den = np.zeros(len(edges)); var_sum = 0.0; var_n = 0
    for a in range(R.shape[1]):
        r = R[:, a]; v = np.isfinite(r)
        if v.sum() < 10:
            continue
        r = r - np.nanmean(r)                                                       # remove the per-AP offset
        var_sum += np.nansum(r ** 2); var_n += v.sum()
        rr = np.outer(np.nan_to_num(r), np.nan_to_num(r))[iu]; vv = np.outer(v, v)[iu]
        k = np.digitize(dd[iu], edges) - 1
        np.add.at(num, k[vv], rr[vv]); np.add.at(den, k[vv], 1)
    var = var_sum / max(var_n, 1)
    corr = np.where(den > 0, num / np.maximum(den, 1) / var, np.nan)
    pos_common = np.nanmean(R - np.nanmean(R, 0), 1)                               # same-position cross-AP mean
    return dict(bins=list(edges), corr=corr.tolist(), pairs=den.tolist(), resid_sd=float(np.sqrt(var)),
                ap_offset_sd=float(np.nanstd(np.nanmean(R, 0))),
                position_common_sd=float(np.nanstd(pos_common)))
