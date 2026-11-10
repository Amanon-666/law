"""Our method on law_bench, dataset-agnostic: M1 (cross-floor observation operator, H-only, trained on source
buildings under randomised deployment history conditions) -> M2 (Bayesian S conditioning) -> M3 (censored
whole-scan reader). Works on the loader structure {building: {floor: dict(rss[N,A], xy[N,2], ...)}}.

Reliability inputs (via the main thread's history_reliability.py, a30566b): per floor-offset slot,
log(1+raw scan count of the matched history position) and log(1+raw scans within the smoothing radius). Raw counts
are evidence descriptors, not effective sample sizes. Flag reliability=False reproduces v1 inputs.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_VT = os.environ.get('VT_DIR', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'claude_hurdle_20261110'))
sys.path.insert(0, os.path.abspath(_VT))
import numpy as np
from scipy.special import logsumexp
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
import translator as T
sys.path.insert(0, os.path.abspath(os.path.join(_VT, '..')))            # current_20261110/src
from history_reliability import position_evidence, neighborhood_evidence   # main thread module (a30566b)

S0, U_DET, U_MISS = 3.0, 1.0 / 60.0, 0.5
OFFSETS = np.array([-8.0, -4.0, 0.0, 4.0, 8.0])
BETA, EPS = 128.0, 0.02
CONDITIONS, COND_P, JITTER = ('full', 'single', 'single_jitter'), (0.4, 0.3, 0.3), 3.0


def det_stats(rss, xy):
    P, inv = np.unique(xy, axis=0, return_inverse=True); inv = inv.ravel()
    n = np.bincount(inv, minlength=len(P)).astype(float)
    det = (rss != 100).astype(float); val = np.where(rss != 100, rss, 0.0)
    nd = np.zeros((len(P), rss.shape[1])); s1 = np.zeros_like(nd); s2 = np.zeros_like(nd)
    np.add.at(nd, inv, det); np.add.at(s1, inv, val); np.add.at(s2, inv, val ** 2)
    mean = np.where(nd > 0, s1 / np.maximum(nd, 1), np.nan)
    var = np.where(nd > 1, s2 / np.maximum(nd, 1) - mean ** 2, np.nan)
    return P, n, nd, mean, np.maximum(var, 0)


def degrade(rss, xy, cond, rng):
    if cond == 'full':
        return rss, xy
    P, inv = np.unique(xy, axis=0, return_inverse=True); inv = inv.ravel()
    keep = np.array([rng.choice(np.flatnonzero(inv == k)) for k in range(len(P))])
    rss, xy = rss[keep], xy[keep].copy()
    if cond == 'single_jitter':
        xy = xy + rng.normal(0, JITTER, xy.shape)
    return rss, xy


def make_maps(floors, rng=None, cond='full'):
    """floors {f: dict(rss, xy)} -> {f: (P, M, R, N)}"""
    out = {}
    for f, d in floors.items():
        rss, xy = degrade(d['rss'], d['xy'], cond, rng) if cond != 'full' else (d['rss'], d['xy'])
        ev = position_evidence(rss, xy)
        out[f] = (ev['P'], ev['M'], ev['R'], ev['n_scans'].astype(float))
    return out


def sub_maps(maps, aps):
    return {f: (P, M[:, aps], R[:, aps], N) for f, (P, M, R, N) in maps.items()}


def features(X, t, maps, reliability=True):
    base, ref = T.history_features(X, t, {f: m[:3] for f, m in maps.items()})
    if not reliability:
        return base.astype(np.float32)
    n, A, _ = base.shape; extra = []
    for d in T.SLOTS:
        if t + d not in maps:
            extra += [np.full((n, A), np.nan)] * 2; continue
        P, _, _, N = maps[t + d]
        D = np.linalg.norm(X[:, None] - P[None], axis=2); j = D.argmin(1); ok = D.min(1) <= T.R_SAME
        nb = neighborhood_evidence(X, dict(P=P, n_scans=N, n_detected=np.zeros((len(P), 1))))['n_scans']
        extra += [np.broadcast_to(np.where(ok, np.log1p(N[j]), np.nan)[:, None], (n, A)),
                  np.broadcast_to(np.log1p(nb.astype(float))[:, None], (n, A))]
    return np.concatenate([base, np.stack(extra, 2)], 2).astype(np.float32)


class _Fit:
    def __init__(self, est, X, y, w):
        self.cols = np.flatnonzero(~np.isnan(X).all(0)); self.est = est.fit(X[:, self.cols], y, sample_weight=w)
    def predict(self, X): return self.est.predict(X[:, self.cols])
    def predict_proba(self, X): return self.est.predict_proba(X[:, self.cols])


def _reg(seed):
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.06, max_leaf_nodes=63, min_samples_leaf=40,
                                         l2_regularization=1.0, random_state=seed)


def episodes(buildings, rng, draws, reliability, randomise, max_rows=18000, targets=None):
    """buildings: list of {floor: dict(rss, xy)} (one AP axis per building). One episode = (building, target floor,
    history condition). Labels: real scans of the target floor (sufficient statistics at the nearest position)."""
    X, n, nd, mu, var = [], [], [], [], []
    for floors in buildings:
        fl = sorted(floors)
        if len(fl) < 2:
            continue
        stats = {f: det_stats(floors[f]['rss'], floors[f]['xy']) for f in fl}
        for t in fl:
            if targets is not None and t not in targets:
                continue
            Pt, nt, ndt, mt, vt = stats[t]
            for _ in range(draws):
                cond = rng.choice(CONDITIONS, p=COND_P) if randomise else 'full'
                maps = make_maps({f: floors[f] for f in fl if f != t}, rng, cond)
                Xc = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
                D = np.linalg.norm(Xc[:, None] - Pt[None], axis=2); j = D.argmin(1); ok = D.min(1) <= T.R_LABEL
                if not ok.any():
                    continue
                hdet = np.zeros(ndt.shape[1], bool)
                for m in maps.values():
                    hdet |= m[2].max(0) > 0
                aps = np.flatnonzero(hdet | (ndt > 0).any(0))
                pos = np.flatnonzero(ok)
                if len(pos) > 1500:                                   # memory: cap candidate positions per episode
                    pos = rng.choice(pos, 1500, replace=False)
                F = features(Xc[pos], t, sub_maps(maps, aps), reliability); F2 = F.reshape(-1, F.shape[2]); jj = j[pos]
                cn = np.repeat(nt[jj], len(aps)); cnd = ndt[jj][:, aps].ravel()
                cm = mt[jj][:, aps].ravel(); cv = vt[jj][:, aps].ravel()
                info = (np.nan_to_num(F2[:, 0:40:5], nan=T.ND).max(1) > T.ND) | (cnd > 0)
                idx = np.flatnonzero(info | (rng.random(len(F2)) < 0.05))
                if len(idx) > max_rows:
                    idx = rng.choice(idx, max_rows, replace=False)
                X.append(F2[idx]); n.append(cn[idx]); nd.append(cnd[idx]); mu.append(cm[idx]); var.append(cv[idx])
    return np.vstack(X), *(np.concatenate(a) for a in (n, nd, mu, var))


class Operator:
    def __init__(self, buildings, seed, draws=8, reliability=True, randomise=True):
        self.rel = reliability
        half = [buildings[0::2], buildings[1::2]] if len(buildings) > 1 else None
        if half is None:   # single source building: cross-fit halves = disjoint TARGET floors (labels never shared)
            fl = sorted(buildings[0])
            parts = [episodes(buildings, np.random.default_rng([seed, i]), draws, reliability, randomise,
                              targets=set(fl[i::2])) for i in (0, 1)]
        else:
            parts = [episodes(h, np.random.default_rng([seed, i]), draws, reliability, randomise) for i, h in enumerate(half)]
        X = np.vstack([p[0] for p in parts]); n, nd, mu, var = (np.concatenate([p[k] for p in parts]) for k in (1, 2, 3, 4))
        Xd = np.vstack([X, X]); yd = np.r_[np.ones(len(X)), np.zeros(len(X))]; wd = np.r_[nd, n - nd]; k = wd > 0
        self.det = _Fit(HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0, random_state=seed), Xd[k], yd[k], wd[k]); del Xd
        d = nd > 0
        self.mean = _Fit(_reg(seed), X[d], mu[d], nd[d])
        Xs, ys, ws, Xt, yt, Xn, yn, wn, Xp, yp, wp = [], [], [], [], [], [], [], [], [], [], []
        for i in (0, 1):
            Xi, ni, ndi, mi, vi = parts[i]; Xo, no, ndo, mo, vo = parts[1 - i]
            do, di = ndo > 0, ndi > 0
            mh = _Fit(_reg(seed), Xo[do], mo[do], ndo[do]).predict(Xi[di])
            Xa = np.hstack([Xi[di], mh[:, None].astype(np.float32)]); vv = np.nan_to_num(vi[di], nan=0.0)
            Xs.append(Xa); ys.append(np.log(vv + (mi[di] - mh) ** 2 + S0 ** 2)); ws.append(ndi[di])
            # M2 v2 heads (2026-11-11): field-prediction error tau2 vs single-scan noise sig2, kept separate
            Xt.append(Xa); yt.append(np.log(np.maximum((mi[di] - mh) ** 2 - vv / np.maximum(ndi[di], 1), 1.0)))
            two = ndi[di] >= 2
            Xn.append(Xa[two]); yn.append(np.log(vv[two] + S0 ** 2)); wn.append(ndi[di][two])
            # cross-fitted detection-rate error -> prior variance of the detection logit
            Xdo = np.vstack([Xo, Xo]); ydo = np.r_[np.ones(len(Xo)), np.zeros(len(Xo))]; wdo = np.r_[ndo, no - ndo]; kk = wdo > 0
            ph = _Fit(HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=63,
                min_samples_leaf=40, l2_regularization=1.0, random_state=seed), Xdo[kk], ydo[kk], wdo[kk]).predict_proba(Xi)[:, 1]
            rate = ndi / np.maximum(ni, 1); ph = np.clip(ph, 1e-3, 1 - 1e-3)
            Xp.append(np.hstack([Xi, ph[:, None].astype(np.float32)]))
            yp.append(np.log(np.maximum((rate - ph) ** 2 - ph * (1 - ph) / np.maximum(ni, 1), 1e-4))); wp.append(ni)
        self.scale = _Fit(_reg(seed), np.vstack(Xs), np.concatenate(ys), np.concatenate(ws))
        self.terr = _Fit(_reg(seed), np.vstack(Xt), np.concatenate(yt), np.concatenate(ws))
        self.noise = _Fit(_reg(seed), np.vstack(Xn), np.concatenate(yn), np.concatenate(wn))
        self.perr = _Fit(_reg(seed), np.vstack(Xp), np.concatenate(yp), np.concatenate(wp))

    def field(self, maps, t, aps, X=None):
        """-> Xc, dict(p, m, s2 [v1 scale head], tau2 [field error], sig2 [scan noise], vp [detection-rate error]).
        X: explicit positions (e.g. the S coordinates, legal at deployment); default = union of history positions"""
        Xc = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0) if X is None else np.asarray(X, float)
        sm = sub_maps(maps, aps); out = {k: [] for k in ('p', 'm', 's2', 'tau2', 'sig2', 'vp')}
        for k in range(0, len(Xc), 1500):                              # memory: chunk candidates
            X = features(Xc[k:k + 1500], t, sm, self.rel); X = X.reshape(-1, X.shape[2])
            p = self.det.predict_proba(X)[:, 1]; m = self.mean.predict(X)
            Xa = np.hstack([X, m[:, None].astype(np.float32)])
            vals = dict(p=p, m=m, s2=np.exp(self.scale.predict(Xa)), tau2=np.exp(self.terr.predict(Xa)),
                        sig2=np.exp(self.noise.predict(Xa)),
                        vp=np.exp(self.perr.predict(np.hstack([X, np.clip(p, 1e-3, 1 - 1e-3)[:, None].astype(np.float32)]))))
            for key, v in vals.items():
                out[key].append(v.reshape(-1, len(aps)))
        return Xc, {k: np.vstack(v) for k, v in out.items()}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4); return np.log(p) - np.log1p(-p)


def condition(Xc, p0, m0, s20, s_rss, s_xy, theta, r_prior=4.0):
    ell, c, g = theta[:3]; o = theta[3] if len(theta) > 3 else 0.0
    if len(s_xy) == 0 or ell <= 0 or (c <= 0 and g <= 0):
        return p0, m0, s20
    Ps, n, nd, ybar, _ = det_stats(s_rss, s_xy)
    D = np.linalg.norm(Ps[:, None] - Xc[None], axis=2); j = D.argmin(1); far = D.min(1) > r_prior
    rho_SS = o + (1 - o) * np.exp(-np.linalg.norm(Ps[:, None] - Ps[None], axis=2) / ell)
    rho_xS = o + (1 - o) * np.exp(-np.linalg.norm(Xc[:, None] - Ps[None], axis=2) / ell)
    K_ = len(Ps); big = 1e6; out_p, out_m, out_s2 = p0.copy(), m0.copy(), s20.copy()
    if c > 0:
        tau_x = np.sqrt(c * s20); tau_S = tau_x[j]; det = nd > 0
        r = np.where(det, np.nan_to_num(ybar) - m0[j], 0.0)
        noise = np.where(det & ~far[:, None], (1 - c) * s20[j] / np.maximum(nd, 1), big)
        KSS = tau_S.T[:, :, None] * tau_S.T[:, None, :] * rho_SS[None] + np.eye(K_)[None] * noise.T[:, :, None]
        W = np.linalg.inv(KSS); kxS = tau_x[:, :, None] * tau_S.T[None] * rho_xS[:, None, :]
        out_m = m0 + np.einsum('nak,ak->na', kxS, np.einsum('akl,la->ak', W, r))
        out_s2 = (1 - c) * s20 + np.maximum(c * s20 - np.einsum('nak,akl,nal->na', kxS, W, kxS), 0.0)
    if g > 0:
        ph = (nd + 0.5) / (n[:, None] + 1.0)
        zr = np.where(far[:, None], 0.0, _logit(ph) - _logit(p0[j]))
        nz = np.where(far[:, None], big, 1.0 / ((n[:, None] + 1.0) * ph * (1 - ph)))
        alpha = np.linalg.solve(g * rho_SS[None] + np.eye(K_)[None] * nz.T[:, :, None], zr.T[:, :, None])[..., 0]
        out_p = 1 / (1 + np.exp(-(_logit(p0) + g * rho_xS @ alpha.T)))
    return out_p, out_m, out_s2


def ensemble_field(ops, maps, t, aps, X=None):
    outs = [op.field(maps, t, aps, X) for op in ops]
    f = {k: np.stack([o[1][k] for o in outs]).mean(0) for k in outs[0][1]}
    mv = np.stack([o[1]['m'] for o in outs]).var(0)
    f['tau2'] = f['tau2'] + mv; f['s2'] = f['s2'] + mv                   # ensemble disagreement is field error
    return outs[0][0], f


NAIVE_TAU2, NAIVE_SIG2 = 64.0, 25.0


def calibrate_naive(source_buildings):
    """return (tau2, sig2) for the copied field from SOURCE data: leave-one-floor-out residual of the copied field at real
    positions (squared detected-mean error minus sampling variance) and mean within-position variance + S0^2"""
    res, wv = [], []
    for floors in source_buildings:
        fl = sorted(floors)
        for t in fl:
            if len(fl) < 2:
                continue
            P, n, nd, mean, var = det_stats(floors[t]['rss'], floors[t]['xy'])
            maps = make_maps({f: floors[f] for f in fl if f != t})
            aps = np.flatnonzero((nd > 0).any(0))
            f = naive_field(maps, t, aps, P)
            d = nd[:, aps] > 1
            res.append(((mean[:, aps] - f['m']) ** 2 - np.nan_to_num(var[:, aps]) / np.maximum(nd[:, aps], 1))[d])
            wv.append(np.nan_to_num(var[:, aps])[d])
    return float(max(np.mean(np.concatenate(res)), 1.0)), float(np.mean(np.concatenate(wv)) + S0 ** 2)


def naive_field(maps, t, aps, X, tau2=None, sig2=None):
    """Copied (no-learning) field. For the nearest available lower and upper history floor (the same sides as
    translator.history_features' 'ref'), take the same-position value (within R_SAME) or else the R_SMOOTH
    neighbourhood mean, for BOTH the encoded mean M and the detection rate R with identical weights and missing
    handling; average the available sides. Then p = copied R and the detected mean m = (ref - (1-p) ND) / p.
    tau2/sig2: constants calibrated on sources (passed explicitly; stored on the model object)."""
    sm = sub_maps(maps, aps)
    below = [d for d in T.SLOTS if d < 0 and t + d in sm]; above = [d for d in T.SLOTS if d > 0 and t + d in sm]
    vm, vr = [], []
    for side in (below[-1:], above[:1]):
        if not side:
            continue
        P, M, R, _ = sm[t + side[0]]
        D = np.linalg.norm(X[:, None] - P[None], axis=2); j = D.argmin(1); ok = (D.min(1) <= T.R_SAME)[:, None]
        W = (D <= T.R_SMOOTH).astype(float); cnt = W.sum(1, keepdims=True)
        with np.errstate(all='ignore'):
            smM = np.where(cnt > 0, W @ M / np.maximum(cnt, 1), np.nan); smR = np.where(cnt > 0, W @ R / np.maximum(cnt, 1), np.nan)
        vm.append(np.where(ok, M[j], smM)); vr.append(np.where(ok, R[j], smR))
    A = len(aps)
    with np.errstate(all='ignore'):
        ref = np.nanmean(np.stack(vm), 0) if vm else np.full((len(X), A), np.nan)
        pr = np.nanmean(np.stack(vr), 0) if vr else np.full((len(X), A), np.nan)
    ref = np.where(np.isnan(ref), T.ND, ref); p = np.clip(np.where(np.isnan(pr), 0.0, pr), 0.02, 0.98)
    m = np.clip(np.where(p > 0.05, (ref - (1 - p) * T.ND) / p, -90.0), -100.0, -30.0)
    tau2 = NAIVE_TAU2 if tau2 is None else tau2; sig2 = NAIVE_SIG2 if sig2 is None else sig2
    return dict(p=p, m=m, s2=np.full(p.shape, tau2 + sig2), tau2=np.full(p.shape, tau2),
                sig2=np.full(p.shape, sig2), vp=np.full(p.shape, 0.05))


def condition_v2(Xc, f, s_rss, s_xy, theta, fS=None, r_prior=4.0):
    """M2 v2. f: ensemble field dict. Prior field error lam*tau2 (spatial shape rho), S noise sig2(S)/n_d, so the
    anchor gain lam*tau2/(lam*tau2 + sig2/n_d) grows when the history-based field is unreliable. Detection logit
    prior variance lam_g * vp / (p(1-p))^2 (delta method). theta = (ell, o, lam, lam_g). Returns p, m, s2."""
    ell, o, lam, lamg = theta
    p0, m0, tau2, sig2, vp = f['p'], f['m'], lam * f['tau2'], f['sig2'], f['vp']
    if len(s_xy) == 0:
        return p0, m0, sig2 + tau2
    Ps, n, nd, ybar, _ = det_stats(s_rss, s_xy)
    if fS is None:     # legacy: prior at S = nearest history candidate (S farther than r_prior is ignored)
        D = np.linalg.norm(Ps[:, None] - Xc[None], axis=2); j = D.argmin(1); far = D.min(1) > r_prior
        pS, mS, tS, sS, vS = p0[j], m0[j], tau2[j], sig2[j], vp[j]
    else:              # prior evaluated by the operator AT the S coordinates (rows ordered like det_stats' unique Ps)
        far = np.zeros(len(Ps), bool)
        pS, mS, tS, sS, vS = fS['p'], fS['m'], lam * fS['tau2'], fS['sig2'], fS['vp']
    rho_SS = o + (1 - o) * np.exp(-np.linalg.norm(Ps[:, None] - Ps[None], axis=2) / ell)
    rho_xS = o + (1 - o) * np.exp(-np.linalg.norm(Xc[:, None] - Ps[None], axis=2) / ell)
    K_ = len(Ps); big = 1e6
    tau_x = np.sqrt(tau2); tau_S = np.sqrt(tS); det = nd > 0
    r = np.where(det, np.nan_to_num(ybar) - mS, 0.0)
    noise = np.where(det & ~far[:, None], sS / np.maximum(nd, 1), big)
    KSS = tau_S.T[:, :, None] * tau_S.T[:, None, :] * rho_SS[None] + np.eye(K_)[None] * noise.T[:, :, None]
    W = np.linalg.inv(KSS); kxS = tau_x[:, :, None] * tau_S.T[None] * rho_xS[:, None, :]
    m = m0 + np.einsum('nak,ak->na', kxS, np.einsum('akl,la->ak', W, r))
    s2 = sig2 + np.maximum(tau2 - np.einsum('nak,akl,nal->na', kxS, W, kxS), 0.0)
    p = p0
    if lamg > 0:
        pc = np.clip(p0, 1e-3, 1 - 1e-3); vz = np.clip(lamg * vp / (pc * (1 - pc)) ** 2, 1e-3, 25.0)
        pcS = np.clip(pS, 1e-3, 1 - 1e-3); vz_S = np.clip(lamg * vS / (pcS * (1 - pcS)) ** 2, 1e-3, 25.0)
        ph = (nd + 0.5) / (n[:, None] + 1.0)
        zr = np.where(far[:, None], 0.0, _logit(ph) - _logit(pS))
        nz = np.where(far[:, None], big, 1.0 / ((n[:, None] + 1.0) * ph * (1 - ph)))
        sz_x, sz_S = np.sqrt(vz), np.sqrt(vz_S)
        KZ = sz_S.T[:, :, None] * sz_S.T[:, None, :] * rho_SS[None] + np.eye(K_)[None] * nz.T[:, :, None]
        alpha = np.linalg.solve(KZ, zr.T[:, :, None])[..., 0]
        kz = sz_x[:, :, None] * sz_S.T[None] * rho_xS[:, None, :]
        p = 1 / (1 + np.exp(-(_logit(p0) + np.einsum('nak,ak->na', kz, alpha))))
    return p, m, s2


def read(cand_xy, p, m, s2, s_rss, s_xy, q_rss, informative=False):
    Ps, n, nd, mean, var = det_stats(s_rss, s_xy)
    xy = np.vstack([cand_xy, Ps]); p = np.clip(np.vstack([p, (nd + 0.5) / (n[:, None] + 1.0)]), 1e-4, 1 - 1e-4)
    m = np.vstack([m, np.where(np.isnan(mean), -95.0, mean)]); s2 = np.vstack([s2, np.nan_to_num(var, nan=0.0) + S0 ** 2 + 16.0])
    lmiss = np.log((1 - EPS) * (1 - p) + EPS * U_MISS); logp = np.log(p); lnorm = -0.5 * np.log(2 * np.pi * s2)
    A = p.shape[1]; pmax = p.max(0) > 0.05; out = np.empty((len(q_rss), 2))
    bd = np.stack([np.floor(xy.min(0) / 2) * 2, np.ceil(xy.max(0) / 2) * 2])
    for i, row in enumerate(q_rss):
        d = row != 100; v = row[d][None]
        per = [np.logaddexp(np.log(1 - EPS) + logp[:, d] + lnorm[:, d] - 0.5 * (v - o - m[:, d]) ** 2 / s2[:, d],
                            np.log(EPS * U_DET)).sum(1) for o in OFFSETS]
        L = (logsumexp(np.stack(per), 0) - np.log(len(OFFSETS)) + lmiss[:, ~d].sum(1)) * BETA / ((d | pmax).sum() if informative else A)
        w = np.exp(L - L.max()); w /= w.sum(); out[i] = w @ xy
    return np.clip(out, bd[0], bd[1])


class Ours:
    """law_bench Method. variant: 'full' (M1 v2 + M2 v2 + M3) | 'noM2' | 'C1' (M1 v2 + original C) |
    'v1' (no reliability inputs, no history randomisation, original C). theta (ell, o, lam, lam_g) from select_theta."""
    def __init__(self, theta=(16.0, 0.0, 1.0, 1.0), seeds=(1, 2, 3), variant='full', informative=False):
        self.theta, self.seeds, self.variant, self.inf = theta, seeds, variant, informative
        self.name = 'ours' if variant == 'full' else 'ours_' + variant

    def fit_source(self, source_buildings):
        rel = rnd = self.variant != 'v1'
        self.ops = [Operator(source_buildings, s, reliability=rel, randomise=rnd) for s in self.seeds]

    def predict(self, H, t, s_rss, s_xy, q_rss):
        """H {floor: dict(rss, xy)} other floors of the task building; one AP axis shared with S and Q"""
        maps = make_maps(H)
        seen = (s_rss != 100).any(0)
        for d in H.values():
            seen |= (d['rss'] != 100).any(0)
        aps = np.flatnonzero(seen)
        sr, qr = s_rss[:, aps], q_rss[:, aps]
        Ps = det_stats(sr, s_xy)[0]
        if self.variant in ('naive', 'naive_noM2'):   # ablation: M1 replaced by copying the nearest history floors (no learning)
            Xc = np.unique(np.concatenate([m[0] for m in maps.values()]), axis=0)
            tv, sv = self.naive_var                                    # stored on the object (survives joblib)
            f, fS = naive_field(maps, t, aps, Xc, tv, sv), (naive_field(maps, t, aps, Ps, tv, sv) if len(Ps) else None)
        else:
            Xc, f = ensemble_field(self.ops, maps, t, aps)
            fS = ensemble_field(self.ops, maps, t, aps, Ps)[1] if len(Ps) else None
        if self.variant in ('noM2', 'naive_noM2'):
            p, m, s2 = f['p'], f['m'], f['sig2'] + f['tau2']
        elif self.variant in ('C1', 'v1'):                              # original C conditioning (s2-proportional)
            p, m, s2 = condition(Xc, f['p'], f['m'], f['s2'], sr, s_xy, (16.0, 0.3, 1.0, 0.0))
        else:
            p, m, s2 = condition_v2(Xc, f, sr, s_xy, self.theta, fS)
        return read(Xc, p, m, s2, sr, s_xy, qr, self.inf)


# ---------------------------------------------------------------- theta selection on SOURCE data only
THETA_GRID = [(l, o, lam, lg) for l in (8.0, 16.0, 32.0) for o in (0.0, 0.5) for lam in (0.5, 1.0, 2.0) for lg in (0.0, 1.0)]


def scan_nll(p, m, s2, rss):
    p = np.clip(p, 1e-4, 1 - 1e-4); d = rss != 100
    ld = np.log(p) - 0.5 * np.log(2 * np.pi * s2) - 0.5 * (np.where(d, rss, 0) - m) ** 2 / s2
    ld = np.logaddexp(np.log(1 - EPS) + ld, np.log(EPS * U_DET))
    return -np.where(d, ld, np.log((1 - EPS) * (1 - p) + EPS * U_MISS)).mean()


def select_theta(source_buildings, seed=1, draws=8, reliability=True, randomise=True, grid=THETA_GRID, max_eval=400):
    """Held-out predictive NLL of real target-floor scans, with the field evaluated AT those scans' coordinates
    (no nearest-candidate proxy, no R_LABEL filter). >=2 source buildings: operators trained on one half of
    the buildings are evaluated on the other half; 1 building: leave-one-floor-out. S: random K in {1,3,5}
    positions (<=3 scans each); history condition drawn like training. Returns (best theta, scores)."""
    rng = np.random.default_rng([seed, 99]); scores = {th: [] for th in grid}
    if len(source_buildings) >= 2:
        splits = [(source_buildings[0::2], source_buildings[1::2]), (source_buildings[1::2], source_buildings[0::2])]
        jobs = [(Operator(tr, seed, draws, reliability, randomise), ev, None) for tr, ev in splits]
    else:
        fl = sorted(source_buildings[0]); jobs = []
        for t in fl:
            tr = [{f: source_buildings[0][f] for f in fl if f != t}]
            jobs.append((Operator(tr, seed, draws, reliability, randomise), source_buildings, t))
    for op, ev_buildings, only_t in jobs:
        for floors in ev_buildings:
            for t in sorted(floors):
                if (only_t is not None and t != only_t) or len(floors) < 2:
                    continue
                cond = rng.choice(CONDITIONS, p=COND_P) if randomise else 'full'
                maps = make_maps({f: floors[f] for f in floors if f != t}, rng, cond)
                tr_ = floors[t]; P, inv = np.unique(tr_['xy'], axis=0, return_inverse=True); inv = inv.ravel()
                for K in (1, 3, 5):
                    if len(P) <= K + 5:
                        continue
                    sp = rng.choice(len(P), K, replace=False)
                    srows = np.concatenate([rng.permutation(np.flatnonzero(inv == k))[:3] for k in sp])
                    erows = np.flatnonzero(~np.isin(inv, sp)); erows = rng.choice(erows, min(max_eval, len(erows)), replace=False)
                    seen = (tr_['rss'][srows] != 100).any(0)
                    for mp in maps.values():
                        seen |= mp[2].max(0) > 0
                    aps = np.flatnonzero(seen)
                    # field evaluated directly at the held-out real scans' coordinates (source data, legal)
                    Pe, inv_e = np.unique(tr_['xy'][erows], axis=0, return_inverse=True); inv_e = inv_e.ravel()
                    fe = op.field(maps, t, aps, Pe)[1]
                    Ps_ = det_stats(tr_['rss'][srows][:, aps], tr_['xy'][srows])[0]
                    fS = op.field(maps, t, aps, Ps_)[1]
                    ev_rss = tr_['rss'][erows][:, aps]
                    for th in grid:
                        p, m, s2 = condition_v2(Pe, fe, tr_['rss'][srows][:, aps], tr_['xy'][srows], th, fS)
                        scores[th].append(scan_nll(p[inv_e], m[inv_e], s2[inv_e], ev_rss))
    mean = {th: float(np.mean(v)) for th, v in scores.items() if v}
    return min(mean, key=mean.get), mean


def select_theta_naive(source_buildings, tau2, sig2, seed=1, grid=THETA_GRID, max_eval=400):
    """Same held-out NLL criterion as select_theta, for the copied (no-learning) field: leave-one-floor-out on every
    source building, field evaluated at the held-out scans' coordinates, history in its full condition."""
    rng = np.random.default_rng([seed, 99]); scores = {th: [] for th in grid}
    for floors in source_buildings:
        for t in sorted(floors):
            if len(floors) < 2:
                continue
            maps = make_maps({f: floors[f] for f in floors if f != t})
            tr_ = floors[t]; P, inv = np.unique(tr_['xy'], axis=0, return_inverse=True); inv = inv.ravel()
            for K in (1, 3, 5):
                if len(P) <= K + 5:
                    continue
                sp = rng.choice(len(P), K, replace=False)
                srows = np.concatenate([rng.permutation(np.flatnonzero(inv == k))[:3] for k in sp])
                erows = np.flatnonzero(~np.isin(inv, sp)); erows = rng.choice(erows, min(max_eval, len(erows)), replace=False)
                seen = (tr_['rss'][srows] != 100).any(0)
                for mp in maps.values():
                    seen |= mp[2].max(0) > 0
                aps = np.flatnonzero(seen)
                Pe, inv_e = np.unique(tr_['xy'][erows], axis=0, return_inverse=True); inv_e = inv_e.ravel()
                fe = naive_field(maps, t, aps, Pe, tau2, sig2)
                Ps_ = det_stats(tr_['rss'][srows][:, aps], tr_['xy'][srows])[0]
                fS = naive_field(maps, t, aps, Ps_, tau2, sig2)
                ev_rss = tr_['rss'][erows][:, aps]
                for th in grid:
                    p, m, s2 = condition_v2(Pe, fe, tr_['rss'][srows][:, aps], tr_['xy'][srows], th, fS)
                    scores[th].append(scan_nll(p[inv_e], m[inv_e], s2[inv_e], ev_rss))
    mean = {th: float(np.mean(v)) for th, v in scores.items() if v}
    return min(mean, key=mean.get), mean
