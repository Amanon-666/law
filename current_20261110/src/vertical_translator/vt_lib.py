"""Vertical translator library: training-set construction, model, translation of an event, readers."""
import sys
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
import core as C, translator as T

FLOORS = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3, 4]}
MAPS = {(b, f): T.position_map(C.RSS[(C.BLD == b) & (C.FLR == f)], C.XY[(C.BLD == b) & (C.FLR == f)])
        for b in FLOORS for f in FLOORS[b]}
_HF = {}


def hist_maps(b, t):
    return {f: MAPS[(b, f)] for f in FLOORS[b] if f != t}


def candidates(b, t):
    return np.unique(np.concatenate([MAPS[(b, f)][0] for f in FLOORS[b] if f != t]), axis=0)


def hist_feats(b, t):
    """cached history-only features on the candidate positions of (b, t)"""
    if (b, t) not in _HF:
        Xc = candidates(b, t)
        F, ref = T.history_features(Xc, t, hist_maps(b, t))
        _HF[(b, t)] = (Xc, F.astype(np.float32), ref)
    return _HF[(b, t)]


def supp_feats(b, t, s_rows):
    Xc, F, ref = hist_feats(b, t)
    if len(s_rows) == 0:
        return np.full((len(Xc), F.shape[1], 12), np.nan, np.float32)
    s_ref = T.naive_ref_at(C.XY[s_rows], t, hist_maps(b, t))
    return T.support_features(Xc, ref, C.XY[s_rows], C.RSS[s_rows], s_ref, F).astype(np.float32)


def sample_support(b, t, K, rng, single_acquisition):
    trows = np.flatnonzero((C.BLD == b) & (C.FLR == t))
    if single_acquisition:
        keys = np.stack([C.USER[trows], C.PHONE[trows]], 1)
        groups, ginv = np.unique(keys, axis=0, return_inverse=True); ginv = ginv.ravel()
        order = rng.permutation(len(groups))
        for g in order:
            r = trows[ginv == g]
            pos, pinv, cnt = np.unique(C.XY[r], axis=0, return_inverse=True, return_counts=True); pinv = pinv.ravel()
            good = np.flatnonzero(cnt >= 3)
            if len(good) >= K:
                sel = rng.choice(good, K, replace=False)
                return np.concatenate([r[pinv == p][np.argsort(C.TS[r[pinv == p]], kind='stable')][:3] for p in sel])
    pos, pinv = np.unique(C.XY[trows], axis=0, return_inverse=True); pinv = pinv.ravel()
    sel = rng.choice(len(pos), K, replace=False)
    return np.concatenate([rng.choice(trows[pinv == p], min(3, (pinv == p).sum()), replace=False) for p in sel])


def training_set(buildings, rng, draws=8, Ks=(1, 3, 5), max_rows=6000, single_acquisition=True, with_support=True):
    Xh, Xs, y = [], [], []
    for b in buildings:
        for t in FLOORS[b]:
            Pt, Mt, _ = MAPS[(b, t)]
            Xc, F, ref = hist_feats(b, t)
            D = np.linalg.norm(Xc[:, None] - Pt[None], axis=2); j = D.argmin(1); ok = D.min(1) <= T.R_LABEL
            hrows = np.flatnonzero((C.BLD == b) & (C.FLR != t))
            hdet = C.DET[hrows].any(0)
            for K in Ks:
                for _ in range(draws):
                    s_rows = sample_support(b, t, K, rng, single_acquisition)
                    aps = np.flatnonzero(hdet | C.DET[s_rows].any(0))
                    Fe = F[ok][:, aps]; lab = Mt[j[ok]][:, aps]
                    F2 = Fe.reshape(-1, Fe.shape[2]); l2 = lab.ravel()
                    S2 = supp_feats(b, t, s_rows)[ok][:, aps].reshape(len(l2), -1) if with_support else np.zeros((len(l2), 0), np.float32)
                    info = (np.nan_to_num(F2[:, 0:40:5], nan=T.ND).max(1) > T.ND) | (l2 > T.ND)
                    if with_support:
                        info |= np.nan_to_num(S2[:, 1], nan=T.ND) > T.ND
                    idx = np.flatnonzero(info | (rng.random(len(l2)) < 0.05))
                    if len(idx) > max_rows:
                        idx = rng.choice(idx, max_rows, replace=False)
                    Xh.append(F2[idx]); Xs.append(S2[idx]); y.append(l2[idx])
    return np.vstack(Xh), np.vstack(Xs), np.concatenate(y)


class Model:
    """HGB regressor that drops feature columns with no observed value in training."""
    def __init__(self, X, y, iters=400):
        self.cols = np.flatnonzero(~np.isnan(X).all(0))
        self.m = HistGradientBoostingRegressor(max_iter=iters, learning_rate=0.06, max_leaf_nodes=63,
            min_samples_leaf=40, l2_regularization=1.0, random_state=0).fit(X[:, self.cols], y)

    def predict(self, X):
        return self.m.predict(X[:, self.cols])


def translate(model, b, t, s_rows, aps, with_support=True):
    """-> candidate positions [n,2], predicted target-floor encoded RSS map [n, len(aps)]"""
    Xc, F, ref = hist_feats(b, t)
    Fe = F[:, aps]; n, A, _ = Fe.shape
    X = Fe.reshape(n * A, -1)
    if with_support:
        X = np.hstack([X, supp_feats(b, t, s_rows)[:, aps].reshape(n * A, -1)])
    return Xc, np.clip(model.predict(X), T.ND, 0).reshape(n, A)


def dist(q_enc, m_enc):
    return np.sqrt(((q_enc[:, None] - m_enc[None]) ** 2).mean(-1))


def read_knn(map_xy, map_enc, s_rows, q_rows, aps, bounds, k=3):
    a = np.vstack([map_enc, T.encode(C.RSS[s_rows][:, aps])]); xy = np.vstack([map_xy, C.XY[s_rows]])
    D = dist(T.encode(C.RSS[q_rows][:, aps]), a)
    idx = np.argsort(D, 1, kind='stable')[:, :k]; w = 1 / np.maximum(np.take_along_axis(D, idx, 1), 1e-8)
    w /= w.sum(1, keepdims=True)
    return np.clip((w[:, :, None] * xy[idx]).sum(1), bounds[0], bounds[1])
