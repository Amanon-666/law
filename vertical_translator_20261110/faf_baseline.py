"""Hand-designed translation baseline: multi-floor attenuation factor (as in multi-wall multi-floor models).

Prediction for candidate x, AP a on target floor t:
    mu(x, a) = max_h [ v_h(x, a) - alpha * |t - h| ]     over history floors h with a value at x
(nearest history position within 5 units, else 10-unit neighbourhood mean; -105 = not detected; result
clipped to [-105, 0]). alpha (dB per floor) is fitted on the SOURCE buildings by map RMSE against their real
floor maps (each floor in turn as target). Optional S calibration: per-AP mean of (S - mu) at the S positions
over APs seen there, added to mu where mu is detected (classic offset calibration).
Same 3NN reader and same events as run_nine_floor.py, so the gap to the learned translator is the value of
learning the translation instead of fixing its form.

  LAW_UJI_CSV=... python faf_baseline.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core as C, translator as T, vt_lib as V

ALPHAS = np.arange(0, 31, 1.0)


def faf_map(b, t, alpha, Xc=None):
    Xc = V.candidates(b, t) if Xc is None else Xc
    best = None
    for h, (P, M, R) in V.hist_maps(b, t).items():
        D = np.linalg.norm(Xc[:, None] - P[None], axis=2)
        j = D.argmin(1); ok = D.min(1) <= T.R_SAME
        W = (D <= T.R_SMOOTH).astype(float); cnt = W.sum(1, keepdims=True)
        sm = np.where(cnt > 0, W @ M / np.maximum(cnt, 1), T.ND)
        v = np.where(ok[:, None], M[j], sm)
        cand = np.where(v > T.ND, v - alpha * abs(t - h), T.ND)
        best = cand if best is None else np.maximum(best, cand)
    return Xc, np.clip(best, T.ND, 0)


def fit_alpha(buildings):
    err = np.zeros(len(ALPHAS)); n = 0
    for b in buildings:
        for t in V.FLOORS[b]:
            Pt, Mt, _ = V.MAPS[(b, t)]
            Xc = V.candidates(b, t)
            D = np.linalg.norm(Xc[:, None] - Pt[None], axis=2); j = D.argmin(1); ok = D.min(1) <= T.R_LABEL
            lab = Mt[j[ok]]
            for i, a in enumerate(ALPHAS):
                _, m = faf_map(b, t, a, Xc)
                m = m[ok]; inf = (m > T.ND) | (lab > T.ND)
                err[i] += ((m - lab)[inf] ** 2).sum()
            n += 1
    return float(ALPHAS[err.argmin()])


def s_offset(mu, Xc, s_rows, aps):
    s = T.encode(C.RSS[s_rows][:, aps]); sxy = C.XY[s_rows]
    j = np.linalg.norm(sxy[:, None] - Xc[None], axis=2).argmin(1)
    ref = mu[j]; seen = (s > T.ND) & (ref > T.ND)
    off = np.where(seen.any(0), np.where(seen, s - ref, 0).sum(0) / np.maximum(seen.sum(0), 1), 0.0)
    return np.where(mu > T.ND, np.clip(mu + off[None], T.ND, 0), mu)


TRAIN = dict(lobo={1: [0, 2], 2: [0, 1]}, b0only={1: [0], 2: [0]})
out = {}
for tag, tr in TRAIN.items():
    alpha = {hb: fit_alpha(src) for hb, src in tr.items()}
    print(tag, 'alpha dB/floor', alpha, flush=True)
    res = {}
    for e in C.MAN['events']:
        b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
        H, S, Q = C.event_arrays(e); ctx = np.concatenate([H, S])
        aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx])
        Xc, m = faf_map(b, t, alpha[b]); m = m[:, aps]
        res.setdefault('FAF_H', {})[e['id']] = C.event_error(e, V.read_knn(Xc, m, S, Q, aps, bd))
        res.setdefault('FAF_HS_offset', {})[e['id']] = C.event_error(e, V.read_knn(Xc, s_offset(m, Xc, S, aps), S, Q, aps, bd))
        Xc0, m0 = faf_map(b, t, 0.0); m0 = m0[:, aps]
        res.setdefault('max_over_floors_alpha0', {})[e['id']] = C.event_error(e, V.read_knn(Xc0, m0, S, Q, aps, bd))
    rows = []
    for k, r in res.items():
        s = C.summarize(r)
        rows.append(dict(train=tag, method=k, mean=s['mean'], **{f'K{a}': x for a, x in s['by_K'].items()}, **s['by_floor']))
    print(pd.DataFrame(rows).round(2).to_string(), flush=True)
    out[tag] = dict(alpha=alpha, events=res)
json.dump(out, open('faf_baseline.json', 'w'))
