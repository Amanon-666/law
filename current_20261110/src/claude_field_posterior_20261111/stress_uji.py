"""Controlled reproduction of the IPIN2025 failure on UJI (no IPIN data, no parameter selection).

IPIN training-side facts (ipin_structure_20261111.md): history = ONE receipt per position (UJI: ~20 scans averaged),
3-5x sparser same-position vertical correspondence (positions interpolated along trajectories, not a shared grid),
AP axis ~1000 (UJI events ~200). The learned operator was trained only on UJI-style history. Here the DEPLOYMENT
history of the nine-floor events (B1/B2 floors other than the target) is degraded one condition at a time while the
operator (trained on B0, unchanged) and S/Query stay the same:
  full      original history (anchor: must reproduce A 9.4325 / B 9.3275 / C 9.2249)
  single    one random receipt per history position
  jitter    single + every history position moved by N(0, 3 m) per axis, independently per floor (breaks the grid)
  pad       full history, AP axis padded with every UJI column never detected in H or S (A ~200 -> 520)
  ipinlike  single + jitter + pad
Methods per condition: A adopted operator, B H-only operator, C H-only + Bayesian S (theta 16 0.3 1 0, frozen),
WKNN3 on the raw (degraded) history + S, and B with the reader temperature normalised by informative APs instead of
all APs (A_info = APs detected in the query or with max predicted p > 0.05): tests whether the beta/A scale breaks.

  LAW_UJI_CSV=... VT_DIR=... nice -n 10 python stress_uji.py --out stress_uji
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import logsumexp
import field_posterior as FP
from field_posterior import C, V, Hd
from run_nine_floor_field_posterior import project_summary

BETA, EPS = 128.0, 0.02
THETA_C = (16.0, 0.3, 1.0, 0.0)
CONDS = ('full', 'single', 'jitter', 'pad', 'ipinlike')
T = Hd.T if hasattr(Hd, 'T') else __import__('translator')


def degraded_rows_and_xy(b, cond, rng):
    """returns {floor: (rows, xy)} for building b's floors under the condition"""
    out = {}
    for f in V.FLOORS[b]:
        rows = np.flatnonzero((C.BLD == b) & (C.FLR == f)); xy = C.XY[rows].copy()
        if cond in ('single', 'jitter', 'ipinlike'):
            P, inv = np.unique(xy, axis=0, return_inverse=True); inv = inv.ravel()
            keep = np.array([rng.choice(np.flatnonzero(inv == k)) for k in range(len(P))])
            rows, xy = rows[keep], xy[keep]
            if cond in ('jitter', 'ipinlike'):
                xy = xy + rng.normal(0, 3.0, xy.shape)
        out[f] = (rows, xy)
    return out


def set_history(b, deg):
    """replace the building's position maps (used only as HISTORY of its other floors) and clear the feature cache"""
    for f, (rows, xy) in deg.items():
        V.MAPS[(b, f)] = T.position_map(C.RSS[rows], xy)
    V._HF.clear()


def read(xy_c, p, m, s2, S, Q, aps, bd, informative=False):
    Ps, ps, ms, s2s = Hd.support_candidates(S, aps)
    xy = np.vstack([xy_c, Ps]); p = np.clip(np.vstack([p, ps]), 1e-4, 1 - 1e-4)
    m = np.vstack([m, ms]); s2 = np.vstack([s2, s2s])
    lmiss = np.log((1 - EPS) * (1 - p) + EPS * Hd.U_MISS); logp = np.log(p); lnorm = -0.5 * np.log(2 * np.pi * s2)
    q = C.RSS[Q][:, aps].astype(float); A = len(aps); pmax = p.max(0) > 0.05
    out = np.empty((len(q), 2))
    for i, row in enumerate(q):
        d = row != 100; v = row[d][None]
        per = [np.logaddexp(np.log(1 - EPS) + logp[:, d] + lnorm[:, d] - 0.5 * (v - o - m[:, d]) ** 2 / s2[:, d],
                            np.log(EPS * Hd.U_DET)).sum(1) for o in Hd.OFFSETS]
        L = logsumexp(np.stack(per), 0) - np.log(len(Hd.OFFSETS)) + lmiss[:, ~d].sum(1)
        L = L * BETA / ((d | pmax).sum() if informative else A)
        w = np.exp(L - L.max()); w /= w.sum(); out[i] = w @ xy
    return np.clip(out, bd[0], bd[1])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5]); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False); t0 = time.time()
    fields = [(Hd.HurdleField([0], seed=s), FP.HurdleFieldX([0], seed=s, use_support=False)) for s in a.seeds]
    print('trained %ds' % (time.time() - t0), flush=True)
    orig = {k: v for k, v in V.MAPS.items()}
    errs = {}
    for cond in CONDS:
        rng = np.random.default_rng(20261111)
        deg = {b: degraded_rows_and_xy(b, cond, rng) for b in (1, 2)}
        for b in (1, 2):
            set_history(b, deg[b])
        for e in C.MAN['events']:
            b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
            H0, S, Q = C.event_arrays(e)
            Hrows = np.concatenate([deg[b][f][0] for f in V.FLOORS[b] if f != t])
            Hxy = np.concatenate([deg[b][f][1] for f in V.FLOORS[b] if f != t])
            det = C.DET[np.concatenate([Hrows, S])].any(0)
            aps = np.arange(C.RSS.shape[1]) if cond in ('pad', 'ipinlike') else np.flatnonzero(det)
            bd = C.bounds_of(np.vstack([Hxy, C.XY[S]])); Xc = V.candidates(b, t); truth = C.XY[Q]
            fa = FP.ensemble([f[0].predict(b, t, S, aps)[1:] for f in fields])
            fb = FP.ensemble([f[1].predict(b, t, S, aps)[1:] for f in fields])
            fc = FP.condition(Xc, *fb, S, aps, THETA_C)
            res = {'A_adopted': read(Xc, *fa, S, Q, aps, bd), 'B_H_only': read(Xc, *fb, S, Q, aps, bd),
                   'C_H_cond': read(Xc, *fc, S, Q, aps, bd),
                   'B_H_only_Ainfo': read(Xc, *fb, S, Q, aps, bd, informative=True),
                   'C_H_cond_Ainfo': read(Xc, *fc, S, Q, aps, bd, informative=True)}
            # WKNN3 on raw degraded history + S (history rows at their degraded coordinates)
            rr = np.vstack([C.RSS[Hrows], C.RSS[S]]); xx = np.vstack([Hxy, C.XY[S]])
            res['WKNN3_raw_HS'] = C.nearest3(rr, xx, C.RSS[Q], np.flatnonzero(det), bd)
            for k, pred in res.items():
                errs.setdefault(cond, {}).setdefault(k, {})[e['id']] = np.linalg.norm(pred - truth, axis=1)
        print(cond, {k: round(project_summary(v)['mean'], 3) for k, v in errs[cond].items()},
              'elapsed %ds' % (time.time() - t0), flush=True)
        V.MAPS.update(orig); V._HF.clear()
    summ = {c: {k: project_summary(v) for k, v in errs[c].items()} for c in errs}
    json.dump(dict(summary=summ, theta_c=THETA_C, seeds=a.seeds, seconds=time.time() - t0,
                   note='deployment history of B1/B2 degraded; operator trained on B0 unchanged; no IPIN data'),
              open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
