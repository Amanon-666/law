"""Zero-training diagnostics on the 54 nine-floor events (no model, no parameter, no new fit).

D0  New ceiling after the reader changed. The 9.19 m 'true target map' bound was computed with 3NN; the adopted
    reader is the Hurdle likelihood. Read the TRUE target-floor observation distribution (scans of the target floor
    from acquisitions other than the event's S and Query acquisitions) with the SAME frozen reader:
      a) on the translator's candidates (history positions; each takes its nearest true position's statistics)
      b) on all true target-floor positions
    single-path 9.4325 minus (a) = what a perfect field could still give; (a) minus (b) = candidate-set loss.
D1  Where does the GP pool gain come from? From the adopted run's saved per-query outputs
    (results/hurdle_gp_pool_20261110/<event>_scored.npz), gain = hurdle error - probability_half error per query,
    binned by distance from the query's true position to the nearest S position and by K.
    Spatial S-conditioning predicts gain concentrated near S and growing with K; independent-error averaging
    predicts no distance dependence.

  LAW_UJI_CSV=... VT_DIR=... python diag_oracle_and_gp_gain.py --pool-dir <src>/results/hurdle_gp_pool_20261110 --out diag
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import spearmanr
from field_posterior import C, V, Hd
from run_nine_floor_field_posterior import posterior, project_summary


def acq_of(rows):
    day = np.array([np.datetime64(int(x), 's').astype('datetime64[D]').astype(str) for x in C.TS[rows]])
    return C.USER[rows], C.PHONE[rows], day


def true_field(b, t, e, aps):
    rows = np.flatnonzero((C.BLD == b) & (C.FLR == t))
    u, ph, day = acq_of(rows)
    drop = np.zeros(len(rows), bool)
    for acq in (e['query_acquisition'],):     # v2: drop only the Query acquisition (v1 also dropped S -> 18 positions left)
        drop |= (u == acq[0]) & (ph == acq[1]) & (day == acq[2])
    rows = rows[~drop]
    P, n, nd, mu, var = Hd.det_stats(C.RSS[rows][:, aps], C.XY[rows])
    p = (nd + 0.5) / (n[:, None] + 1.0)
    m = np.where(np.isnan(mu), -95.0, mu)
    s2 = np.nan_to_num(var, nan=0.0) + Hd.S0 ** 2 + 4.0 ** 2          # same scale floor as hurdle.support_candidates
    return P, p, m, s2, len(rows)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--pool-dir', required=True); ap.add_argument('--out', required=True)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=False)
    errA, errB, info = {}, {}, {}
    rows_gain = []
    for e in C.MAN['events']:
        b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
        H, S, Q = C.event_arrays(e); ctx = np.concatenate([H, S])
        aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx]); truth = C.XY[Q]
        P, p, m, s2, nrows = true_field(b, t, e, aps)
        Xc = V.candidates(b, t)
        D = np.linalg.norm(Xc[:, None] - P[None], axis=2); j = D.argmin(1)
        xy, L = posterior(Xc, p[j], m[j], s2[j], S, Q, aps)
        errA[e['id']] = np.linalg.norm(np.clip(np.exp(L) @ xy, bd[0], bd[1]) - truth, axis=1)
        xy, L = posterior(P, p, m, s2, S, Q, aps)
        errB[e['id']] = np.linalg.norm(np.clip(np.exp(L) @ xy, bd[0], bd[1]) - truth, axis=1)
        info[e['id']] = dict(true_rows=int(nrows), true_positions=int(len(P)),
                             cand_far_frac=float((D.min(1) > 4).mean()))
        # D1
        f = os.path.join(a.pool_dir, e['id'] + '_scored.npz')
        if os.path.exists(f):
            with np.load(f) as z:
                assert np.allclose(z['truth'], truth)
                Ps = np.unique(C.XY[S], axis=0)
                dS = np.linalg.norm(truth[:, None] - Ps[None], axis=2).min(1)
                for i in range(len(truth)):
                    rows_gain.append((e['id'], e['K'], e['floor'], float(dS[i]), float(z['hurdle'][i]),
                                      float(z['probability_half'][i]), float(z['GP'][i])))
        print(e['id'], 'oracle-cand %.2f oracle-all %.2f' % (errA[e['id']].mean(), errB[e['id']].mean()), flush=True)
    out = dict(D0=dict(oracle_on_candidates=project_summary(errA), oracle_on_true_positions=project_summary(errB),
                       reference=dict(adopted_single=9.4325, adopted_pool=8.7792, knn3_true_map=9.19), events=info))
    if rows_gain:
        g = np.array([(r[1], r[3], r[4] - r[5], r[4], r[6]) for r in rows_gain])
        bins = [0, 3, 6, 12, 24, 1e9]; k = np.digitize(g[:, 1], bins) - 1
        tab = {}
        for K in (1, 3, 5):
            for bi in range(len(bins) - 1):
                sel = (g[:, 0] == K) & (k == bi)
                if sel.any():
                    tab[f'K{K}_d{bins[bi]}-{bins[bi+1]:g}'] = dict(n=int(sel.sum()), gain=float(g[sel, 2].mean()),
                        hurdle=float(g[sel, 3].mean()), gp=float(g[sel, 4].mean()))
        rho, pv = spearmanr(g[:, 1], g[:, 2])
        out['D1'] = dict(table=tab, spearman_dist_vs_gain=float(rho), p=float(pv), n=len(g),
                         note='query-level, unweighted; gain = hurdle error - probability_half error (m)')
    else:
        out['D1'] = 'scored npz not found in --pool-dir'
    json.dump(out, open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k: (v if k != 'D0' else {x: v[x]['mean'] if isinstance(v[x], dict) and 'mean' in v[x] else None
                                                for x in ('oracle_on_candidates', 'oracle_on_true_positions')})
                      for k, v in out.items()}, indent=1)[:3000], flush=True)


if __name__ == '__main__':
    main()
