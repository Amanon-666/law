"""Nine-floor 54-event run of the learned-prior + Bayesian-S field, paired with the adopted system.

Run ONLY if dev_field_posterior_b0only.py passed its gate; theta is the NLL-selected value printed there (frozen,
never tuned on these events). Training source: B0 only, seeds 1-5, same recipe as the adopted five-model Hurdle.

Variants, all read by the frozen Hurdle reader (beta=128, eps=0.02, offsets on), all fused with the SAME cached GP
score maps by the UNCHANGED hurdle_gp_pool.fuse_hurdle_gp (imported, not modified):
  A  adopted operator (H+S features), seeds 1-5             -> must reproduce 9.4325 / 8.7792 (anchor)
  B  H-only operator, no conditioning                       -> history alone
  C  H-only operator + Bayesian S conditioning (theta)      -> main variant
  D  H+S operator + Bayesian S conditioning (theta)         -> control (double use of S)

Decision (written before running):
  adopt C (or D) as the single-field system if its single path < 9.4325 AND its GP pool < 8.7792 with no overall
  or K1 p90 regression; record whether the GP pool gain (single - pool) SHRINKS versus A's 0.653 m. Only
  'single better AND pool gain shrinks' supports 'the learner absorbed GP's S-conditioning'.

  LAW_UJI_CSV=... VT_DIR=... POOL_SRC=<current_20261110/src> \
  nice -n 10 python run_nine_floor_field_posterior.py --theta 8 0.7 1 --gp-records <.../position_fusion_20261109_continued/target/complete.json> --out nine_fp
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import logsumexp
import field_posterior as FP
from field_posterior import C, V, Hd

BETA, EPS = 128.0, 0.02
VARIANTS = ('A_adopted', 'B_H_only', 'C_H_cond', 'D_HS_cond')


def posterior(cand_xy, p, m, s2, s_rows, q_rows, aps):
    """identical to run_nine_floor_hurdle.posterior (frozen reader), returning candidates and normalised log mass"""
    Ps, ps, ms, s2s = Hd.support_candidates(s_rows, aps)
    xy = np.vstack([cand_xy, Ps]); p = np.clip(np.vstack([p, ps]), 1e-4, 1 - 1e-4)
    m = np.vstack([m, ms]); s2 = np.vstack([s2, s2s])
    lmiss = np.log((1 - EPS) * (1 - p) + EPS * Hd.U_MISS)
    logp = np.log(p); lnorm = -0.5 * np.log(2 * np.pi * s2)
    q = C.RSS[q_rows][:, aps].astype(float); A = len(aps)
    L = np.empty((len(q), len(xy)))
    for i, row in enumerate(q):
        d = row != 100; v = row[d][None]
        per = [np.logaddexp(np.log(1 - EPS) + logp[:, d] + lnorm[:, d] - 0.5 * (v - o - m[:, d]) ** 2 / s2[:, d],
                            np.log(EPS * Hd.U_DET)).sum(1) for o in Hd.OFFSETS]
        L[i] = (logsumexp(np.stack(per), 0) - np.log(len(Hd.OFFSETS)) + lmiss[:, ~d].sum(1)) * BETA / A
    return xy, L - logsumexp(L, 1, keepdims=True)


def project_summary(errors):
    """errors {event_id: per-query error array}; project metric: position-balanced, equal K/building/floor/direction,
    mean and pooled position_mean_p90 (same weights as hurdle_gp_pool_trial.summarize)."""
    ev = {e['id']: e for e in C.MAN['events']}
    floors = sorted({e['floor'] for e in C.MAN['events']})
    def agg(ids):
        vals, wts = [], []
        for i in ids:
            e = ev[i]; _, _, Q = C.event_arrays(e)
            _, g = np.unique(C.XY[Q], axis=0, return_inverse=True); g = g.ravel()
            per_pos = np.bincount(g, weights=errors[i]) / np.bincount(g)
            nf = sum(f.startswith(e['building']) for f in floors)
            vals.append(per_pos); wts.append(np.full(len(per_pos), .5 / nf / 2 / 3 / len(per_pos)))
        v = np.concatenate(vals); w = np.concatenate(wts); w /= w.sum(); o = np.argsort(v)
        return dict(mean=float(w @ v), p90=float(v[o[np.searchsorted(np.cumsum(w[o]), .9)]]))
    out = agg(list(errors))
    out['by_K'] = {k: agg([i for i in errors if ev[i]['K'] == k]) for k in (1, 3, 5)}
    out['by_floor'] = {f: agg([i for i in errors if ev[i]['floor'] == f]) for f in floors}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--theta', type=float, nargs=3, required=True, metavar=('ELL', 'C', 'G'))
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5])
    ap.add_argument('--gp-records', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    sys.path.insert(0, os.environ['POOL_SRC'])
    from hurdle_gp_pool import fuse_hurdle_gp                      # unchanged adopted fusion
    gp = json.load(open(a.gp_records)); gp_root = os.environ['POOL_SRC']   # gp_path is relative to src (trial ROOT)
    gp_rec = {r['event']: r for r in gp['records']}
    os.makedirs(a.out, exist_ok=False)
    theta = tuple(a.theta); t0 = time.time()
    accA, accB = {}, {}
    for seed in a.seeds:
        fA = Hd.HurdleField([0], seed=seed)
        fB = FP.HurdleFieldX([0], seed=seed, use_support=False)
        print('seed', seed, 'trained %ds' % (time.time() - t0), flush=True)
        for e in C.MAN['events']:
            b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
            H, S, Q = C.event_arrays(e); aps = np.flatnonzero(C.DET[np.concatenate([H, S])].any(0))
            accA.setdefault(e['id'], []).append(fA.predict(b, t, S, aps)[1:])
            accB.setdefault(e['id'], []).append(fB.predict(b, t, S, aps)[1:])
        del fA, fB
    single, pooled = {v: {} for v in VARIANTS}, {v: {} for v in VARIANTS}
    for e in C.MAN['events']:
        b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
        H, S, Q = C.event_arrays(e); ctx = np.concatenate([H, S])
        aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx]); Xc = V.candidates(b, t)
        fa, fb = FP.ensemble(accA[e['id']]), FP.ensemble(accB[e['id']])
        fields = {'A_adopted': fa, 'B_H_only': fb,
                  'C_H_cond': FP.condition(Xc, *fb, S, aps, theta), 'D_HS_cond': FP.condition(Xc, *fa, S, aps, theta)}
        rec = gp_rec[e['id']]
        with np.load(os.path.join(gp_root, rec['gp_path'])) as g:
            assert np.array_equal(g['query_rows'], e['query_rows']) and np.array_equal(g['aps'], aps), e['id']
            grid, scores, raw = g['grid'], g['scores'], g['raw']
        truth = C.XY[Q]; save = {}
        for v, (p, m, s2) in fields.items():
            xy, L = posterior(Xc, p, m, s2, S, Q, aps)
            r = fuse_hurdle_gp(xy, L, grid, scores, raw, bd)['predictions']
            single[v][e['id']] = np.linalg.norm(r['hurdle'] - truth, axis=1)
            pooled[v][e['id']] = np.linalg.norm(r['probability_half'] - truth, axis=1)
            save[v + '_log_post'] = L.astype(np.float32); save[v + '_cand_xy'] = xy
            save[v + '_pred'] = r['hurdle']; save[v + '_pool'] = r['probability_half']
        np.savez_compressed(os.path.join(a.out, e['id'] + '.npz'), query_rows=np.asarray(e['query_rows']), aps=aps, **save)
        print(e['id'], {v: round(float(single[v][e['id']].mean()), 2) for v in VARIANTS}, flush=True)
    summ = {v: dict(single=project_summary(single[v]), pool=project_summary(pooled[v])) for v in VARIANTS}
    for v in VARIANTS:
        summ[v]['pool_gain'] = summ[v]['single']['mean'] - summ[v]['pool']['mean']
    json.dump(dict(theta=theta, seeds=a.seeds, reader=dict(beta=BETA, eps=EPS, offsets=True), summary=summ,
                   anchor=dict(adopted_single=9.4325, adopted_pool=8.7792), seconds=time.time() - t0),
              open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    for v in VARIANTS:
        s = summ[v]
        print(v, 'single %.3f p90 %.2f | pool %.3f p90 %.2f K1p90 %.2f | gain %.3f' % (
            s['single']['mean'], s['single']['p90'], s['pool']['mean'], s['pool']['p90'],
            s['pool']['by_K'][1]['p90'], s['pool_gain']), flush=True)


if __name__ == '__main__':
    main()
