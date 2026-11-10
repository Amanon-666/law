"""Source-only development of the learned-prior + Bayesian-S field (field_posterior.py). B0 ONLY, never B1/B2.

Same leave-one-floor-out as dev_hurdle_b0only.py (seed 20261110): for each B0 floor t the operators are trained
on B0 without t, then the B0 development events of floor t are read with the FROZEN reader (beta=128, eps=0.02,
offsets on). Outputs, all on B0:
  D2   residual correlogram of the H-only and H+S operators on the held-out floor (is there spatial structure in
       the translation error that S could propagate? which length?)
  NLL  predictive log-likelihood of held-out real target-floor scans (not S acquisition) for every theta
       -> theta is SELECTED BY NLL (not by localization) and frozen for the nine-floor run
  LOC  localization of the dev events:
         A  adopted operator (H+S features), no conditioning           (reference, should be ~5.93)
         B  H-only operator, no conditioning                           (what history alone gives)
         C  H-only operator + Bayesian S conditioning  [main variant]  (S used once, by conditioning)
         D  H+S operator + Bayesian S conditioning     [control]       (S used twice)
       C and D are reported for every theta (transparency) and for the NLL-selected theta (the decision).

Decision (written before running):
  - D2: if same-AP residual correlation at 3-6 m is < 0.1, the spatial-conditioning hypothesis is refuted; stop.
  - LOC gate: C at the NLL-selected theta < A (5.93) on the B0 dev fold, else stop; nine-floor is not run.

  LAW_UJI_CSV=... VT_DIR=<dir with core.py, hurdle.py ...> nice -n 10 python dev_field_posterior_b0only.py --out dev_fp_b0
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import field_posterior as FP
from field_posterior import C, V, Hd

BETA, EPS = 128.0, 0.02
ELLS, CS, GS = (4.0, 8.0, 16.0, 32.0), (0.3, 0.7), (0.0, 1.0)
THETAS = [(l, c, g) for l in ELLS for c in CS for g in GS]
tkey = lambda th: 'l%g_c%g_g%g' % th


def eval_scans(t, e, S, rng, cap=400):
    """real target-floor scans of B0 floor t outside the S acquisition and outside S itself"""
    rows = np.flatnonzero((C.BLD == 0) & (C.FLR == t))
    acq = tuple(e['support_acquisition'])
    day = np.array([np.datetime64(int(x), 's').astype('datetime64[D]').astype(str) for x in C.TS[rows]])
    same = (C.USER[rows] == acq[0]) & (C.PHONE[rows] == acq[1]) & (day == acq[2])
    rows = rows[~same & ~np.isin(rows, S)]
    return rows if len(rows) <= cap else rng.choice(rows, cap, replace=False)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    man = C.DEV; FULL = list(V.FLOORS[0]); rng = np.random.default_rng(20261111)
    loc, nll, d2 = {}, {}, {}
    t_start = time.time()
    for t in FULL:
        V.FLOORS[0] = [f for f in FULL if f != t]; V._HF.clear()
        t0 = time.time()
        fHS = Hd.HurdleField([0], seed=20261110)                       # exact adopted recipe
        fH = FP.HurdleFieldX([0], seed=20261110, use_support=False)    # same recipe, no S columns
        print('floor', t, 'operators trained without it', round(time.time() - t0), 's', flush=True)
        V.FLOORS[0] = FULL; V._HF.clear()
        for e in man['events']:
            if int(e['floor'].split('F')[1]) != t:
                continue
            H, S, Q = C.event_arrays_m(man, e); ctx = np.concatenate([H, S])
            aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx])
            Xc, pA, mA, vA = fHS.predict(0, t, S, aps)
            _, pB, mB, vB = fH.predict(0, t, S, aps)
            err = lambda p, m, v: C.event_error_m(man, e, Hd.read_hurdle(Xc, p, m, v, S, Q, aps, bd, BETA, EPS, True))
            loc.setdefault('A_adopted_HS', {})[e['id']] = err(pA, mA, vA)
            loc.setdefault('B_H_only', {})[e['id']] = err(pB, mB, vB)
            if e['K'] == 5:
                d2[e['id']] = dict(H_only=FP.residual_correlogram(Xc, mB, 0, t, aps),
                                   HS=FP.residual_correlogram(Xc, mA, 0, t, aps))
            ev = eval_scans(t, e, S, rng)
            D = np.linalg.norm(C.XY[ev][:, None] - Xc[None], axis=2); jj = D.argmin(1); okk = D.min(1) <= FP.R_PRIOR
            ev, jj = ev[okk], jj[okk]; rss = C.RSS[ev][:, aps].astype(float)
            for th in THETAS:
                for name, (p0, m0, v0) in (('C_H_cond', (pB, mB, vB)), ('D_HS_cond', (pA, mA, vA))):
                    p, m, v = FP.condition(Xc, p0, m0, v0, S, aps, th)
                    nll.setdefault(name, {}).setdefault(tkey(th), {})[e['id']] = \
                        float(-FP.scan_loglik(p[jj], m[jj], v[jj], rss).mean())
                    loc.setdefault(name + '_' + tkey(th), {})[e['id']] = err(p, m, v)
            for name, (p0, m0, v0) in (('A_adopted_HS', (pA, mA, vA)), ('B_H_only', (pB, mB, vB))):
                nll.setdefault(name, {}).setdefault('prior', {})[e['id']] = \
                    float(-FP.scan_loglik(p0[jj], m0[jj], v0[jj], rss).mean())
            print(e['id'], 'A %.2f B %.2f' % (loc['A_adopted_HS'][e['id']], loc['B_H_only'][e['id']]),
                  'elapsed %ds' % (time.time() - t_start), flush=True)
        del fHS, fH
    rows = []
    for k, r in loc.items():
        s = C.summarize_m(man, r)
        rows.append(dict(method=k, mean=s['mean'], **{f'K{x}': y for x, y in s['by_K'].items()}, **s['by_floor']))
    table = pd.DataFrame(rows).sort_values('mean')
    sel = {}
    for name in ('C_H_cond', 'D_HS_cond'):
        scores = {k: float(np.mean(list(v.values()))) for k, v in nll[name].items()}
        best = min(scores, key=scores.get); sel[name] = dict(theta=best, nll=scores[best], all=scores)
    base_nll = {n: float(np.mean(list(nll[n]['prior'].values()))) for n in ('A_adopted_HS', 'B_H_only')}
    pick = {n: C.summarize_m(man, loc[n + '_' + sel[n]['theta']]) for n in sel}
    print(table.round(2).to_string(), flush=True)
    print('NLL prior', base_nll, 'selected', {n: (sel[n]['theta'], round(sel[n]['nll'], 4)) for n in sel})
    print('selected-theta localization', {n: round(pick[n]['mean'], 3) for n in pick})
    corr = {k: np.nanmean([np.array(v[k]['corr'], float) for v in d2.values()], 0).round(3).tolist() for k in ('H_only', 'HS')}
    print('D2 mean correlogram over K5 events (bins 0-3,3-6,6-12,12-24,24-48,>48 m):', corr, flush=True)
    json.dump(dict(loc=loc, nll=nll, nll_prior=base_nll, selected=sel,
                   selected_localization={n: pick[n] for n in pick}, d2=d2, d2_mean=corr,
                   reader=dict(beta=BETA, eps=EPS, offsets=True), thetas=THETAS,
                   seconds=time.time() - t_start), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    table.to_csv(os.path.join(a.out, 'localization_table.csv'), index=False)


if __name__ == '__main__':
    main()
