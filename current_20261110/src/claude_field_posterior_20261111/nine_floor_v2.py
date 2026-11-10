"""Decisive same-information increment on the 54-event nine-floor protocol: best old -> new complete mechanism.

Same source (UJI B0 only), same events, same H/S/Q, same reader, same cached GP and unchanged fuse_hurdle_gp.
Rows (operator ensembles seeds 1-5):
  C_old       f397b71 C path, NOT retrained: per-event predictions reloaded from results/nine_fp_v2_20261111/<event>.npz
              (C_H_cond_pred / C_H_cond_pool) after checking query_rows and aps are identical
  v1_newcode  same model class re-implemented in law_bench_ours (no reliability, no randomisation) + same
              conditioning: measures the code-path difference so that v2 rows are attributable
  v2_noM2     M1 v2 (reliability inputs + randomised history conditions), no conditioning
  v2_C1       M1 v2 + original conditioning (isolates M1 change)
  v2_full     M1 v2 + M2 v2 (separate tau2 / sig2 / detection-error heads), theta selected on B0 by held-out NLL
Decision written before running: v2_full replaces C as our single path if its single-path mean < C_old AND its
single-path p90 and K1 p90 do not exceed C_old's by more than 0.3 m; the GP-pool row is reported, not used to adopt.
Anchors: A_frozen (adopted five-seed assets loaded via FrozenHurdleTranslator, no retraining) must reproduce
9.4325 single / 8.7792 pool; C_old must reproduce 9.2249 within 0.02. The first new-mechanism increment is read
against BOTH A_frozen (adopted) and C_old (best single path).

  LAW_UJI_CSV=... VT_DIR=... POOL_SRC=<main worktree root> nice -n 10 python nine_floor_v2.py --gp-records ... --out nine_v2
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import law_bench_ours as O
import field_posterior as FP
from field_posterior import C, V
from run_nine_floor_field_posterior import posterior, project_summary


def floors_of(b, rows=None):
    out = {}
    for f in V.FLOORS[b]:
        r = np.flatnonzero((C.BLD == b) & (C.FLR == f))
        out[f] = dict(rss=C.RSS[r], xy=C.XY[r])
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--gp-records', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3, 4, 5]); a = ap.parse_args()
    sys.path.insert(0, os.environ['POOL_SRC'])
    from hurdle_gp_pool import fuse_hurdle_gp
    gp = json.load(open(a.gp_records)); gp_rec = {r['event']: r for r in gp['records']}
    os.makedirs(a.out, exist_ok=False); t0 = time.time()
    src = [floors_of(0)]
    import joblib, hashlib
    mdir = os.path.join(a.out, 'models'); os.makedirs(mdir, exist_ok=True)
    code_sha = {f: hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f), 'rb').read()).hexdigest()
                for f in ('law_bench_ours.py', 'nine_floor_v2.py')}
    theta, scores = O.select_theta(src, seed=a.seeds[0])
    json.dump(dict(theta=theta, scores={str(k): v for k, v in scores.items()}, code_sha256=code_sha,
                   source='UJI B0', csv=os.environ.get('LAW_UJI_CSV')), open(os.path.join(mdir, 'theta_selection.json'), 'w'), indent=1)
    print('theta (B0 held-out NLL)', theta, 'elapsed %ds' % (time.time() - t0), flush=True)
    ops_v2, ops_v1 = [], []
    for s_ in a.seeds:                                                   # dump each model as soon as it is fitted
        for tag, rel, lst in (('v2', True, ops_v2), ('v1', False, ops_v1)):
            op = O.Operator(src, s_, reliability=rel, randomise=rel); lst.append(op)
            joblib.dump(dict(model=op, seed=s_, source='UJI B0', csv=os.environ.get('LAW_UJI_CSV'), theta=theta,
                             code_sha256=code_sha), os.path.join(mdir, f'{tag}_seed{s_}.joblib'))
    cdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results', 'nine_fp_v2_20261111')
    print('trained', 'elapsed %ds' % (time.time() - t0), flush=True)
    from frozen_hurdle_translator import FrozenHurdleTranslator          # adopted A: saved five-seed assets, no retraining
    frozen = FrozenHurdleTranslator(os.path.join(os.environ['POOL_SRC'], 'results/claude_hurdle_b0_ens5_20261110'))
    rows = ('A_frozen', 'C_old', 'v1_newcode', 'v2_noM2', 'v2_C1', 'v2_full')
    single, pooled = {r: {} for r in rows}, {r: {} for r in rows}
    for e in C.MAN['events']:
        b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
        H, S, Q = C.event_arrays(e)
        Hf = {f: dict(rss=C.RSS[H][C.FLR[H] == f], xy=C.XY[H][C.FLR[H] == f]) for f in np.unique(C.FLR[H])}
        aps = np.flatnonzero(C.DET[np.concatenate([H, S])].any(0)); bd = C.bounds_of(C.XY[np.concatenate([H, S])])
        maps = O.make_maps(Hf); sr, sx, truth = C.RSS[S][:, aps], C.XY[S], C.XY[Q]
        X1, f1 = O.ensemble_field(ops_v1, maps, t, aps); X2, f2 = O.ensemble_field(ops_v2, maps, t, aps)
        Ps = O.det_stats(sr, sx)[0]; fS2 = O.ensemble_field(ops_v2, maps, t, aps, Ps)[1]
        with np.load(os.path.join(cdir, e['id'] + '.npz')) as z:
            assert np.array_equal(z['query_rows'], np.asarray(e['query_rows'])) and np.array_equal(z['aps'], aps), e['id']
            single['C_old'][e['id']] = np.linalg.norm(z['C_H_cond_pred'] - truth, axis=1)
            pooled['C_old'][e['id']] = np.linalg.norm(z['C_H_cond_pool'] - truth, axis=1)
        fields = {'v1_newcode': (X1, O.condition(X1, f1['p'], f1['m'], f1['s2'], sr, sx, (16.0, 0.3, 1.0, 0.0))),
                  'v2_noM2': (X2, (f2['p'], f2['m'], f2['sig2'] + f2['tau2'])),
                  'v2_C1': (X2, O.condition(X2, f2['p'], f2['m'], f2['s2'], sr, sx, (16.0, 0.3, 1.0, 0.0))),
                  'v2_full': (X2, O.condition_v2(X2, f2, sr, sx, theta, fS2))}
        with np.load(os.path.join(os.environ['POOL_SRC'], gp_rec[e['id']]['gp_path'])) as g:
            assert np.array_equal(g['aps'], aps), e['id']
            grid, sc, raw = g['grid'], g['scores'], g['raw']
        hist = {f"{e['building']}F{f}": dict(rss=C.RSS[H][C.FLR[H] == f], xy=C.XY[H][C.FLR[H] == f]) for f in np.unique(C.FLR[H])}
        prep = frozen.prepare(hist, dict(rss=C.RSS[S], xy=C.XY[S]), e['floor'])
        hz = prep.predict(C.RSS[Q])
        resA = fuse_hurdle_gp(hz['cand_xy'], hz['log_position_mass'], grid, sc, raw, prep.bounds)['predictions']
        single['A_frozen'][e['id']] = np.linalg.norm(hz['projected'] - truth, axis=1)
        pooled['A_frozen'][e['id']] = np.linalg.norm(resA['probability_half'] - truth, axis=1)
        for r, (Xc, (p, m, s2)) in fields.items():
            xy, L = posterior(Xc, p, m, s2, S, Q, aps)
            res = fuse_hurdle_gp(xy, L, grid, sc, raw, bd)['predictions']
            single[r][e['id']] = np.linalg.norm(res['hurdle'] - truth, axis=1)
            pooled[r][e['id']] = np.linalg.norm(res['probability_half'] - truth, axis=1)
        print(e['id'], {r: round(float(single[r][e['id']].mean()), 2) for r in rows}, flush=True)
    summ = {r: dict(single=project_summary(single[r]), pool=project_summary(pooled[r])) for r in rows}
    c, v = summ['C_old']['single'], summ['v2_full']['single']
    adopt = (v['mean'] < c['mean']) and (v['p90'] <= c['p90'] + 0.3) and (v['by_K'][1]['p90'] <= c['by_K'][1]['p90'] + 0.3)
    json.dump(dict(theta=theta, theta_scores={str(k): s for k, s in scores.items()}, summary=summ, adopt_v2_full=bool(adopt),
                   anchor=dict(C_old_expected=9.2249), seconds=time.time() - t0), open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    for r in rows:
        s = summ[r]
        print(r, 'single %.3f p90 %.2f K1p90 %.2f | pool %.3f p90 %.2f' % (s['single']['mean'], s['single']['p90'],
              s['single']['by_K'][1]['p90'], s['pool']['mean'], s['pool']['p90']), flush=True)
    print('adopt v2_full:', adopt)


if __name__ == '__main__':
    main()
