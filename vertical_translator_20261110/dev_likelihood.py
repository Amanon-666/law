"""Development fold only (B0 targets, translator trained on B1+B2): choose beta, s0 of the likelihood reader.
No B1/B2 target score is computed here.

  LAW_UJI_CSV=... python dev_likelihood.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core as C, vt_lib as V, likelihood as L

man = C.DEV
t0 = time.time()
ht = L.HeteroTranslator([1, 2], seed=20261110)
print('trained', round(time.time() - t0), 's', flush=True)
GRID = [(beta, s0) for beta in (0.5, 1, 2, 4, 8, 16) for s0 in (2.0, 4.0, 6.0)]
res = {}
for e in man['events']:
    b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
    H, S, Q = C.event_arrays_m(man, e); ctx = np.concatenate([H, S])
    aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx])
    Xc, mu, v = ht.translate(b, t, S, aps)
    res.setdefault('knn3_mean_map', {})[e['id']] = C.event_error_m(man, e, V.read_knn(Xc, mu, S, Q, aps, bd))
    for beta, s0 in GRID:
        p = L.read_likelihood(Xc, mu, v, S, Q, aps, bd, beta, s0)
        res.setdefault(f'lik_b{beta}_s{s0}', {})[e['id']] = C.event_error_m(man, e, p)
        p = L.read_likelihood(Xc, mu, np.zeros_like(v), S, Q, aps, bd, beta, s0)
        res.setdefault(f'lik_const_b{beta}_s{s0}', {})[e['id']] = C.event_error_m(man, e, p)
rows = []
for k, r in res.items():
    s = C.summarize_m(man, r)
    rows.append(dict(method=k, mean=s['mean'], **{f'K{a}': x for a, x in s['by_K'].items()}, **s['by_floor']))
df = pd.DataFrame(rows).sort_values('mean')
print(df.round(2).to_string(), flush=True)
json.dump(res, open('dev_likelihood.json', 'w'))
