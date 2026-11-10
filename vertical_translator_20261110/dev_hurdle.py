"""Development fold only (B0 targets, operator trained on B1+B2): choose beta / eps / offset marginalisation of
the hurdle reader. No B1/B2 target score is computed here.

  LAW_UJI_CSV=... python dev_hurdle.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core as C, vt_lib as V, hurdle as Hd

man = C.DEV
t0 = time.time()
hf = Hd.HurdleField([1, 2], seed=20261110)
print('trained', round(time.time() - t0), 's', flush=True)
GRID = [(beta, eps, off) for beta in (16, 32, 64, 128, 256, None) for eps in (0.02, 0.1) for off in (True, False)]
res = {}
for e in man['events']:
    b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
    H, S, Q = C.event_arrays_m(man, e); ctx = np.concatenate([H, S])
    aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx])
    Xc, p, m, s2 = hf.predict(b, t, S, aps)
    mean_map = p * m + (1 - p) * -105.0
    res.setdefault('knn3_on_p_times_m', {})[e['id']] = C.event_error_m(man, e, V.read_knn(Xc, mean_map, S, Q, aps, bd))
    for beta, eps, off in GRID:
        pr = Hd.read_hurdle(Xc, p, m, s2, S, Q, aps, bd, beta, eps, off)
        res.setdefault(f'hurdle_b{beta}_e{eps}_off{int(off)}', {})[e['id']] = C.event_error_m(man, e, pr)
rows = []
for k, r in res.items():
    s = C.summarize_m(man, r)
    rows.append(dict(method=k, mean=s['mean'], **{f'K{a}': x for a, x in s['by_K'].items()}, **s['by_floor']))
print(pd.DataFrame(rows).sort_values('mean').round(2).to_string(), flush=True)
json.dump(res, open('dev_hurdle.json', 'w'))
