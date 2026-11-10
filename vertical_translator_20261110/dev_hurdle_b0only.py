"""Reader selection using building B0 ONLY (answers the permission concern that dev_hurdle.py trained on B1+B2).

Leave-one-floor-out inside B0: for each target floor t of B0, the hurdle operator is trained on B0 with floor t
removed entirely (neither target, history nor label), then the B0 development events of floor t are read with
every reader configuration. B1/B2 data are not touched. If the selected configuration agrees with the frozen one
(beta=128, eps=0.02, offsets on), the nine-floor numbers need no re-selection.

  LAW_UJI_CSV=... python dev_hurdle_b0only.py
"""
import sys, os, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core as C, vt_lib as V, hurdle as Hd

man = C.DEV
GRID = [(beta, eps, off) for beta in (16, 32, 64, 128, 256, None) for eps in (0.02, 0.1) for off in (True, False)]
FULL = list(V.FLOORS[0])
res = {}
for t in FULL:
    V.FLOORS[0] = [f for f in FULL if f != t]; V._HF.clear()
    t0 = time.time(); hf = Hd.HurdleField([0], seed=20261110)
    print('floor', t, 'trained without it', round(time.time() - t0), 's', flush=True)
    V.FLOORS[0] = FULL; V._HF.clear()                      # deployment on floor t sees all other B0 floors as history
    for e in man['events']:
        if int(e['floor'].split('F')[1]) != t:
            continue
        H, S, Q = C.event_arrays_m(man, e); ctx = np.concatenate([H, S])
        aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx])
        Xc, p, m, s2 = hf.predict(0, t, S, aps)
        res.setdefault('knn3_on_p_times_m', {})[e['id']] = C.event_error_m(man, e, V.read_knn(Xc, p * m + (1 - p) * -105.0, S, Q, aps, bd))
        for beta, eps, off in GRID:
            res.setdefault(f'hurdle_b{beta}_e{eps}_off{int(off)}', {})[e['id']] = \
                C.event_error_m(man, e, Hd.read_hurdle(Xc, p, m, s2, S, Q, aps, bd, beta, eps, off))
    del hf
rows = []
for k, r in res.items():
    s = C.summarize_m(man, r)
    rows.append(dict(method=k, mean=s['mean'], **{f'K{a}': x for a, x in s['by_K'].items()}, **s['by_floor']))
print(pd.DataFrame(rows).sort_values('mean').round(2).to_string(), flush=True)
json.dump(res, open('dev_hurdle_b0only.json', 'w'))
