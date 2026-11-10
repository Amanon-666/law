"""Same-corpus attribution: translator trained on B0F0-F2 only (the floors ordinary30 / full24 may use), versus
the four-floor B0 version. B0F3 is removed entirely (neither target nor history during training).
Protected source-validation rows inside F0-F2 are NOT removed here (the catalog lives on the server); the exact
version is for Codex. Evaluation: 54 nine-floor events, 3NN reader, seeds 1-5 map ensemble, as run_nine_floor.

  LAW_UJI_CSV=... python source_floor_attribution.py
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core as C, vt_lib as V

SEEDS = (1, 2, 3, 4, 5)
FULL = list(V.FLOORS[0])
out = {}
for tag, floors0 in (('B0F0-F2', [0, 1, 2]),):
    V.FLOORS[0] = floors0
    V._HF.clear()
    maps = {}
    for seed in SEEDS:
        Xh, Xs, y = V.training_set([0], np.random.default_rng(seed), draws=8, single_acquisition=False)
        mH, mHS = V.Model(Xh, y), V.Model(np.hstack([Xh, Xs]), y)
        del Xh, Xs, y
        for e in C.MAN['events']:
            b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
            H, S, Q = C.event_arrays(e); aps = np.flatnonzero(C.DET[np.concatenate([H, S])].any(0))
            _, a = V.translate(mH, b, t, S, aps, False); _, c = V.translate(mHS, b, t, S, aps, True)
            maps.setdefault(e['id'], []).append((a.astype(np.float32), c.astype(np.float32)))
        print(tag, 'seed', seed, 'done', flush=True)
    res = {}
    for e in C.MAN['events']:
        b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
        H, S, Q = C.event_arrays(e); ctx = np.concatenate([H, S])
        aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx]); Xc = V.candidates(b, t)
        for k, i in (('H', 0), ('HS', 1)):
            m = np.mean([x[i] for x in maps[e['id']]], 0)
            res.setdefault(k, {})[e['id']] = C.event_error(e, V.read_knn(Xc, m, S, Q, aps, bd))
    rows = []
    for k, r in res.items():
        s = C.summarize(r)
        rows.append(dict(source=tag, method=k + '_ens5', mean=s['mean'], **{f'K{a}': x for a, x in s['by_K'].items()}, **s['by_floor']))
    print(pd.DataFrame(rows).round(2).to_string(), flush=True)
    out[tag] = res
V.FLOORS[0] = FULL
json.dump(out, open('source_floor_attribution.json', 'w'))
