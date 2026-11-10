"""Counter-case: history that does not belong to the target building/floor.

For each of the 54 nine-floor events the history floors' AP columns are permuted (one random permutation per
event, applied to every history floor, so the history stays internally consistent but no longer matches
the target's APs). The translator (unchanged, trained as in run_nine_floor --train lobo, seed 1) then
translates this wrong history together with the real S. Reference readings on the same events:
  S only        3NN on the K real S positions alone (no history at all)
  correct       translator on the real history
  wrong H only  translator without S on the permuted history
  wrong HS      translator with S on the permuted history
A robust history-use rule should fall back towards "S only" and not far below it when history is wrong.

  LAW_UJI_CSV=... python wrong_history.py
"""
import sys, os, json, zlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
import core as C, translator as T, vt_lib as V

TRAIN = {1: [0, 2], 2: [0, 1]}
models = {}
for hb, tr in TRAIN.items():
    Xh, Xs, y = V.training_set(tr, np.random.default_rng(1), draws=8, single_acquisition=False)
    models[hb] = (V.Model(Xh, y), V.Model(np.hstack([Xh, Xs]), y))
    print('trained for B%d' % hb, flush=True)


def translate_with_maps(model, hmaps, t, Xc, s_rows, aps, with_support):
    F, ref = T.history_features(Xc, t, hmaps)
    F = F.astype(np.float32)
    Fe = F[:, aps]; n, A, _ = Fe.shape
    X = Fe.reshape(n * A, -1)
    if with_support:
        s_ref = T.naive_ref_at(C.XY[s_rows], t, hmaps)
        Sf = T.support_features(Xc, ref, C.XY[s_rows], C.RSS[s_rows], s_ref, F).astype(np.float32)
        X = np.hstack([X, Sf[:, aps].reshape(n * A, -1)])
    return np.clip(model.predict(X), T.ND, 0).reshape(n, A)


res = {}
for e in C.MAN['events']:
    b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
    H, S, Q = C.event_arrays(e); ctx = np.concatenate([H, S])
    aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx])
    Xc = V.candidates(b, t)
    rng = np.random.default_rng(zlib.crc32(e['id'].encode()))
    perm = np.arange(C.RSS.shape[1]); perm[aps] = rng.permutation(aps)   # shuffle identities among event APs
    good = V.hist_maps(b, t)
    bad = {f: (P, M[:, perm], R[:, perm]) for f, (P, M, R) in good.items()}
    out = {
        'S_only_3NN': C.nearest3(C.RSS[S], C.XY[S], C.RSS[Q], aps, bd),
        'correct_H': V.read_knn(Xc, translate_with_maps(models[b][0], good, t, Xc, S, aps, False), S, Q, aps, bd),
        'correct_HS': V.read_knn(Xc, translate_with_maps(models[b][1], good, t, Xc, S, aps, True), S, Q, aps, bd),
        'wrong_H': V.read_knn(Xc, translate_with_maps(models[b][0], bad, t, Xc, S, aps, False), S, Q, aps, bd),
        'wrong_HS': V.read_knn(Xc, translate_with_maps(models[b][1], bad, t, Xc, S, aps, True), S, Q, aps, bd),
    }
    for k, p in out.items():
        res.setdefault(k, {})[e['id']] = C.event_error(e, p)
rows = []
for k, r in res.items():
    s = C.summarize(r)
    rows.append(dict(method=k, mean=s['mean'], **{f'K{a}': x for a, x in s['by_K'].items()}, **s['by_floor']))
print(pd.DataFrame(rows).round(2).to_string(), flush=True)
json.dump(res, open('wrong_history.json', 'w'))
