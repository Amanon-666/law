"""Train the vertical translator and evaluate it on the project's 54 nine-floor events.

  LAW_UJI_CSV=/path/to/trainingData.csv python run_nine_floor.py --train lobo --seeds 20261110 --out results_lobo

--train lobo    : B1 targets use a translator trained on B0+B2, B2 targets one trained on B0+B1
--train b0only  : both use a translator trained on B0 only (same source building as full24 / GP params)
Several seeds are averaged at the map level (ensemble). numpy + pandas + scikit-learn only; CPU, minutes.

Per event the script saves <out>/<event_id>.npz with
  cand_xy [n,2]            candidate positions (all history positions of the building's other floors)
  map_H, map_HS [n,A]      predicted target-floor encoded RSS (dBm, -105 = not detected), history only / with S
  aps [A]                  zero-based WAP indices (detected in H or S, exactly the project's AP rule)
  pred_3nn_H, pred_3nn_HS  coordinates from the project's 3NN rule on the translated map + real S scans
  query_rows               one-based CSV row ids, as in the project manifest
Query coordinates are read only for scoring, after predictions are stored in memory.
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import core as C, vt_lib as V

TRAIN = dict(lobo={1: [0, 2], 2: [0, 1]}, b0only={1: [0], 2: [0]})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', choices=tuple(TRAIN), default='lobo')
    ap.add_argument('--seeds', type=int, nargs='+', default=[20261110])
    ap.add_argument('--draws', type=int, default=8)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=False)
    maps = {}
    for seed in args.seeds:
        models = {}
        for hb, tr in TRAIN[args.train].items():
            rng = np.random.default_rng(seed); t0 = time.time()
            Xh, Xs, y = V.training_set(tr, rng, draws=args.draws, single_acquisition=False)
            models[hb] = (V.Model(Xh, y), V.Model(np.hstack([Xh, Xs]), y))
            print(f'seed {seed} held-out B{hb}: trained on B{tr}, {len(y)} cells, {time.time()-t0:.0f}s', flush=True)
        for e in C.MAN['events']:
            b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
            H, S, Q = C.event_arrays(e)
            aps = np.flatnonzero(C.DET[np.concatenate([H, S])].any(0))
            _, mH = V.translate(models[b][0], b, t, S, aps, False)
            _, mHS = V.translate(models[b][1], b, t, S, aps, True)
            maps.setdefault(e['id'], []).append((mH, mHS))
    errs = dict(translated_H={}, translated_HS={})
    for e in C.MAN['events']:
        b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
        H, S, Q = C.event_arrays(e); ctx = np.concatenate([H, S])
        aps = np.flatnonzero(C.DET[ctx].any(0)); bounds = C.bounds_of(C.XY[ctx])
        Xc = V.candidates(b, t)
        mH = np.mean([m[0] for m in maps[e['id']]], 0); mHS = np.mean([m[1] for m in maps[e['id']]], 0)
        pH = V.read_knn(Xc, mH, S, Q, aps, bounds); pHS = V.read_knn(Xc, mHS, S, Q, aps, bounds)
        np.savez_compressed(os.path.join(args.out, e['id'] + '.npz'), cand_xy=Xc, map_H=mH.astype(np.float32),
                            map_HS=mHS.astype(np.float32), aps=aps, pred_3nn_H=pH, pred_3nn_HS=pHS,
                            query_rows=np.asarray(e['query_rows']), bounds=bounds)
        errs['translated_H'][e['id']] = C.event_error(e, pH); errs['translated_HS'][e['id']] = C.event_error(e, pHS)
    summary = {k: C.summarize(v) for k, v in errs.items()}
    json.dump(dict(train=args.train, seeds=args.seeds, draws=args.draws, summary=summary, events=errs,
                   reference=dict(project_3NN_HS=16.8254, GP=11.8983, ordinary_Tab_HS=12.1148, GP_ordinary_half=10.7880)),
              open(os.path.join(args.out, 'summary.json'), 'w'), indent=1)
    for k, s in summary.items():
        print(k, 'mean %.3f' % s['mean'], {a: round(x, 2) for a, x in s['by_K'].items()})


if __name__ == '__main__':
    main()
