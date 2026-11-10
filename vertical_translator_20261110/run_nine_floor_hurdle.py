"""Nine-floor 54-event evaluation of the hurdle observation-distribution operator with the reader configuration
frozen on the B0 development fold (dev_hurdle.py): beta=128, eps=0.02, device-offset marginalisation on.

Several seeds are averaged at the distribution level (p, m, s2). Per event the script saves
<out>/<event_id>.npz with cand_xy (translated candidates + S positions, the reader's candidate set),
log_post [n_query, n_cand] (tempered, normalised log posterior), pred (posterior mean), query_rows, so that the
GP path can be fused by posterior product without re-running anything here.

  LAW_UJI_CSV=... python run_nine_floor_hurdle.py --train lobo --seeds 1 2 3 --out hurdle_lobo
"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.special import logsumexp
import core as C, translator as T, hurdle as Hd

BETA, EPS, OFFSETS = 128.0, 0.02, True
TRAIN = dict(lobo={1: [0, 2], 2: [0, 1]}, b0only={1: [0], 2: [0]})


def posterior(cand_xy, p, m, s2, s_rows, q_rows, aps):
    """same computation as hurdle.read_hurdle, returning candidates and normalised log posterior"""
    Ps, ps, ms, s2s = Hd.support_candidates(s_rows, aps)
    xy = np.vstack([cand_xy, Ps]); p = np.clip(np.vstack([p, ps]), 1e-4, 1 - 1e-4)
    m = np.vstack([m, ms]); s2 = np.vstack([s2, s2s])
    lmiss = np.log((1 - EPS) * (1 - p) + EPS * Hd.U_MISS)
    logp = np.log(p); lnorm = -0.5 * np.log(2 * np.pi * s2)
    q = C.RSS[q_rows][:, aps].astype(float); A = len(aps)
    offs = Hd.OFFSETS if OFFSETS else np.array([0.0])
    L = np.empty((len(q), len(xy)))
    for i, row in enumerate(q):
        d = row != 100; v = row[d][None]
        per = [np.logaddexp(np.log(1 - EPS) + logp[:, d] + lnorm[:, d] - 0.5 * (v - o - m[:, d]) ** 2 / s2[:, d],
                            np.log(EPS * Hd.U_DET)).sum(1) for o in offs]
        L[i] = (logsumexp(np.stack(per), 0) - np.log(len(offs)) + lmiss[:, ~d].sum(1)) * BETA / A
    return xy, L - logsumexp(L, 1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', choices=tuple(TRAIN), default='lobo')
    ap.add_argument('--seeds', type=int, nargs='+', default=[1])
    ap.add_argument('--draws', type=int, default=4)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=False)
    acc = {}
    for seed in a.seeds:
        fields, built = {}, {}
        for hb, tr in TRAIN[a.train].items():
            t0 = time.time()
            if tuple(tr) not in built:                     # b0only: one field serves both held-out buildings
                built[tuple(tr)] = Hd.HurdleField(tr, seed=seed, draws=a.draws)
            fields[hb] = built[tuple(tr)]
            print(f'seed {seed} held-out B{hb}: trained on B{tr} {time.time()-t0:.0f}s', flush=True)
        for e in C.MAN['events']:
            b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
            H, S, Q = C.event_arrays(e); aps = np.flatnonzero(C.DET[np.concatenate([H, S])].any(0))
            Xc, p, m, s2 = fields[b].predict(b, t, S, aps)
            acc.setdefault(e['id'], []).append((p, m, s2))
        del fields, built
    errs = {}
    for e in C.MAN['events']:
        b = int(e['building'][1]); t = int(e['floor'].split('F')[1])
        H, S, Q = C.event_arrays(e); ctx = np.concatenate([H, S])
        aps = np.flatnonzero(C.DET[ctx].any(0)); bd = C.bounds_of(C.XY[ctx])
        Xc = Hd.V.candidates(b, t)
        ps, ms, vs = (np.stack([x[k] for x in acc[e['id']]]) for k in range(3))
        p, m = ps.mean(0), ms.mean(0); s2 = vs.mean(0) + ms.var(0)
        xy, L = posterior(Xc, p, m, s2, S, Q, aps)
        pred = np.clip(np.exp(L) @ xy, bd[0], bd[1])
        np.savez_compressed(os.path.join(a.out, e['id'] + '.npz'), cand_xy=xy, log_post=L.astype(np.float32),
                            pred=pred, query_rows=np.asarray(e['query_rows']), aps=aps, bounds=bd)
        errs[e['id']] = C.event_error(e, pred)
    s = C.summarize(errs)
    json.dump(dict(train=a.train, seeds=a.seeds, draws=a.draws, reader=dict(beta=BETA, eps=EPS, offsets=OFFSETS),
                   summary=s, events=errs,
                   reference=dict(GP=11.8983, GP_ordinary_half=10.7880, translated_3NN_b0only_ens5=10.5648,
                                  GP_translated_3NN_half=9.9607)),
              open(os.path.join(a.out, 'summary.json'), 'w'), indent=1)
    print('hurdle', a.train, 'mean %.3f' % s['mean'], {k: round(v, 2) for k, v in s['by_K'].items()},
          {k: round(v, 2) for k, v in s['by_floor'].items()}, flush=True)


if __name__ == '__main__':
    main()
