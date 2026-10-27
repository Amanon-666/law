"""Training-free floor-structure prior (PROTOCOL_physics.md). ray-web only.

Stage: event --event uji:ID|tau:ID|gia:ID ; summarize.
"""
import argparse
import json
from pathlib import Path
import socket

import numpy as np
import torch

import run_operator_lodo as L
import run_operator_v11 as V
from data import group_scans
from full_ap_study import metrics
from history_bank import BankPrior
from inference import observation_log_prob
from operator_prior import history_moments, marginal_loglik
from run import ap_stats, tensor
from run_multi_history import write
from source_weight_mixture import field_from_prior
from transfer import rbf

OUT = Path(__file__).resolve().parent / 'results/physics_prior_20261026'
KAPPA = np.arange(0, 25, 2.)
SIGMA, PEAK_DB, PEAK_M = 6., 6., 10.


def own_flag(hs, dz):
    if any(h is None for h in hs): return False
    p = [h.m.max().item() for h in hs]; loc = [h.z[h.m.argmax()] for h in hs]
    near = float((loc[0] - loc[1]).norm()) <= PEAK_M
    if sorted(dz) == [-1, 1]:
        return near and abs(p[0] - p[1]) <= PEAK_DB
    return near and min(dz) < 0 < max(dz)


class FloorPrior(BankPrior):
    def __init__(self, hs, w, base, kappa, own):
        super().__init__(hs, w, base); self.hs, self.kappa, self.own = hs, kappa, own

    def features(self, x):
        m, f = super().features(x)
        if self.own:
            top = torch.stack([history_moments(h, x, self.base)[0] for h in self.hs]).max(0).values
            m = top + self.kappa
        return m, f

    def covariance(self, x, y, fx, fy):
        c = super().covariance(x, y, fx, fy)
        return c + (SIGMA ** 2 * rbf(x, y, 1., self.base['length']) if self.own else 0)

    def diagonal(self, x, fx):
        return super().diagonal(x, fx) + (SIGMA ** 2 if self.own else 0)


def gia_case(event_id):
    meta = next(e for e in json.loads((V.OUT / 'manifest.json').read_text())['events'] if e['id'] == event_id)
    fl = V.floors_of('GIA'); t, srcs, origin = meta['target'], meta['sources'], np.array(meta['origin'])
    txy, trss = fl[t]
    support = group_scans(trss[meta['support']], txy[meta['support']] - origin)
    hpos = np.concatenate([fl[j][0] - origin for j in srcs]); sxy = support['positions']
    hdet = np.zeros(trss.shape[1], bool)
    for j in srcs: hdet |= (fl[j][1] != 100).any(0)
    aps = np.flatnonzero(hdet | (support['stats']['k'].sum(0) > 0)).tolist()
    lo = np.minimum(hpos.min(0), sxy.min(0)); hi = np.maximum(hpos.max(0), sxy.max(0))
    axes = [np.linspace(lo[d], hi[d], int(np.ceil((hi[d] - lo[d]) / 2)) + 1) for d in range(2)]
    base, _ = V.params(); dirs = [V.OUT / 'gia_histories' / meta['config'] / f'F{j}' for j in srcs]
    hist = {a: [V.history_or_none(d / f'ap{a}.npz', base) for d in dirs] for a in aps}
    return dict(support=support, aps=aps, hist=hist, grid=tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2)),
                qrss=trss[meta['internal']], qxy=txy[meta['internal']] - origin, split=np.array(['internal'] * len(meta['internal'])),
                dz=[j - t for j in srcs], seed=meta['seed'], dataset='GIA', config=meta['config'])


def event(spec):
    kind, eid = spec.split(':'); dest = OUT / f'{kind}_{eid}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'completed.json').exists(): return
    case = {'uji': L.uji_case, 'tau': L.tau_case, 'gia': gia_case}[kind](eid)
    base, obs = V.params(); sup = case['support']; sx = tensor(sup['positions']); aps = case['aps']; hist = case['hist']
    stats = {k: torch.stack([ap_stats(sup, a)[k] for a in aps]) for k in ('n', 'k', 's1', 's2')}
    w, _ = V.fit_w(hist, aps, sx, stats, base, obs)
    own = {a: own_flag(hist[a], case['dz']) for a in aps}
    scores = np.zeros(len(KAPPA))
    with torch.no_grad():
        for a in aps:
            if not own[a]: continue
            st = ap_stats(sup, a)
            for i, k in enumerate(KAPPA):
                p = FloorPrior(hist[a], w, base, k, True); m, f = p.features(sx)
                scores[i] += marginal_loglik(m, p.diagonal(sx, f), st['n'], st['k'], st['s1'], st['s2'], obs).sum().item()
    kappa = float(KAPPA[int(np.argmax(scores))])
    pred = {}
    for name, use in (('adapted_bank', False), ('floor_prior', True)):
        sc = torch.zeros((len(case['qrss']), len(case['grid'])))
        for a in aps:
            hs = hist[a]
            prior = FloorPrior(hs, w, base, kappa, use and own[a]) if any(h is not None for h in hs) else BankPrior([], [], base)
            with torch.no_grad():
                r = field_from_prior(prior, obs, sx, ap_stats(sup, a), case['grid'], 261026 + 10000 * case['seed'] + a, defensive_fraction=.1)
                for s in range(0, len(sc), 128):
                    sc[s:s + 128] += observation_log_prob(tensor(case['qrss'][s:s + 128, a].astype(float))[:, None], r['mean'][None], r['variance'][None], obs)
        pred[name] = (torch.softmax(sc, 1) @ case['grid']).numpy()
    res = []
    for sp in sorted(set(case['split'])):
        sel = case['split'] == sp; xy = case['qxy'][sel]
        _, g, c = np.unique(np.round(xy, 3), axis=0, return_inverse=True, return_counts=True); g = g.ravel()
        for m, p in pred.items(): res.append(dict(method=m, split=sp, **metrics(p[sel], dict(xy=xy, scan_weights=1 / (len(c) * c[g])))))
    write(dest / 'completed.json', dict(event=spec, dataset=case['dataset'], kappa=kappa, own=int(sum(own.values())), aps=len(aps), results=res))
    print(json.dumps(dict(event=spec, kappa=kappa, own=int(sum(own.values())), res=[(r['method'], r['split'], round(r['mde'], 3)) for r in res])), flush=True)


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/completed.json'))]
    out = {}
    for ds in sorted({r['dataset'] for r in rows}):
        rs = [r for r in rows if r['dataset'] in ([ds] if ds != 'TIE1' and ds != 'SAH1' else [ds])]
        a = [next(x['mde'] for x in r['results'] if x['method'] == 'adapted_bank' and x['split'] == 'internal') for r in rs]
        b = [next(x['mde'] for x in r['results'] if x['method'] == 'floor_prior' and x['split'] == 'internal') for r in rs]
        out[ds] = dict(n=len(rs), adapted=float(np.mean(a)), floor=float(np.mean(b)), improved=int(sum(x > y for x, y in zip(a, b))), kappa=[r['kappa'] for r in rs])
    write(OUT / 'summary.json', out); print(json.dumps(out))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web': raise RuntimeError('ray-web only')
    p = argparse.ArgumentParser(); p.add_argument('stage'); p.add_argument('--event'); a = p.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    event(a.event) if a.stage == 'event' else summarize()
