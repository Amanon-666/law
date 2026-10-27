"""Operator prior v1.1: g-free source weight in training and deployment; offset-only ablation.

Stages: histories --event BUILDING | train --event full|offset | manifest | event --event ID | summarize.
Train on UJI B0-B2 + Tampere TIE1/SAH1; confirm on giaIndoorLoc. See PROTOCOL_v11.md. ray-web only.
"""
import argparse
import itertools
import json
from pathlib import Path
import pickle
import socket

import numpy as np
import torch

from data import group_scans
from full_ap_study import metrics, wknn
from history_bank import BankPrior, fit_two_source_weight
from inference import observation_log_prob
from multi_history import single_history
from operator_prior import OperatorBankPrior, OperatorNet, history_moments, marginal_loglik, source_tokens
from run import ap_stats, tensor
from run_multi_history import digest, write
from run_operator_prior import load_history
from run_operator_tampere import raw as tampere_raw
from source_weight_mixture import field_from_prior
from transfer import Prior, Transfer, rbf
from uji_probability_study import load_raw

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'results/uji_history_bank_v1_20261026'
V1 = ROOT / 'results/uji_operator_prior_v1_20261026'
OUT = ROOT / 'results/operator_v11_20261026'
GIA = Path.home() / 'projects/panyushuo/multifloor-check2-20261026/cache/gia_v2.pkl'
CELL = 4.
TRAIN = {'UJI_B0': [0, 1, 2, 3], 'UJI_B1': [0, 1, 2, 3], 'UJI_B2': [0, 1, 2, 3, 4], 'TIE1': [0, 1, 2, 3], 'SAH1': [1, 2, 3]}
CONFIGS = {'GIA_F1': (1, [0, 2]), 'GIA_F2': (2, [1, 3]), 'GIA_F3': (3, [2, 4])}
SEEDS, K, QUERIES = (0, 1, 2), 5, 300
STEPS, LR, BATCH, MAX_POS, MAX_AP = 1500, 1e-3, 16, 200, 150
W_GRID = np.linspace(0, 1, 17)
OFFSET_MASK = torch.tensor([0., 0., 1., 0., 0., 1.])
METHODS = ['adapted_bank', 'v11_full', 'v11_offset', 'v1', 'v11_scaled']
ZETA = np.linspace(0, 2, 17)


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def params():
    p = json.loads((OLD / 'parameters.json').read_text()); return p['base'], p['obs']


class ScaledNet(torch.nn.Module):
    def __init__(self, net, zeta):
        super().__init__(); self.net, self.zeta = net, zeta

    def forward(self, tokens):
        g, s = self.net(tokens); return self.zeta * g, s


def choose_zeta(hist, aps, sx, support, w, base, obs, net, dz):
    """Shared scale of g from S only: sum of independent per-position prior marginal likelihoods."""
    scores = np.zeros(len(ZETA))
    with torch.no_grad():
        for a in aps:
            hs = hist[a]
            if all(h is None for h in hs): continue
            tok = torch.stack([source_tokens(h, d, sx, base) for h, d in zip(hs, dz)]); g, s = net(tok)
            mv = [history_moments(h, sx, base) for h in hs]
            convex = w[0] * mv[0][0] + w[1] * mv[1][0]
            var = base['drift'] ** 2 + w[0] ** 2 * mv[0][1] + w[1] ** 2 * mv[1][1] + s.square()
            st = ap_stats(support, a)
            for i, z in enumerate(ZETA):
                scores[i] += marginal_loglik(convex + z * g, var, st['n'], st['k'], st['s1'], st['s2'], obs).sum().item()
    return float(ZETA[int(np.argmax(scores))]), scores.tolist()


class MaskedNet(torch.nn.Module):
    def __init__(self, mask=None):
        super().__init__(); self.net = OperatorNet().double(); self.mask = mask

    def forward(self, tokens):
        return self.net(tokens if self.mask is None else tokens * self.mask.to(tokens))


def floors_of(building):
    """{floor: (xy, rss)} raw scans, consistent AP columns within the building."""
    if building.startswith('UJI_'):
        b = building[-1]; names = {f'B{b}F{f}' for f in TRAIN[building]}
        d = load_raw('trainingData.csv', names)
        return {int(n[-1]): (d[n]['xy'], d[n]['rss']) for n in names}
    if building in ('TIE1', 'SAH1'):
        c, r = tampere_raw(building)
        return {f: (c[c[:, 3] == f][:, :2], r[c[:, 3] == f]) for f in TRAIN[building]}
    if building == 'GIA':
        fl = pickle.load(open(GIA, 'rb'))[0]['floors']
        return {int(f): (np.asarray(v[0], float), np.asarray(v[1]).astype(int)) for f, v in fl.items()}
    raise KeyError(building)


def aggregate(building, xy, rss):
    if building.startswith('UJI_'):
        return group_scans(rss, xy)        # original reference points
    key = np.floor(xy / CELL).astype(int)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    centroid = np.stack([np.bincount(inv, xy[:, d]) / np.bincount(inv) for d in range(2)], -1)
    return group_scans(rss, centroid[inv])


def histories(building):
    base, obs = params(); fl = floors_of(building)
    origin = np.concatenate([v[0] for v in fl.values()]).mean(0)
    for f, (xy, rss) in fl.items():
        dest = OUT / 'histories' / building / f'F{f}'
        if (dest / 'done.json').exists(): continue
        dest.mkdir(parents=True, exist_ok=True)
        scans = aggregate(building, xy - origin, rss)
        np.savez_compressed(dest / 'target.npz', positions=scans['positions'], **scans['stats'])
        for ap in np.flatnonzero(scans['stats']['k'].sum(0) > 0):
            if (dest / f'ap{ap}.npz').exists(): continue
            h = single_history(dict(scans=scans), int(ap), base, obs, steps=640)
            arrays = dict(empty=np.array(h is None))
            if h is not None: arrays.update(z=h.z.numpy(), mean=h.m.numpy(), cov=h.c.numpy())
            np.savez_compressed(dest / f'ap{ap}.npz', **arrays)
        write(dest / 'done.json', dict(origin=origin.tolist(), positions=len(scans['positions'])))
        log(stage='history', building=building, floor=f, positions=len(scans['positions']))


def history_or_none(path, base):
    return load_history(path, base) if path.exists() else None


def build_tasks(base, obs, rng):
    tasks = []
    for building, fl in TRAIN.items():
        root = OUT / 'histories' / building
        for t in fl:
            with np.load(root / f'F{t}' / 'target.npz') as a:
                pos = a['positions']; st = {k: a[k] for k in ('n', 'k', 's1', 's2')}
            keep_pos = np.sort(rng.permutation(len(pos))[:MAX_POS])
            x = tensor(pos[keep_pos])
            for pair in itertools.combinations([f for f in fl if f != t], 2):
                cand = sorted({int(p.stem[2:]) for j in pair for p in (root / f'F{j}').glob('ap*.npz')})
                hs_all = {a: [history_or_none(root / f'F{j}' / f'ap{a}.npz', base) for j in pair] for a in cand}
                cand = [a for a in cand if any(h is not None for h in hs_all[a])]
                aps = sorted(rng.permutation(cand)[:MAX_AP].tolist())
                tok, means, vars_ = [], [], []
                for a in aps:
                    hs = hs_all[a]
                    tok.append(torch.stack([source_tokens(h, j - t, x, base) for h, j in zip(hs, pair)]))
                    mv = [history_moments(h, x, base) for h in hs]
                    means.append(torch.stack([m for m, _ in mv])); vars_.append(torch.stack([v for _, v in mv]))
                idx = np.array(aps)
                stats = {k: tensor(np.broadcast_to(v[keep_pos], (len(aps), len(keep_pos))).copy()) if k == 'n'
                         else tensor(v[keep_pos][:, idx].T) for k, v in st.items()}
                task = dict(building=building, target=t, sources=list(pair), tokens=torch.stack(tok, 1),
                            means=torch.stack(means, 1), vars=torch.stack(vars_, 1), stats=stats)
                with torch.no_grad():
                    task['w'] = float(W_GRID[int(np.argmin([loss(None, task, base, obs, w).item() for w in W_GRID]))])
                tasks.append(task)
        log(stage='tasks', building=building, total=len(tasks))
    return tasks


def loss(net, task, base, obs, w=None):
    w = task['w'] if w is None else w
    J, A, n, _ = task['tokens'].shape
    if net is None:
        g = s = task['means'].new_zeros((A, n))
    else:
        g, s = net(task['tokens'].reshape(J, A * n, -1)); g, s = g.reshape(A, n), s.reshape(A, n)
    mean = w * task['means'][0] + (1 - w) * task['means'][1] + g
    var = base['drift'] ** 2 + w ** 2 * task['vars'][0] + (1 - w) ** 2 * task['vars'][1] + s.square()
    st = task['stats']
    return -marginal_loglik(mean, var, st['n'], st['k'], st['s1'], st['s2'], obs).mean()


def train(kind):
    base, obs = params(); torch.manual_seed(261027); rng = np.random.default_rng(261027)
    tasks = build_tasks(base, obs, rng)
    net = MaskedNet(OFFSET_MASK if kind == 'offset' else None)
    opt = torch.optim.Adam(net.parameters(), lr=LR); curve = []
    with torch.no_grad():
        convex = float(np.mean([loss(None, t, base, obs).item() for t in tasks]))
    for step in range(STEPS + 1):
        if step % 100 == 0:
            with torch.no_grad():
                curve.append(dict(step=step, loss=float(np.mean([loss(net, t, base, obs).item() for t in tasks])), convex_only=convex))
            log(stage='train', kind=kind, **curve[-1])
        if step == STEPS: break
        batch = rng.choice(len(tasks), BATCH, replace=False)
        opt.zero_grad(); l = sum(loss(net, tasks[i], base, obs) for i in batch) / BATCH; l.backward(); opt.step()
    torch.save(net.state_dict(), OUT / f'operator_{kind}.pt')
    write(OUT / f'training_{kind}.json', dict(kind=kind, steps=STEPS, curve=curve, tasks=[dict(building=t['building'], target=t['target'],
          sources=t['sources'], w=t['w'], aps=int(t['tokens'].shape[1]), positions=int(t['tokens'].shape[2])) for t in tasks],
          checkpoint_sha256=digest(OUT / f'operator_{kind}.pt')))


def manifest():
    fl = floors_of('GIA'); events = []
    for name, (t, srcs) in CONFIGS.items():
        origin = np.concatenate([fl[j][0] for j in srcs]).mean(0)
        n = len(fl[t][0])
        for seed in SEEDS:
            order = np.random.default_rng(5000 + 100 * seed + t).permutation(n)
            sup = order[:K]; sxy = fl[t][0][sup]
            rest = order[K:]; rest = rest[~(np.round(fl[t][0][rest], 3)[:, None] == np.round(sxy, 3)[None]).all(-1).any(1)][:QUERIES]
            events.append(dict(id=f'{name}_s{seed}', config=name, target=t, sources=srcs, seed=seed, origin=origin.tolist(),
                               support=sup.tolist(), internal=rest.tolist()))
    write(OUT / 'manifest.json', dict(events=events, code_sha256=digest(Path(__file__)),
          checkpoints={k: digest(OUT / f'operator_{k}.pt') for k in ('full', 'offset')}, v1=digest(V1 / 'operator.pt')))


def gia_histories(meta, base, obs):
    fl = floors_of('GIA'); origin = np.array(meta['origin'])
    for j in meta['sources']:
        dest = OUT / 'gia_histories' / meta['config'] / f'F{j}'
        if (dest / 'done.json').exists(): continue
        dest.mkdir(parents=True, exist_ok=True)
        scans = aggregate('GIA', fl[j][0] - origin, fl[j][1])
        for ap in np.flatnonzero(scans['stats']['k'].sum(0) > 0):
            h = single_history(dict(scans=scans), int(ap), base, obs, steps=640)
            arrays = dict(empty=np.array(h is None))
            if h is not None: arrays.update(z=h.z.numpy(), mean=h.m.numpy(), cov=h.c.numpy())
            np.savez_compressed(dest / f'ap{ap}.npz', **arrays)
        write(dest / 'done.json', dict(positions=len(scans['positions'])))


def fit_w(hist, aps, sx, stats, base, obs, net=None, dz=None):
    means, covs, residuals = [], [], []
    with torch.no_grad():
        for a in aps:
            hs = hist[a]; g = s = sx.new_zeros(len(sx))
            if net is not None:
                op = OperatorBankPrior(hs, [.5, .5], base, net, dz)
                if op.active: g, s = op.operator(sx)
            mm, cc = [], []
            for h in hs:
                p = Prior(Transfer('identity'), h, dict(base, drift=0.)); m, f = p.features(sx)
                mm.append(m + g); cc.append(p.covariance(sx, sx, f, f))
            if any(h is not None for h in hs):
                res = rbf(sx, sx, base['drift'], base['length']) + s[:, None] * s[None] * rbf(sx, sx, 1., base['length'])
            else:
                cc = [torch.zeros_like(x) for x in cc]; res = rbf(sx, sx, base['amplitude'], base['length'])
            means.append(torch.stack(mm)); covs.append(torch.stack(cc)); residuals.append(res)
    return fit_two_source_weight(torch.stack(means), torch.stack(covs), torch.stack(residuals), stats, obs)


def event(event_id):
    base, obs = params()
    meta = next(e for e in json.loads((OUT / 'manifest.json').read_text())['events'] if e['id'] == event_id)
    dest = OUT / event_id; dest.mkdir(exist_ok=True)
    if (dest / 'completed.json').exists(): return
    gia_histories(meta, base, obs)
    fl = floors_of('GIA'); t, srcs, origin = meta['target'], meta['sources'], np.array(meta['origin'])
    txy, trss = fl[t]
    support = group_scans(trss[meta['support']], txy[meta['support']] - origin); sx = tensor(support['positions'])
    qxy, qrss = txy[meta['internal']] - origin, trss[meta['internal']]
    hdirs = [OUT / 'gia_histories' / meta['config'] / f'F{j}' for j in srcs]
    hpos = np.concatenate([fl[j][0] - origin for j in srcs])
    hdet = np.zeros(trss.shape[1], bool)
    for j in srcs: hdet |= (fl[j][1] != 100).any(0)
    aps = np.flatnonzero(hdet | (support['stats']['k'].sum(0) > 0)).tolist()
    lo = np.minimum(hpos.min(0), sx.numpy().min(0)); hi = np.maximum(hpos.max(0), sx.numpy().max(0))
    axes = [np.linspace(lo[d], hi[d], int(np.ceil((hi[d] - lo[d]) / 2)) + 1) for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))
    dz = [j - t for j in srcs]
    nets = {}
    for kind in ('full', 'offset'):
        m = MaskedNet(OFFSET_MASK if kind == 'offset' else None); m.load_state_dict(torch.load(OUT / f'operator_{kind}.pt', weights_only=True)); m.eval()
        nets['v11_' + kind] = m
    v1 = OperatorNet().double(); v1.load_state_dict(torch.load(V1 / 'operator.pt', weights_only=True)); v1.eval(); nets['v1'] = v1
    hist = {a: [history_or_none(d / f'ap{a}.npz', base) for d in hdirs] for a in aps}
    stats = {key: torch.stack([ap_stats(support, a)[key] for a in aps]) for key in ('n', 'k', 's1', 's2')}
    wpath = dest / 'weights.json'
    if not wpath.exists():
        w_free, tr_free = fit_w(hist, aps, sx, stats, base, obs)
        w_v1, tr_v1 = fit_w(hist, aps, sx, stats, base, obs, nets['v1'], dz)
        write(wpath, dict(g_free=w_free, v1_refit=w_v1, trials=dict(g_free=tr_free, v1=tr_v1)))
    W = json.loads(wpath.read_text())
    weights = dict(adapted_bank=W['g_free'], v11_full=W['g_free'], v11_offset=W['g_free'], v1=W['v1_refit'], v11_scaled=W['g_free'])
    zpath = dest / 'zeta.json'
    if not zpath.exists():
        z, sc = choose_zeta(hist, aps, sx, support, W['g_free'], base, obs, nets['v11_full'], dz)
        write(zpath, dict(zeta=z, grid=ZETA.tolist(), scores=sc))
    zeta = json.loads(zpath.read_text())['zeta']; nets['v11_scaled'] = ScaledNet(nets['v11_full'], zeta)
    log(stage='adapted', event=event_id, weights=weights, aps=len(aps), grid=len(grid))
    for i, a in enumerate(aps):
        path = dest / f'ap{a}.npz'
        if path.exists(): continue
        hs = hist[a]; arrays = {}
        for m in METHODS:
            prior = BankPrior(hs, weights[m], base) if m == 'adapted_bank' else OperatorBankPrior(hs, weights[m], base, nets[m], dz)
            if all(h is None for h in hs): prior = BankPrior([], [], base)
            with torch.no_grad():
                res = field_from_prior(prior, obs, sx, ap_stats(support, a), grid, 261026 + 10000 * meta['seed'] + a, defensive_fraction=.1)
            arrays[m + '_mean'] = res['mean'].numpy(); arrays[m + '_variance'] = res['variance'].numpy()
            arrays[m + '_ess'] = np.array(np.nan if res['effective_samples'] is None else res['effective_samples'])
        np.savez_compressed(path, **arrays)
    fields = {a: np.load(dest / f'ap{a}.npz') for a in aps}
    low = {m: int(sum(float(fields[a][m + '_ess']) < 1000 for a in aps)) for m in METHODS}
    prediction = {m: [] for m in METHODS}
    with torch.no_grad():
        for start in range(0, len(qrss), 64):
            scans = tensor(qrss[start:start + 64].astype(float))
            for m in METHODS:
                sc = scans.new_zeros((len(scans), len(grid)))
                for a in aps:
                    sc += observation_log_prob(scans[:, a, None], tensor(fields[a][m + '_mean'])[None], tensor(fields[a][m + '_variance'])[None], obs)
                prediction[m].append((torch.softmax(sc, 1) @ grid).numpy())
    prediction = {m: np.concatenate(v) for m, v in prediction.items()}
    sup_raw = dict(rss=trss[meta['support']], xy=txy[meta['support']] - origin)
    bank = dict(rss=np.concatenate([fl[j][1] for j in srcs] + [sup_raw['rss']]), xy=np.concatenate([fl[j][0] - origin for j in srcs] + [sup_raw['xy']]))
    prediction['wknn_local'] = wknn(sup_raw, dict(rss=qrss), aps, False)
    prediction['wknn_bank'] = wknn(bank, dict(rss=qrss), aps, False)
    _, grp, cnt = np.unique(np.round(qxy, 3), axis=0, return_inverse=True, return_counts=True); grp = grp.ravel()
    results = [dict(method=m, split='internal', **metrics(p, dict(xy=qxy, scan_weights=1 / (len(cnt) * cnt[grp])))) for m, p in prediction.items()]
    np.savez_compressed(dest / 'predictions.npz', **prediction, query_xy=qxy)
    write(dest / 'completed.json', dict(event=event_id, weights=weights, zeta=zeta, aps=len(aps), low_ess=low,
          low_ess_fraction={m: low[m] / len(aps) for m in METHODS}, results=results))
    log(stage='completed', event=event_id, results=[(r['method'], round(r['mde'], 3)) for r in results])


def summarize():
    events = json.loads((OUT / 'manifest.json').read_text())['events']
    rows = {e['id']: json.loads((OUT / e['id'] / 'completed.json').read_text()) for e in events}
    names = METHODS + ['wknn_local', 'wknn_bank']
    per = {m: [next(r for r in rows[e['id']]['results'] if r['method'] == m) for e in events] for m in names}
    out = dict(events=[e['id'] for e in events],
               mean={m: float(np.mean([r['mde'] for r in v])) for m, v in per.items()},
               median_of_medians={m: float(np.median([r['median'] for r in v])) for m, v in per.items()},
               gt10={m: float(np.mean([r['fraction_error_gt10'] for r in v])) for m, v in per.items()},
               per_event={m: [r['mde'] for r in v] for m, v in per.items()},
               improved_vs_adapted={m: int(sum(a['mde'] > b['mde'] for a, b in zip(per['adapted_bank'], per[m]))) for m in names},
               full_vs_offset=int(sum(a['mde'] > b['mde'] for a, b in zip(per['v11_offset'], per['v11_full']))),
               low_ess={i: v['low_ess_fraction'] for i, v in rows.items()})
    write(OUT / 'summary.json', out); log(stage='summary', mean=out['mean'], improved=out['improved_vs_adapted'])


def giahist():
    base, obs = params()
    seen = set()
    for e in json.loads((OUT / 'manifest.json').read_text())['events']:
        if e['config'] in seen: continue
        seen.add(e['config']); gia_histories(e, base, obs); log(stage='gia_histories', config=e['config'])


def main():
    if socket.gethostname() != 'ray-web': raise RuntimeError('ray-web only')
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['histories', 'train', 'manifest', 'giahist', 'event', 'summarize']); p.add_argument('--event')
    a = p.parse_args(); torch.set_default_dtype(torch.float64); torch.set_num_threads(2); OUT.mkdir(exist_ok=True)
    {'histories': lambda: histories(a.event), 'train': lambda: train(a.event), 'manifest': manifest, 'giahist': giahist,
     'event': lambda: event(a.event), 'summarize': summarize}[a.stage]()


if __name__ == '__main__':
    main()
