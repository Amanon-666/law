"""Independent confirmation of the frozen operator prior v1 on Tampere TIE1/SAH1.

Stages: manifest | prepare --event CONFIG | event --event ID | summarize. Run on ray-web only. See PROTOCOL_tampere.md.
"""
import argparse
import json
from pathlib import Path
import socket

import numpy as np
import torch

from data import group_scans
from full_ap_study import metrics, wknn
from history_bank import BankPrior, fit_two_source_weight
from inference import observation_log_prob
from multi_history import single_history
from operator_prior import OperatorBankPrior, OperatorNet
from run import ap_stats, tensor
from run_multi_history import digest, write
from run_operator_prior import load_history
from source_weight_mixture import field_from_prior
from transfer import Prior, Transfer, rbf

ROOT = Path(__file__).resolve().parent
TAU = Path.home() / 'datasets/tampere_5174851/WiFi RSS measurements in Tampere University multi-building campus 2017 - Zenodo 5174851'
TRAIN = ROOT / 'results/uji_operator_prior_v1_20261026'
OLD = ROOT / 'results/uji_history_bank_v1_20261026'
OUT = ROOT / 'results/tampere_operator_confirm_v1_20261026'
CONFIGS = {'TIE1_F1': ('TIE1', 1, [0, 2]), 'TIE1_F2': ('TIE1', 2, [1, 3]), 'SAH1_F2': ('SAH1', 2, [1, 3]),
           'TIE1_F3': ('TIE1', 3, [1, 2])}
PRIMARY = ['TIE1_F1', 'TIE1_F2', 'SAH1_F2']
SEEDS, K, QUERIES, CELL = (0, 1, 2), 5, 300, 4.
METHODS = ['adapted_bank', 'operator_bank']


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def params():
    p = json.loads((OLD / 'parameters.json').read_text()); return p['base'], p['obs']


def raw(building, split='training'):
    c = np.loadtxt(TAU / f'{building}_{split}_coordinates.csv', delimiter=',')
    r = np.loadtxt(TAU / f'{building}_{split}_rss.csv', delimiter=',').astype(int)
    return c, r


def cells(xy, rss):
    """4 m cells; a scan's position is replaced by its cell centroid (R=1 data)."""
    key = np.floor(xy / CELL).astype(int)
    _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    centroid = np.stack([np.bincount(inv, xy[:, d]) / np.bincount(inv) for d in range(2)], -1)
    return group_scans(rss, centroid[inv])


def config_origin(c, srcs):
    return np.concatenate([c[c[:, 3] == j][:, :2] for j in srcs]).mean(0)


def prepare(name):
    """Histories are cached per configuration because each uses its own origin."""
    base, obs = params(); b, t, srcs = CONFIGS[name]
    c, r = raw(b); origin = config_origin(c, srcs)
    for j in srcs:
        dest = OUT / 'histories' / name / f'{b}F{j}'
        if (dest / 'done.json').exists(): continue
        dest.mkdir(parents=True, exist_ok=True)
        m = c[:, 3] == j
        scans = cells(c[m][:, :2] - origin, r[m])
        for ap in np.flatnonzero(scans['stats']['k'].sum(0) > 0):
            if (dest / f'ap{ap}.npz').exists(): continue
            h = single_history(dict(scans=scans), int(ap), base, obs, steps=640)
            arrays = dict(empty=np.array(h is None))
            if h is not None: arrays.update(z=h.z.numpy(), mean=h.m.numpy(), cov=h.c.numpy())
            np.savez_compressed(dest / f'ap{ap}.npz', **arrays)
        write(dest / 'done.json', dict(origin=origin.tolist(), cells=len(scans['positions'])))
        log(stage='history', config=name, floor=f'{b}F{j}', cells=len(scans['positions']))


def manifest():
    OUT.mkdir(exist_ok=True); events = []
    for name, (b, t, srcs) in CONFIGS.items():
        c, r = raw(b); origin = config_origin(c, srcs)
        tm = np.flatnonzero(c[:, 3] == t)
        for seed in SEEDS:
            rng = np.random.default_rng(1000 * seed + 17 * t + len(b))
            order = rng.permutation(tm)
            support, rest = order[:K], order[K:]
            sxy = c[support, :2]
            rest = rest[~(c[rest, :2][:, None] == sxy[None]).all(-1).any(1)][:QUERIES]
            events.append(dict(id=f'{name}_s{seed}', config=name, building=b, target=t, sources=srcs, seed=seed,
                               origin=origin.tolist(), support=support.tolist(), internal=rest.tolist()))
    write(OUT / 'manifest.json', dict(events=events, cell=CELL, k=K, queries=QUERIES,
          checkpoint_sha256=digest(TRAIN / 'operator.pt'), code_sha256=digest(Path(__file__))))


def event(event_id):
    base, obs = params()
    meta = next(e for e in json.loads((OUT / 'manifest.json').read_text())['events'] if e['id'] == event_id)
    dest = OUT / event_id; dest.mkdir(exist_ok=True)
    if (dest / 'completed.json').exists(): return
    b, t, srcs, origin = meta['building'], meta['target'], meta['sources'], np.array(meta['origin'])
    c, r = raw(b)
    support = group_scans(r[meta['support']], c[meta['support'], :2] - origin); sx = tensor(support['positions'])
    qxy, qrss, split = c[meta['internal'], :2] - origin, r[meta['internal']], ['internal'] * len(meta['internal'])
    if (b, t) == ('SAH1', 2):
        tc, tr = raw(b, 'test'); m = tc[:, 3] == t
        qxy, qrss = np.concatenate([qxy, tc[m][:, :2] - origin]), np.concatenate([qrss, tr[m]]); split += ['official'] * int(m.sum())
    split = np.array(split)
    hist_dirs = [OUT / 'histories' / meta['config'] / f'{b}F{j}' for j in srcs]
    hpos = []
    for j in srcs:
        mm = c[:, 3] == j; hpos.append(c[mm][:, :2] - origin)
    hpos = np.concatenate(hpos)
    hdet = np.zeros(r.shape[1], bool)
    for j in srcs: hdet |= (r[c[:, 3] == j] != 100).any(0)
    aps = np.flatnonzero(hdet | (support['stats']['k'].sum(0) > 0)).tolist()
    lo = np.minimum(hpos.min(0), sx.numpy().min(0)); hi = np.maximum(hpos.max(0), sx.numpy().max(0))
    axes = [np.linspace(lo[d], hi[d], int(np.ceil((hi[d] - lo[d]) / 2)) + 1) for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))
    net = OperatorNet().double(); net.load_state_dict(torch.load(TRAIN / 'operator.pt', weights_only=True)); net.eval()
    dz = [j - t for j in srcs]
    hist = {a: [load_history(d / f'ap{a}.npz', base) if (d / f'ap{a}.npz').exists() else None for d in hist_dirs] for a in aps}
    stats = {key: torch.stack([ap_stats(support, a)[key] for a in aps]) for key in ('n', 'k', 's1', 's2')}

    weights = {}
    for method in METHODS:
        path = dest / f'adaptation_{method}.json'
        if not path.exists():
            means, covs, residuals = [], [], []
            with torch.no_grad():
                for a in aps:
                    hs = hist[a]; g = s = sx.new_zeros(len(sx))
                    if method == 'operator_bank':
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
            w, trials = fit_two_source_weight(torch.stack(means), torch.stack(covs), torch.stack(residuals), stats, obs)
            write(path, dict(weights=w, trials=trials))
        weights[method] = json.loads(path.read_text())['weights']
    log(stage='adapted', event=event_id, weights=weights, aps=len(aps), grid=len(grid))

    ess = []
    for i, a in enumerate(aps):
        path = dest / f'ap{a}.npz'
        if path.exists(): continue
        hs = hist[a]
        priors = dict(adapted_bank=BankPrior(hs, weights['adapted_bank'], base),
                      operator_bank=OperatorBankPrior(hs, weights['operator_bank'], base, net, dz))
        if all(h is None for h in hs): priors = {m: BankPrior([], [], base) for m in METHODS}
        arrays = {}
        for method, prior in priors.items():
            with torch.no_grad():
                res = field_from_prior(prior, obs, sx, ap_stats(support, a), grid, 261026 + 10000 * meta['seed'] + a, defensive_fraction=.1)
            arrays[method + '_mean'] = res['mean'].numpy(); arrays[method + '_variance'] = res['variance'].numpy()
            arrays[method + '_ess'] = np.array(np.nan if res['effective_samples'] is None else res['effective_samples'])
        np.savez_compressed(path, **arrays)
        if i % 50 == 0: log(stage='field', event=event_id, done=i + 1, total=len(aps))

    prediction = {m: [] for m in METHODS}; low = {m: 0 for m in METHODS}
    fields = {a: np.load(dest / f'ap{a}.npz') for a in aps}
    for m in METHODS:
        low[m] = int(sum(float(fields[a][m + '_ess']) < 1000 for a in aps))
    with torch.no_grad():
        for start in range(0, len(qrss), 64):
            scans = tensor(qrss[start:start + 64].astype(float))
            for m in METHODS:
                scores = scans.new_zeros((len(scans), len(grid)))
                for a in aps:
                    f = fields[a]
                    scores += observation_log_prob(scans[:, a, None], tensor(f[m + '_mean'])[None], tensor(f[m + '_variance'])[None], obs)
                prediction[m].append((torch.softmax(scores, 1) @ grid).numpy())
    prediction = {m: np.concatenate(v) for m, v in prediction.items()}
    sup_raw = dict(rss=r[meta['support']], xy=c[meta['support'], :2] - origin)
    bank = dict(rss=np.concatenate([r[c[:, 3] == j] for j in srcs] + [r[meta['support']]]),
                xy=np.concatenate([c[c[:, 3] == j][:, :2] - origin for j in srcs] + [sup_raw['xy']]))
    prediction['wknn_local'] = wknn(sup_raw, dict(rss=qrss), aps, False)
    prediction['wknn_bank'] = np.concatenate([wknn(bank, dict(rss=qrss[s:s + 64]), aps, False) for s in range(0, len(qrss), 64)])
    results = []
    for sp in sorted(set(split)):
        sel = split == sp; xy = qxy[sel]
        _, grp, cnt = np.unique(xy, axis=0, return_inverse=True, return_counts=True); grp = grp.ravel()
        for m, p in prediction.items():
            results.append(dict(method=m, split=sp, **metrics(p[sel], dict(xy=xy, scan_weights=1 / (len(cnt) * cnt[grp])))))
    np.savez_compressed(dest / 'predictions.npz', **prediction, query_xy=qxy, split=split)
    write(dest / 'completed.json', dict(event=event_id, weights=weights, aps=len(aps), low_ess_fields=low,
          low_ess_fraction={m: low[m] / len(aps) for m in METHODS}, results=results))
    log(stage='completed', event=event_id, results=[(x['method'], x['split'], round(x['mde'], 3)) for x in results])


def summarize():
    events = json.loads((OUT / 'manifest.json').read_text())['events']
    rows = {e['id']: json.loads((OUT / e['id'] / 'completed.json').read_text()) for e in events if (OUT / e['id'] / 'completed.json').exists()}
    def table(ids, split):
        per = {m: [next(x['mde'] for x in rows[i]['results'] if x['method'] == m and x['split'] == split) for i in ids]
               for m in METHODS + ['wknn_local', 'wknn_bank']}
        return dict(events=ids, per_event=per, mean={m: float(np.mean(v)) for m, v in per.items()},
                    improved=int(sum(a > b for a, b in zip(per['adapted_bank'], per['operator_bank']))))
    primary = [e['id'] for e in events if e['config'] in PRIMARY and e['id'] in rows]
    out = dict(primary_internal=table(primary, 'internal'),
               stress_TIE1_F3=table([e['id'] for e in events if e['config'] == 'TIE1_F3' and e['id'] in rows], 'internal'),
               SAH1_F2_official=table([e['id'] for e in events if e['config'] == 'SAH1_F2' and e['id'] in rows], 'official'),
               by_config={cfg: table([e['id'] for e in events if e['config'] == cfg and e['id'] in rows], 'internal') for cfg in CONFIGS},
               low_ess={i: v['low_ess_fraction'] for i, v in rows.items()})
    write(OUT / 'summary.json', out); log(stage='summary', primary=out['primary_internal']['mean'], improved=out['primary_internal']['improved'])


def main():
    if socket.gethostname() != 'ray-web': raise RuntimeError('ray-web only')
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['manifest', 'prepare', 'event', 'summarize']); p.add_argument('--event')
    a = p.parse_args(); torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    {'manifest': manifest, 'prepare': lambda: prepare(a.event), 'event': lambda: event(a.event), 'summarize': summarize}[a.stage]()


if __name__ == '__main__':
    main()
