"""Leave-one-dataset-out evaluation of the fixed v1.1+zeta method. See PROTOCOL_lodo.md. ray-web only.

Stages: train --event noUJI|noTAU | event --event uji:ID|tau:ID | summarize.
"""
import argparse
import json
from pathlib import Path
import socket

import numpy as np
import torch

import run_operator_v11 as V
from data import group_scans
from full_ap_study import metrics
from history_bank import BankPrior
from inference import observation_log_prob
from operator_prior import OperatorBankPrior
from run import ap_stats, tensor
from run_multi_history import digest, write
from run_operator_prior import load_history
from run_operator_tampere import raw as tampere_raw
from source_weight_mixture import field_from_prior
from uji_probability_study import load_raw

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/operator_lodo_20261026'
UJI_OLD = ROOT / 'results/uji_history_bank_v1_20261026'
TAU = ROOT / 'results/tampere_operator_confirm_v1_20261026'
SETS = {'noUJI': ['TIE1', 'SAH1', 'GIA'], 'noTAU': ['UJI_B0', 'UJI_B1', 'UJI_B2', 'GIA']}
V.TRAIN['GIA'] = [0, 1, 2, 3, 4]
ARMS = ['adapted_bank', 'lodo_full', 'lodo_scaled']


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def train(kind):
    base, obs = V.params(); torch.manual_seed(261028); rng = np.random.default_rng(261028)
    full = dict(V.TRAIN); V.TRAIN.clear(); V.TRAIN.update({b: full[b] for b in SETS[kind]})
    tasks = V.build_tasks(base, obs, rng)
    net = V.MaskedNet(None); opt = torch.optim.Adam(net.parameters(), lr=V.LR); curve = []
    with torch.no_grad():
        convex = float(np.mean([V.loss(None, t, base, obs).item() for t in tasks]))
    for step in range(V.STEPS + 1):
        if step % 100 == 0:
            with torch.no_grad():
                curve.append(dict(step=step, loss=float(np.mean([V.loss(net, t, base, obs).item() for t in tasks])), convex_only=convex))
            log(stage='train', kind=kind, **curve[-1])
        if step == V.STEPS: break
        batch = rng.choice(len(tasks), V.BATCH, replace=False)
        opt.zero_grad(); l = sum(V.loss(net, tasks[i], base, obs) for i in batch) / V.BATCH; l.backward(); opt.step()
    OUT.mkdir(exist_ok=True); torch.save(net.state_dict(), OUT / f'operator_{kind}.pt')
    write(OUT / f'training_{kind}.json', dict(kind=kind, buildings=SETS[kind], curve=curve, tasks=len(tasks),
          checkpoint_sha256=digest(OUT / f'operator_{kind}.pt')))


def load_net(kind):
    net = V.MaskedNet(None); net.load_state_dict(torch.load(OUT / f'operator_{kind}.pt', weights_only=True)); net.eval(); return net


def uji_case(event_id):
    meta = next(e for e in json.loads((UJI_OLD / 'manifest.json').read_text())['events'] if e['id'] == event_id)
    sources = ['B1F0', 'B1F3']
    f = load_raw('trainingData.csv', {meta['floor']})[meta['floor']]; q = load_raw('validationData.csv', {meta['floor']})[meta['floor']]
    lk = {int(r): i for i, r in enumerate(f['row_ids'])}; ql = {int(r): i for i, r in enumerate(q['row_ids'])}
    sr = np.array([lk[r] for r in meta['support_rows']]); qi = np.array([lk[r] for r in meta['internal_rows']]); qo = np.array([ql[r] for r in meta['official_rows']])
    origin = np.array(json.loads((UJI_OLD / 'histories/completed.json').read_text())['origin'])
    support = group_scans(f['rss'][sr], f['xy'][sr] - origin)
    b = np.array(meta['grid_bounds']) - origin
    axes = [np.linspace(b[0, d], b[1, d], int(np.ceil((b[1, d] - b[0, d]) / 2)) + 1) for d in range(2)]
    base, _ = V.params()
    aps = meta['ap_columns']
    hist = {a: [load_history(UJI_OLD / 'histories' / n / f'ap{a}.npz', base) for n in sources] for a in aps}
    return dict(support=support, aps=aps, hist=hist, grid=tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2)),
                qrss=np.concatenate([f['rss'][qi], q['rss'][qo]]), qxy=np.concatenate([f['xy'][qi], q['xy'][qo]]) - origin,
                split=np.array(['internal'] * len(qi) + ['official'] * len(qo)),
                dz=[int(n[-1]) - int(meta['floor'][-1]) for n in sources], seed=meta['seed'], net='noUJI', dataset='UJI_B1')


def tau_case(event_id):
    meta = next(e for e in json.loads((TAU / 'manifest.json').read_text())['events'] if e['id'] == event_id)
    b, t, srcs, origin = meta['building'], meta['target'], meta['sources'], np.array(meta['origin'])
    c, r = tampere_raw(b)
    support = group_scans(r[meta['support']], c[meta['support'], :2] - origin)
    qxy, qrss, split = c[meta['internal'], :2] - origin, r[meta['internal']], ['internal'] * len(meta['internal'])
    if (b, t) == ('SAH1', 2):
        tc, tr = tampere_raw(b, 'test'); m = tc[:, 3] == t
        qxy, qrss = np.concatenate([qxy, tc[m][:, :2] - origin]), np.concatenate([qrss, tr[m]]); split += ['official'] * int(m.sum())
    hpos = np.concatenate([c[c[:, 3] == j][:, :2] - origin for j in srcs])
    hdet = np.zeros(r.shape[1], bool)
    for j in srcs: hdet |= (r[c[:, 3] == j] != 100).any(0)
    aps = np.flatnonzero(hdet | (support['stats']['k'].sum(0) > 0)).tolist()
    sxy = support['positions']
    lo = np.minimum(hpos.min(0), sxy.min(0)); hi = np.maximum(hpos.max(0), sxy.max(0))
    axes = [np.linspace(lo[d], hi[d], int(np.ceil((hi[d] - lo[d]) / 2)) + 1) for d in range(2)]
    base, _ = V.params()
    dirs = [TAU / 'histories' / meta['config'] / f'{b}F{j}' for j in srcs]
    hist = {a: [load_history(d / f'ap{a}.npz', base) if (d / f'ap{a}.npz').exists() else None for d in dirs] for a in aps}
    return dict(support=support, aps=aps, hist=hist, grid=tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2)),
                qrss=qrss, qxy=qxy, split=np.array(split), dz=[j - t for j in srcs], seed=meta['seed'], net='noTAU',
                dataset=b, config=meta['config'])


def event(spec):
    kind, event_id = spec.split(':')
    dest = OUT / f'{kind}_{event_id}'; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'completed.json').exists(): return
    case = uji_case(event_id) if kind == 'uji' else tau_case(event_id)
    base, obs = V.params(); support = case['support']; sx = tensor(support['positions']); aps = case['aps']; hist = case['hist']
    net = load_net(case['net'])
    stats = {k: torch.stack([ap_stats(support, a)[k] for a in aps]) for k in ('n', 'k', 's1', 's2')}
    if not (dest / 'weights.json').exists():
        w, trials = V.fit_w(hist, aps, sx, stats, base, obs)
        z, scores = V.choose_zeta(hist, aps, sx, support, w, base, obs, net, case['dz'])
        write(dest / 'weights.json', dict(w=w, zeta=z, zeta_scores=scores, trials=trials))
    W = json.loads((dest / 'weights.json').read_text()); w, zeta = W['w'], W['zeta']
    nets = dict(lodo_full=net, lodo_scaled=V.ScaledNet(net, zeta))
    log(stage='adapted', event=spec, w=w, zeta=zeta, aps=len(aps))
    for a in aps:
        path = dest / f'ap{a}.npz'
        if path.exists(): continue
        hs = hist[a]; arrays = {}
        for m in ARMS:
            prior = BankPrior(hs, w, base) if m == 'adapted_bank' else OperatorBankPrior(hs, w, base, nets[m], case['dz'])
            if all(h is None for h in hs): prior = BankPrior([], [], base)
            with torch.no_grad():
                res = field_from_prior(prior, obs, sx, ap_stats(support, a), case['grid'], 261026 + 10000 * case['seed'] + a, defensive_fraction=.1)
            arrays[m + '_mean'] = res['mean'].numpy(); arrays[m + '_variance'] = res['variance'].numpy()
            arrays[m + '_ess'] = np.array(np.nan if res['effective_samples'] is None else res['effective_samples'])
        np.savez_compressed(path, **arrays)
    fields = {a: np.load(dest / f'ap{a}.npz') for a in aps}
    low = {m: int(sum(float(fields[a][m + '_ess']) < 1000 for a in aps)) for m in ARMS}
    grid, qrss = case['grid'], case['qrss']; pred = {m: [] for m in ARMS}
    with torch.no_grad():
        for start in range(0, len(qrss), 64):
            scans = tensor(qrss[start:start + 64].astype(float))
            for m in ARMS:
                sc = scans.new_zeros((len(scans), len(grid)))
                for a in aps:
                    sc += observation_log_prob(scans[:, a, None], tensor(fields[a][m + '_mean'])[None], tensor(fields[a][m + '_variance'])[None], obs)
                pred[m].append((torch.softmax(sc, 1) @ grid).numpy())
    pred = {m: np.concatenate(v) for m, v in pred.items()}
    results = []
    for sp in sorted(set(case['split'])):
        sel = case['split'] == sp; xy = case['qxy'][sel]
        _, grp, cnt = np.unique(np.round(xy, 3), axis=0, return_inverse=True, return_counts=True); grp = grp.ravel()
        for m, p in pred.items():
            results.append(dict(method=m, split=sp, **metrics(p[sel], dict(xy=xy, scan_weights=1 / (len(cnt) * cnt[grp])))))
    np.savez_compressed(dest / 'predictions.npz', **pred, query_xy=case['qxy'], split=case['split'])
    write(dest / 'completed.json', dict(event=spec, dataset=case['dataset'], config=case.get('config', 'UJI_B1'), w=w, zeta=zeta,
          low_ess_fraction={m: low[m] / len(aps) for m in ARMS}, results=results))
    log(stage='completed', event=spec, zeta=zeta, results=[(r['method'], r['split'], round(r['mde'], 3)) for r in results])


def summarize():
    rows = [json.loads(p.read_text()) for p in sorted(OUT.glob('*/completed.json'))]
    gia = json.loads((V.OUT / 'summary.json').read_text())
    out = {}
    def pick(r, m, sp):
        return next((x['mde'] for x in r['results'] if x['method'] == m and x['split'] == sp), None)
    for name, sel in [('UJI_B1', lambda r: r['dataset'] == 'UJI_B1'),
                      ('Tampere_primary', lambda r: r['config'] in ('TIE1_F1', 'TIE1_F2', 'SAH1_F2')),
                      ('TIE1', lambda r: r['config'] in ('TIE1_F1', 'TIE1_F2')), ('SAH1', lambda r: r['config'] == 'SAH1_F2'),
                      ('TIE1_F3_stress', lambda r: r['config'] == 'TIE1_F3')]:
        for sp in ('internal', 'official'):
            rs = [r for r in rows if sel(r) and pick(r, 'adapted_bank', sp) is not None]
            if not rs: continue
            vals = {m: [pick(r, m, sp) for r in rs] for m in ARMS}
            out[f'{name}/{sp}'] = dict(n=len(rs), mean={m: float(np.mean(v)) for m, v in vals.items()},
                improved={m: int(sum(a > b for a, b in zip(vals['adapted_bank'], v))) for m, v in vals.items()},
                zeta=[r['zeta'] for r in rs])
    out['GIA/internal'] = dict(n=9, mean=dict(adapted_bank=gia['mean']['adapted_bank'], lodo_full=gia['mean']['v11_full'], lodo_scaled=gia['mean']['v11_scaled']),
                               improved=dict(lodo_full=gia['improved_vs_adapted']['v11_full'], lodo_scaled=gia['improved_vs_adapted']['v11_scaled']))
    out['low_ess_max'] = max(max(r['low_ess_fraction'].values()) for r in rows)
    write(OUT / 'summary.json', out); log(stage='summary', **{k: v for k, v in out.items() if k != 'low_ess_max'})


def main():
    if socket.gethostname() != 'ray-web': raise RuntimeError('ray-web only')
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['train', 'event', 'summarize']); p.add_argument('--event')
    a = p.parse_args(); torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    {'train': lambda: train(a.event), 'event': lambda: event(a.event), 'summarize': summarize}[a.stage]()


if __name__ == '__main__':
    main()
