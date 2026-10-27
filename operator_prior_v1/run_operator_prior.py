"""Operator prior v1: train on UJI B0/B2 only, evaluate on the original B1 events.

Stages: check | prepare | train | event --event ID | summarize. Run on ray-web only.
See PROTOCOL.md (written before any run).
"""
import argparse
import itertools
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from data import group_scans
from full_ap_study import metrics, wknn
from history_bank import BankPrior, fit_two_source_weight
from inference import observation_log_prob, predict_from_posterior
from multi_history import single_history
from operator_prior import OperatorBankPrior, OperatorNet, history_moments, marginal_loglik, source_tokens
from reference_pmc import setup, sample_reference
from run import ap_stats, tensor
from run_multi_history import digest, write
from transfer import History, Prior, Transfer, rbf
from uji_probability_study import load_raw

ROOT = Path(__file__).resolve().parent
OLD = ROOT / 'results/uji_history_bank_v1_20261026'
OUT = ROOT / 'results/uji_operator_prior_v1_20261026'
SOURCES = ['B1F0', 'B1F3']
TRAIN_FLOORS = {0: [0, 1, 2, 3], 2: [0, 1, 2, 3, 4]}
STEPS, LR, W_EVERY = 1500, 1e-3, 50
W_GRID = np.linspace(0, 1, 17)


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def params():
    p = json.loads((OLD / 'parameters.json').read_text())
    return p['base'], p['obs']


def floor_dz(name, target):
    return int(name[-1]) - int(target[-1])


def check():
    base, obs = params()
    torch.manual_seed(26)
    x = tensor([[0., 0.], [4., 1.], [9., 3.], [15., -2.]])
    mean = tensor([-70., -80., -92., -99.])
    h0 = History(x, dict(mean=mean, cov=.3 * rbf(x, x, base['amplitude'], base['length'])), base)
    h1 = History(x, dict(mean=mean - 12, cov=.5 * rbf(x, x, base['amplitude'], base['length'])), base)
    y = tensor([[1., 1.], [7., 0.], [20., 5.]])
    errors = {}
    old, new = BankPrior([h0, h1], [.4, .6], base), OperatorBankPrior([h0, h1], [.4, .6], base, None, [-1, 2])
    (om, of), (nm, nf) = old.features(y), new.features(y)
    errors['net_none'] = max(float((om - nm).abs().max()), float((old.covariance(y, y, of, of) - new.covariance(y, y, nf, nf)).abs().max()),
                             float((old.diagonal(y, of) - new.diagonal(y, nf)).abs().max()))
    net = OperatorNet().double()
    g0, _ = net(torch.stack([source_tokens(h0, -1, y, base), source_tokens(h1, 2, y, base)]))
    errors['initial_g'] = float(g0.detach().abs().max())
    with torch.no_grad():
        for p in net.parameters(): p.add_(.3 * torch.randn_like(p))
    ga, sa = net(torch.stack([source_tokens(h0, -1, y, base), source_tokens(h1, 2, y, base)]))
    gb, sb = net(torch.stack([source_tokens(h1, 2, y, base), source_tokens(h0, -1, y, base)]))
    errors['permutation'] = max(float((ga - gb).abs().max()), float((sa - sb).abs().max()))
    prior = OperatorBankPrior([h0, h1], [.4, .6], base, net, [-1, 2])
    m, f = prior.features(y)
    errors['min_eig'] = -float(torch.linalg.eigvalsh(prior.covariance(y, y, f, f)).min())
    errors['diag_vs_cov'] = float((prior.diagonal(y, f) - prior.covariance(y, y, f, f).diagonal()).abs().max())
    mu, var = tensor([-80., -95., -60.]), tensor([30., 120., 9.])
    for rss in (-78., 100.):
        r = tensor([rss] * 3)
        k = tensor([0. if rss == 100. else 1.] * 3)
        ml = marginal_loglik(mu, var, torch.ones(3), k, k * rss, k * rss ** 2, obs)
        ref = observation_log_prob(r, mu, var, obs)
        errors[f'single_scan_{rss}'] = float((ml - ref).abs().max())
    m0, v0 = history_moments(h0, y, base)
    p0 = Prior(Transfer('identity'), h0, dict(base, drift=0.)); pm, pf = p0.features(y)
    errors['moments'] = max(float((m0 - pm).abs().max()), float((v0 - p0.diagonal(y, pf)).abs().max()))
    log(stage='check', errors=errors)
    limits = dict(net_none=1e-10, initial_g=1e-12, permutation=1e-10, min_eig=0., diag_vs_cov=1e-8,
                  moments=1e-6, **{'single_scan_-78.0': 1e-8, 'single_scan_100.0': 1e-3})
    bad = {k: v for k, v in errors.items() if v > limits[k] and not (k == 'min_eig' and v < 0)}
    if bad: raise RuntimeError(f'check failed: {bad}')


def prepare(out):
    """Fit historical GPs for every B0/B2 floor and AP with the deployment rule."""
    base, obs = params()
    names = {f'B{b}F{f}' for b, fl in TRAIN_FLOORS.items() for f in fl}
    floors = load_raw('trainingData.csv', names)
    for b, fl in TRAIN_FLOORS.items():
        origin = np.concatenate([floors[f'B{b}F{f}']['positions'] for f in fl]).mean(0)
        detected = np.zeros(520, bool)
        for f in fl: detected |= floors[f'B{b}F{f}']['stats']['k'].sum(0) > 0
        for f in fl:
            name = f'B{b}F{f}'; floor = floors[name]
            scans = group_scans(floor['rss'], floor['xy'] - origin)
            dest = out / 'train_histories' / name; dest.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dest / 'target.npz', positions=scans['positions'], origin=origin, **scans['stats'])
            started = time.perf_counter()
            for ap in np.flatnonzero(detected):
                path = dest / f'ap{ap}.npz'
                if path.exists(): continue
                h = single_history(dict(scans=scans), int(ap), base, obs, steps=640)
                arrays = dict(empty=np.array(h is None))
                if h is not None: arrays.update(z=h.z.numpy(), mean=h.m.numpy(), cov=h.c.numpy())
                np.savez_compressed(path, **arrays)
            log(stage='prepare', floor=name, aps=int(detected.sum()), seconds=time.perf_counter() - started)
    write(out / 'train_histories/completed.json', dict(floors=sorted(names)))


def load_history(path, base):
    with np.load(path) as a:
        if bool(a['empty']): return None
        return History(tensor(a['z']), dict(mean=tensor(a['mean']), cov=tensor(a['cov'])), base)


def build_tasks(out, base):
    tasks = []
    for b, fl in TRAIN_FLOORS.items():
        for t in fl:
            tname = f'B{b}F{t}'; root = out / 'train_histories'
            with np.load(root / tname / 'target.npz') as a:
                x = tensor(a['positions']); stats = {k: tensor(a[k]) for k in ('n', 'k', 's1', 's2')}
            aps = sorted(int(p.stem[2:]) for p in (root / tname).glob('ap*.npz'))
            for pair in itertools.combinations([f for f in fl if f != t], 2):
                tok, means, vars_, keep = [], [], [], []
                for ap in aps:
                    hs = [load_history(root / f'B{b}F{j}' / f'ap{ap}.npz', base) for j in pair]
                    if all(h is None for h in hs): continue
                    tok.append(torch.stack([source_tokens(h, j - t, x, base) for h, j in zip(hs, pair)]))
                    mv = [history_moments(h, x, base) for h in hs]
                    means.append(torch.stack([m for m, _ in mv])); vars_.append(torch.stack([v for _, v in mv])); keep.append(ap)
                ap_index = tensor(keep).long()
                tasks.append(dict(target=tname, sources=[f'B{b}F{j}' for j in pair],
                    tokens=torch.stack(tok, 1),            # [J, A, n, TOKEN]
                    means=torch.stack(means, 1), vars=torch.stack(vars_, 1),  # [J, A, n]
                    stats={k: v[None].expand(len(keep), -1) if k == 'n' else v[:, ap_index].T for k, v in stats.items()},
                    w=.5))
    return tasks


def task_loss(net, task, base, obs, w=None):
    w = task['w'] if w is None else w
    J, A, n, _ = task['tokens'].shape
    g, s = net(task['tokens'].reshape(J, A * n, -1))
    g, s = g.reshape(A, n), s.reshape(A, n)
    mean = w * task['means'][0] + (1 - w) * task['means'][1] + g
    var = base['drift'] ** 2 + w ** 2 * task['vars'][0] + (1 - w) ** 2 * task['vars'][1] + s.square()
    st = task['stats']
    ll = marginal_loglik(mean, var, st['n'], st['k'], st['s1'], st['s2'], obs)
    return -ll.mean()


def train(out):
    base, obs = params()
    torch.manual_seed(261026)
    tasks = build_tasks(out, base)
    log(stage='tasks', count=len(tasks), aps=[t['tokens'].shape[1] for t in tasks])
    net = OperatorNet().double()
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    curve = []
    for step in range(STEPS + 1):
        if step % W_EVERY == 0:
            with torch.no_grad():
                for t in tasks:
                    t['w'] = float(W_GRID[int(np.argmin([task_loss(net, t, base, obs, w).item() for w in W_GRID]))])
                curve.append(dict(step=step, loss=float(np.mean([task_loss(net, t, base, obs).item() for t in tasks])),
                                  convex_only=float(np.mean([task_loss(ZERO, t, base, obs).item() for t in tasks])),
                                  weights=[t['w'] for t in tasks]))
                log(stage='train', **{k: v for k, v in curve[-1].items() if k != 'weights'})
        if step == STEPS: break
        opt.zero_grad()
        loss = sum(task_loss(net, t, base, obs) for t in tasks) / len(tasks)
        loss.backward(); opt.step()
    torch.save(net.state_dict(), out / 'operator.pt')
    write(out / 'training.json', dict(steps=STEPS, lr=LR, curve=curve, tasks=[dict(target=t['target'], sources=t['sources'],
          aps=int(t['tokens'].shape[1]), w=t['w']) for t in tasks], checkpoint_sha256=digest(out / 'operator.pt')))


class ZeroNet(torch.nn.Module):
    """g=0, s=0: the convex BankPrior term only (training-loss reference)."""
    def forward(self, tokens):
        z = tokens.new_zeros(tokens.shape[1]); return z, z


ZERO = ZeroNet()


def event(out, event_id):
    base, obs = params()
    dest = out / event_id; dest.mkdir(exist_ok=True)
    if (dest / 'completed.json').exists(): return
    net = OperatorNet().double(); net.load_state_dict(torch.load(out / 'operator.pt')); net.eval()
    meta = next(e for e in json.loads((OLD / 'manifest.json').read_text())['events'] if e['id'] == event_id)
    floors = load_raw('trainingData.csv', {meta['floor'], *SOURCES})
    f = floors[meta['floor']]; q = load_raw('validationData.csv', {meta['floor']})[meta['floor']]
    lookup = {int(r): i for i, r in enumerate(f['row_ids'])}; qlookup = {int(r): i for i, r in enumerate(q['row_ids'])}
    sr = np.array([lookup[r] for r in meta['support_rows']]); qi = np.array([lookup[r] for r in meta['internal_rows']])
    qo = np.array([qlookup[r] for r in meta['official_rows']])
    origin = np.array(json.loads((OLD / 'histories/completed.json').read_text())['origin'])
    support = group_scans(f['rss'][sr], f['xy'][sr] - origin)
    sx = tensor(support['positions']); aps = meta['ap_columns']
    bounds = np.array(meta['grid_bounds']) - origin
    axes = [np.linspace(bounds[0, d], bounds[1, d], int(np.ceil((bounds[1, d] - bounds[0, d]) / 2)) + 1) for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))
    dz = [floor_dz(n, meta['floor']) for n in SOURCES]
    hist = {ap: [load_history(OLD / 'histories' / n / f'ap{ap}.npz', base) for n in SOURCES] for ap in aps}
    old_w = json.loads((OLD / event_id / 'adaptation.json').read_text())['weights']

    adapted = dest / 'adaptation.json'
    if not adapted.exists():
        means, covariances, residuals = [], [], []
        with torch.no_grad():
            for ap in aps:
                hs = hist[ap]; hm, hc = [], []
                op = OperatorBankPrior(hs, [.5, .5], base, net, dz)
                g, s = op.operator(sx) if op.active else (sx.new_zeros(len(sx)), sx.new_zeros(len(sx)))
                for h in hs:
                    p = Prior(Transfer('identity'), h, dict(base, drift=0.))
                    m, feat = p.features(sx); hm.append(m + g); hc.append(p.covariance(sx, sx, feat, feat))
                if all(h is None for h in hs):
                    hc = [torch.zeros_like(c) for c in hc]; residuals.append(rbf(sx, sx, base['amplitude'], base['length']))
                else:
                    residuals.append(rbf(sx, sx, base['drift'], base['length']) + s[:, None] * s[None] * rbf(sx, sx, 1., base['length']))
                means.append(torch.stack(hm)); covariances.append(torch.stack(hc))
        stats = {key: torch.stack([ap_stats(support, ap)[key] for ap in aps]) for key in ('n', 'k', 's1', 's2')}
        weights, trials = fit_two_source_weight(torch.stack(means), torch.stack(covariances), torch.stack(residuals), stats, obs)
        write(adapted, dict(weights=weights, trials=trials, old_weights=old_w))
    weights = json.loads(adapted.read_text())['weights']
    log(stage='adapted', event=event_id, weights=weights, old_weights=old_w)

    replay_error = 0.
    for index, ap in enumerate(aps):
        path = dest / f'ap{ap}.npz'
        if path.exists(): continue
        hs = hist[ap]
        priors = dict(adapted_bank=BankPrior(hs, old_w, base), operator_bank=OperatorBankPrior(hs, weights, base, net, dz))
        if all(h is None for h in hs):
            priors = {k: BankPrior([], [], base) for k in priors}
        saved = {}
        for method, prior in priors.items():
            with torch.no_grad():
                sm, sf = prior.features(sx); sk = prior.covariance(sx, sx, sf, sf)
                target, proposal, center, root, _ = setup(sm, sk, ap_stats(support, ap), obs)
                samples, logw, info = sample_reference(target, proposal, center, root, 261026 + meta['seed'])
                w = logw.exp(); pm = w @ samples; c = samples - pm
                post = dict(mean=pm, cov=(c.T * w) @ c)
                gm, gf = prior.features(grid); gv = prior.diagonal(grid, gf)
                pred = predict_from_posterior(gm, gv, prior.covariance(grid, sx, gf, sf), sm, sk, post)
            if not torch.isfinite(pred['mean']).all() or (pred['variance'] < 0).any():
                raise RuntimeError(f'Invalid field {event_id} {method} {ap}')
            saved[method + '_mean'] = pred['mean'].numpy(); saved[method + '_variance'] = pred['variance'].numpy()
        with np.load(OLD / event_id / f'ap{ap}.npz') as cached:
            err = float(np.abs(cached['adapted_bank_mean'] - saved['adapted_bank_mean']).max())
        replay_error = max(replay_error, err)
        np.savez_compressed(path, **saved, replay_error=np.array(err))
        if index % 20 == 0: log(stage='field', event=event_id, ap_index=index + 1, total=len(aps), replay_error=replay_error)

    query = group_scans(np.concatenate([f['rss'][qi], q['rss'][qo]]), np.concatenate([f['xy'][qi], q['xy'][qo]]) - origin)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    prediction = {}
    with torch.no_grad():
        for method in ('adapted_bank', 'operator_bank'):
            scores = torch.zeros((len(query['rss']), len(grid)), device=device)
            for ap in aps:
                with np.load(dest / f'ap{ap}.npz') as a:
                    m = tensor(a[method + '_mean']).to(device); v = tensor(a[method + '_variance']).to(device)
                rss = tensor(query['rss'][:, ap]).to(device)
                for start in range(0, len(rss), 128):
                    scores[start:start + 128] += observation_log_prob(rss[start:start + 128, None], m[None], v[None], obs)
            prediction[method] = (torch.softmax(scores, 1) @ grid.to(device)).cpu().numpy()
    results = []
    for split, sl in [('internal', slice(0, len(qi))), ('official', slice(len(qi), None))]:
        xy = query['xy'][sl]
        _, group, count = np.unique(xy, axis=0, return_inverse=True, return_counts=True); group = group.ravel()
        for method, pred in prediction.items():
            results.append(dict(method=method, split=split, weighting='position',
                                **metrics(pred[sl], dict(xy=xy, scan_weights=1 / (len(count) * count[group])))))
    np.savez_compressed(dest / 'predictions.npz', **{k: v + origin for k, v in prediction.items()}, query_xy=query['xy'] + origin)
    errs = [float(np.load(dest / f'ap{ap}.npz')['replay_error']) for ap in aps]
    write(dest / 'completed.json', dict(event=event_id, weights=weights, old_weights=old_w, results=results,
          replay_vs_old_cache=dict(max=max(errs), median=float(np.median(errs)))))
    log(stage='completed', event=event_id, results=results)


def summarize(out):
    events = [e['id'] for e in json.loads((OLD / 'manifest.json').read_text())['events']]
    rows = {e: json.loads((out / e / 'completed.json').read_text())['results'] for e in events}
    summary = {}
    for split in ('internal', 'official'):
        per = {m: [next(r['mde'] for r in rows[e] if r['method'] == m and r['split'] == split) for e in events]
               for m in ('adapted_bank', 'operator_bank')}
        summary[split] = dict(per_event=per, mean={m: float(np.mean(v)) for m, v in per.items()},
                              improved_events=int(sum(a > b for a, b in zip(per['adapted_bank'], per['operator_bank']))))
    write(out / 'summary.json', summary); log(stage='summary', **summary)


def main():
    if socket.gethostname() != 'ray-web': raise RuntimeError('Numerical work runs on ray-web only')
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['check', 'prepare', 'train', 'event', 'summarize'])
    p.add_argument('--event'); args = p.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(4)
    OUT.mkdir(exist_ok=True)
    {'check': lambda: check(), 'prepare': lambda: prepare(OUT), 'train': lambda: train(OUT),
     'event': lambda: event(OUT, args.event), 'summarize': lambda: summarize(OUT)}[args.stage]()


if __name__ == '__main__':
    main()
