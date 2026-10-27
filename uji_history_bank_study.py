"""Bounded UJI history-bank experiment. Run numerical work on ray-web only."""
import argparse
import json
from pathlib import Path
import shutil
import socket
import time

import numpy as np
import torch

import inference
from data import group_scans
from full_ap_study import metrics, wknn
from history_bank import BankPrior, fit_two_source_weight, gaussian_batch
from inference import gaussian_update, observation_log_prob, predict_from_posterior
from multi_history import single_history
from reference_pmc import setup, sample_reference
from run import ap_stats, tensor
from run_multi_history import digest, write
from transfer import History, Prior, Transfer, rbf
from uji_probability_study import DATA, load_raw


ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results/uji_history_bank_v1_20261026'
SOURCES = ['B1F0', 'B1F3']
METHODS = ['local_gp', 'pooled_history', 'equal_bank', 'adapted_bank']
CONTEXT = {}


def log(**row):
    print(json.dumps(row, allow_nan=False), flush=True)


def check():
    """Only checks that can invalidate this new numerical construction."""
    torch.manual_seed(26)
    obs = dict(alpha=.8, beta=-85., detect_scale=4., noise_std=5.)
    x = tensor([[0., 0.], [3., 1.], [7., 2.]])
    base = dict(mean=-95., amplitude=15., length=8., drift=6.)
    cov = rbf(x, x, 15., 8.)
    mean = tensor([-86., -90., -96.])
    stats = {k: tensor(v) for k, v in dict(n=[3, 3, 3], k=[3, 0, 1],
             s1=[-255, 0, -98], s2=[21677, 0, 9604]).items()}
    a = gaussian_update(mean, cov, stats, obs)
    b = gaussian_batch(mean[None], cov[None], {k: v[None] for k, v in stats.items()}, 5.)
    errors = [float((a[name] - value[0]).abs().max()) for name, value in zip(('mean', 'cov', 'loss'), b)]
    if max(errors) > 1e-4:
        raise RuntimeError(f'Batched RSS evidence differs from existing solver: {errors}')
    h0 = History(x, dict(mean=mean, cov=.3 * cov), base)
    h1 = History(x, dict(mean=mean + 5, cov=.5 * cov), base)
    for weights, histories in [([1., 0.], [h0, h1]), ([.3, .7], [h0, h1]), ([.7, .3], [h1, h0])]:
        prior = BankPrior(histories, weights, base)
        m, f = prior.features(x)
        c = prior.covariance(x, x, f, f)
        if torch.linalg.eigvalsh(c).min() <= 0:
            raise RuntimeError('Invalid multi-source covariance')
        if weights == [1., 0.]:
            old = Prior(Transfer('identity'), h0, base); om, of = old.features(x)
            torch.testing.assert_close(m, om)
            torch.testing.assert_close(c, old.covariance(x, x, of, of))
        elif histories[0] is h0:
            reference = (m, c)
        else:
            torch.testing.assert_close(m, reference[0]); torch.testing.assert_close(c, reference[1])
    prior = BankPrior([], [], base); m, f = prior.features(x)
    torch.testing.assert_close(prior.covariance(x, x, f, f), cov)
    truth = tensor([[-55., -65., -60.], [-65., -50., -60.], [-58., -68., -55.]])
    wrong = truth + tensor([15., -15., 15.])
    means = torch.stack((truth, wrong), dim=1)
    covariance = torch.eye(3).expand(3, 2, 3, 3)
    counts = dict(n=torch.full_like(truth, 3), k=torch.full_like(truth, 3),
                  s1=3 * truth, s2=3 * truth.square())
    weights, trials = fit_two_source_weight(means, covariance, torch.eye(3), counts, obs)
    if weights[0] < .9:
        raise RuntimeError(f'Known correct source not recovered: {weights}')
    log(stage='checks_passed', batch_rss_max_errors=errors,
        checked=['RSS likelihood/zero detections', 'single-history reduction', 'source permutation',
                 'positive covariance', 'empty-bank ordinary GP', 'known-correct source coefficient'],
        synthetic_weights=weights, fit_trials=trials, host=socket.gethostname())


def prepare(out):
    out.mkdir(exist_ok=False)
    old = ROOT / 'results/uji_probability_v1'
    shutil.copy2(old / 'parameters.json', out / 'parameters.json')
    # This record is written before history fitting, adaptation, or query scoring.
    freeze = dict(created_unix=time.time(), primary_input='history_bank + target Support + query RSS',
        model='independent-source linear autoregressive GP; f=w*h0+(1-w)*h1+delta',
        prior_art=['https://arxiv.org/abs/2512.10258', 'https://proceedings.mlr.press/v70/wei17a.html'],
        reproduction='mature linear GP component adapted to LAW likelihood; NOT full R2-HGP or TC_MSStack',
        hypothesis='Support-learned one-global-weight improves localization over same-bank equal blend and pooling',
        falsification='Better support evidence alone is insufficient; require paired query localization improvement',
        source_parameter_floors=['B0F0', 'B0F1', 'B0F2'], historical_bank=SOURCES,
        evaluation_floors=['B1F1', 'B1F2'], evaluation_role='previously inspected development, not untouched evidence',
        correspondence='Original UJI global XY and shared 520 WAP identities. No inferred alignment.',
        cross_building='Not tested. Historical target-building data are available.',
        k=5, scans_per_position=3, seeds=[0, 1, 2], methods=METHODS + ['wknn_local', 'wknn_bank'],
        initial_source_weights=[.25, .5, .75], weight_fit='S-only full-Gaussian variational empirical Bayes',
        trainable='one floor-wide source coefficient shared across APs; source parameters reused frozen',
        new_source_training_steps=0, weight_fit_maxiter=800,
        ap_policy='Union detected in historical bank or S; same AP columns for all methods; no Query selection',
        history_absence='Existing common fallback: AP never detected in a source uses unconditioned base GP',
        historical_solver_maxiter=640, support_solver='Existing M-PMC 6x4096 + 32768, unchanged',
        location_readout='Gaussian posterior moments, existing detection/RSS likelihood, AP sum, posterior mean XY',
        grid='2-unit rectangular grid from historical bank coordinates plus S only; no Query XY',
        query_access='Query RSS enters after weight fit. Query XY enters scoring only.',
        extra_unlabelled_target_data=False,
        budget_limit='K=5 initial direction test; existing exact reference supports K<=10, no K20 claim',
        counterexamples=dict(empty_bank='ordinary local GP', one_source='original identity LAW',
          zero_residual='historical uncertainty remains', source_permutation='invariant with permuted weights',
          identical_sources='weight not identifiable; no expected gain; duplicate acquisitions excluded',
          wrong_histories='convex fusion cannot invent correct spatial pattern; residual S coverage limits repair',
          missing_AP='existing common fallback; no synthetic measurements or automatic AP matching',
          weak_support='global weight can remain uncertain; point empirical Bayes does not integrate weight uncertainty',
          noise_scale='fixed source-fitted raw-dBm observation model; no K-dependent rescaling',
          coordinate_range='historical XY defines usable range; result is not no-map localization'),
        reverse_audit=['H raw rows -> per-AP historical posterior, no target query data',
          'S stats + historical SS blocks -> one weight using complete RSS/detection ELBO',
          'Same learned prior SS/GS/GG -> existing M-PMC S posterior -> conditional field',
          'Query scan -> grid likelihood -> predicted coordinate -> scored against withheld XY'],
        code_sha256={p.name: digest(p) for p in ROOT.glob('*.py')},
        inputs={str(p): digest(p) for p in [DATA / 'trainingData.csv', DATA / 'validationData.csv', old / 'parameters.json']})
    write(out / 'freeze.json', freeze)
    (out / 'code').mkdir()
    for p in ROOT.glob('*.py'): shutil.copy2(p, out / 'code' / p.name)
    floors = load_raw('trainingData.csv', set(SOURCES + freeze['evaluation_floors']))
    official = load_raw('validationData.csv', set(freeze['evaluation_floors']))
    history_xy = np.concatenate([floors[name]['positions'] for name in SOURCES])
    history_rss = np.concatenate([floors[name]['rss'] for name in SOURCES])
    events = []
    for name in freeze['evaluation_floors']:
        f, q = floors[name], official[name]
        for seed in freeze['seeds']:
            rng = np.random.default_rng(seed)
            chosen = np.sort(rng.choice(np.flatnonzero(f['stats']['n'] >= 3), 5, replace=False))
            sr = np.concatenate([rng.choice(np.flatnonzero(f['group'] == p), 3, replace=False) for p in chosen])
            qi = np.flatnonzero(~np.isin(f['group'], chosen))
            overlap = (q['xy'][:, None] == f['positions'][chosen][None]).all(-1).any(-1)
            qo = np.flatnonzero(~overlap)
            aps = np.flatnonzero((history_rss != 100).any(0) | (f['rss'][sr] != 100).any(0))
            known = np.concatenate([history_xy, f['xy'][sr]])
            events.append(dict(id=f'{name}_s{seed}', floor=name, seed=seed,
                support_rows=f['row_ids'][sr].tolist(), internal_rows=f['row_ids'][qi].tolist(),
                official_rows=q['row_ids'][qo].tolist(), ap_columns=aps.tolist(),
                grid_bounds=[known.min(0).tolist(), known.max(0).tolist()],
                support_positions=f['positions'][chosen].tolist(), official_overlaps_excluded=int(overlap.sum())))
    write(out / 'manifest.json', dict(events=events, history={name: dict(scans=len(floors[name]['rss']),
         positions=len(floors[name]['positions']), row_ids=floors[name]['row_ids'].tolist()) for name in SOURCES}))
    log(stage='prepared', events=len(events), histories=SOURCES, target_positions=5, target_scans=15)


def history_cache(out):
    params = json.loads((out / 'parameters.json').read_text()); base, obs = params['base'], params['obs']
    manifest = json.loads((out / 'manifest.json').read_text())
    floors = load_raw('trainingData.csv', set(SOURCES))
    origin = np.concatenate([f['positions'] for f in floors.values()]).mean(0)
    pools = {name: group_scans(floors[name]['rss'], floors[name]['xy'] - origin) for name in SOURCES}
    pools['pooled'] = group_scans(np.concatenate([pools[n]['rss'] for n in SOURCES]),
                                 np.concatenate([pools[n]['xy'] for n in SOURCES]))
    aps = sorted(set(ap for e in manifest['events'] for ap in e['ap_columns']))
    for name, floor in pools.items():
        destination = out / 'histories' / name; destination.mkdir(parents=True, exist_ok=True)
        for index, ap in enumerate(aps):
            path = destination / f'ap{ap}.npz'
            if path.exists(): continue
            CONTEXT.update(stage='history', source=name, ap=ap)
            started = time.perf_counter()
            h = single_history(dict(scans=floor), ap, base, obs, steps=640)
            arrays = dict(empty=np.array(h is None), origin=origin)
            if h is not None: arrays.update(z=h.z.numpy(), mean=h.m.numpy(), cov=h.c.numpy())
            np.savez_compressed(path, **arrays)
            if index % 10 == 0: log(stage='history', source=name, ap_index=index + 1, total=len(aps), seconds=time.perf_counter() - started)
    write(out / 'histories/completed.json', dict(sources=list(pools), aps=len(aps), origin=origin.tolist()))


def cached_history(out, name, ap, base):
    with np.load(out / 'histories' / name / f'ap{ap}.npz') as arrays:
        if bool(arrays['empty']): return None
        return History(tensor(arrays['z']), dict(mean=tensor(arrays['mean']), cov=tensor(arrays['cov'])), base)


def event(out, event_id):
    destination = out / event_id; destination.mkdir(exist_ok=True)
    if (destination / 'completed.json').exists(): return
    params = json.loads((out / 'parameters.json').read_text()); base, obs = params['base'], params['obs']
    meta = next(e for e in json.loads((out / 'manifest.json').read_text())['events'] if e['id'] == event_id)
    floors = load_raw('trainingData.csv', {meta['floor'], *SOURCES})
    f = floors[meta['floor']]; q = load_raw('validationData.csv', {meta['floor']})[meta['floor']]
    lookup = {int(r): i for i, r in enumerate(f['row_ids'])}; qlookup = {int(r): i for i, r in enumerate(q['row_ids'])}
    sr = np.array([lookup[r] for r in meta['support_rows']]); qi = np.array([lookup[r] for r in meta['internal_rows']])
    qo = np.array([qlookup[r] for r in meta['official_rows']])
    origin = np.array(json.loads((out / 'histories/completed.json').read_text())['origin'])
    support = group_scans(f['rss'][sr], f['xy'][sr] - origin)
    sx = tensor(support['positions']); aps = meta['ap_columns']
    bounds = np.array(meta['grid_bounds']) - origin
    axes = [np.linspace(bounds[0, d], bounds[1, d], int(np.ceil((bounds[1, d] - bounds[0, d]) / 2)) + 1) for d in range(2)]
    grid = tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))

    # No query RSS or XY is used while learning this coefficient.
    adapted = destination / 'adaptation.json'
    if not adapted.exists():
        means, covariances, residuals = [], [], []
        for ap in aps:
            hm, hc = [], []
            histories = [cached_history(out, name, ap, base) for name in SOURCES]
            for history in histories:
                prior = Prior(Transfer('identity'), history, dict(base, drift=0.))
                m, features = prior.features(sx); hm.append(m); hc.append(prior.covariance(sx, sx, features, features))
            if all(h is None for h in histories):
                hc = [torch.zeros_like(c) for c in hc]
                residuals.append(rbf(sx, sx, base['amplitude'], base['length']))
            else:
                residuals.append(rbf(sx, sx, base['drift'], base['length']))
            means.append(torch.stack(hm)); covariances.append(torch.stack(hc))
        stats = {key: torch.stack([ap_stats(support, ap)[key] for ap in aps]) for key in ('n', 'k', 's1', 's2')}
        started = time.perf_counter()
        weights, trials = fit_two_source_weight(torch.stack(means), torch.stack(covariances),
            torch.stack(residuals), stats, obs)
        write(adapted, dict(weights=weights, source_names=SOURCES, trials=trials,
              seconds=time.perf_counter() - started, target_positions=len(sx), target_scans=len(sr)))
        log(stage='adapted', event=event_id, weights=weights, seconds=time.perf_counter() - started)
    weights = json.loads(adapted.read_text())['weights']

    for index, ap in enumerate(aps):
        path = destination / f'ap{ap}.npz'; diagnostics = destination / f'ap{ap}.json'
        if path.exists() and diagnostics.exists(): continue
        histories = [cached_history(out, name, ap, base) for name in SOURCES]
        priors = dict(local_gp=BankPrior([], [], base),
            pooled_history=Prior(Transfer('identity'), cached_history(out, 'pooled', ap, base), base),
            equal_bank=BankPrior(histories, [.5, .5], base), adapted_bank=BankPrior(histories, weights, base))
        if all(h is None for h in histories):
            priors['equal_bank'] = priors['adapted_bank'] = priors['local_gp']
        saved, rows = {}, []
        for method, prior in priors.items():
            started = time.perf_counter()
            with torch.no_grad():
                sm, sf = prior.features(sx); sk = prior.covariance(sx, sx, sf, sf)
                counts = ap_stats(support, ap)
                target, proposal, center, root, _ = setup(sm, sk, counts, obs)
                samples, logw, info = sample_reference(target, proposal, center, root, 261026 + meta['seed'])
                w = logw.exp(); pm = w @ samples; c = samples - pm
                post = dict(mean=pm, cov=(c.T * w) @ c)
                gm, gf = prior.features(grid); gv = prior.diagonal(grid, gf)
                cross = prior.covariance(grid, sx, gf, sf)
                prediction = predict_from_posterior(gm, gv, cross, sm, sk, post)
                if not torch.isfinite(prediction['mean']).all() or not torch.isfinite(prediction['variance']).all() or (prediction['variance'] < 0).any():
                    raise RuntimeError(f'Invalid field: {event_id} {method} AP {ap}')
                saved[method + '_mean'] = prediction['mean'].numpy()
                saved[method + '_variance'] = prediction['variance'].numpy()
                # This remains a diagnostic of the frozen fitted weight, not a second selection criterion.
                rss_loss = gaussian_update(sm, sk, counts, obs)['loss'].item()
                rows.append(dict(method=method, support_log_evidence=info['log_detection_evidence'] - rss_loss,
                     effective_samples=info['effective_samples'], seconds=time.perf_counter() - started))
        np.savez_compressed(path, **saved); write(diagnostics, dict(ap=ap, results=rows))
        if index % 10 == 0: log(stage='field', event=event_id, ap_index=index + 1, total=len(aps))

    query = group_scans(np.concatenate([f['rss'][qi], q['rss'][qo]]),
                        np.concatenate([f['xy'][qi], q['xy'][qo]]) - origin)
    prediction = {}
    device = torch.device('cuda:0')
    with torch.no_grad():
        for method in METHODS:
            scores = torch.zeros((len(query['rss']), len(grid)), device=device)
            for ap in aps:
                with np.load(destination / f'ap{ap}.npz') as arrays:
                    m = tensor(arrays[method + '_mean']).to(device)
                    v = tensor(arrays[method + '_variance']).to(device)
                rss = tensor(query['rss'][:, ap]).to(device)
                for start in range(0, len(rss), 128):
                    scores[start:start + 128] += observation_log_prob(rss[start:start + 128, None], m[None], v[None], obs)
            prediction[method] = (torch.softmax(scores, 1) @ grid.to(device)).cpu().numpy()
    prediction['wknn_local'] = wknn(support, query, aps, False)
    all_data = group_scans(np.concatenate([floors[n]['rss'] for n in SOURCES] + [support['rss']]),
        np.concatenate([floors[n]['xy'] - origin for n in SOURCES] + [support['xy']]))
    prediction['wknn_bank'] = np.concatenate([
        wknn(all_data, dict(rss=query['rss'][start:start + 64]), aps, False)
        for start in range(0, len(query['rss']), 64)])
    results = []
    for split, sl in [('internal', slice(0, len(qi))), ('official', slice(len(qi), None))]:
        xy = query['xy'][sl]
        _, group, count = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
        group = group.ravel()
        for method, pred in prediction.items():
            for weighting, scan_weights in [('scan', np.ones(len(xy)) / len(xy)),
                                            ('position', 1 / (len(count) * count[group]))]:
                results.append(dict(method=method, split=split, weighting=weighting,
                     **metrics(pred[sl], dict(xy=xy, scan_weights=scan_weights))))
    np.savez_compressed(destination / 'predictions.npz', **{k: v + origin for k, v in prediction.items()},
        query_xy=query['xy'] + origin, internal_count=len(qi), support_xy=support['positions'] + origin,
        grid=grid.numpy() + origin, ap_columns=aps)
    write(destination / 'completed.json', dict(completed=True, event=event_id, weights=weights, results=results,
          predictions_sha256=digest(destination / 'predictions.npz')))
    log(stage='completed', event=event_id, weights=weights,
        official=[r for r in results if r['split'] == 'official' and r['weighting'] == 'position'])


def summarize(out):
    manifest = json.loads((out / 'manifest.json').read_text())
    rows = []
    for meta in manifest['events']:
        result = json.loads((out / meta['id'] / 'completed.json').read_text())
        rows.extend(dict(event=meta['id'], floor=meta['floor'], seed=meta['seed'], **r) for r in result['results'])
    summary = []
    for method in METHODS + ['wknn_local', 'wknn_bank']:
        for split in ('internal', 'official'):
            for weighting in ('scan', 'position'):
                group = [r for r in rows if (r['method'], r['split'], r['weighting']) == (method, split, weighting)]
                summary.append(dict(method=method, split=split, weighting=weighting,
                    mean_error=float(np.mean([r['mde'] for r in group])),
                    std_across_events=float(np.std([r['mde'] for r in group], ddof=1)), events=len(group)))
    write(out / 'summary.json', dict(completed=True, rows=rows, summary=summary))
    log(stage='summary', results=[r for r in summary if r['split'] == 'official' and r['weighting'] == 'position'])


def main():
    if socket.gethostname() != 'ray-web': raise RuntimeError('Numerical work runs on ray-web only')
    p = argparse.ArgumentParser(); p.add_argument('stage', choices=['check', 'prepare', 'history', 'event', 'summarize'])
    p.add_argument('--event'); p.add_argument('--out', type=Path, default=OUT); args = p.parse_args()
    torch.set_default_dtype(torch.float64); torch.set_num_threads(1)
    real = inference.minimize
    def tracked(*a, **kw):
        result = real(*a, **kw)
        row = dict(context=dict(CONTEXT), success=bool(result.success), iterations=int(result.nit),
                   gradient=float(np.abs(result.jac).max()), message=str(result.message))
        with (args.out / 'history_solver.jsonl').open('a') as stream: stream.write(json.dumps(row) + '\n')
        if not result.success: raise RuntimeError(f'Unfinished historical VI: {row}')
        return result
    if args.stage == 'check': check(); return
    if args.stage == 'prepare': prepare(args.out); return
    freeze = json.loads((args.out / 'freeze.json').read_text())
    for name in ('history_bank.py', 'uji_history_bank_study.py', 'inference.py', 'transfer.py', 'reference_pmc.py', 'multi_history.py'):
        if digest(ROOT / name) != freeze['code_sha256'][name]: raise RuntimeError(f'Frozen code changed: {name}')
    inference.minimize = tracked
    if args.stage == 'history': history_cache(args.out)
    elif args.stage == 'event': event(args.out, args.event)
    else: summarize(args.out)


if __name__ == '__main__': main()
