"""Fixed-query K curves for existing methods; no new source training.

History, AP permissions, support rows, domain and readout are shared within
each event. The learned arm is the existing K5-trained paired10 model: this
tests its deployment across K, not a newly tuned model for each budget.
"""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import socket
import sys
import time

import numpy as np
import torch

# Reuse the existing frozen data loader and its dependency search path.
from residual_probe.prepare import InterceptBankPrior, choose_intercept, history_features, save_case
from public_corpus import k_curve_episodes as episodes
from data import group_scans
from transfer import History
from history_bank import BankPrior
from conditional_operator_learning import fit_episode
from residual_probe.paired_features import features as paired_features
from residual_probe.support_calibration import fit as fit_support
from inference import observation_log_prob, predict_from_posterior
from reference_pmc import sample_reference
from bounded_pmc import setup
from full_ap_study import metrics, wknn
from run import ap_stats, tensor

ROOT, OUT = episodes.ROOT, episodes.OUT
SOURCE = ROOT / 'results/source_residual_probe_v1_20261026/learning_v2'
PROBABILISTIC = ('GP_S', 'H_base', 'H_source', 'H_support')
METHODS = ('WKNN_S', 'WKNN_HS', *PROBABILISTIC)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')


def log(**value):
    print(json.dumps(dict(time=datetime.now(timezone.utc).isoformat(), **value)), flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def manifest():
    return read(OUT / 'manifest.json')


def source_model(building):
    path = SOURCE / building / 'paired/model.json'
    model = read(path)
    assert building not in model['training_buildings']
    assert model['feature_indices'] == list(range(10))
    return path, model


@lru_cache(maxsize=2048)
def load_history(path, shift, base_items):
    with np.load(path, allow_pickle=False) as a:
        if bool(a['empty']):
            return None
        return History(tensor(a['z']) + tensor(shift),
                       dict(mean=tensor(a['mean']), cov=tensor(a['cov'])), dict(base_items))


def support_cases(meta, task, base):
    """No Q fields enter this interface or either support fit."""
    cases = {}
    for column, ap in enumerate(meta['AP_indices']):
        histories = []
        for floor in meta['histories']:
            info = meta['history_metadata'][str(floor)]
            shift = tuple(np.asarray(info['cache_origin']) - np.asarray(meta['origin']))
            histories.append(load_history(str(Path(info['cache']) / f'ap{ap}.npz'),
                                          shift, tuple(base.items())))
        cases[ap] = dict(histories=histories, dz=task['dz'],
                         sx=tensor(task['S']['positions']), stats=ap_stats(task['S'], column))
    return cases


def grid(meta):
    g = meta['grid']
    axes = [np.linspace(g['lower'][i], g['upper'][i], g['shape_xy'][i]) for i in range(2)]
    return tensor(np.stack(np.meshgrid(*axes), -1).reshape(-1, 2))


def phi(case, points, base):
    return torch.cat((history_features(case, points, base), paired_features(case, points, base)), -1)


def adapt(meta, cases, base, obs, destination):
    path = destination / 'adaptation.json'
    if path.exists():
        return read(path)
    started = time.perf_counter()
    weights, trials = fit_episode(cases, base, obs)
    sigma = choose_intercept(cases, weights, base, obs)
    means, variances, features, observations = [], [], [], []
    with torch.no_grad():
        for case in cases.values():
            prior = InterceptBankPrior(case['histories'], weights, base, sigma['intercept_std'])
            sm, sf = prior.features(case['sx'])
            means.append(sm)
            variances.append(prior.diagonal(case['sx'], sf))
            features.append(phi(case, case['sx'], base))
            stats = case['stats']
            assert bool((stats['n'] == 1).all())
            observations.append(torch.where(stats['k'] == 1, stats['s1'], 100.))
    calibration = fit_support(tuple(torch.stack(v) for v in
                                    (means, variances, features, observations)),
                              obs, torch.zeros(10), [torch.zeros(10)])
    result = dict(weights=weights, weight_trials=trials, sigma=sigma,
                  support_calibration=calibration, query_used=False,
                  seconds=time.perf_counter() - started)
    write(path, result)
    log(event=meta['id'], stage='support_fitted', seconds=result['seconds'])
    return result


@torch.no_grad()
def field(prior, case, points, obs, seed, correction=None):
    sx, stats = case['sx'], case['stats']
    sm, sf = prior.features(sx)
    sk = prior.covariance(sx, sx, sf, sf)
    qm, qf = prior.features(points)
    qv = prior.diagonal(points, qf)
    cross = prior.covariance(points, sx, qf, sf)
    if correction is not None:
        sm = sm + correction(sx)
        qm = qm + correction(points)
    target, proposal, center, root, modes = setup(sm, sk, stats, obs)
    samples, logw, info = sample_reference(target, proposal, center, root, seed, defensive_fraction=.1)
    if info['effective_samples'] < 1000:
        raise RuntimeError(f"Insufficient ESS: {info['effective_samples']}")
    weights = logw.exp()
    pm = weights @ samples
    residual = samples - pm
    posterior = dict(mean=pm, cov=(residual.T * weights) @ residual)
    result = predict_from_posterior(qm, qv, cross, sm, sk, posterior)
    if not all(torch.isfinite(v).all() for v in result.values()) or bool((result['variance'] < 0).any()):
        raise FloatingPointError('Invalid predicted moments')
    return dict(mean=result['mean'].numpy(), variance=result['variance'].numpy(),
                ess=np.array(info['effective_samples']), modes=np.array(len(modes)))


def freeze():
    m = manifest()
    data_check = read(OUT / 'data_check.json')
    assert data_check['events'] == 60 and data_check['nested_S'] and data_check['fixed_Q']
    assert data_check['disjoint_S_Q'] and data_check['complete_H_union_current_S_AP']
    check = OUT / 'solver_check/summary.json'
    numerical = read(check)
    if not numerical['accepted']:
        raise RuntimeError('Numerical check is not accepted')
    modules = ('bounded_pmc', 'public_corpus.k_curve_episodes', 'public_corpus.probe_episodes',
               'public_corpus.adapter', 'data', 'history_bank', 'transfer', 'inference',
               'reference_pmc', 'multi_source_adaptation', 'conditional_operator_learning',
               'residual_probe.prepare', 'residual_probe.paired_features',
               'residual_probe.support_calibration', 'operator_prior', 'full_ap_study', 'run')
    files = [Path(__file__), *[Path(sys.modules[name].__file__) for name in modules]]
    models = [source_model(b)[0] for b in sorted({e['building'] for e in m['events']})]
    target = OUT / 'freeze.json'
    if target.exists():
        raise FileExistsError(target)
    write(target, dict(created=datetime.now(timezone.utc).isoformat(), methods=METHODS,
        code_sha256={str(p): digest(p) for p in files},
        input_sha256={str(p): digest(p) for p in [OUT/'manifest.json', OUT/'data_check.json', check, *models]},
        hypothesis='The existing source-trained history correction must improve localization over both the unchanged historical base and the same features fitted only on current S to establish a useful learning increment.',
        methods_definition=dict(WKNN_S='Raw S scans, k=min(3,number of scans)',
            WKNN_HS='Exact retained raw H plus the same S, same raw WKNN',
            GP_S='Ordinary spatial GP and existing censored observation model; H-derived AP/domain metadata retained; no intercept',
            H_base='Mature shared S-only weights, existing S-selected random intercept; theta=0',
            H_source='Same H_base and posterior path plus existing held-building paired10 source theta',
            H_support='Same H_base and posterior path plus paired10 fitted on S only; zero center and zero initialization; existing .001 ridge'),
        inference='Same bounded initialization for all K and probabilistic methods, original target density and 6x4096/32768 M-PMC, defensive .1, ESS>=1000; no failed AP omission or automatic retries',
        forward='H functions plus current S fit w/sigma/optional theta; full-AP Q RSS likelihood on the common H+current-S domain; posterior mean location.',
        reverse='Q coordinates enter scoring only. Query rows fixed, S nested; K budget counts distinct locations, R1. Source theta fixed before curves, no per-K source retraining or query tuning.',
        comparisons='H_source vs H_base isolates this frozen learned mean correction; H_source vs H_support tests source experience against local fitting of identical features. GP_S vs H_base is an end-to-end method comparison, not a pure causal H ablation.',
        limitations='Four repeatedly examined development buildings; B0-fitted base/obs; existing g trained at K5. Three support draws with 12 fixed Q per draw are not 60 independent environments. Public labels are walking trajectories. AP/domain can expand with permitted S; whole-budget K effect includes coverage. Within K, all methods have identical information. Do not pool different coordinate units.',
        decision='Report every building/K/draw, including failures. No choice of arm, checkpoint, K, feature or source from Q. No new source training in this curve.',
        runtime='CPU2 per building, at most four concurrent building processes; existing local server caches/env, no downloads; small outputs only return to Mac'))


def verify():
    f = read(OUT / 'freeze.json')
    for name in ('code_sha256', 'input_sha256'):
        for p, expected in f[name].items():
            if digest(p) != expected:
                raise ValueError(f'Frozen input changed: {p}')
    return f


def run_event(meta, base, obs):
    destination = OUT / meta['id']
    destination.mkdir(exist_ok=True)
    if (destination / 'completed.json').exists():
        return
    task = episodes.build_task(meta)
    points = grid(meta)
    q = task['Q']
    results = {}
    for kind in METHODS[:2]:
        path = destination / f'{kind}.json'
        if path.exists():
            results[kind] = read(path)
            continue
        references = task['S'] if kind == 'WKNN_S' else group_scans(
            np.concatenate([h['rss'] for h in task['H']] + [task['S']['rss']]),
            np.concatenate([h['xy'] for h in task['H']] + [task['S']['xy']]))
        predictions = wknn(references, dict(rss=q['rss']), np.arange(len(meta['AP_indices'])), False)
        results[kind] = dict(**metrics(predictions, q), predictions=predictions.tolist(),
                             errors=np.linalg.norm(predictions-q['xy'], axis=1).tolist())
        write(path, results[kind])
    cases = support_cases(meta, task, base)
    adaptation = adapt(meta, cases, base, obs, destination)
    coefficients = dict(H_base=torch.zeros(10), H_source=tensor(source_model(meta['building'])[1]['theta']),
                        H_support=tensor(adaptation['support_calibration']['theta']))
    for kind in PROBABILISTIC:
        result_path = destination / f'{kind}.json'
        if result_path.exists():
            results[kind] = read(result_path)
            continue
        start = time.perf_counter()
        directory = destination / kind
        directory.mkdir(exist_ok=True)
        scores = torch.zeros((len(q['rss']), len(points)))
        effective = []
        identical = {}
        for column, (ap, case) in enumerate(cases.items()):
            path = directory / f'ap{ap}.npz'
            if not path.exists():
                seed = int.from_bytes(hashlib.sha256(f'kcurve|{meta["id"]}|{ap}'.encode()).digest()[:4], 'little')
                if kind == 'GP_S':
                    key = b''.join(case['stats'][k].numpy().tobytes() for k in ('n','k','s1','s2'))
                    seed = int.from_bytes(hashlib.sha256(meta['id'].encode()+key).digest()[:4], 'little')
                    if key not in identical:
                        identical[key] = field(BankPrior([], [], base), case, points, obs, seed)
                    values = identical[key]
                else:
                    prior = InterceptBankPrior(case['histories'], adaptation['weights'], base,
                                               adaptation['sigma']['intercept_std'])
                    theta = coefficients[kind]
                    values = field(prior, case, points, obs, seed,
                                   None if kind == 'H_base' else lambda x: 20 * (phi(case, x, base) @ theta))
                save_case(path, values)
            with np.load(path, allow_pickle=False) as a:
                mean, variance = tensor(a['mean']), tensor(a['variance'])
                effective.append(float(a['ess']))
            scores += observation_log_prob(tensor(q['rss'][:, column])[:, None],
                                            mean[None], variance[None], obs)
            if column % 50 == 0:
                log(stage='AP', event=meta['id'], method=kind, done=column+1, total=len(cases))
        if not torch.isfinite(scores).all():
            raise FloatingPointError('Invalid full-scan scores')
        predictions = (scores.softmax(1) @ points).numpy()
        errors = np.linalg.norm(predictions-q['xy'], axis=1)
        results[kind] = dict(**metrics(predictions, q), predictions=predictions.tolist(), errors=errors.tolist(),
                             min_ESS=min(effective), seconds=time.perf_counter()-start)
        save_case(directory / 'predictions.npz', dict(scores=scores.numpy(), predictions=predictions,
                  query_xy=q['xy'], query_scan_ids=q['scan_ids'], errors=errors))
        write(result_path, results[kind])
        log(stage='method_done', event=meta['id'], method=kind, mde=results[kind]['mde'])
    write(destination / 'completed.json', dict(id=meta['id'], building=meta['building'], K=meta['K'],
        seed=meta['seed'], unit=meta['coordinate_unit'], query_scan_ids=meta['query_scan_ids'],
        APs=len(cases), grid_points=len(points), outside_grid=meta['query_outside_grid_positions'],
        outside_history_grid=meta['query_outside_H_grid_positions'],
        results=results, adaptation_seconds=adaptation['seconds']))


def run(building):
    verify()
    m = manifest()
    for e in m['events']:
        if e['building'] != building:
            continue
        try:
            run_event(e, m['base'], m['obs'])
        except Exception as error:
            write(OUT / f'{building}_failure.json', dict(event=e['id'], error=repr(error),
                  time=datetime.now(timezone.utc).isoformat()))
            raise
    write(OUT / f'{building}_completed.json', dict(complete=True, building=building))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical work stays on ray-web')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['freeze', 'run'])
    parser.add_argument('--building')
    args = parser.parse_args()
    freeze() if args.stage == 'freeze' else run(args.building)
