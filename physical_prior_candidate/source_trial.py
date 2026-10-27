"""Six frozen source events: existing BankPrior, random intercept, floor mixture."""
import argparse
from datetime import datetime
from functools import lru_cache
import json
from pathlib import Path
import shutil
import socket
import sys
import time
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'operator_prior_v1'))

import numpy as np
from scipy.spatial import ConvexHull
import torch

import conditional_operator_learning as C
from data import group_scans
from floor_profile import FloorProfile, fit_attenuation, field_from_profile, scan_log_prob
from full_ap_study import metrics
from history_bank import BankPrior
from inference import observation_log_prob
from run import tensor
from run_multi_history import digest, write
from source_weight_mixture import field_from_prior
from transfer import Prior, Transfer

OUT = ROOT / 'results/physical_prior_source_v1_20261026'
HERE = Path(__file__).resolve().parent
GRID_A = list(range(0, 21, 2))
METHODS = ['adapted_bank', 'random_intercept', 'floor_profile']


def now():
    return datetime.now(ZoneInfo('Asia/Singapore')).isoformat()


def log(**row):
    print(json.dumps(dict(time=now(), **row), allow_nan=False), flush=True)


class RandomInterceptPrior(BankPrior):
    def covariance(self, x, y, fx, fy):
        return super().covariance(x, y, fx, fy) + self.base['drift'] ** 2

    def diagonal(self, x, fx):
        return super().diagonal(x, fx) + self.base['drift'] ** 2


def intersection_probes(left, right):
    """Convex polygon clipping followed by a fixed 8x4 uniform lattice."""
    a, b = ConvexHull(left), ConvexHull(right)
    poly = left[a.vertices]
    for normal in b.equations:
        result = []
        for p, q in zip(poly, np.roll(poly, -1, axis=0)):
            dp, dq = p @ normal[:2] + normal[2], q @ normal[:2] + normal[2]
            pin, qin = dp <= 1e-9, dq <= 1e-9
            if pin:
                result.append(p)
            if pin != qin:
                result.append(p + dp / (dp - dq) * (q - p))
        if not result:
            return np.empty((0, 2)), np.empty((0, 2))
        poly = np.array(result)
    if len(poly) < 3:
        return np.empty((0, 2)), poly
    axes = [np.linspace(poly[:, d].min(), poly[:, d].max(), n) for d, n in enumerate([8, 4])]
    points = np.stack(np.meshgrid(*axes), -1).reshape(-1, 2)
    equations = np.concatenate([a.equations, b.equations])
    inside = ((points @ equations[:, :2].T + equations[:, 2]) <= 1e-8).all(1)
    return points[inside], poly


def code_files():
    return [Path(__file__), HERE / 'floor_profile.py', HERE / 'PROTOCOL_source_v1.md',
            *[ROOT / name for name in ['conditional_operator_learning.py', 'multi_source_adaptation.py',
                'history_bank.py', 'source_weight_mixture.py', 'reference_pmc.py', 'inference.py',
                'transfer.py', 'run_operator_prior.py', 'operator_prior.py', 'data.py',
                'uji_probability_study.py', 'full_ap_study.py', 'run.py', 'run_multi_history.py']]]


def prepare():
    OUT.mkdir(exist_ok=False)
    raw = C.raw_floors()
    assert all(name.startswith(('B0', 'B2')) for name in raw)
    events = []
    for target in ['B0F1', 'B2F2']:
        building = int(target[1]); floor = raw[target]
        names = [f'B{building}F0', f'B{building}F3']
        with np.load(C.BANK / names[0] / 'target.npz') as data:
            origin = data['origin']
        with np.load(C.BANK / target / 'target.npz') as data:
            np.testing.assert_array_equal(origin, data['origin'])
        hp = [raw[name]['positions'] - origin for name in names]
        probes, polygon = intersection_probes(*hp)
        bounds = np.concatenate(hp)
        axes = [np.linspace(bounds[:, d].min(), bounds[:, d].max(),
                            int(np.ceil(np.ptp(bounds[:, d]) / 2)) + 1) for d in range(2)]
        grid = np.stack(np.meshgrid(*axes), -1).reshape(-1, 2)
        np.savez_compressed(OUT / f'{target}_geometry.npz', origin=origin, probes=probes,
                            polygon=polygon, grid=grid, source0=hp[0], source1=hp[1])
        for seed in [261026, 261027, 261028]:
            rng = np.random.default_rng(seed)
            chosen = rng.permutation(np.flatnonzero(floor['stats']['n'] >= 3))[:17]
            assert len(chosen) == 17
            rows = [rng.choice(np.flatnonzero(floor['group'] == g), 3, replace=False) for g in chosen]
            sr, qr = np.concatenate(rows[:5]), np.concatenate(rows[5:])
            seen = (floor['rss'][sr] != 100).any(0)
            for name in names:
                seen |= (raw[name]['rss'] != 100).any(0)
            events.append(dict(id=f'{target}_s{seed}', target=target, seed=seed, histories=names,
                candidate_floors=C.TRAIN_FLOORS[building], support_rows=floor['row_ids'][sr].tolist(),
                query_rows=floor['row_ids'][qr].tolist(), APs=np.flatnonzero(seen).tolist(),
                support_positions=(floor['positions'][chosen[:5]] - origin).tolist(),
                source_rows={name: raw[name]['row_ids'].tolist() for name in names},
                probes=len(probes), grid_points=len(grid)))
    write(OUT / 'manifest.json', dict(created=now(), events=events, Q_evaluated=False))
    files = code_files()
    inputs = [C.PARAMETERS, C.DATA / 'trainingData.csv']
    for target in ['B0F1', 'B2F2']:
        inputs += [OUT / f'{target}_geometry.npz', C.BANK / target / 'target.npz']
    for name in ['B0F0', 'B0F3', 'B2F0', 'B2F3']:
        inputs += sorted((C.BANK / name).glob('*.npz'))
    write(OUT / 'prepared.json', dict(created=now(), manifest_sha256=digest(OUT / 'manifest.json'),
        code_sha256={str(p): digest(p) for p in files}, inputs={str(p): digest(p) for p in inputs},
        base=C.parameters()[0], obs=C.parameters()[1], methods=METHODS, attenuation_grid=GRID_A,
        scope='Six B0/B2 source development events, no B1 RSS, no training, no target experiment'))
    (OUT / 'code').mkdir()
    for p in files:
        shutil.copy2(p, OUT / 'code' / p.name)
    log(stage='prepared', events=[dict(id=e['id'], APs=len(e['APs']), probes=e['probes'],
                                     grid=e['grid_points']) for e in events])


def verify():
    frozen = json.loads((OUT / 'prepared.json').read_text())
    assert digest(OUT / 'manifest.json') == frozen['manifest_sha256']
    for section in ['code_sha256', 'inputs']:
        for path, expected in frozen[section].items():
            assert digest(Path(path)) == expected, path


@lru_cache(maxsize=None)
def geometry(target):
    with np.load(OUT / f'{target}_geometry.npz') as data:
        return {k: data[k] for k in data.files}


def profiles_for(meta, cases, weights, base):
    probes = tensor(geometry(meta['target'])['probes'])
    return [FloorProfile(c['histories'], weights, base, [0, 3], int(meta['target'][-1]),
                         meta['candidate_floors'], probes) for c in cases.values()]


def audit():
    verify(); base, obs = C.parameters()
    events = json.loads((OUT / 'manifest.json').read_text())['events']
    rows = []
    for meta in [events[0], events[3]]:
        cases = C.episode(meta)
        assert list(cases) == meta['APs']
        ap = max(cases, key=lambda a: float(cases[a]['stats']['k'].sum()))
        case = cases[ap]; weights = [.4, .6]
        profile = profiles_for(meta, {ap: case}, weights, base)[0]
        sx = case['sx']; bank = profile.bank
        normal = RandomInterceptPrior(profile.histories, profile.weights, base)
        bm, bf = bank.features(sx); rm, rf = normal.features(sx)
        torch.testing.assert_close(bm, rm, atol=0, rtol=0)
        difference = normal.covariance(sx, sx, rf, rf) - bank.covariance(sx, sx, bf, bf)
        torch.testing.assert_close(difference, torch.full_like(difference, base['drift'] ** 2))
        assert torch.linalg.eigvalsh(normal.covariance(sx, sx, rf, rf)).min() > 0
        zero = field_from_profile(profile, 0., obs, sx, case['stats'], sx, meta['seed'] + ap)
        direct = field_from_prior(bank, obs, sx, case['stats'], sx, meta['seed'] + ap, defensive_fraction=.1)
        torch.testing.assert_close(zero['mean'], direct['mean'], atol=0, rtol=0)
        torch.testing.assert_close(zero['variance'], direct['variance'], atol=0, rtol=0)
        mixed = field_from_profile(profile, 6., obs, sx, case['stats'], sx, meta['seed'] + ap)
        ess = [mixed['effective_samples']] if np.isscalar(mixed['effective_samples']) else mixed['effective_samples']
        assert min(ess) >= 1000
        # A pure S-only audit: chosen point/scan label does not read Q.
        probe_rss = torch.tensor([[-65.], [100.]])
        mix = scan_log_prob(probe_rss, mixed, obs).exp()
        direct_mix = sum(float(w) * observation_log_prob(probe_rss, m[None], v[None], obs).exp()
                         for w, m, v in zip(mixed['log_mass'].exp(), mixed['component_means'], mixed['component_variances']))
        torch.testing.assert_close(mix, direct_mix)
        rows.append(dict(event=meta['id'], AP=ap, random_intercept_covariance=True,
                         zero_attenuation_exact=True, mixture_probability=True, min_ESS=min(ess)))
    write(OUT / 'audit.json', dict(completed=now(), Q_evaluated=False, source_checks=rows,
        reverse_trace='Q RSS/true positions never enter H probes, AP union, w, or a. w is fitted once from all AP S. Each floor component sees S once through existing evidence; scan likelihood mixes components before AP sum. Localization bounds use H alone.',
        no_source_training=True, original_synthetic_checks=str(HERE / 'synthetic_results.json')))
    log(stage='audited', checks=rows)


def freeze():
    verify()
    data = json.loads((OUT / 'audit.json').read_text()); assert not data['Q_evaluated']
    write(OUT / 'freeze.json', dict(frozen=now(), prepared_sha256=digest(OUT / 'prepared.json'),
        manifest_sha256=digest(OUT / 'manifest.json'), audit_sha256=digest(OUT / 'audit.json'),
        protocol_sha256=digest(HERE / 'PROTOCOL_source_v1.md'), Q_evaluated=False,
        decision='Proceed with exactly the six authorized source events and frozen three-arm comparison'))
    log(stage='frozen')


def flat_ess(value):
    return [float(value)] if np.isscalar(value) else [float(v) for v in value]


@torch.no_grad()
def event(meta):
    dest = OUT / meta['id']; dest.mkdir(exist_ok=True)
    if (dest / 'completed.json').exists():
        return
    started = time.perf_counter(); base, obs = C.parameters(); cases = C.episode(meta)
    assert list(cases) == meta['APs']
    adapted = dest / 'adaptation.json'
    if not adapted.exists():
        # Existing scipy adapter needs autograd even though prediction is no-grad.
        with torch.enable_grad():
            weights, trials = C.fit_episode(cases, base, obs)
        profiles = profiles_for(meta, cases, weights, base)
        fitted = fit_attenuation(profiles, next(iter(cases.values()))['sx'],
                                [c['stats'] for c in cases.values()], obs, GRID_A, 0.)
        write(adapted, dict(completed=now(), weights=weights, trials=trials, attenuation=fitted,
                           boundary_hit=fitted['attenuation'] in [0, 20], APs=list(cases), Q_used=False))
    fit = json.loads(adapted.read_text()); weights = fit['weights']; attenuation = fit['attenuation']['attenuation']
    profiles = profiles_for(meta, cases, weights, base)
    first = next(iter(cases.values())); qx, sx = first['qx'], first['sx']; grid = tensor(geometry(meta['target'])['grid'])
    points = torch.cat([qx, grid]); qp = len(qx); count = len(first['rss'])
    scores = {m: torch.zeros((count, len(grid))) for m in METHODS}
    scan_lp = {m: torch.zeros(count) for m in METHODS}
    query_mean = {m: [] for m in METHODS}; query_variance = {m: [] for m in METHODS}
    errors = {m: [] for m in METHODS}; ess = {m: [] for m in METHODS}
    null_mass, target_mass, availability, source_uncertainty = [], [], [], []
    for index, ((ap, c), profile) in enumerate(zip(cases.items(), profiles)):
        bank = profile.bank
        priors = dict(adapted_bank=bank, random_intercept=RandomInterceptPrior(profile.histories, profile.weights, base))
        fields = {m: field_from_prior(p, obs, sx, c['stats'], points, meta['seed'] + ap, defensive_fraction=.1)
                  for m, p in priors.items()}
        fields['floor_profile'] = field_from_profile(profile, attenuation, obs, sx, c['stats'], points, meta['seed'] + ap)
        for method, value in fields.items():
            if method == 'floor_profile':
                lp = scan_log_prob(c['rss'][:, None], value, obs)
            else:
                lp = observation_log_prob(c['rss'][:, None], value['mean'][None], value['variance'][None], obs)
            scan_lp[method] += lp[torch.arange(count), c['group']]
            scores[method] += lp[:, qp:]
            query_mean[method].append(value['mean'][:qp].numpy())
            query_variance[method].append(value['variance'][:qp].numpy())
            detected = c['rss'] != 100
            errors[method].extend((value['mean'][c['group']][detected] - c['rss'][detected]).tolist())
            ess[method].extend(flat_ess(value['effective_samples']))
            if min(ess[method][-len(flat_ess(value['effective_samples'])):]) < 1000:
                raise RuntimeError(f'Low ESS {meta["id"]} AP{ap} {method}')
        value = fields['floor_profile']
        mass = profile.history_log_mass(attenuation).exp() if value['labels'] == ['identity'] else value['log_mass'].exp()
        null_mass.append(float(mass[-1])); target_mass.append(float(mass[meta['candidate_floors'].index(int(meta['target'][-1]))]))
        availability.append(sum(h is not None for h in c['histories']))
        for h in c['histories']:
            if h is not None and meta['probes']:
                p = Prior(Transfer('identity'), h, dict(base, drift=0.))
                px = tensor(geometry(meta['target'])['probes']); _, pf = p.features(px)
                source_uncertainty.append(float(p.diagonal(px, pf).mean().sqrt()))
        if index % 20 == 0:
            log(stage='APs', event=meta['id'], complete=index + 1, total=len(cases), attenuation=attenuation)
    qxy = qx[first['group']].numpy(); qw = first['query_weights'].numpy()
    results, arrays = [], dict(query_xy=qxy, query_rows=np.array(meta['query_rows']), APs=np.array(meta['APs']),
                              null_mass=np.array(null_mass), target_floor_mass=np.array(target_mass), availability=np.array(availability))
    for method in METHODS:
        prediction = (scores[method].softmax(1) @ grid).numpy()
        error = np.array(errors[method])
        results.append(dict(method=method, scan_NLL=float(-(first['query_weights'] * scan_lp[method]).sum()),
            NLL_per_AP=float(-(first['query_weights'] * scan_lp[method]).sum()) / len(cases),
            detected_RSS_RMSE=float(np.sqrt(np.mean(error ** 2))), detected_RSS_bias=float(error.mean()),
            min_ESS=min(ess[method]), **metrics(prediction, dict(xy=qxy, scan_weights=qw))))
        arrays[method + '_prediction'] = prediction
        arrays[method + '_query_mean'] = np.stack(query_mean[method])
        arrays[method + '_query_variance'] = np.stack(query_variance[method])
        arrays[method + '_scan_log_prob'] = scan_lp[method].numpy()
    np.savez_compressed(dest / 'predictions.npz', **arrays)
    coverage = dict(APs=len(cases), by_available_histories={str(k): availability.count(k) for k in range(3)},
                    H_probes=meta['probes'], probe_history_posterior_sd_mean=float(np.mean(source_uncertainty)),
                    null_mean=float(np.mean(null_mass)), null_median=float(np.median(null_mass)),
                    target_floor_mass_mean=float(np.mean(target_mass)))
    hxy = np.concatenate([geometry(meta['target'])['source0'], geometry(meta['target'])['source1']])
    coverage['Q_nearest_H_mean_distance'] = float(np.linalg.norm(qxy[:, None] - hxy[None], axis=-1).min(1).mean())
    coverage['Q_outside_H_box_fraction'] = float(((qxy < hxy.min(0)) | (qxy > hxy.max(0))).any(1).mean())
    write(dest / 'completed.json', dict(event=meta['id'], target=meta['target'], seed=meta['seed'],
        completed=now(), seconds=time.perf_counter() - started, results=results, weights=weights,
        attenuation=fit['attenuation'], boundary_hit=fit['boundary_hit'], coverage=coverage,
        source_only=True, fields_used_S=True, localization_region='H-only 2m grid posterior mean'))
    log(stage='completed', event=meta['id'], seconds=time.perf_counter() - started, results=results, coverage=coverage)


def run():
    verify(); frozen = json.loads((OUT / 'freeze.json').read_text())
    assert digest(OUT / 'prepared.json') == frozen['prepared_sha256']
    for meta in json.loads((OUT / 'manifest.json').read_text())['events']:
        verify(); event(meta)


def summarize():
    events = json.loads((OUT / 'manifest.json').read_text())['events']
    rows = [json.loads((OUT / e['id'] / 'completed.json').read_text()) for e in events]
    summaries = {}
    for target in ['B0F1', 'B2F2', 'all']:
        selected = [r for r in rows if target == 'all' or r['target'] == target]
        values = {m: {k: float(np.mean([next(v for v in r['results'] if v['method'] == m)[k] for r in selected]))
                      for k in ['scan_NLL', 'NLL_per_AP', 'detected_RSS_RMSE', 'detected_RSS_bias', 'mde', 'median', 'fraction_error_gt10']}
                  for m in METHODS}
        contrasts = {}
        for control in METHODS[:2]:
            deltas = {key: [next(v for v in r['results'] if v['method'] == 'floor_profile')[key] -
                            next(v for v in r['results'] if v['method'] == control)[key] for r in selected]
                      for key in ['scan_NLL', 'mde']}
            contrasts[control] = {key: dict(mean_change=float(np.mean(d)), improved_events=sum(v < 0 for v in d)) for key, d in deltas.items()}
        summaries[target] = dict(events=len(selected), means=values, floor_versus=contrasts)
    write(OUT / 'summary.json', dict(completed=now(), summary=summaries, events=rows,
        scope='Six source-development events; three seeds are not independent buildings; no target B1 or cross-dataset claim'))
    log(stage='summary', summary=summaries)


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical execution is restricted to ray-web')
    torch.set_default_dtype(torch.float64); torch.set_num_threads(2)
    parser = argparse.ArgumentParser(); parser.add_argument('stage', choices=['prepare', 'audit', 'freeze', 'run', 'summarize'])
    args = parser.parse_args(); globals()[args.stage]()
