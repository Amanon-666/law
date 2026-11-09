"""24 real source episodes for a later residual-learning probe.

Freeze source choices, then fit g-free S-only weights. No Q score, neural
training, new H fits, confirmation corpora or large proposal/field caches.
"""
import argparse
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from conditional_operator_learning import fit_episode, prior_for
from run import ap_stats, tensor
from transfer import History
from uji_probability_study import load_raw, DATA
from public_corpus.adapter import load_building, source_task
from public_corpus.history_cache import write


ROOT = Path('/home/panyushuo/projects/panyushuo/LAW-source-resume-20261025-01a164d0/Law-Probabilistic-Transfer-20261023')
PREPARATION = ROOT / 'results/source_residual_probe_v1_20261026/preparation'
UJI_BANK = ROOT / 'results/uji_operator_prior_v1_20261026/train_histories'
PUBLIC_DATA = ROOT / 'results/public_corpus_adapter_v1_20261026/data'
PUBLIC_BANK = ROOT / 'results/public_corpus_history_v1_20261026'
PARAMETERS = ROOT / 'results/uji_history_bank_v1_20261026/parameters.json'
CONFIGS = {'UJI_B0': (1, [0, 2, 3]), 'UJI_B2': (2, [0, 1, 3, 4]),
           'UAH2016-b20': (1, [0, 2, 3]), 'Pisa2019-b60': (1, [0, 2])}
SEEDS = (261026, 261027)


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


@lru_cache(maxsize=4)
def building_data(name):
    if not name.startswith('UJI_'):
        return load_building(PUBLIC_DATA / name)
    b = int(name[-1])
    t, hs = CONFIGS[name]
    raw = load_raw('trainingData.csv', {f'B{b}F{f}' for f in [t, *hs]})
    floors = {}
    for label, values in raw.items():
        floors[int(label.split('F')[1])] = dict(xy=values['xy'], rss=values['rss'],
            raw_latlon=values['xy'], scan_ids=np.array([f'trainingData.csv@row{r}' for r in values['row_ids']]),
            original_row_ids=values['row_ids'])
    return dict(metadata=dict(site=name, name=name, ap_namespace=name,
                              position_kind='original_UJI_reference_point',
                              coordinate_unit='UJI original coordinate distance; no 0.766 conversion'),
                floors=floors, ap_ids=np.array([f'WAP{i:03}' for i in range(1, 521)]))


def h_selection(name, floor):
    if name.startswith('UJI_'):
        raw = building_data(name)['floors'][floor]
        path = UJI_BANK / f'B{name[-1]}F{floor}'
        with np.load(path / 'target.npz') as a:
            origin = a['origin'].tolist()
        return dict(cache=str(path), cache_origin=origin, rows=np.arange(len(raw['rss'])).tolist(),
                    scans=len(raw['rss']), positions=len(np.unique(raw['xy'], axis=0)),
                    scan_ids=raw['scan_ids'].tolist(), raw_scope='all original floor scans')
    path = PUBLIC_BANK / name / f'F{floor}'
    s = json.loads((path / 'selection.json').read_text())
    return dict(cache=str(path), cache_origin=[0., 0.], rows=s['row_indices'],
                scans=s['used_scans'], positions=s['used_positions'], scan_ids=s['scan_ids'],
                available_scans=s['available_scans'], available_positions=s['available_positions'],
                raw_scope='existing coordinate-FPS cache; max 128 positions, one real scan/position')


def task_for(meta):
    name = meta['building']
    rows = {f: np.array(h_selection(name, f)['rows']) for f in meta['histories']}
    return source_task(building_data(name), meta['target'], meta['histories'],
                       5, 1, 12, meta['seed'], history_rows=rows)


def grid_definition(task):
    hxy = [h['positions'] for h in task['H']]
    xy = np.concatenate([*hxy, task['S']['positions']])
    low = np.floor(xy.min(0) / 2) * 2
    high = np.ceil(xy.max(0) / 2) * 2
    shape = np.rint((high - low) / 2).astype(int) + 1
    return dict(lower=low.tolist(), upper=high.tolist(), spacing=2.,
                shape_xy=shape.tolist(), points=int(np.prod(shape)),
                flatten_order='meshgrid(indexing=xy), C-order; y varies fastest',
                boundary_source='actual retained H positions union S only; snapped outward to spacing 2',
                query_involved=False, coordinate_frame='raw public frame minus episode origin')


def freeze(output=PREPARATION):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    params = json.loads(PARAMETERS.read_text())
    old_public = json.loads((PUBLIC_BANK / 'freeze.json').read_text())
    if params['base'] != old_public['base'] or params['obs'] != old_public['obs']:
        raise ValueError('Public H cache and fixed probe parameters differ')
    inputs = {DATA / 'trainingData.csv', PARAMETERS, PUBLIC_BANK / 'freeze.json'}
    events = []
    for name, (target, available) in CONFIGS.items():
        b = building_data(name)
        for f in [target, *available]:
            if name.startswith('UJI_'):
                inputs.add(UJI_BANK / f'B{name[-1]}F{f}' / 'target.npz')
            else:
                inputs.add(PUBLIC_DATA / name / f'F{f}.npz')
                inputs.add(PUBLIC_DATA / name / 'metadata.json')
        for f in available:
            select = h_selection(name, f)
            path = Path(select['cache'])
            inputs.update(path.glob('ap*.npz'))
            if not name.startswith('UJI_'):
                inputs.add(path / 'selection.json')
        for seed in SEEDS:
            # Independent stream from target position/row sampling. No RSS used.
            order = np.random.default_rng(seed).permutation(available).tolist()
            fixed_rows = None
            for J in range(1, len(order) + 1):
                meta = dict(id=f'{name}_F{target}_s{seed}_J{J}', building=name,
                            target=target, seed=seed, histories=order[:J], history_order=order)
                task = task_for(meta)
                S, Q = task['S'], task['Q']
                ids = (S['scan_ids'].tolist(), Q['scan_ids'].tolist())
                if fixed_rows is not None and fixed_rows != ids:
                    raise AssertionError('S/Q changed with J')
                fixed_rows = ids
                assert not (S['positions'][:, None] == Q['positions'][None]).all(-1).any()
                meta.update(K=5, R=1, Q_positions=12, support_scan_ids=ids[0], query_scan_ids=ids[1],
                            support_rows=S['row_indices'].tolist(), query_rows=Q['row_indices'].tolist(),
                            AP_indices=task['ap_indices'].tolist(), AP_ids=task['ap_ids'].tolist(),
                            origin=task['metadata']['origin'], grid=grid_definition(task),
                            history_metadata={str(f): h_selection(name, f) for f in order[:J]},
                            coordinate_unit='UJI original coordinate distance' if name.startswith('UJI_') else 'metres',
                            position_kind=b['metadata']['position_kind'],
                            episode_training_weight=1 / (4 * 2 * len(available)),
                            AP_training_weight=1 / len(task['ap_indices']),
                            position_training_weight=1 / 12,
                            weighting='sum over episodes of 1/(4*events_per_building) * mean_AP * mean_Q_position',
                            learning_identity_policy='AP/source/position IDs are metadata only; no identity embeddings',
                            query_role='source supervision only; not weight/AP/grid selection or scoring',
                            source_scope='development sources; B0 already used for fixed base/obs')
                events.append(meta)
    assert len(events) == 24
    assert abs(sum(e['episode_training_weight'] for e in events) - 1) < 1e-12
    manifest = dict(events=events, base=params['base'], obs=params['obs'], parameters=str(PARAMETERS),
                    parameter_scope='B0 source-fitted base/obs; holding out B0 later holds out new g only',
                    future_building_split='all 4 are source development; not blind confirmation',
                    readout='existing cases and g-free Bank; no VP dependency')
    code = [Path(__file__), ROOT / 'data.py', ROOT / 'transfer.py', ROOT / 'inference.py',
            ROOT / 'conditional_operator_learning.py', ROOT / 'multi_source_adaptation.py',
            ROOT / 'history_bank.py', ROOT / 'uji_probability_study.py',
            Path(source_task.__code__.co_filename)]
    record = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
                  scope=__doc__, code_sha256={str(p): digest(p) for p in code},
                  inputs_sha256={str(p): digest(p) for p in sorted(inputs)},
                  manifest_sha256=hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest(),
                  candidates='2 original units; actual H union S rectangle; Q cannot expand bounds',
                  stages='freeze before all 24 S-only weight fits; no Q scoring/training')
    if (output / 'freeze.json').exists():
        raise FileExistsError('Preparation already frozen; use fit/check to resume')
    write(output / 'manifest.json', manifest)
    write(output / 'freeze.json', record)
    return manifest


def verify_freeze(output):
    output = Path(output)
    f = json.loads((output / 'freeze.json').read_text())
    for key in ('code_sha256', 'inputs_sha256'):
        for path, expected in f[key].items():
            if digest(Path(path)) != expected:
                raise ValueError(f'Frozen input/code changed: {path}')
    manifest = json.loads((output / 'manifest.json').read_text())
    if hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest() != f['manifest_sha256']:
        raise ValueError('Frozen episode manifest changed')
    return manifest


def load_probe_episode(id, preparation=PREPARATION):
    """Read one frozen source scenario, lazily reconstructing H from old NPZs.

    Returns existing conditional_operator cases and metadata, never a pickle.
    Caller may obtain fitted weights from meta['adaptation'] after preparation.
    All returned Q labels/signals are supervised source data, not adaptation.
    """
    preparation = Path(preparation)
    manifest = json.loads((preparation / 'manifest.json').read_text())
    meta = dict(next(e for e in manifest['events'] if e['id'] == id))
    task = task_for(meta)
    S, Q = task['S'], task['Q']
    assert S['scan_ids'].tolist() == meta['support_scan_ids']
    assert Q['scan_ids'].tolist() == meta['query_scan_ids']
    assert task['ap_indices'].tolist() == meta['AP_indices']
    assert grid_definition(task) == meta['grid']
    base = manifest['base']
    cases = {}
    shift = tensor(meta['origin'])
    for local, ap in enumerate(meta['AP_indices']):
        histories = []
        for floor in meta['histories']:
            info = meta['history_metadata'][str(floor)]
            with np.load(Path(info['cache']) / f'ap{ap}.npz', allow_pickle=False) as a:
                h = None if bool(a['empty']) else History(
                    tensor(a['z']) + tensor(info['cache_origin']) - shift,
                    dict(mean=tensor(a['mean']), cov=tensor(a['cov'])), base)
            histories.append(h)
        cases[ap] = dict(histories=histories, sx=tensor(S['positions']), qx=tensor(Q['positions']),
                         stats=ap_stats(S, local), dz=task['dz'], rss=tensor(Q['rss'][:, local]),
                         group=torch.tensor(Q['group']), query_weights=tensor(Q['scan_weights']))
    meta.update(base=base, obs=manifest['obs'], parameters=manifest['parameters'],
                parameter_scope=manifest['parameter_scope'])
    apath = preparation / id / 'adaptation.json'
    if apath.exists():
        meta['adaptation'] = json.loads(apath.read_text())
    return cases, meta


def fit_all(output=PREPARATION):
    output = Path(output)
    manifest = verify_freeze(output)
    started = time.perf_counter()
    summaries = []
    for event in manifest['events']:
        dest = output / event['id']
        dest.mkdir(exist_ok=True)
        if (dest / 'adaptation.json').exists():
            summaries.append(json.loads((dest / 'adaptation.json').read_text()))
            continue
        t0 = time.perf_counter()
        cases, meta = load_probe_episode(event['id'], output)
        # Pass no Q arrays to weight estimation; fits cannot see its RSS/region.
        support_cases = {ap: {k: v for k, v in c.items() if k in ('histories', 'sx', 'stats', 'dz')}
                         for ap, c in cases.items()}
        weights, trials = fit_episode(support_cases, manifest['base'], manifest['obs'])
        # Interface/geometry checks use H and S only, with the existing prior.
        assert all(len(c['sx']) == 5 and len(c['qx']) == 12 for c in cases.values())
        assert all(torch.equal(c['stats']['n'], torch.ones(5)) for c in cases.values())
        for ap, c in cases.items():
            prior = prior_for(None, c, weights, manifest['base'])
            m, features = prior.features(c['sx'])
            cov = prior.covariance(c['sx'], c['sx'], features, features)
            assert torch.isfinite(m).all() and torch.isfinite(cov).all()
            np.testing.assert_allclose(np.bincount(c['group'].numpy(), weights=c['query_weights'].numpy()), 1 / 12)
        adaptation = dict(id=event['id'], building=event['building'], J=len(weights),
                          APs=len(cases), weights=weights, trials=trials,
                          seconds=time.perf_counter() - t0,
                          fit_inputs=['histories', 'sx', 'stats', 'dz'],
                          query_used_for_weights=False, query_scored=False, shared_updates=0,
                          grid_points=event['grid']['points'], interface_checked=True)
        write(dest / 'adaptation.json', adaptation)
        summaries.append(adaptation)
        print(json.dumps({k: adaptation[k] for k in ('id', 'APs', 'weights', 'seconds', 'grid_points')}), flush=True)
    result = dict(complete=len(summaries) == 24, events=len(summaries),
                  event_counts={b: sum(e['building'] == b for e in summaries) for b in CONFIGS},
                  seconds=time.perf_counter() - started, fits_seconds=sum(e['seconds'] for e in summaries),
                  AP_range=[min(e['APs'] for e in summaries), max(e['APs'] for e in summaries)],
                  query_scored=False, shared_updates=0, history_refits=0,
                  weighting='each building 1/4; its episodes equal; each AP and Q position equal',
                  isolation_checks='S/Q IDs common across J; no duplicate S/Q positions; H union S AP/grid; no Q keys passed to fit',
                  events_summary=summaries)
    write(output / 'summary.json', result)
    verify_freeze(output)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['freeze', 'fit'])
    p.add_argument('--output', type=Path, default=PREPARATION)
    a = p.parse_args()
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical source data work stays on ray-web')
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    torch.manual_seed(261026)
    result = freeze(a.output) if a.stage == 'freeze' else fit_all(a.output)
    print(json.dumps(dict(stage=a.stage, events=len(result['events']) if a.stage == 'freeze' else result['events'])), flush=True)
