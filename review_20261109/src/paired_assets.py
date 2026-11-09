"""Fresh ReFi inputs for the protected same-position acquisition protocol.

Reuses the original UJI observation/transfer calibration and finite 32-update
base fit. No old fitted parameters, history cache, or learned checkpoint is
loaded. Protected RSS is never parsed; validation queries remain row IDs.
"""
import csv
import importlib.util
import json
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import history_fit
from episodic_tasks import task_inputs
from fields import History, rbf
from paired_acquisition_tasks import build_pairs, make_quartet, verify_quartet
from recomposition_trial import ROOT, CSV, digest, write, now, subset
from source_episode_catalog import read_metadata, verify_catalog
from uji import ap_stats, tensor

CATALOG = ROOT/'results/source_episode_catalog_20261105'
OUT = ROOT/'results/paired_assets_20261106'
LEGACY = ROOT.parent/'FeMLoc-Signal-Calibrated-Map/models'


def read_allowed(path, allowed):
    """Filter CSV row IDs before interpreting any RSS column."""
    floors = {}
    with path.open(newline='') as stream:
        for row_id, row in enumerate(csv.DictReader(stream), 1):
            if row_id not in allowed:
                continue
            name = f"B{row['BUILDINGID']}F{row['FLOOR']}"
            floors.setdefault(name, []).append((row_id,
                [int(row[f'WAP{i:03}']) for i in range(1, 521)],
                [float(row['LONGITUDE']), float(row['LATITUDE'])]))
    assert {r[0] for rows in floors.values() for r in rows} == set(allowed)
    return {name: dict(row_ids=np.array([r[0] for r in rows]),
        rss=np.array([r[1] for r in rows], dtype=np.int16),
        xy=np.array([r[2] for r in rows])) for name, rows in floors.items()}


def legacy_module(name):
    spec = importlib.util.spec_from_file_location('paired_'+name, LEGACY/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def finite_parameters(base, obs):
    assert np.isfinite(list(base.values())+list(obs.values())).all()
    assert all(base[k] > 0 for k in ('amplitude', 'length', 'drift'))
    assert 0 < obs['alpha'] < 1 and obs['noise_std'] > 0 and obs['detect_scale'] > 0


def run():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    OUT.mkdir(exist_ok=False)
    catalog = json.loads((CATALOG/'catalog.json').read_text())
    parent = json.loads((CATALOG/'freeze.json').read_text())
    assert parent['inputs'][str(CSV)] == digest(CSV)
    metadata = read_metadata(CSV, set(catalog['source_floors']))
    verify_catalog(metadata, catalog)
    protected = set(catalog['protected_rows'])
    allowed = set(metadata)-protected
    pair = build_pairs(catalog)[0]  # Lexical selection, never a score selection.
    events = {phase: make_quartet(catalog, pair, 0, phase=phase)
              for phase in ('fit', 'validation')}
    for phase, quartet in events.items():
        verify_quartet(metadata, catalog, quartet, phase)
    names = events['fit'][0]['histories']
    history_rows = events['fit'][0]['history_rows']
    assert all(set(ids) <= allowed for ids in history_rows.values())
    # The common coordinate origin is historical geometry only.
    origin = np.mean([metadata[i]['xy'] for ids in history_rows.values() for i in ids], axis=0)
    source_files = list(ROOT.glob('*.py'))+[LEGACY/'radio_map.py', LEGACY/'cross_floor.py']
    freeze = dict(created=now(), pair=pair, events=events, origin=origin.tolist(),
        inputs={str(p): digest(p) for p in (CSV, CATALOG/'catalog.json', CATALOG/'freeze.json')},
        code={str(p): digest(p) for p in source_files}, allowed_rows=sorted(allowed),
        protected_rows=sorted(protected), source_floors=catalog['source_floors'],
        method='Original UJI source observation/transfer fit; source base Adam 32 updates, 12 AP cases; fresh full-Gaussian histories',
        initialization=dict(mean=-95., amplitude=20., length=20.),
        base_seed=251025, base_updates=32, learning_rate=.02,
        inner_maxiter=640, inner_max_projected_gradient=1e-3,
        scope='One paired floor; 2 H/S inputs shared by 8 fit/validation event descriptions. No query scores or learning-network updates.',
        limitations=['Finite base calibration, not a convergence claim.',
            'Source training queries may enter shared source calibration; protected source positions do not.',
            'Source-position validation is not unseen-floor/device generalization.',
            'Original pooled noise includes acquisition variation; this is not a new device-noise estimator.',
            'Same-position acquisitions confound user, phone and date; no causal device interpretation.'],
        resources=dict(cpu=2, gpu=0, seconds=600, downloads=0))
    write(OUT/'freeze.json', freeze)
    (OUT/'code').mkdir()
    for p in source_files:
        label = 'legacy_'+p.name if p.parent == LEGACY else p.name
        (OUT/'code'/label).write_bytes(p.read_bytes())
    raw = read_allowed(CSV, allowed)
    assert set(raw) == set(catalog['source_floors'])
    floors = {f: subset(r, r['row_ids'], origin) for f, r in raw.items()}
    cells = {f: dict(xy=v['positions'], **v['stats']) for f, v in floors.items()}
    context = {'stage': 'observation'}
    solver_rows = []

    def monitored(original, *args, **kwargs):
        result = original(*args, **kwargs)
        grad = np.asarray(result.jac).copy()
        for j, (low, high) in enumerate(kwargs.get('bounds') or [(None, None)]*len(grad)):
            if low is not None and result.x[j] <= low+1e-10 and grad[j] > 0:
                grad[j] = 0
            if high is not None and result.x[j] >= high-1e-10 and grad[j] < 0:
                grad[j] = 0
        row = dict(context, success=bool(result.success), iterations=int(result.nit),
                   projected_gradient=float(np.abs(grad).max()), message=str(result.message))
        solver_rows.append(row)
        with (OUT/'solver.jsonl').open('a') as stream:
            stream.write(json.dumps(row)+'\n')
        if not (np.isfinite(result.fun) and np.isfinite(result.x).all() and np.isfinite(grad).all()):
            raise RuntimeError(f'Nonfinite solve: {row}')
        if context['stage'] == 'observation':
            if not result.success:
                raise RuntimeError(f'Unfinished original transfer calibration: {row}')
        elif row['projected_gradient'] > freeze['inner_max_projected_gradient']:
            raise RuntimeError(f'Unfinished VI solve: {row}')
        return result

    radio, cross = legacy_module('radio_map'), legacy_module('cross_floor')
    obs0 = radio.fit_observation(list(cells.values()))
    original = cross.minimize
    with patch.object(cross, 'minimize', lambda *a, **kw: monitored(original, *a, **kw)):
        law = cross.fit_transfer([(cells['B0F0'], cells['B0F1']),
                                  (cells['B0F1'], cells['B0F2'])], obs0)
    write(OUT/'observation_fit.json', {k: v.tolist() if isinstance(v, np.ndarray) else v for k, v in law.items()})
    obs = dict(alpha=law['h'], beta=law['theta'], detect_scale=law['s'], noise_std=law['sigma_total'])
    rng = np.random.default_rng(freeze['base_seed'])
    cases, case_meta = [], []
    for name in catalog['source_floors']:
        f = floors[name]
        eligible = np.flatnonzero((f['stats']['k'] > 0).any(0))
        for ap in rng.choice(eligible, 4, replace=False):
            cases.append((tensor(f['positions']), ap_stats(f, int(ap))))
            case_meta.append(dict(floor=name, ap=int(ap), positions=len(f['positions'])))
    write(OUT/'base_cases.json', case_meta)
    shift = torch.nn.Parameter(torch.tensor(0.))
    logs = torch.nn.Parameter(torch.tensor(np.log([20., 20.])))
    optimizer = torch.optim.Adam([shift, logs], lr=freeze['learning_rate'])
    original_vi = history_fit.minimize
    with patch.object(history_fit, 'minimize', lambda *a, **kw: monitored(original_vi, *a, **kw)):
        for step in range(freeze['base_updates']):
            ci = step % len(cases)
            context.clear(); context.update(stage='base', step=step, **case_meta[ci])
            x, stats = cases[ci]
            mean = -95+20*shift
            amplitude, length = logs.exp().unbind()
            optimizer.zero_grad()
            posterior = history_fit.infer(mean.expand(len(x)), rbf(x, x, amplitude, length),
                stats, obs, steps=freeze['inner_maxiter'], hyperparameter_fit=True)
            loss = posterior['loss']/stats['n'].sum()
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in (shift, logs))
            optimizer.step()
            with (OUT/'base_training.jsonl').open('a') as stream:
                stream.write(json.dumps(dict(step=step, case=ci, objective=float(loss.detach()),
                    mean=float((-95+20*shift).detach()), scales=logs.detach().exp().tolist()))+'\n')
        base = dict(mean=float((-95+20*shift).detach()), amplitude=float(logs[0].detach().exp()),
            length=float(logs[1].detach().exp()), drift=float(np.sqrt(np.median(law['sb2']+law['s_w2']))))
        finite_parameters(base, obs)
        write(OUT/'parameters.json', dict(base=base, obs=obs, updates=32,
            freeze_sha256=digest(OUT/'freeze.json'), converged_claim=False))
        print(json.dumps(dict(stage='calibrated', elapsed=time.monotonic()-started)), flush=True)
        histories = {h: subset(raw[h], history_rows[h], origin) for h in names}
        records = []
        for h, scans in histories.items():
            folder = OUT/'histories'/h
            folder.mkdir(parents=True)
            for ap in np.flatnonzero((scans['stats']['k'] > 0).any(0)):
                ap = int(ap)
                context.clear(); context.update(stage='history', floor=h, ap=ap)
                field = history_fit.single_history(dict(scans=scans), ap, base, obs,
                                                   steps=freeze['inner_maxiter'])
                path = folder/f'ap{ap}.npz'
                np.savez_compressed(path, z=field.z.numpy(), mean=field.m.numpy(),
                                    cov=field.c.numpy(), origin=origin)
                records.append(dict(floor=h, ap=ap, path=str(path.relative_to(OUT)), sha256=digest(path)))
            print(json.dumps(dict(stage='history', floor=h, fields=sum(r['floor']==h for r in records),
                                  elapsed=time.monotonic()-started)), flush=True)
    write(OUT/'histories.json', dict(parameters_sha256=digest(OUT/'parameters.json'),
                                    history_rows=history_rows, fields=records))
    tasks = []
    for side, event in zip(('A', 'B'), (events['fit'][0], events['fit'][2])):
        support = subset(raw[event['floor']], event['support_rows'], origin)
        # Reload saved histories: prove tasks use persisted assets, not hidden fit state.
        loaded = {}
        for record in records:
            path = OUT/record['path']
            assert digest(path) == record['sha256']
            with np.load(path) as arrays:
                loaded[record['floor'], record['ap']] = History(tensor(arrays['z']),
                    dict(mean=tensor(arrays['mean']), cov=tensor(arrays['cov'])), base)
        task = task_inputs(histories, support, base, obs, lambda h, ap: loaded[h, ap])
        path = OUT/f'support_{side}.pt'
        torch.save(task, path)
        tasks.append(dict(side=side, path=path.name, sha256=digest(path),
            support_rows=event['support_rows'], APs=list(task['cases']), grid_points=len(task['grid'])))
    # No protected RSS enters this function, including task geometry/AP selection.
    assert all(set(t['support_rows']) <= allowed for t in tasks)
    write(OUT/'complete.json', dict(complete=True, created=now(), elapsed=time.monotonic()-started,
        freeze_sha256=digest(OUT/'freeze.json'), parameters_sha256=digest(OUT/'parameters.json'),
        histories_sha256=digest(OUT/'histories.json'), tasks=tasks, pair=pair['id'],
        parsed_RSS_rows=len(allowed), protected_RSS_rows_parsed=0, history_fields=len(records),
        query_scores=0, learning_network_updates=0, base_calibration_updates=32,
        max_VI_projected_gradient=max(r['projected_gradient'] for r in solver_rows if r['stage']!='observation')))
    print((OUT/'complete.json').read_text(), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    run()
