"""Frozen position-ranking comparison on the existing four K1 events."""
import ast
import json
import os
from pathlib import Path
import signal
import socket
import time

for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '2'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import numpy as np
import torch
import xgboost as xgb

import acquisition_bank_transfer as gp
import common_update_transport_trial as assets
from history_position_ranking import (SEED, ROUNDS, PARAMS, candidate_groups,
    training_matrix, fit_ranker, margin_contract, residual_scores)
from observations import scan_log_prob
from prior_mean_transport_trial import bind
from recomposition_trial import ROOT, PLAN, digest, write, now

OUT = ROOT/'results/history_position_ranking_fixed_20261109'
FAILED = ROOT/'results/history_position_ranking_20261109'
PARENT = ROOT/'results/history_ratio_20261109'
FIELDS = ROOT/'results/forest_observation_20261109'
WALL_SECONDS = 600
BASELINES = ('GP', 'GP_mean', 'retained', 'retained_mean', 'full24')
ARMS = ('primary', 'raw', 'no_offset', 'synthetic')


def code_inputs():
    pending, found = [Path(__file__).name], {}
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT/name
        found[name] = digest(path)
        for node in ast.walk(ast.parse(path.read_text())):
            modules = ([a.name for a in node.names] if isinstance(node, ast.Import)
                else [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            pending.extend(m.split('.')[0]+'.py' for m in modules
                if (ROOT/(m.split('.')[0]+'.py')).is_file())
    return found


def progress(stage, **values):
    row = dict(created=now(), stage=stage, **values)
    write(OUT/'progress.json', row)
    print(json.dumps(row), flush=True)


def deadline(signum, frame):
    raise TimeoutError('Fixed 600-second position-ranking budget exhausted; no retry')


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    _, _, _, inputs, package = gp.source_assets()
    bind(inputs, FAILED/'complete.json')
    bind(inputs, FAILED/'freeze.json')
    complete = json.loads((PARENT/'complete.json').read_text())
    assert complete['complete'] and complete['status'] == 'evaluated'
    bind(inputs, PARENT/'complete.json')
    bind(inputs, PARENT/'freeze.json', complete['freeze_sha256'])
    frozen = json.loads((PARENT/'freeze.json').read_text())
    assert frozen['package'] == package
    for name, value in frozen['code'].items():
        assert digest(ROOT/name) == value, name
    for path, value in frozen['inputs'].items():
        bind(inputs, Path(path), value)
    field_complete = json.loads((FIELDS/'complete.json').read_text())
    assert field_complete['complete'] and field_complete['status'] == 'evaluated'
    bind(inputs, FIELDS/'complete.json')
    bind(inputs, FIELDS/'freeze.json', field_complete['freeze_sha256'])
    field_freeze = json.loads((FIELDS/'freeze.json').read_text())
    assert field_freeze['events'] == frozen['events']
    assert field_freeze['GP_state'] == frozen['GP_state']
    dependency = Path(xgb.__file__).parent
    wheel = ROOT/'.deps/wheels/xgboost_cpu-3.1.3-py3-none-manylinux_2_28_x86_64.whl'
    bind(inputs, wheel, 'a7079aeca33e27b8b6d9967f84201b33942ed565178eda8db83ad0b59f17c033')
    bind(inputs, Path(xgb.__file__))
    bind(inputs, dependency/'lib/libxgboost.so')
    assert xgb.__version__ == '3.1.3' and not xgb.build_info()['USE_CUDA']
    events = frozen['events']
    assert [e['id'] for e in events] == [f'B1F{f}_acquisition_{a}' for f in (1, 2) for a in (0, 1)]
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    prepared = []
    for index, event in enumerate(events):
        eid = event['id']
        assert event['K'] == 1 and event['R'] == 3
        assert not set(event['query_rows']) & set(event['support_rows']+
            [i for rows in event['history_rows'].values() for i in rows])
        old = next(r for r in complete['records'] if r['event'] == eid)
        prediction_path = PARENT/f'{eid}.npz'
        bind(inputs, prediction_path, old['prediction_sha256'])
        source = FIELDS/f'{eid}_HS_proxy.npz'
        bind(inputs, source, field_freeze['HS_proxies'][eid])
        with np.load(source) as saved:
            field = {k:saved[k].copy() for k in ('mean', 'variance', 'grid', 'aps')}
        fit_path = PARENT/f'{eid}_fit_inputs.npz'
        bind(inputs, fit_path, frozen['fit_inputs'][eid])
        with np.load(fit_path) as saved:
            assert np.array_equal(saved['aps'], field['aps'])
            indices = saved['grid_indices'].copy()
            scans = {name:saved[key].copy() for name, key in
                     (('primary', 'primary'), ('raw', 'raw'), ('synthetic', 'negative'))}
        assert all(r.shape == (len(indices), len(field['aps'])) for r in scans.values())
        candidates = candidate_groups(indices, len(field['grid']), SEED+index)
        candidate_path = OUT/f'{eid}_candidates.npy'
        np.save(candidate_path, candidates)
        prepared.append(dict(event=event, old=old, prediction_path=prediction_path,
            field=field, bounds=np.stack((field['grid'].min(0), field['grid'].max(0))),
            scans=scans, candidates=candidates, candidate_path=candidate_path,
            candidate_sha256=digest(candidate_path), models={}, model_hashes={}, fit_seconds={}))
    code = code_inputs()
    freeze = dict(created=now(), events=events, code=code, inputs=inputs, package=package,
        parent=str(PARENT), fields=str(FIELDS), prior_failed_run=str(FAILED),
        fix='Contract check uses distinct DMatrix handles to avoid stale prediction cache; method unchanged.',
        GP_state=frozen['GP_state'], seed=SEED,
        xgboost=dict(version=xgb.__version__, build=xgb.build_info(), path=str(dependency)),
        ranker=dict(params=PARAMS, rounds=ROUNDS, other_settings='Installed 3.1.3 defaults'),
        candidate_hashes={p['event']['id']:p['candidate_sha256'] for p in prepared},
        target='One positive proxy grid position and four uniform distinct alternatives per scan; shared by all arms.',
        features=['mask', 'GP-standardized detected residual', 'GP detection probability', 'normalized XY', 'complete raw GP logscore'],
        offset='Group-centered GP score during fit; explicit zero margin for residual inference, then original float64 GP added once.',
        controls=['raw H plus GP offset', 'same calibrated H/features without offset', 'GP synthetic scans plus offset'],
        readout='Softmax score and geometric median primary; coordinate mean secondary, never selected after Q.',
        adoption='Each floor primary >=3% better than all five fixed GP/full24 readouts; all events no worse.',
        attribution='Separate per-control gates: each floor strictly better, all events no worse.',
        query_isolation='Candidates/config frozen before fitting; all 16 rankers saved before Q access.',
        limitations=['Four exposed K1 development events, no independent validation or K-growth claim.',
            'Positive XY is an RBF proxy pairing, not the original scan measurement location.',
            'Rank scores are not a calibrated density or posterior and do not optimize meters directly.',
            'Offset can be cancelled by learned trees and gives no GP marginal or risk guarantee.',
            'Old ratio method differs in learner and features; not a pure objective ablation.',
            'No hyperparameter search, source pretraining, base model tuning, GP refit or retry.'],
        resources=dict(cpu=2, gpu=0, seconds=WALL_SECONDS, context_ranker_fits=16, downloads_during_run=0))
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    check = None
    for p in prepared:
        for scan_name in ('primary', 'raw', 'synthetic'):
            before = time.monotonic()
            matrix, offset = training_matrix(p['scans'][scan_name], p['candidates'],
                                            p['field'], plan['obs'], p['bounds'])
            matrix_seconds = time.monotonic()-before
            for name in (('primary', 'no_offset') if scan_name == 'primary' else (scan_name,)):
                before = time.monotonic()
                model = fit_ranker(matrix, offset, name != 'no_offset')
                if check is None:
                    check = margin_contract(model, matrix, offset)
                path = OUT/f"{p['event']['id']}_{name}.ubj"
                model.save_model(path)
                p['models'][name] = model
                p['model_hashes'][name] = digest(path)
                p['fit_seconds'][name] = dict(matrix=matrix_seconds, fit=time.monotonic()-before)
                progress('context_ranker_fitted', event=p['event']['id'], arm=name,
                         seconds=time.monotonic()-started)
            del matrix
        del p['scans']
    write(OUT/'models.json', dict(created=now(), query_opened=False, margin_contract=check,
        models={p['event']['id']:p['model_hashes'] for p in prepared}))
    query_rows = assets.read_selected([i for e in events for i in e['query_rows']])
    records = []
    for p in prepared:
        event, field = p['event'], p['field']
        query = assets.collect(query_rows, event['query_rows'], origin)
        assert all(query_rows[i]['floor'] == event['floor'] for i in event['query_rows'])
        assert len(query['positions']) == 12 and np.all(query['stats']['n'] == 3)
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        predictions = {}
        with np.load(p['prediction_path']) as saved:
            for key, value in (('query_rows', event['query_rows']), ('truth', query['xy']),
                               ('weights', weights), ('group', query['group'])):
                assert np.array_equal(saved[key], value)
            for name in BASELINES+('old_common', 'primary', 'primary_mean'):
                key = {'primary':'old_ratio', 'primary_mean':'old_ratio_mean'}.get(name, name)
                predictions[key] = saved[name].copy()
                assert abs(float(weights@np.linalg.norm(predictions[key]-query['xy'], axis=1))
                           - p['old']['mde'][name]) < 1e-9
            wknn = saved['WKNN_errors'].copy()
        score = torch.zeros((len(query['rss']), len(field['grid'])))
        qrss = query['rss'][:, field['aps']]
        for column in range(len(field['aps'])):
            score += scan_log_prob(torch.tensor(qrss[:, column]), torch.tensor(field['mean'][:, column]),
                                   torch.tensor(field['variance'][:, column]), plan['obs'])
        grid = torch.tensor(field['grid'])
        base = gp.actions(score.log_softmax(1), grid)
        replay = max(float(np.abs(base[key]-predictions[name]).max())
                     for key, name in (('mean', 'retained_mean'), ('median', 'retained')))
        assert replay <= 1e-6, replay
        timings, residual_ranges = {}, {}
        for name in ARMS:
            before = time.monotonic()
            residual = residual_scores(p['models'][name], qrss, score.numpy(), field, plan['obs'], p['bounds'])
            logscore = torch.tensor(residual)+(score if name != 'no_offset' else 0.)
            result = gp.actions(logscore.log_softmax(1), grid)
            predictions[name], predictions[name+'_mean'] = result['median'], result['mean']
            timings[name] = time.monotonic()-before
            residual_ranges[name] = float(np.ptp(residual, axis=1).mean())
        errors = {name:np.linalg.norm(xy-query['xy'], axis=1) for name, xy in predictions.items()}
        errors['WKNN'] = wknn
        mde = {name:float(weights@error) for name, error in errors.items()}
        path = OUT/f"{event['id']}.npz"
        np.savez_compressed(path, **predictions, WKNN_errors=wknn, truth=query['xy'], weights=weights,
                            group=query['group'], query_rows=event['query_rows'])
        records.append(dict(event=event['id'], floor=event['floor'], mde=mde,
            per_position_errors={name:[float(error[query['group'] == j].mean())
                for j in range(len(query['positions']))] for name, error in errors.items()},
            fit_seconds=p['fit_seconds'], predict_seconds=timings, residual_spatial_ranges=residual_ranges,
            GP_replay_max_coordinate_error=replay, prediction_sha256=digest(path), model_hashes=p['model_hashes']))
        progress('event_evaluated', event=event['id'], seconds=time.monotonic()-started)
    floor_means = {floor:{name:float(np.mean([r['mde'][name] for r in records if r['floor'] == floor]))
        for name in records[0]['mde']} for floor in ('B1F1', 'B1F2')}
    mean_gate = all(v['primary'] <= .97*v[n] for v in floor_means.values() for n in BASELINES)
    event_gate = all(r['mde']['primary'] <= r['mde'][n] for r in records for n in BASELINES)
    def increment(name):
        return (all(v['primary'] < v[name] for v in floor_means.values()) and
                all(r['mde']['primary'] <= r['mde'][name] for r in records))
    assert all(digest(ROOT/name) == value for name, value in code.items())
    assert all(digest(Path(path)) == value for path, value in inputs.items())
    assert all(digest(p['candidate_path']) == p['candidate_sha256'] for p in prepared)
    assert all(digest(OUT/f"{p['event']['id']}_{name}.ubj") == value
               for p in prepared for name, value in p['model_hashes'].items())
    result = dict(complete=True, status='evaluated', created=now(), seconds=time.monotonic()-started,
        records=records, floor_means=floor_means,
        gate=bool(mean_gate and event_gate), floor_mean_gate=bool(mean_gate), event_gate=bool(event_gate),
        mechanism_gates={name:bool(increment(name)) for name in ARMS[1:]},
        structural_increment_gate=bool(increment('old_common')), old_ratio_gate=bool(increment('old_ratio')),
        context_ranker_fits=16, context_ranker_trees=16*ROUNDS, base_model_training=0, GP_refits=0,
        margin_contract=check, freeze_sha256=digest(OUT/'freeze.json'), models_sha256=digest(OUT/'models.json'))
    text = json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(text.encode()) < 50000
    (OUT/'complete.json').write_text(text)
    print(text, end='', flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web', 'Server-only entry'
    OUT.mkdir(exist_ok=False)
    signal.signal(signal.SIGALRM, deadline)
    signal.setitimer(signal.ITIMER_REAL, WALL_SECONDS)
    try:
        run()
    except Exception as error:
        write(OUT/'complete.json', dict(complete=False, status='failed', created=now(), error=repr(error)))
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
