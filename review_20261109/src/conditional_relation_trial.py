"""Frozen within-source/location interaction trial with fixed GP marginals."""
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

import numpy as np
import torch

import acquisition_bank_transfer as gp
import common_update_transport_trial as assets
from marginal_relation import gp_bins, correction
from conditional_relation import interactions, relation_table, finite_check
from observations import observation_log_prob
from prior_mean_transport_trial import bind
from recomposition_trial import ROOT, PLAN, digest, write, now

OUT = ROOT/'results/conditional_relation_20261109'
REFERENCE = ROOT/'results/marginal_relation_gpu_20261109'
PARENT = ROOT/'results/forest_observation_20261109'
WALL_SECONDS = 600
BASELINES = ('GP', 'GP_mean', 'retained', 'retained_mean', 'full24')
ARMS = ('primary', 'total', 'between', 'raw')


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
    raise TimeoutError('Fixed 600-second conditional-relation budget exhausted; no retry')


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_default_dtype(torch.float64)
    torch.set_num_threads(2)
    check = finite_check()
    _, _, _, inputs, package = gp.source_assets()
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
    events = frozen['events']
    assert [e['id'] for e in events] == [f'B1F{f}_acquisition_{a}' for f in (1, 2) for a in (0, 1)]
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    reference = json.loads((REFERENCE/'complete.json').read_text())
    assert reference['complete'] and reference['status'] == 'evaluated'
    bind(inputs, REFERENCE/'complete.json')
    bind(inputs, REFERENCE/'freeze.json', reference['freeze_sha256'])
    prepared = []
    for event in events:
        eid = event['id']
        assert event['K'] == 1 and event['R'] == 3
        assert not set(event['query_rows']) & set(event['support_rows']+
            [i for rows in event['history_rows'].values() for i in rows])
        old = next(r for r in reference['records'] if r['event'] == eid)
        prediction_path = REFERENCE/f'{eid}.npz'
        bind(inputs, prediction_path, old['prediction_sha256'])
        source = PARENT/f'{eid}_HS_proxy.npz'
        bind(inputs, source, frozen['HS_proxies'][eid])
        with np.load(source) as saved:
            field = {k:saved[k].copy() for k in ('mean', 'variance', 'grid', 'aps')}
            edges = saved['edges'].copy()
            proxies = {name:{k:saved[name+'_'+k].copy() for k in ('rss', 'xy', 'group', 'spatial')}
                       for name in ('calibrated', 'raw')}
        for proxy in proxies.values():
            proxy['bandwidth'] = frozen['observation_proxy']['gaussian_RSS_bandwidth']
        probabilities = gp_bins(field['mean'], field['variance'], plan['obs'])
        within, between, groups = interactions(proxies['calibrated'], field['mean'], edges, event)
        ratios, diagnostics = {}, {}
        for name, delta in (('primary', within), ('total', within+between), ('between', between)):
            ratios[name], diagnostics[name] = relation_table(delta, probabilities, edges)
        del within, between, delta
        raw, _, raw_groups = interactions(proxies['raw'], field['mean'], edges, event, between=False)
        assert groups == raw_groups
        ratios['raw'], diagnostics['raw'] = relation_table(raw, probabilities, edges)
        path = OUT/f'{eid}_HS_tables.npz'
        np.savez_compressed(path, **ratios, GP_bins=probabilities)
        prepared.append(dict(event=event, old=old, prediction_path=prediction_path,
            field=field, edges=edges, table_path=path, table_sha256=digest(path),
            diagnostics=diagnostics, groups=groups))
        progress('HS_tables_complete', event=eid, seconds=time.monotonic()-started)
        del proxies, ratios, raw, probabilities
    code = code_inputs()
    freeze = dict(created=now(), events=events, code=code, inputs=inputs, package=package,
        parent=str(PARENT), reference=str(REFERENCE), GP_state=frozen['GP_state'],
        partition='Same GP-mean three-state observed partition as section150; full GP continuous law retained.',
        grouping='Source identity and physical XY; same source-order concatenation as signed section149 proxy.',
        interaction='within=E_g Cov_g(kernel); between=Cov_g(E(kernel|g)); total=within+between.',
        positivity='B=p_a*p_b; lambda=.5/max(1,max_cell(-delta/B)); C=B+lambda*delta.',
        law='Full GP node densities times forest cell ratios C/(p_a*p_b).',
        HS_tables={p['event']['id']:p['table_sha256'] for p in prepared},
        table_diagnostics={p['event']['id']:p['diagnostics'] for p in prepared},
        groups={p['event']['id']:p['groups'] for p in prepared},
        controls=['total: same law, retain within and between group interactions.',
                  'between: same law, retain only group mean/coverage interactions.',
                  'raw: within-group interactions before historical amplitude calibration.',
                  'retained GP: same complete marginals, no interaction.'],
        readout='Geometric median primary; coordinate mean secondary, never selected after Q.',
        adoption='Each floor primary >=3% better than all five GP/full24 readouts; all events no worse.',
        mechanism='Each floor strictly better than each matched control; all events no worse.',
        structural='Each floor strictly better than old common four-permutation output; all events no worse.',
        query_isolation='All H/S interaction tables saved and frozen before Q access.',
        limitations=['Four exposed K1 development events, no independent validation or K-growth claim.',
            'Within-group device/time heterogeneity and finite-sample interaction noise remain.',
            'One-row groups and group-constant AP kernels have zero within interaction.',
            'No latent copula recovery, GP risk guarantee, or global half-GP mixture lower bound.',
            'Within-bin history dependence, cycles and higher-order relations are omitted.',
            'Between-group relationships may be informative; removal is a tested hypothesis, not established truth.',
            'Small target probabilities may constrain whole edges close to independence.',
            'Mature covariance decomposition; no fitting, tuning or automatic retry.'],
        finite_check=check, resources=dict(cpu=2, gpu=1, seconds=WALL_SECONDS, downloads=0))
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    query_rows = assets.read_selected([i for e in events for i in e['query_rows']])
    records = []
    for p in prepared:
        event, field, edges = p['event'], p['field'], p['edges']
        query = assets.collect(query_rows, event['query_rows'], origin)
        assert all(query_rows[i]['floor'] == event['floor'] for i in event['query_rows'])
        assert len(query['positions']) == 12 and np.all(query['stats']['n'] == 3)
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        predictions = {}
        with np.load(p['prediction_path']) as saved:
            for key, value in (('query_rows', event['query_rows']), ('truth', query['xy']),
                               ('weights', weights), ('group', query['group'])):
                assert np.array_equal(saved[key], value)
            for name in BASELINES+('old_common', 'old_forest', 'old_forest_mean', 'primary', 'primary_mean'):
                key = {'primary':'old_marginal', 'primary_mean':'old_marginal_mean'}.get(name, name)
                predictions[key] = saved[name].copy()
                assert abs(float(weights@np.linalg.norm(predictions[key]-query['xy'], axis=1))
                           - p['old']['mde'][name]) < 1e-9
            wknn = saved['WKNN_errors'].copy()
        with np.load(p['table_path']) as saved:
            ratios = {name:saved[name].copy() for name in ARMS}
        before = time.monotonic()
        scores = {name:[] for name in ('retained',)+ARMS}
        for row in query['rss'][:, field['aps']]:
            base_score = observation_log_prob(row, field['mean'], field['variance'], plan['obs']).numpy().sum(1)
            scores['retained'].append(base_score)
            for name, ratio in ratios.items():
                scores[name].append(base_score+correction(row, field, edges, ratio))
        scores = {name:np.stack(value) for name, value in scores.items()}
        grid = torch.tensor(field['grid'])
        base = gp.actions(torch.tensor(scores['retained']).log_softmax(1), grid)
        replay = max(float(np.abs(base[key]-predictions[name]).max())
                     for key, name in (('mean', 'retained_mean'), ('median', 'retained')))
        assert replay <= 1e-6, replay
        for name in ARMS:
            result = gp.actions(torch.tensor(scores[name]).log_softmax(1), grid)
            predictions[name], predictions[name+'_mean'] = result['median'], result['mean']
        elapsed = time.monotonic()-before
        errors = {name:np.linalg.norm(xy-query['xy'], axis=1) for name, xy in predictions.items()}
        errors['WKNN'] = wknn
        mde = {name:float(weights@error) for name, error in errors.items()}
        path = OUT/f"{event['id']}.npz"
        np.savez_compressed(path, **predictions, WKNN_errors=wknn, truth=query['xy'], weights=weights,
                            group=query['group'], query_rows=event['query_rows'])
        records.append(dict(event=event['id'], floor=event['floor'], mde=mde,
            per_position_errors={name:[float(error[query['group'] == j].mean())
                for j in range(len(query['positions']))] for name, error in errors.items()},
            predict_seconds=elapsed, GP_replay_max_coordinate_error=replay, prediction_sha256=digest(path)))
        progress('event_evaluated', event=event['id'], seconds=time.monotonic()-started)
        del ratios
    floor_means = {floor:{name:float(np.mean([r['mde'][name] for r in records if r['floor'] == floor]))
        for name in records[0]['mde']} for floor in ('B1F1', 'B1F2')}
    mean_gate = all(v['primary'] <= .97*v[n] for v in floor_means.values() for n in BASELINES)
    event_gate = all(r['mde']['primary'] <= r['mde'][n] for r in records for n in BASELINES)
    def increment(name):
        return (all(v['primary'] < v[name] for v in floor_means.values()) and
                all(r['mde']['primary'] <= r['mde'][name] for r in records))
    assert all(digest(ROOT/name) == value for name, value in code.items())
    assert all(digest(Path(path)) == value for path, value in inputs.items())
    assert all(digest(p['table_path']) == p['table_sha256'] for p in prepared)
    result = dict(complete=True, status='evaluated', created=now(), seconds=time.monotonic()-started,
        records=records, floor_means=floor_means,
        gate=bool(mean_gate and event_gate), floor_mean_gate=bool(mean_gate), event_gate=bool(event_gate),
        mechanism_gates={name:bool(increment(name)) for name in ARMS[1:]},
        structural_increment_gate=bool(increment('old_common')), prior_forest_gate=bool(increment('old_forest')), prior_marginal_gate=bool(increment('old_marginal')),
        finite_check=check, GPU_peak_MiB=float(torch.cuda.max_memory_allocated()/1024**2),
        table_diagnostics={p['event']['id']:p['diagnostics'] for p in prepared},
        groups={p['event']['id']:p['groups'] for p in prepared},
        base_model_training=0, GP_refits=0, freeze_sha256=digest(OUT/'freeze.json'))
    text = json.dumps(result, separators=(',', ':'), allow_nan=False)+'\n'
    assert len(text.encode()) < 45000
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
