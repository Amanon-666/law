"""Fixed strong-GP conditional scan transport into the existing full24 Tab.

Two previously exposed B1 K1 events, no training or hyperparameter selection.
All contexts are completed with H/S only before current query arrays are read.
"""
import ast
import csv
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

from conditional_budget_trial import history_reader
from coupled_scan_transport import build_transport
from episodic_tasks import task_inputs
from recomposition_trial import ROOT, OLD, PLAN, CSV, digest, write, now
from tabicl_history import PACKAGE, FrozenHistoryRegressor, state_hash
from uji import group_scans

PARENT = ROOT/'results/acquisition_bank_transfer_20261108'
SOURCE = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
OUT = ROOT/'results/coupled_scan_transport_20261108_numeric_replay'
FAILED = ROOT/'results/coupled_scan_transport_20261108'
CHECK = ROOT/'results/coupled_scan_transport_structure_20261108/complete.json'
SEED = 1150541
TRANSPORT_SEED = 11508135
WORLDS = 8
REPLAY_ATOL = 1e-4  # Coordinate units; Euclidean-error effect <= sqrt(2)*atol.


def code_inputs():
    pending = [Path(__file__).name]
    found = {}
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


def read_selected(ids):
    """Parse RSS only from the explicitly requested rows, retaining row order."""
    selected = set(ids)
    rows = {}
    with open(CSV, newline='') as stream:
        for row_id, row in enumerate(csv.DictReader(stream), 1):
            if row_id in selected:
                rows[row_id] = dict(rss=[int(row[f'WAP{i:03}']) for i in range(1, 521)],
                    xy=[float(row['LONGITUDE']), float(row['LATITUDE'])],
                    floor=f"B{row['BUILDINGID']}F{row['FLOOR']}")
    assert set(rows) == selected
    return rows


def collect(rows, ids, origin):
    return group_scans(np.asarray([rows[i]['rss'] for i in ids]),
                       np.asarray([rows[i]['xy'] for i in ids])-origin)


def progress(stage, **values):
    value = dict(created=now(), stage=stage, **values)
    write(OUT/'progress.json', value)
    print(json.dumps(value), flush=True)


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    check = json.loads(CHECK.read_text())
    assert check['complete'] and check['code_sha256'] == digest(ROOT/'coupled_scan_transport.py')
    parent = json.loads((PARENT/'freeze.json').read_text())
    result = json.loads((PARENT/'complete.json').read_text())
    assert result['complete'] and result['status'] == 'evaluated'
    assert digest(PARENT/'freeze.json') == result['freeze_sha256']
    for path, value in parent['inputs'].items():
        assert digest(Path(path)) == value, path
    source = json.loads((SOURCE/'complete.json').read_text())
    checkpoint = SOURCE/'full_step_24.ckpt'
    assert source['formal_step'] == 24 and source['complete']
    assert digest(checkpoint) == source['checkpoint_sha256']['24']
    package = {str(p.relative_to(PACKAGE)):digest(p)
               for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == parent['package']
    events = parent['events']
    assert [e['id'] for e in events] == ['B1F1_acquisition_0', 'B1F1_acquisition_1']
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    inputs = dict(parent['inputs'])
    inputs.update({str(p):digest(p) for p in (PARENT/'freeze.json', PARENT/'complete.json',
        SOURCE/'complete.json', checkpoint, CHECK, FAILED/'freeze.json',
        FAILED/'failure.json', FAILED/'replay_diagnostic.json')})
    prepared = []
    for event in events:
        assert event['K'] == 1 and event['R'] == 3 and event['floor'] == 'B1F1'
        ids = event['support_rows']+[i for values in event['history_rows'].values() for i in values]
        assert not set(ids).intersection(event['query_rows'])
        rows = read_selected(ids)
        histories = {name:collect(rows, values, origin) for name, values in event['history_rows'].items()}
        support = collect(rows, event['support_rows'], origin)
        assert len(support['positions']) == 1
        assert all(rows[i]['floor'] == name for name, values in event['history_rows'].items() for i in values)
        meta_path = OLD/f"results/decision_local_pilot_20261030/{event['parent_event']}/meta.json"
        inputs[str(meta_path)] = digest(meta_path)
        task = task_inputs(histories, support, plan['base'], plan['obs'],
                          history_reader(json.loads(meta_path.read_text()), plan['base'], origin, inputs))
        fitted = result['numerical'][event['id']]['GP_fit']
        assert fitted['passed']
        selected = fitted['selected']
        for name in ('full24', 'retained_256'):
            p = PARENT/f'{event["id"]}_{name}.npz'
            inputs[str(p)] = digest(p)
        prepared.append(dict(event=event, histories=histories, support=support, task=task,
            selected=selected, aps=np.array(list(task['cases'])),
            bounds=np.stack([task['grid'].min(0).values.numpy(), task['grid'].max(0).values.numpy()])))
    code = code_inputs()
    freeze = dict(created=now(), code=code, inputs=inputs, events=events, package=package,
        GP_state={p['event']['id']:p['selected'] for p in prepared}, worlds=WORLDS,
        transport_seed=TRANSPORT_SEED, tab_preprocessing_seed=SEED,
        baseline_replay_atol=REPLAY_ATOL,
        execution_revision='Original attempt stopped at identity replay: identical inputs and deterministic current predictions differed from old cache by 7.62939453125e-6. Retain old failure; allow 1e-4 numerical coordinate tolerance, with no change to method or performance gate.',
        primary='Mean of eight separately projected full24 coordinates from coupled transported contexts',
        control='One context using exact posterior mean displacement; no selection between readouts',
        unchanged='Original H row counts, source floor features, masks, S, APs, grid bounds, full24 weights',
        relation='M=sum(w*h); only persistent history difference M-h_j transported, no new acquisition residual',
        criterion='Both directions no worse than full24/retained; mean improves >=3% against both',
        resources=dict(cpu=2, gpu=1, seconds=600, downloads=0, new_training=0),
        limitations=['Previously exposed development events, not blind generalization',
            'Mask preservation does not reconstruct target detection coverage',
            'Empirical source residual transfer is an additional unverified assumption',
            'Conditional context ensemble, not a joint Bayesian position posterior',
            'Eight worlds are a fixed finite estimator, not an accuracy certificate',
            'Existing history VI and GP source fits are reused without claiming exact true fields'])
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    for index, p in enumerate(prepared):
        tick = time.monotonic()
        p['transport'] = build_transport(p['histories'], p['support'], p['task'], p['selected'],
            worlds=WORLDS, seed=TRANSPORT_SEED+index)
        write(OUT/f'{p["event"]["id"]}_transport.json', p['transport']['diagnostics'])
        progress('contexts_prepared', event=p['event']['id'], seconds=time.monotonic()-tick)
        del p['task']
    # Only now read the current queries. They did not construct transport states.
    query_rows = read_selected([i for e in events for i in e['query_rows']])
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
    model.loaded.model_.requires_grad_(False).eval()
    expected = source['trained_state_sha256']
    assert state_hash(model.loaded.model_) == expected
    records = []
    for p in prepared:
        event = p['event']; eid = event['id']
        query = collect(query_rows, event['query_rows'], origin)
        assert not set(map(tuple, query['positions'])) & set(map(tuple, p['support']['positions']))
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        cached = {}
        for name, suffix, key in (('full24', 'full24', 'projected'), ('retained', 'retained_256', 'median')):
            with np.load(PARENT/f'{eid}_{suffix}.npz') as saved:
                assert np.array_equal(saved['query_rows'], event['query_rows'])
                cached[name] = saved[key].copy()
            value = float(weights@np.linalg.norm(cached[name]-query['xy'], axis=1))
            old = next(r for r in result['records'] if r['event'] == eid)
            assert abs(value-old['metrics'][name][key]) < 1e-9
        contexts = [('identity', p['histories']), ('mean_transport', p['transport']['mean_histories'])]
        contexts += [(f'world_{i}', h) for i,h in enumerate(p['transport']['world_histories'])]
        predictions = {}
        for label, histories in contexts:
            model.fit(histories, p['support'], event['floor'], p['aps'], p['bounds'], SEED)
            assert all(m.n_samples_in_ == sum(len(h['rss']) for h in p['histories'].values())+3
                       for m in model.estimators)
            prediction = model.predict(query['rss'])
            predictions[label] = prediction['projected']
            if label == 'identity':
                replay_error = float(np.max(np.abs(predictions[label]-cached['full24'])))
                assert replay_error <= REPLAY_ATOL
            progress('predicted', event=eid, context=label, seconds=time.monotonic()-started)
        predictions['primary'] = np.mean([predictions[f'world_{i}'] for i in range(WORLDS)], axis=0)
        predictions.update(cached)
        errors = {name:np.linalg.norm(x-query['xy'], axis=1) for name,x in predictions.items()}
        mde = {name:float(weights@e) for name,e in errors.items()}
        mde['WKNN'] = old['metrics']['WKNN']['mde']
        saved_path = OUT/f'{eid}.npz'
        np.savez_compressed(saved_path, **predictions, truth=query['xy'], weights=weights,
                            group=query['group'], query_rows=event['query_rows'])
        records.append(dict(event=eid, mde=mde,
            baseline_replay_max_coordinate_error=replay_error,
            per_position_errors={name:[float(e[query['group']==j].mean()) for j in range(len(query['positions']))]
                                 for name,e in errors.items() if name in ('primary','mean_transport','full24','retained')},
            prediction_sha256=digest(saved_path)))
    means = {name:float(np.mean([r['mde'][name] for r in records]))
             for name in ('primary','mean_transport','full24','retained','WKNN')}
    gains = {name:1-means['primary']/means[name] for name in ('full24','retained')}
    mean_gate = all(v>=.03 for v in gains.values())
    direction_gate = all(r['mde']['primary']<=r['mde'][name] for r in records for name in gains)
    assert state_hash(model.loaded.model_) == expected
    assert all(digest(ROOT/n)==h for n,h in code.items())
    complete = dict(complete=True, created=now(), seconds=time.monotonic()-started,
        records=records, means=means, relative_gains=gains, gate=bool(mean_gate and direction_gate),
        mean_gate=bool(mean_gate), direction_gate=bool(direction_gate),
        worlds=WORLDS, model_state_unchanged=True, training_updates=0, blind_test=False,
        peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20,
        freeze_sha256=digest(OUT/'freeze.json'))
    write(OUT/'complete.json', complete)
    print(json.dumps(complete), flush=True)


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    OUT.mkdir(exist_ok=False)
    try:
        run()
    except Exception as error:
        write(OUT/'failure.json', dict(created=now(), error=repr(error), complete=False))
        raise
