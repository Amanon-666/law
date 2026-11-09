"""Change only the consumer floor feature for raw/transported H contexts.

Source-specific GP inputs stay intact. Target-role historical rows are derived
contexts, not additional measured target observations. No model training.
"""
import ast
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

import common_transport_pairing_trial as pairing
import common_update_transport_trial as assets
from common_update_transport import build_transport
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from recomposition_trial import ROOT, OLD, PLAN, digest, write, now
from tabicl_history import PACKAGE, FrozenHistoryRegressor, features, state_hash
from uji import group_scans

PARENT, SOURCE = pairing.PARENT, pairing.SOURCE
CACHE = pairing.OUT
OUT = ROOT/'results/target_role_transport_20261108'
SEEDS = pairing.SEEDS


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


def target_role(histories, support, target_floor, aps):
    """Concatenate in consumer order; verify the complete fit feature change.

group_scans aggregates statistics without reordering its raw RSS/XY rows. S is
not merged here: FrozenHistoryRegressor.fit appends it separately exactly once.
"""
    target = int(target_floor.split('F')[1])
    rss = np.concatenate([h['rss'] for h in histories.values()])
    xy = np.concatenate([h['xy'] for h in histories.values()])
    combined = group_scans(rss, xy)
    assert np.array_equal(combined['rss'], rss)
    assert np.array_equal(combined['xy'], xy)
    source_features = np.concatenate([features(h['rss'], int(name.split('F')[1]), aps)
                                      for name,h in histories.items()])
    target_features = features(combined['rss'], target, aps)
    support_features = features(support['rss'], target, aps)
    before = np.concatenate([source_features, support_features])
    after = np.concatenate([target_features, support_features])
    assert before.shape == after.shape
    assert np.array_equal(before[:, :-1], after[:, :-1])
    assert np.array_equal(after[-len(support['rss']):], support_features)
    assert np.all(target_features[:, -1] == target)
    spans, start = [], 0
    for name,h in histories.items():
        stop = start+len(h['rss'])
        spans.append(dict(source=name, start_inclusive=start, stop_exclusive=stop,
                          physical_source_floor=int(name.split('F')[1]), consumer_floor=target))
        assert np.array_equal(combined['rss'][start:stop], h['rss'])
        assert np.array_equal(combined['xy'][start:stop], h['xy'])
        start = stop
    return {target_floor:combined}, dict(history_rows=len(rss), support_rows=len(support['rss']),
        fit_rows=len(before), source_row_spans=spans,
        changed_floor_cells=int(np.count_nonzero(before[:, -1] != after[:, -1])),
        nonfloor_features_identical=True, RSS_XY_order_identical=True,
        support_separate=True, role='Derived historical context; not measured target scans.')


def progress(stage, **values):
    row = dict(created=now(), stage=stage, **values)
    write(OUT/'progress.json', row)
    print(json.dumps(row), flush=True)


@torch.no_grad()
def run():
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float64)
    cache = json.loads((CACHE/'complete.json').read_text())
    cache_freeze = json.loads((CACHE/'freeze.json').read_text())
    assert cache['complete'] and cache['model_state_unchanged']
    assert digest(CACHE/'freeze.json') == cache['freeze_sha256']
    for name,value in cache_freeze['code'].items():
        assert digest(ROOT/name) == value, name
    for path,value in cache_freeze['inputs'].items():
        assert digest(Path(path)) == value, path
    assert cache_freeze['permutation_seeds'] == list(SEEDS)
    assert cache_freeze['generated_worlds'] == assets.WORLDS == 8
    assert cache_freeze['transport_seed'] == assets.TRANSPORT_SEED
    assert cache_freeze['tab_preprocessing_seed'] == assets.SEED
    parent = json.loads((PARENT/'freeze.json').read_text())
    reference = json.loads((PARENT/'complete.json').read_text())
    assert reference['complete'] and reference['status'] == 'evaluated'
    assert digest(PARENT/'freeze.json') == reference['freeze_sha256']
    events = cache_freeze['events']
    assert events == parent['events']
    assert [e['id'] for e in events] == ['B1F1_acquisition_0', 'B1F1_acquisition_1']
    checkpoint = SOURCE/'full_step_24.ckpt'
    source = json.loads((SOURCE/'complete.json').read_text())
    assert source['complete'] and source['formal_step'] == 24
    assert digest(checkpoint) == source['checkpoint_sha256']['24']
    package = {str(p.relative_to(PACKAGE)):digest(p)
               for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == cache_freeze['package'] == parent['package']
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    inputs = dict(cache_freeze['inputs'])
    inputs.update({str(p):digest(p) for p in (CACHE/'freeze.json', CACHE/'complete.json')})
    prepared = []
    for event in events:
        assert event['K'] == 1 and event['R'] == 3 and event['floor'] == 'B1F1'
        ids = event['support_rows']+[i for rows in event['history_rows'].values() for i in rows]
        assert not set(ids).intersection(event['query_rows'])
        rows = assets.read_selected(ids)
        histories = {name:assets.collect(rows, ids, origin) for name,ids in event['history_rows'].items()}
        assert all(rows[i]['floor'] == name for name,ids in event['history_rows'].items() for i in ids)
        support = assets.collect(rows, event['support_rows'], origin)
        assert len(support['positions']) == 1
        meta_path = OLD/f"results/decision_local_pilot_20261030/{event['parent_event']}/meta.json"
        inputs[str(meta_path)] = digest(meta_path)
        task = task_inputs(histories, support, plan['base'], plan['obs'],
            history_reader(json.loads(meta_path.read_text()), plan['base'], origin, inputs))
        fitted = reference['numerical'][event['id']]['GP_fit']
        assert fitted['passed']
        record = next(r for r in cache['records'] if r['event'] == event['id'])
        prediction_path = CACHE/f"{event['id']}.npz"
        assert digest(prediction_path) == record['prediction_sha256']
        inputs[str(prediction_path)] = digest(prediction_path)
        spans, start = [], 0
        for name,h in histories.items():
            stop = start+len(h['rss'])
            spans.append(dict(source=name, start_inclusive=start, stop_exclusive=stop,
                              physical_source_floor=int(name.split('F')[1]),
                              consumer_floor=int(event['floor'].split('F')[1])))
            start = stop
        prepared.append(dict(event=event, histories=histories, support=support, task=task,
            selected=fitted['selected'], aps=np.array(list(task['cases'])), source_row_spans=spans,
            bounds=np.stack([task['grid'].min(0).values.numpy(), task['grid'].max(0).values.numpy()])))
    code = code_inputs()
    freeze = dict(created=now(), code=code, inputs=inputs, events=events, package=package,
        source_role_cache=str(CACHE), permutation_seeds=list(SEEDS),
        source_row_spans={p['event']['id']:p['source_row_spans'] for p in prepared},
        GP_state={p['event']['id']:p['selected'] for p in prepared},
        transport_seed=assets.TRANSPORT_SEED, generated_worlds=assets.WORLDS,
        tab_preprocessing_seed=assets.SEED, baseline_replay_atol=assets.REPLAY_ATOL,
        primary='Mean of four target-role H1 shuffled-context projected predictions',
        control='Target-role H0 four-shuffle ensemble; paired target-role H0/H1 each once, no selection',
        intervention='Only final floor feature changes; concat history RSS/XY in original dictionary order. GP tasks keep original sources.',
        unchanged='RSS/XY/order/count/AP/bounds/model/S; S appended once by existing consumer',
        estimands='Matched-seed target-minus-source MDE, individual mean/range, ensemble differences, H1-minus-H0; report these separately.',
        criterion='Primary both directions no worse than full24/retained; equal-direction mean >=3% better than both.',
        resources=dict(cpu=2, gpu=1, gpu_memory_fraction=.5, seconds=600,
                       fits_per_event=11, downloads=0, new_training=0),
        limitations=['Floor reassignment is a consumer-role intervention, not new true target observations.',
            'A raw-H benefit can reflect removal of source-floor dependence, not calibration.',
            'Permutation and ensemble gains remain distinct; no independence significance claim.',
            'Two exposed development directions; no coverage reconstruction or generalization claim.'])
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    for index,p in enumerate(prepared):
        transport = build_transport(p['histories'], p['support'], p['task'], p['selected'],
            worlds=assets.WORLDS, seed=assets.TRANSPORT_SEED+index)
        contexts = [('source_H0_replay', p['histories'])]
        diagnostics = {}
        for family,h in (('H0',p['histories']), ('H1',transport['mean_histories'])):
            mapped, d = target_role(h, p['support'], p['event']['floor'], p['aps'])
            label = f'target_paired_{family}'
            contexts.append((label,mapped)); diagnostics[label] = d
            for seed in SEEDS:
                shuffled, permutation_diagnostic = pairing.permute_pairing(h, p['aps'], seed)
                mapped, d = target_role(shuffled, p['support'], p['event']['floor'], p['aps'])
                label = f'target_{family}_seed_{seed}'
                contexts.append((label,mapped))
                diagnostics[label] = dict(d, permutation=permutation_diagnostic)
        assert len(contexts) == 11
        p['contexts'] = contexts
        write(OUT/f"{p['event']['id']}_role_invariants.json", diagnostics)
        del transport, p['task']
        progress('all_role_invariants_complete', event=p['event']['id'])
    # Current Q is inaccessible to every preceding role/permutation operation.
    query_rows = assets.read_selected([i for event in events for i in event['query_rows']])
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
    model.loaded.model_.requires_grad_(False).eval()
    expected = source['trained_state_sha256']
    assert state_hash(model.loaded.model_) == expected
    records = []
    for p in prepared:
        event = p['event']; eid = event['id']
        query = assets.collect(query_rows, event['query_rows'], origin)
        assert not set(map(tuple, query['positions'])) & set(map(tuple, p['support']['positions']))
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        cached_record = next(r for r in cache['records'] if r['event'] == eid)
        old = next(r for r in reference['records'] if r['event'] == eid)
        predictions = {}
        with np.load(CACHE/f'{eid}.npz') as saved:
            for key,value in (('query_rows',event['query_rows']), ('truth',query['xy']),
                              ('weights',weights), ('group',query['group'])):
                assert np.array_equal(saved[key], value)
            for name in saved.files:
                if name in ('query_rows','truth','weights','group'):
                    continue
                predictions['source_'+name] = saved[name].copy()
                value = float(weights@np.linalg.norm(saved[name]-query['xy'], axis=1))
                assert abs(value-cached_record['mde'][name]) < 1e-9
        assert abs(cached_record['mde']['WKNN']-old['metrics']['WKNN']['mde']) < 1e-9
        for name in ('full24','retained'):
            key = 'projected' if name == 'full24' else 'median'
            assert abs(cached_record['mde'][name]-old['metrics'][name][key]) < 1e-9
        replay = None
        for label,histories in p['contexts']:
            model.fit(histories, p['support'], event['floor'], p['aps'], p['bounds'], assets.SEED)
            assert all(m.n_samples_in_ == sum(len(h['rss']) for h in p['histories'].values())+3
                       for m in model.estimators)
            predictions[label] = model.predict(query['rss'])['projected']
            if label == 'source_H0_replay':
                replay = float(np.max(np.abs(predictions[label]-predictions['source_paired_H0'])))
                assert replay <= assets.REPLAY_ATOL
            progress('predicted', event=eid, context=label, seconds=time.monotonic()-started)
        for family,output in (('H0','target_H0_ensemble'), ('H1','primary')):
            predictions[output] = np.mean([predictions[f'target_{family}_seed_{seed}'] for seed in SEEDS], axis=0)
        errors = {name:np.linalg.norm(value-query['xy'], axis=1) for name,value in predictions.items()}
        mde = {name:float(weights@value) for name,value in errors.items()}
        mde['WKNN'] = cached_record['mde']['WKNN']
        role_effect = {}
        for family,target_ensemble,source_ensemble in (('H0','target_H0_ensemble','source_H0_ensemble'),
                                                       ('H1','primary','source_primary')):
            target_values = [mde[f'target_{family}_seed_{seed}'] for seed in SEEDS]
            source_values = [mde[f'source_{family}_seed_{seed}'] for seed in SEEDS]
            differences = [t-s for t,s in zip(target_values,source_values)]
            role_effect[family] = dict(target_single_mde=target_values, source_single_mde=source_values,
                target_single_mean=float(np.mean(target_values)), source_single_mean=float(np.mean(source_values)),
                target_single_range=[min(target_values),max(target_values)],
                source_single_range=[min(source_values),max(source_values)],
                per_seed_target_minus_source=differences, mean_target_minus_source=float(np.mean(differences)),
                improved_seeds=sum(d < 0 for d in differences),
                ensemble_target_minus_source=mde[target_ensemble]-mde[source_ensemble],
                paired_target_minus_source=mde[f'target_paired_{family}']-mde[f'source_paired_{family}'])
        calibration = dict(
            target_per_seed_H1_minus_H0=[mde[f'target_H1_seed_{s}']-mde[f'target_H0_seed_{s}'] for s in SEEDS],
            source_per_seed_H1_minus_H0=[mde[f'source_H1_seed_{s}']-mde[f'source_H0_seed_{s}'] for s in SEEDS],
            target_single_mean_H1_minus_H0=role_effect['H1']['target_single_mean']-role_effect['H0']['target_single_mean'],
            source_single_mean_H1_minus_H0=role_effect['H1']['source_single_mean']-role_effect['H0']['source_single_mean'],
            target_ensemble_H1_minus_H0=mde['primary']-mde['target_H0_ensemble'],
            source_ensemble_H1_minus_H0=mde['source_primary']-mde['source_H0_ensemble'],
            target_paired_H1_minus_H0=mde['target_paired_H1']-mde['target_paired_H0'],
            source_paired_H1_minus_H0=mde['source_paired_H1']-mde['source_paired_H0'])
        path = OUT/f'{eid}.npz'
        np.savez_compressed(path, **predictions, truth=query['xy'], weights=weights,
                            group=query['group'], query_rows=event['query_rows'])
        position_keys = [name for name in predictions if name.startswith('target_')]
        position_keys += ['primary','source_primary','source_H0_ensemble','source_paired_H0',
                          'source_paired_H1','source_full24','source_retained']
        records.append(dict(event=eid, mde=mde, role_effect=role_effect, calibration=calibration,
            baseline_replay_max_coordinate_error=replay,
            per_position_errors={name:[float(errors[name][query['group']==j].mean()) for j in range(len(query['positions']))]
                                 for name in position_keys}, prediction_sha256=digest(path)))
    means = {name:float(np.mean([r['mde'][name] for r in records])) for name in records[0]['mde']}
    gains = {name:1-means['primary']/means['source_'+name] for name in ('full24','retained')}
    mean_gate = all(value >= .03 for value in gains.values())
    direction_gate = all(r['mde']['primary'] <= r['mde']['source_'+name] for r in records for name in gains)
    role_effect = {family:{key:float(np.mean([r['role_effect'][family][key] for r in records]))
                          for key in ('target_single_mean','source_single_mean','mean_target_minus_source',
                                      'ensemble_target_minus_source','paired_target_minus_source')}
                   for family in ('H0','H1')}
    calibration = {key:float(np.mean([r['calibration'][key] for r in records]))
                   for key in records[0]['calibration'] if 'per_seed' not in key}
    assert state_hash(model.loaded.model_) == expected
    assert all(digest(ROOT/name) == value for name,value in code.items())
    complete = dict(complete=True, created=now(), seconds=time.monotonic()-started,
        records=records, means=means, role_effect=role_effect, calibration=calibration, relative_gains=gains,
        gate=bool(mean_gate and direction_gate), mean_gate=bool(mean_gate), direction_gate=bool(direction_gate),
        model_state_unchanged=True, training_updates=0, blind_test=False,
        peak_gpu_MiB=torch.cuda.max_memory_allocated()/2**20, freeze_sha256=digest(OUT/'freeze.json'))
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
