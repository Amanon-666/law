"""Fixed within-position AP-pairing intervention on H and mean-transported H.

No training, rank completion or CDF transport. Four individual permutations
measure pairing sensitivity; their coordinate ensemble is a separate output.
"""
import ast
import json
from pathlib import Path
import socket
import time

import numpy as np
import torch

import common_update_transport_trial as previous_trial
from common_update_transport import build_transport
from conditional_budget_trial import history_reader
from episodic_tasks import task_inputs
from recomposition_trial import ROOT, OLD, PLAN, digest, write, now
from tabicl_history import PACKAGE, FrozenHistoryRegressor, state_hash
from uji import group_scans

PARENT = previous_trial.PARENT
CACHE = previous_trial.OUT
SOURCE = previous_trial.SOURCE
OUT = ROOT/'results/common_transport_pairing_20261108'
SEEDS = (115081381, 115081382, 115081383, 115081384)
STATS_RTOL, S1_ATOL, S2_ATOL = 1e-12, 1e-7, 1e-5


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


def permute_pairing(histories, aps, seed):
    """Data-independent permutations in sorted source/group/AP order.

Calling with the same seed on H0/H1 gives identical row permutations. The
non-detection sentinel participates in each permutation as an observed value.
"""
    rng = np.random.default_rng(seed)
    output = {}
    diagnostics = dict(seed=seed, groups=0, groups_with_multiple_rows=0,
                       variable_columns=0, changed_columns=0, changed_cells=0)
    for name in sorted(histories):
        original = histories[name]
        rss = original['rss'].copy()
        for group in range(len(original['positions'])):
            rows = np.flatnonzero(original['group'] == group)
            diagnostics['groups'] += 1
            diagnostics['groups_with_multiple_rows'] += int(len(rows) >= 2)
            for ap in sorted(aps):
                values = original['rss'][rows, ap]
                shuffled = values[rng.permutation(len(rows))]
                assert np.array_equal(np.sort(values), np.sort(shuffled))
                rss[rows, ap] = shuffled
                changes = int(np.count_nonzero(values != shuffled))
                diagnostics['variable_columns'] += int(len(values) > 1 and np.any(values != values[0]))
                diagnostics['changed_columns'] += int(changes > 0)
                diagnostics['changed_cells'] += changes
        rebuilt = dict(original, **group_scans(rss, original['xy']))
        for key in ('xy', 'positions', 'group'):
            assert np.array_equal(rebuilt[key], original[key])
        for key in ('n', 'k'):
            assert np.array_equal(rebuilt['stats'][key], original['stats'][key])
        for key, atol in (('s1', S1_ATOL), ('s2', S2_ATOL)):
            assert np.allclose(rebuilt['stats'][key], original['stats'][key],
                               rtol=STATS_RTOL, atol=atol)
        output[name] = rebuilt
    # Consumer source order is unchanged, independently of RNG traversal order.
    return {name:output[name] for name in histories}, diagnostics


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
    for name, value in cache_freeze['code'].items():
        assert digest(ROOT/name) == value, name
    for path, value in cache_freeze['inputs'].items():
        assert digest(Path(path)) == value, path
    assert cache_freeze['worlds'] == previous_trial.WORLDS == 8
    assert cache_freeze['transport_seed'] == previous_trial.TRANSPORT_SEED
    assert cache_freeze['tab_preprocessing_seed'] == previous_trial.SEED
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
    inputs = dict(cache_freeze['inputs'])
    inputs.update({str(p):digest(p) for p in (CACHE/'freeze.json', CACHE/'complete.json')})
    plan = json.loads(PLAN.read_text())
    origin = np.asarray(plan['origin'])
    prepared = []
    for event in events:
        assert event['K'] == 1 and event['R'] == 3 and event['floor'] == 'B1F1'
        ids = event['support_rows']+[i for rows in event['history_rows'].values() for i in rows]
        assert not set(ids).intersection(event['query_rows'])
        rows = previous_trial.read_selected(ids)
        histories = {name:previous_trial.collect(rows, ids, origin)
                     for name,ids in event['history_rows'].items()}
        assert all(rows[i]['floor'] == name for name,ids in event['history_rows'].items() for i in ids)
        support = previous_trial.collect(rows, event['support_rows'], origin)
        assert len(support['positions']) == 1
        meta_path = OLD/f"results/decision_local_pilot_20261030/{event['parent_event']}/meta.json"
        inputs[str(meta_path)] = digest(meta_path)
        task = task_inputs(histories, support, plan['base'], plan['obs'],
            history_reader(json.loads(meta_path.read_text()), plan['base'], origin, inputs))
        fitted = reference['numerical'][event['id']]['GP_fit']
        assert fitted['passed'] and fitted['selected'] == cache_freeze['GP_state'][event['id']]
        record = next(r for r in cache['records'] if r['event'] == event['id'])
        prediction_path = CACHE/f"{event['id']}.npz"
        assert digest(prediction_path) == record['prediction_sha256']
        inputs[str(prediction_path)] = digest(prediction_path)
        prepared.append(dict(event=event, histories=histories, support=support, task=task,
            selected=fitted['selected'], aps=np.array(list(task['cases'])),
            bounds=np.stack([task['grid'].min(0).values.numpy(), task['grid'].max(0).values.numpy()])))
    code = code_inputs()
    freeze = dict(created=now(), code=code, inputs=inputs, events=events, package=package,
        parent_cache=str(CACHE), permutation_seeds=list(SEEDS),
        transport_seed=previous_trial.TRANSPORT_SEED, generated_worlds=previous_trial.WORLDS,
        used_transport='Only the unchanged deterministic mean_histories; generated worlds discarded.',
        tab_preprocessing_seed=previous_trial.SEED, baseline_replay_atol=previous_trial.REPLAY_ATOL,
        stats_tolerance=dict(rtol=STATS_RTOL, s1_atol=S1_ATOL, s2_atol=S2_ATOL),
        primary='Coordinate mean of four separately projected H1 shuffled-context predictions',
        control='Equivalent H0 four-permutation ensemble; paired H0 and H1 replay once each',
        pairing_estimand='Mean of four individual-permutation MDEs minus paired MDE, per H and direction; ensemble is separate.',
        intervention='Within each source/XY, independently permute each AP column including 100; identical row permutations for H0/H1.',
        invariants='Exact sorted columns, n/k, XY/group, source labels, H row counts, full H/S AP catalog, S and bounds; s1/s2 numerical tolerance.',
        criterion='Primary both directions no worse than full24/retained and equal-direction mean >=3% better than both.',
        resources=dict(cpu=2, gpu=1, gpu_memory_fraction=.5, seconds=600,
                       fits_per_event=10, downloads=0, new_training=0),
        limitations=['Two exposed development directions, no independence significance test.',
            'Singleton/constant columns cannot have their pairing changed; not all dependence is destroyed.',
            'A coordinate ensemble gain is not evidence of a single-context pairing gain.',
            'Transport still preserves per-position detection marginals and does not recover target coverage.'])
    write(OUT/'freeze.json', freeze)
    for name in code:
        (OUT/name).write_bytes((ROOT/name).read_bytes())
    for index, p in enumerate(prepared):
        transport = build_transport(p['histories'], p['support'], p['task'], p['selected'],
            worlds=previous_trial.WORLDS, seed=previous_trial.TRANSPORT_SEED+index)
        contexts = [('paired_H0', p['histories']), ('paired_H1', transport['mean_histories'])]
        diagnostics = {}
        for family, h in (('H0', p['histories']), ('H1', transport['mean_histories'])):
            for seed in SEEDS:
                shuffled, d = permute_pairing(h, p['aps'], seed)
                label = f'{family}_seed_{seed}'
                contexts.append((label, shuffled))
                diagnostics[label] = d
        p['contexts'] = contexts
        write(OUT/f"{p['event']['id']}_intervention.json", diagnostics)
        del transport, p['task']
        progress('contexts_and_invariants_complete', event=p['event']['id'])
    # Every intervention and invariant is complete before reading current Q.
    query_rows = previous_trial.read_selected([i for e in events for i in e['query_rows']])
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_per_process_memory_fraction(.5, 0)
    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
    model.loaded.model_.requires_grad_(False).eval()
    expected = source['trained_state_sha256']
    assert state_hash(model.loaded.model_) == expected
    records = []
    for p in prepared:
        event = p['event']; eid = event['id']
        query = previous_trial.collect(query_rows, event['query_rows'], origin)
        assert not set(map(tuple, query['positions'])) & set(map(tuple, p['support']['positions']))
        weights = 1/(len(query['positions'])*query['stats']['n'][query['group']])
        old = next(r for r in reference['records'] if r['event'] == eid)
        cache_record = next(r for r in cache['records'] if r['event'] == eid)
        predictions = {}
        for name, suffix, key in (('full24','full24','projected'), ('retained','retained_256','median')):
            with np.load(PARENT/f'{eid}_{suffix}.npz') as saved:
                assert np.array_equal(saved['query_rows'], event['query_rows'])
                predictions[name] = saved[key].copy()
            value = float(weights@np.linalg.norm(predictions[name]-query['xy'], axis=1))
            assert abs(value-old['metrics'][name][key]) < 1e-9
        with np.load(CACHE/f'{eid}.npz') as saved:
            for key, value in (('query_rows',event['query_rows']), ('truth',query['xy']), ('weights',weights)):
                assert np.array_equal(saved[key], value)
            for name, key in (('common_worlds','primary'), ('cached_paired_H1','mean_transport'),
                              ('previous_transport','previous_transport'), ('previous_mean','previous_mean')):
                predictions[name] = saved[key].copy()
                value = float(weights@np.linalg.norm(predictions[name]-query['xy'], axis=1))
                assert abs(value-cache_record['mde'][key]) < 1e-9
        replay = {}
        for label, histories in p['contexts']:
            model.fit(histories, p['support'], event['floor'], p['aps'], p['bounds'], previous_trial.SEED)
            assert all(m.n_samples_in_ == sum(len(h['rss']) for h in p['histories'].values())+3
                       for m in model.estimators)
            predictions[label] = model.predict(query['rss'])['projected']
            if label in ('paired_H0','paired_H1'):
                target = 'full24' if label == 'paired_H0' else 'cached_paired_H1'
                replay[label] = float(np.max(np.abs(predictions[label]-predictions[target])))
                assert replay[label] <= previous_trial.REPLAY_ATOL
            progress('predicted', event=eid, context=label, seconds=time.monotonic()-started)
        for family, output in (('H0','H0_ensemble'), ('H1','primary')):
            predictions[output] = np.mean([predictions[f'{family}_seed_{seed}'] for seed in SEEDS], axis=0)
        errors = {name:np.linalg.norm(value-query['xy'], axis=1) for name,value in predictions.items()}
        mde = {name:float(weights@value) for name,value in errors.items()}
        mde['WKNN'] = old['metrics']['WKNN']['mde']
        pairing = {}
        for family, ensemble in (('H0','H0_ensemble'), ('H1','primary')):
            values = [mde[f'{family}_seed_{seed}'] for seed in SEEDS]
            paired = mde[f'paired_{family}']
            pairing[family] = dict(individual_mde=values, individual_mean=float(np.mean(values)),
                individual_range=[min(values),max(values)], paired_mde=paired,
                individual_mean_minus_paired=float(np.mean(values))-paired,
                improved_seeds=sum(value < paired for value in values),
                ensemble_minus_paired=mde[ensemble]-paired)
        path = OUT/f'{eid}.npz'
        np.savez_compressed(path, **predictions, truth=query['xy'], weights=weights,
                            group=query['group'], query_rows=event['query_rows'])
        records.append(dict(event=eid, mde=mde, pairing=pairing, replay_max_coordinate_error=replay,
            per_position_errors={name:[float(value[query['group']==j].mean()) for j in range(len(query['positions']))]
                                 for name,value in errors.items()}, prediction_sha256=digest(path)))
    means = {name:float(np.mean([r['mde'][name] for r in records])) for name in records[0]['mde']}
    gains = {name:1-means['primary']/means[name] for name in ('full24','retained')}
    mean_gate = all(value >= .03 for value in gains.values())
    direction_gate = all(r['mde']['primary'] <= r['mde'][name] for r in records for name in gains)
    pairing = {}
    for family in ('H0','H1'):
        differences = [[r['mde'][f'{family}_seed_{seed}']-r['mde'][f'paired_{family}'] for r in records]
                       for seed in SEEDS]
        pairing[family] = dict(
            individual_mean_minus_paired=float(np.mean([r['pairing'][family]['individual_mean_minus_paired'] for r in records])),
            individual_mean=float(np.mean([r['pairing'][family]['individual_mean'] for r in records])),
            per_seed_direction_differences=differences,
            seeds_improving_both=sum(all(d < 0 for d in ds) for ds in differences),
            seeds_worsening_both=sum(all(d > 0 for d in ds) for ds in differences))
    assert state_hash(model.loaded.model_) == expected
    assert all(digest(ROOT/name) == value for name,value in code.items())
    complete = dict(complete=True, created=now(), seconds=time.monotonic()-started,
        records=records, means=means, pairing=pairing, relative_gains=gains,
        gate=bool(mean_gate and direction_gate), mean_gate=bool(mean_gate), direction_gate=bool(direction_gate),
        model_state_unchanged=True, training_updates=0, blind_test=False,
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
