"""Frozen four-arm B1 evaluation; source completion is a hard prerequisite.

All A/B/Aplus/C final48 checkpoints must share the completed B0+B2 source
protocol, schedule and vector profile. Evaluate original raw H and HS inputs;
augmentation never enters deployment. Four H forwards and 24 HS forwards per
model are sealed before any Q truth is opened, then all event combinations
are sealed before scoring. B2 is source and is never counted as target here.

GP always reads HS. GP+Tab_H is a Tab-side S ablation, not an H-only system.
H->HS changes APs, bounds and preprocessing too, not just supervision labels.
The contrasts are descriptive algorithm comparisons on an exposed development
building, not isolated causal effects, blind generalization or an S-only gate.
No checkpoint/weight selection, GP fitting, automatic next fold or fallback.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import socket
import time

for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_name] = '2'

import numpy as np
import torch

from building_information_tab_trial import context, digest, now, read_scans, read_truth, write
from history_training_attribution_trial import mapped, METHODS as OLD_METHODS
from history_task_vectors import OUT as VECTORS
from tabicl_history import PACKAGE, FrozenHistoryRegressor, state_hash

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
TRAIN = RESULTS/'history_task_training_20261110'
PRE = RESULTS/'history_task_preflight_20261110'
ASSETS = RESULTS/'building_information_assets_20261109'
OLD = RESULTS/'history_training_attribution_20261110'
OUT = RESULTS/'history_task_evaluation_20261110'
ARMS = ('A', 'B', 'Aplus', 'C')
SEED = 11509159
NEW_METHODS = tuple(name for a in ARMS for info in ('H', 'HS')
                    for name in (f'{a}_{info}', f'GP_{a}_{info}_half'))
METHODS = tuple(OLD_METHODS)+NEW_METHODS
CODE = ('history_task_evaluation.py', 'history_task_training.py', 'history_task_vectors.py',
        'history_task_protocol.py', 'history_task_preflight.py', 'history_task_augmentation.py',
        'history_training_attribution_trial.py', 'building_information_tab_trial.py',
        'building_information_protocol.py', 'tabicl_history.py', 'tabicl_history_learning.py')


def read(path):
    return json.loads(Path(path).read_text())


def save_arrays(path, **values):
    temporary = path.with_suffix('.npz.part')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **values)
    temporary.replace(path)


def initialize(args):
    pre, vector = read(PRE/'complete.json'), read(VECTORS/'complete.json')
    assert pre['complete'] and pre['weights_unchanged'] and vector['complete'] and vector['source_only']
    for name in ('freeze', 'source_profile', 'schedule'):
        assert pre[name+'_sha256'] == digest(PRE/f'{name}.json')
    assert vector['preflight_sha256'] == digest(PRE/'complete.json')
    for name in ('freeze', 'profile'):
        assert vector[name+'_sha256'] == digest(VECTORS/f'{name}.json')
    assert vector['schedule_sha256'] == digest(PRE/'schedule.json')
    assert vector['code_sha256'] == digest(ROOT/'history_task_vectors.py')
    pc = read(PRE/'freeze.json')['contract']
    assert pc['source_buildings'] == ['B0', 'B2'] and pc['held_out_building'] == 'B1' and pc['updates'] == 48
    for name, value in pc['code'].items():
        assert digest(ROOT/name) == value
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == pc['package']
    inputs = [PRE/f'{n}.json' for n in ('complete', 'freeze', 'source_profile', 'schedule')]
    inputs += [VECTORS/f'{n}.json' for n in ('complete', 'freeze', 'profile')]
    checkpoints, source = {}, {}
    for arm in ARMS:
        folder = TRAIN/arm
        c, f = read(folder/'complete.json'), read(folder/'freeze.json')
        assert c['complete'] and c['arm'] == f['arm'] == arm and c['source_updates'] == f['updates'] == 48
        assert c['source_tasks'] == 96 and c['held_out_RSS_parsed'] == c['protected_RSS_parsed'] == 0
        assert c['freeze_sha256'] == digest(folder/'freeze.json')
        assert c['training_sha256'] == digest(folder/'training.json')
        assert f['runner_sha256'] == digest(ROOT/'history_task_training.py')
        assert f['preflight_sha256'] == digest(PRE/'complete.json')
        assert f['vector_profile_completion_sha256'] == digest(VECTORS/'complete.json')
        assert f['profile_sha256'] == digest(VECTORS/'profile.json') and f['schedule_sha256'] == digest(PRE/'schedule.json')
        assert f['source_buildings'] == ['B0', 'B2'] and f['held_out_building'] == 'B1'
        assert (f['model_seed'], f['task_and_augmentation_seed']) == (pc['model_seed'], pc['seed'])
        assert f['recipe'] == ('ordinary' if arm in ('A', 'Aplus') else 'episodic')
        assert f['augmented'] == (arm in ('Aplus', 'C'))
        assert c['original_state_sha256'] == pre['initial_state_sha256']
        checkpoint = folder/'final.ckpt'
        assert digest(checkpoint) == c['checkpoint_sha256']
        checkpoints[arm], source[arm] = checkpoint, c
        inputs += [folder/n for n in ('complete.json', 'freeze.json', 'training.json', 'final.ckpt')]
    manifest = read(ASSETS/'manifest.json')
    assert read(ASSETS/'complete.json')['manifest_sha256'] == digest(ASSETS/'manifest.json')
    assert manifest['seed'] == SEED and manifest['K'] == [1, 3, 5]
    csv = Path(manifest['csv_path'])
    assert digest(csv) == manifest['csv_sha256']
    events = [e for e in manifest['events'] if e['building'] == 'B1']
    floors = [f for f in manifest['floors'] if f.startswith('B1F')]
    assert len(events) == 24 and floors == ['B1F0', 'B1F1', 'B1F2', 'B1F3']
    old, oc = read(OLD/'complete.json'), read(OLD/'freeze.json')['contract']
    assert old['complete'] and old['freeze_sha256'] == digest(OLD/'freeze.json')
    assert oc['events'] == manifest['events'] and oc['seed'] == SEED and oc['package'] == package
    for name in ('building_information_tab_trial.py', 'tabicl_history.py'):
        assert oc['code'][name] == digest(ROOT/name)
    assert oc['inputs'][str(csv)] == digest(csv) and oc['inputs'][str(ASSETS/'manifest.json')] == digest(ASSETS/'manifest.json')
    references = {r['event']: r for r in old['records'] if r['event'] in {e['id'] for e in events}}
    assert set(references) == {e['id'] for e in events}
    cached = {}
    for r in references.values():
        for name in ('prediction', 'scored'):
            assert digest(r[name+'_path']) == r[name+'_sha256']
            cached[r[name+'_path']] = r[name+'_sha256']
    plan = []
    for arm in ARMS:
        for floor in floors:
            group = [e for e in events if e['floor'] == floor]
            assert len(group) == 6 and len({tuple(e['histories']) for e in group}) == 1
            for direction in ('AB', 'BA'):
                nested = [e for e in group if e['direction'] == direction]
                assert sorted(e['K'] for e in nested) == [1, 3, 5]
                assert len({tuple(e['query_rows']) for e in nested}) == 1
            plan.append(dict(id=f'{arm}_{floor}_H', arm=arm, information='H',
                event=group[0]['id'], query_rows=list(dict.fromkeys(i for e in group for i in e['query_rows']))))
        plan += [dict(id=f'{arm}_{e["id"]}_HS', arm=arm, information='HS', event=e['id'],
                      query_rows=e['query_rows']) for e in events]
    assert len(plan) == 112
    inputs += [csv, ASSETS/'manifest.json', ASSETS/'complete.json', OLD/'complete.json', OLD/'freeze.json']
    contract = dict(code={n: digest(ROOT/n) for n in CODE}, package=package,
        inputs={str(p): digest(p) for p in inputs}, cached_predictions=cached,
        events=events, floors=floors, plan=plan, seed=SEED, source=source,
        methods=list(METHODS), deployment='Original raw H/HS, physical floor roles, native eight views; no augmentation or training',
        truth_policy='All 112 model forwards and 24 combined event predictions sealed before first query truth read',
        scoring='Equal B1 floor, direction, K, position; position-mean-error weighted p90, same original aggregation',
        comparisons='C-A, C-Aplus, B-A, Aplus-A, C-B and H->HS; errors candidate minus reference, no selection gate',
        limits=['B1 development comparison; B2 was source and is excluded',
            'GP always uses HS; GP+Tab_H is only Tab-side S ablation',
            'H->HS also changes legal APs, coordinate bounds and preprocessing; not isolated causal label calibration',
            'Different task recipes have different supervision/context/compute, despite common corpus and schedule',
            'Neither increased deltaS nor any sole metric determines success; no automatic next fold or rotation fallback',
            'Cross-K query repeats are not independent samples; no confidence interval or model/weight selection'],
        resources=dict(gpu=1, cpu_threads=2, gpu_fraction=.8, seconds=args.seconds,
            boundary='Complete forward or scoring event; explicit resume only', downloads=0))
    if args.resume:
        assert read(args.output/'freeze.json')['contract'] == contract
        progress = read(args.output/'progress.json')
        assert not progress.get('complete', False)
        assert progress['freeze_sha256'] == digest(args.output/'freeze.json')
        for r in progress['forwards']+progress['predictions']+progress['scored']:
            assert digest(r['path']) == r['sha256']
    else:
        args.output.mkdir(exist_ok=False)
        write(args.output/'freeze.json', dict(created=now(), contract=contract))
        progress = dict(freeze_sha256=digest(args.output/'freeze.json'), forwards=[], predictions=[], scored=[], segments=[])
        write(args.output/'progress.json', progress)
    return manifest, csv, events, floors, checkpoints, references, source, old, plan, progress


def seal(args, events, references, progress, started):
    assert len(progress['forwards']) == 112
    forwards = {r['id']: r for r in progress['forwards']}
    sealed = {r['event'] for r in progress['predictions']}
    for event in events:
        if event['id'] in sealed:
            continue
        if time.monotonic()-started >= args.seconds:
            return
        with np.load(references[event['id']]['prediction_path']) as saved:
            assert np.array_equal(saved['query_rows'], event['query_rows'])
            values = {m: saved[m].copy() for m in OLD_METHODS}
            schema = {key: saved[key].copy() for key in ('H_aps', 'H_bounds', 'HS_aps', 'HS_bounds')}
        for arm in ARMS:
            for info in ('H', 'HS'):
                key = f'{arm}_{event["floor"]}_H' if info == 'H' else f'{arm}_{event["id"]}_HS'
                with np.load(forwards[key]['path']) as saved:
                    assert np.array_equal(saved['aps'], schema[info+'_aps'])
                    assert np.array_equal(saved['bounds'], schema[info+'_bounds'])
                    values[f'{arm}_{info}'] = mapped(saved, event['query_rows'])
                values[f'GP_{arm}_{info}_half'] = .5*(values['GP']+values[f'{arm}_{info}'])
        path = args.output/f'{event["id"]}_unscored.npz'
        save_arrays(path, **values, **schema, query_rows=event['query_rows'])
        progress['predictions'].append(dict(event=event['id'], path=str(path), sha256=digest(path)))
        write(args.output/'progress.json', progress)
    assert len(progress['predictions']) == 24


def score_event(args, event, csv, prediction, reference):
    truth = read_truth(csv, event['query_rows'])
    _, group, count = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
    weights = 1/(len(count)*count[group])
    with np.load(prediction['path']) as saved:
        errors = {m: np.linalg.norm(saved[m]-truth, axis=1) for m in METHODS}
    with np.load(reference['scored_path']) as saved:
        assert np.array_equal(truth, saved['truth']) and np.array_equal(weights, saved['weights'])
        for m in OLD_METHODS:
            assert np.allclose(errors[m], saved[m], rtol=0, atol=1e-10)
    path = args.output/f'{event["id"]}_scored.npz'
    save_arrays(path, truth=truth, weights=weights, **errors)
    return dict(event=event['id'], path=str(path), sha256=digest(path))


def finish(args, events, floors, old, progress):
    assert len(progress['scored']) == 24
    by_event = {e['id']: e for e in events}
    samples = []
    for record in progress['scored']:
        event = by_event[record['event']]
        with np.load(record['path']) as saved:
            _, group = np.unique(saved['truth'], axis=0, return_inverse=True)
            n = int(group.max())+1
            # Match the old B1 subset's literal weights; normalization below
            # gives equal floor/direction/K/position without changing p90 ties.
            samples.append(dict(event=event, weight=.5/4/2/n,
                errors={m: np.array([saved[m][group == i].mean() for i in range(n)]) for m in METHODS}))

    def aggregate(subset, method):
        v = np.concatenate([s['errors'][method] for s in subset])
        w = np.concatenate([np.full(len(s['errors'][method]), s['weight']) for s in subset])
        w /= w.sum()
        order = np.argsort(v)
        return dict(mean=float(w@v), position_mean_p90=float(v[order][np.searchsorted(np.cumsum(w[order]), .9)]))

    summaries = {}
    for method in METHODS:
        result = aggregate(samples, method)
        for key, choices in (('K', (1, 3, 5)), ('floor', floors), ('direction', ('AB', 'BA'))):
            result['by_'+key] = {str(c): aggregate([s for s in samples if s['event'][key] == c], method) for c in choices}
        summaries[method] = result
    for m in OLD_METHODS:
        for metric in ('mean', 'position_mean_p90'):
            assert abs(summaries[m][metric]-old['summary'][m]['by_building']['B1'][metric]) <= 1e-9

    def subtract(a, b):
        return {k: subtract(a[k], b[k]) if isinstance(a[k], dict) else a[k]-b[k] for k in a}

    contrasts = {}
    for candidate, reference in (('C', 'A'), ('C', 'Aplus'), ('B', 'A'), ('Aplus', 'A'), ('C', 'B')):
        for info in ('H', 'HS'):
            for prefix, suffix in (('', ''), ('GP_', '_half')):
                contrasts[f'{prefix}{candidate}_minus_{reference}_{info}{suffix}'] = subtract(
                    summaries[f'{prefix}{candidate}_{info}{suffix}'], summaries[f'{prefix}{reference}_{info}{suffix}'])
    for model in (*ARMS, 'public', 'ordinary', 'full24'):
        for prefix, suffix in (('', ''), ('GP_', '_half')):
            contrasts[f'{prefix}{model}_H_to_HS_gain{suffix}'] = subtract(
                summaries[f'{prefix}{model}_H{suffix}'], summaries[f'{prefix}{model}_HS{suffix}'])
    progress['complete'] = True
    write(args.output/'progress.json', progress)
    write(args.output/'complete.json', dict(complete=True, created=now(), freeze_sha256=progress['freeze_sha256'],
        summary=summaries, contrasts=contrasts,
        contrast_sign='candidate-minus-reference errors: negative improves; H-to-HS gain: positive improves. p90 differences are marginal quantile differences.',
        forwards=progress['forwards'], predictions=progress['predictions'], scored=progress['scored'],
        segments=progress['segments'], new_forwards=112, training_updates=0, GP_fits=0,
        target_buildings=['B1'], target_selection=False, blind_test=False,
        limits=read(args.output/'freeze.json')['contract']['limits']))


@torch.no_grad()
def run(args):
    assert socket.gethostname() == 'ray-web' and torch.cuda.is_available()
    assert args.seconds > 0
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    started = time.monotonic()
    manifest, csv, events, floors, checkpoints, references, source, old, plan, progress = initialize(args)
    by_event = {e['id']: e for e in events}
    done = {r['id'] for r in progress['forwards']}
    initial = len(done)
    model, active_arm = None, None
    for task in plan:
        if task['id'] in done:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        arm, event, info = task['arm'], by_event[task['event']], task['information']
        if active_arm != arm:
            del model
            gc.collect()
            torch.cuda.empty_cache()
            model = FrozenHistoryRegressor(checkpoints[arm], device='cuda:0')
            model.loaded.model_.requires_grad_(False)
            active_arm = arm
            assert state_hash(model.loaded.model_) == source[arm]['trained_state_sha256']
        histories, support, rss, _, aps, bounds = context(csv, manifest, event, info)
        with np.load(references[event['id']]['prediction_path']) as saved:
            assert np.array_equal(aps, saved[info+'_aps']) and np.array_equal(bounds, saved[info+'_bounds'])
        qrss = read_scans(csv, task['query_rows'], False)['rss']
        before = state_hash(model.loaded.model_)
        model.fit(histories, support, event['floor'], aps, bounds, SEED)
        result = model.predict(qrss)
        after = state_hash(model.loaded.model_)
        assert after == before
        path = args.output/f'{task["id"]}_forward.npz'
        save_arrays(path, **result, query_rows=task['query_rows'], aps=aps, bounds=bounds)
        progress['forwards'].append(dict(task, path=str(path), sha256=digest(path),
            state_before=before, state_after=after, context_rows=len(rss), APs=len(aps), seconds=time.monotonic()-tick))
        write(args.output/'progress.json', progress)
        done.add(task['id'])
        del histories, support, rss, qrss, result
        print(json.dumps(dict(sealed_forwards=len(done), total=112, truth_read=False)), flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if len(done) == 112:
        seal(args, events, references, progress, started)
        predictions = {r['event']: r for r in progress['predictions']}
        scored = {r['event'] for r in progress['scored']}
        for event in events:
            if len(predictions) != 24:
                break
            if event['id'] in scored:
                continue
            if time.monotonic()-started >= args.seconds:
                break
            progress['scored'].append(score_event(args, event, csv, predictions[event['id']], references[event['id']]))
            write(args.output/'progress.json', progress)
    progress['segments'].append(dict(created=now(), seconds=time.monotonic()-started,
        budget=args.seconds, new_forwards=len(done)-initial, total_forwards=len(done), scored_events=len(progress['scored'])))
    write(args.output/'progress.json', progress)
    if len(progress['scored']) != 24:
        print(json.dumps(dict(complete=False, explicit_resume_required=True)), flush=True)
        return 2
    finish(args, events, floors, old, progress)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    raise SystemExit(run(parser.parse_args()))
