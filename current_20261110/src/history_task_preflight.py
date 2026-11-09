"""Two real-source path/cost checks, with no optimizer or localization score.

The existing B0+B2 catalog and a 48-update metadata schedule are frozen before
RSS parsing. Only its nonprotected historical-row union is decoded. For each
recipe the scheduled task with maximum context+query scans is selected without
RSS or scores. Check raw/augmented official views, then execute exactly one
original augmented backward per recipe. No parameter update, model saving,
target/protected RSS, prediction, performance selection, or automatic retry.
"""
import argparse
from copy import deepcopy
import csv
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time

for _thread_variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[_thread_variable] = '2'

import numpy as np
import torch

from building_information_tab_trial import digest, now, write
from history_task_augmentation import apply_transform, sample_transform
from history_task_protocol import (allowed_rows, building, empirical_target_shifts,
    episodic_event, episodic_pack, ordinary_event, ordinary_pack, training_schedule)
from tabicl_history import CHECKPOINT, PACKAGE, state_hash
from tabicl_history_learning import EpisodicHistoryRegressor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'results/building_source_tasks_20261106'
CSV = ROOT.parent/'UJIIndoorLoc/data/raw/UJIndoorLoc/trainingData.csv'
OUT = ROOT/'results/history_task_preflight_20261110'
SEED = 11510173
MODEL_SEED = 1150546
UPDATES = 48
CODE = ('history_task_preflight.py', 'history_task_augmentation.py', 'history_task_protocol.py',
        'source_episode_catalog.py', 'acquisition_tasks.py', 'uji.py', 'tabicl_history.py',
        'tabicl_history_learning.py', 'check_tabicl_history_learning.py', 'building_information_tab_trial.py')


def read(path):
    return json.loads(Path(path).read_text())


def load_allowed(csv_path, permitted, *, rss=False):
    """Decode only whitelisted rows; even held-out XY is not interpreted."""
    wanted = {i for ids in permitted.values() for i in ids}
    rows = {}
    with open(csv_path, newline='') as stream:
        for row_id, row in enumerate(csv.DictReader(stream), 1):
            if row_id not in wanted:
                continue
            stamp = int(row['TIMESTAMP'])
            value = dict(floor=f"B{row['BUILDINGID']}F{row['FLOOR']}",
                xy=(float(row['LONGITUDE']), float(row['LATITUDE'])), timestamp=stamp,
                acquisition=(int(row['USERID']), int(row['PHONEID']),
                    datetime.fromtimestamp(stamp, timezone.utc).date().isoformat()))
            if rss:
                value['rss'] = [int(row[f'WAP{a:03}']) for a in range(1, 521)]
            rows[row_id] = value
    assert set(rows) == wanted
    return rows


def audit_event(event, rows, permitted, reserved, held_out):
    floor = event['floor']
    assert building(floor) != held_out and event['building'] == building(floor)
    by_role = [(f, ids) for f, ids in event['history_rows'].items()]
    by_role += [(floor, event['support_rows']), (floor, event['query_rows'])]
    for f, ids in by_role:
        assert ids and len(ids) == len(set(ids)) and set(ids) <= permitted[f]
        assert building(f) == event['building']
        assert all(rows[i]['floor'] == f and rows[i]['xy'] not in reserved[f] for i in ids)
    assert floor not in event['history_rows']
    h = {i for ids in event['history_rows'].values() for i in ids}
    s, q = set(event['support_rows']), set(event['query_rows'])
    assert not (h & s or h & q or s & q)
    assert not {rows[i]['xy'] for i in s}.intersection(rows[i]['xy'] for i in q)
    # Across different floors, common physical XY is legal registered history.
    counts = dict(context_scans=len(h)+len(s), query_scans=len(q),
        context_positions=sum(len({rows[i]['xy'] for i in ids}) for _, ids in by_role[:-1]),
        query_positions=len({rows[i]['xy'] for i in q}))
    return counts


def freeze_metadata(args):
    catalog = read(SOURCE/'catalog.json')
    old = read(SOURCE/'freeze.json')
    assert read(SOURCE/'complete.json')['complete']
    assert digest(CSV) == old['CSV_sha256']
    assert catalog['contract']['source_buildings'] == ['B0', 'B2']
    assert catalog['contract']['held_out_building'] == 'B1'
    permitted = allowed_rows(catalog)
    protected = set(catalog['protected_rows'])
    reserved = {f: {tuple(x) for x in catalog['protected_positions'][f]} for f in permitted}
    rows = load_allowed(CSV, permitted)
    assert all(not protected.intersection(ids) for ids in permitted.values())
    assert all(rows[i]['xy'] not in reserved[f] for f, ids in permitted.items() for i in ids)
    schedule, counts = training_schedule(catalog, UPDATES, SEED)
    allowed = {f: set(ids) for f, ids in permitted.items()}
    selected, totals = {}, {r: dict(tasks=0, query_scans=0, query_positions=0) for r in ('ordinary', 'episodic')}
    for batch in schedule:
        assert {s['building'] for s in batch} == {'B0', 'B2'}
        for spec in batch:
            e = episodic_event(catalog, spec)
            o = ordinary_event(rows, catalog, spec, SEED)
            assert e['task_key'] == o['task_key'] == spec['task_key']
            assert set(e['histories']) == set(o['histories'])
            assert e['support_acquisition'][:2] != e['query_acquisition'][:2]
            for recipe, event in (('ordinary', o), ('episodic', e)):
                measured = audit_event(event, rows, allowed, reserved, 'B1')
                totals[recipe]['tasks'] += 1
                for key in ('query_scans', 'query_positions'):
                    totals[recipe][key] += measured[key]
                size = measured['context_scans']+measured['query_scans']
                if recipe not in selected or size > selected[recipe]['total_scans']:
                    selected[recipe] = dict(spec=spec, total_scans=size, **measured)
    args.output.mkdir(exist_ok=False)
    write(args.output/'schedule.json', dict(schedule=schedule, counts=counts))
    contract = dict(code={name: digest(ROOT/name) for name in CODE},
        inputs={str(p): digest(p) for p in (CSV, CHECKPOINT, SOURCE/'catalog.json', SOURCE/'freeze.json', SOURCE/'complete.json')},
        package={str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))},
        source_buildings=['B0', 'B2'], held_out_building='B1', seed=SEED, model_seed=MODEL_SEED,
        updates=UPDATES, schedule_sha256=digest(args.output/'schedule.json'), selected=selected,
        permissions={f: dict(rows=len(ids), positions=len({rows[i]['xy'] for i in ids}),
            row_ids_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest()) for f, ids in permitted.items()},
        planned_exposure=totals,
        source_profile='Exact empirical_target_shifts(rows,catalog) output; nonprotected same-XY fully detected paired R3 contrast pool, target-only application',
        selection='First schedule task attaining maximum context+query scans per recipe; metadata only, no RSS/score-based selection',
        checks='Two real packs only: exact identity/replay/input immutability, shared A+/C transform, official raw/aug views and standardized labels, one original augmented backward per recipe',
        resources=dict(gpu=1, cpu_threads=2, device='cuda:0', max_gpu_fraction=.8, seconds=args.seconds,
            optimizer_steps=0, backward_calls=2, localization_predictions=0, target_RSS=0, protected_RSS=0),
        limits=['Finite source shift pool is descriptive, not a persistent future S/Q shift distribution',
            'Coordinate augmentation may encourage context reading; neither memorization nor unique S necessity is established',
            'Two maximum-scan tasks give a cost proxy, not a measured whole-run bound; attention/AP/preprocessing costs vary',
            'Equal planned updates do not establish equal compute or supervision; metadata exposures are recorded',
            'No learning, target evaluation, performance gate, retry or automatic four-arm launch'])
    write(args.output/'freeze.json', dict(created=now(), contract=contract))
    return catalog, permitted, schedule, selected, totals


def assert_same(a, b):
    """Exact comparison for the two real packs, including identity dtypes."""
    if isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            assert_same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            assert_same(x, y)
    else:
        assert a == b


def transformed_pack(p, pool, spec, catalog, rows):
    before = deepcopy(p)
    kwargs = dict(seed=SEED, task_key=spec['task_key'], range_evidence='source_profile.json, empirical_target_shifts; descriptive source contrast pool')
    transform = sample_transform(p['histories'], p['event']['floor'], target_shift_pool_db=pool, **kwargs)
    augmented = apply_transform(p, transform)
    identity = sample_transform(p['histories'], p['event']['floor'], target_shift_pool_db=[0.], coordinates=False, **kwargs)
    unchanged = apply_transform(p, identity)
    unchanged.pop('augmentation')
    assert_same(p, unchanged)
    replay = apply_transform(p, json.loads(json.dumps(augmented['augmentation'])))
    assert_same(augmented, replay)
    assert_same(p, before)
    pair = [episodic_event(catalog, spec), ordinary_event(rows, catalog, spec, SEED)]
    matched = [sample_transform(e['histories'], e['floor'], target_shift_pool_db=pool, **kwargs) for e in pair]
    assert matched[0] == matched[1] == transform
    for name, h in p['histories'].items():
        assert_same(h['rss'], augmented['histories'][name]['rss'])
    for role in ('support', 'query'):
        a, b = p[role]['rss'], augmented[role]['rss']
        mask = a != 100
        assert np.array_equal(mask, b != 100)
        assert np.allclose(b[mask]-a[mask], transform['target_shift_db'], rtol=0, atol=1e-12)
    assert np.array_equal(p['aps'], augmented['aps'])
    assert_same(p['event'], augmented['event'])
    assert_same(p['query_rows'], augmented['query_rows'])
    return augmented


def fit(model, p):
    model.fit(p['histories'], p['support'], p['event']['floor'], p['aps'], p['bounds'], MODEL_SEED)
    n = sum(len(h['rss']) for h in p['histories'].values())+len(p['support']['rss'])
    assert all(e.n_samples_in_ == n and e.n_estimators == 8 for e in model.estimators)


def view_data(model, p, axis):
    x, y = model.views(p['query']['rss'], axis)
    centered = np.asarray(p['query']['xy'][:, axis]-model.origin[axis], dtype=np.float32)
    target = model.estimators[axis].y_scaler_.transform(centered[:, None]).ravel()
    return x, y, target


def measure(recipe, p, augmented, model):
    torch.cuda.reset_peak_memory_stats()
    tick = time.monotonic()
    fit(model, p)
    raw_fit_seconds = time.monotonic()-tick
    original = [view_data(model, p, axis) for axis in range(2)]
    tick = time.monotonic()
    fit(model, augmented)
    augmented_fit_seconds = time.monotonic()-tick
    differences = []
    for axis in range(2):
        current = view_data(model, augmented, axis)
        old = original[axis]
        differences.append(dict(axis=axis, raw_X_shape=list(old[0].shape), augmented_X_shape=list(current[0].shape),
            X_changed=old[0].shape != current[0].shape or not np.array_equal(old[0], current[0]),
            context_y_max_difference=float(np.max(np.abs(old[1]-current[1]))),
            query_y_max_difference=float(np.max(np.abs(old[2]-current[2])))))
        del current
    del original, old
    # Context labels are network inputs too. RSS X may stay identical when
    # the empirical pool happens to draw zero; that is not a failed rotation.
    assert any(d['X_changed'] or d['context_y_max_difference'] > 1e-6 for d in differences)
    assert any(d['query_y_max_difference'] > 1e-6 for d in differences)
    model.loaded.model_.zero_grad(set_to_none=True)
    scaler = torch.amp.GradScaler('cuda', init_scale=1024.)
    torch.cuda.synchronize()
    tick = time.monotonic()
    loss = model.backward(augmented['query']['rss'], augmented['query']['xy'], scaler)
    torch.cuda.synchronize()
    seconds = time.monotonic()-tick
    norm2 = 0.
    for parameter in model.loaded.model_.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None and bool(torch.isfinite(parameter.grad).all())
            norm2 += float((parameter.grad.double()/scaler.get_scale()).square().sum())
    assert np.isfinite(norm2) and norm2 > 0
    peak = torch.cuda.max_memory_allocated()/2**20
    model.loaded.model_.zero_grad(set_to_none=True)
    result = dict(recipe=recipe, event=p['event']['id'], task_key=p['event']['task_key'],
        context_scans=sum(len(h['rss']) for h in p['histories'].values())+len(p['support']['rss']),
        context_positions=sum(len(h['positions']) for h in p['histories'].values())+len(p['support']['positions']),
        query_scans=len(p['query']['rss']), query_positions=len(p['query']['positions']), APs=len(p['aps']),
        raw_fit_seconds=raw_fit_seconds, augmented_fit_seconds=augmented_fit_seconds,
        augmented_backward_seconds=seconds, peak_GPU_MiB=peak, source_pinball=loss,
        unscaled_gradient_norm=norm2**.5, native_view_changes=differences,
        augmentation=augmented['augmentation'], identity_exact=True, replay_exact=True,
        original_pack_unchanged=True, matched_task_transform=True, optimizer_steps=0)
    del scaler
    return result


def run(args):
    assert socket.gethostname() == 'ray-web' and torch.cuda.is_available(), 'Run only on the authorized GPU server'
    assert args.seconds > 0
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    np.random.seed(MODEL_SEED)
    torch.manual_seed(MODEL_SEED)
    catalog, permitted, schedule, selected, totals = freeze_metadata(args)
    rows = load_allowed(CSV, permitted, rss=True)
    profile = empirical_target_shifts(rows, catalog)
    write(args.output/'source_profile.json', profile)
    model = EpisodicHistoryRegressor(CHECKPOINT, device='cuda:0')
    model.enable_recompute()
    named = dict(model.loaded.model_.named_parameters())
    assert sum(p.numel() for p in named.values() if p.requires_grad) == 28544983
    assert [n for n, p in named.items() if not p.requires_grad] == ['row_interactor.tf_row.rope.freqs']
    initial = state_hash(model.loaded.model_)
    recipes = {}
    for recipe in ('ordinary', 'episodic'):
        if time.monotonic()-started >= args.seconds:
            raise TimeoutError('Preflight budget reached; no automatic continuation')
        spec = selected[recipe]['spec']
        p = (ordinary_pack(rows, catalog, spec, SEED) if recipe == 'ordinary'
             else episodic_pack(rows, catalog, spec))
        pool = profile['by_building'][spec['building']]['pool_db']
        augmented = transformed_pack(p, pool, spec, catalog, rows)
        recipes[recipe] = measure(recipe, p, augmented, model)
        assert state_hash(model.loaded.model_) == initial
        print(json.dumps(dict(recipe=recipe, backward_seconds=recipes[recipe]['augmented_backward_seconds'],
                              peak_GPU_MiB=recipes[recipe]['peak_GPU_MiB'], weights_unchanged=True)), flush=True)
        del p, augmented
        gc.collect()
        torch.cuda.empty_cache()
    estimates = {}
    for recipe, measured in recipes.items():
        n = totals[recipe]['tasks']
        estimates[recipe] = dict(tasks=n,
            raw_arm_seconds_proxy=n*(measured['raw_fit_seconds']+measured['augmented_backward_seconds']),
            augmented_arm_seconds_proxy=n*(measured['augmented_fit_seconds']+measured['augmented_backward_seconds']))
    write(args.output/'complete.json', dict(complete=True, created=now(), seconds=time.monotonic()-started,
        freeze_sha256=digest(args.output/'freeze.json'), source_profile_sha256=digest(args.output/'source_profile.json'),
        schedule_sha256=digest(args.output/'schedule.json'), weights_unchanged=True,
        initial_state_sha256=initial, final_state_sha256=state_hash(model.loaded.model_), recipes=recipes,
        budget_proxy=estimates, four_arm_seconds_proxy=sum(v['raw_arm_seconds_proxy']+v['augmented_arm_seconds_proxy'] for v in estimates.values()),
        budget_limits='Proxy uses maximum-scan task timing for all scheduled tasks, assumes raw backward cost equals measured augmented backward, excludes optimizer/validation/checkpoint/augmentation overhead; not a strict bound or execution authorization',
        protected_RSS_parsed=0, held_out_RSS_parsed=0, optimizer_steps=0, localization_predictions=0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
