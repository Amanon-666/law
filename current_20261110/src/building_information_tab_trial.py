"""Resumable, fixed-public/full24/3NN H/S/HS building-information comparison.

No training, calibration, retrieval, automatic downloads or query-fitted
transforms. A segment stops after a complete context task once its wall budget
is reached; explicit --resume is required to continue the unchanged freeze.
"""
import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from tabicl_history import CHECKPOINT, PACKAGE, FrozenHistoryRegressor, state_hash

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/'results/building_information_assets_20261109'
SOURCE = ROOT/'results/tabicl_multitask_full_trial_20261107_checked'
OUT = ROOT/'results/building_information_tab_20261109'
SEED = 11509159
METHODS = ('public', 'full24', '3NN')
INFORMATION = ('H', 'S', 'HS')


def digest(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def write(path, obj):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')
    temporary.replace(path)


def read_scans(csv_path, ids, coordinates):
    """Interpret only requested rows; query coordinates are a separate read."""
    wanted = set(ids)
    selected = {}
    with open(csv_path, newline='') as stream:
        for row_id, row in enumerate(csv.DictReader(stream), 1):
            if row_id in wanted:
                value = dict(rss=[int(row[f'WAP{a:03}']) for a in range(1, 521)])
                if coordinates:
                    value['xy'] = [float(row['LONGITUDE']), float(row['LATITUDE'])]
                selected[row_id] = value
    assert set(selected) == wanted
    result = dict(rss=np.asarray([selected[i]['rss'] for i in ids], dtype=np.float64))
    if coordinates:
        result['xy'] = np.asarray([selected[i]['xy'] for i in ids], dtype=np.float64)
    return result


def read_truth(csv_path, ids):
    wanted = set(ids)
    selected = {}
    with open(csv_path, newline='') as stream:
        for row_id, row in enumerate(csv.DictReader(stream), 1):
            if row_id in wanted:
                selected[row_id] = [float(row['LONGITUDE']), float(row['LATITUDE'])]
    assert set(selected) == wanted
    return np.asarray([selected[i] for i in ids], dtype=np.float64)


def tasks(manifest, max_k):
    """H is fitted once per floor/method; both directions and all K reuse it."""
    result = []
    for k in (value for value in (1, 3, 5) if value <= max_k):
        for floor in manifest['floors']:
            events = [e for e in manifest['events'] if e['floor'] == floor and e['K'] <= max_k]
            for info in (INFORMATION if k == 1 else ('S', 'HS')):
                groups = [events] if info == 'H' else [[event] for event in events if event['K'] == k]
                for group in groups:
                    key = floor if info == 'H' else group[0]['id']
                    for method in METHODS:
                        result.append(dict(id=f'{key}_{info}_{method}', method=method,
                                           information=info, events=[e['id'] for e in group]))
    return result


def context(csv_path, manifest, event, info):
    histories = ({name: read_scans(csv_path, manifest['history_rows'][name], True)
                  for name in event['histories']} if info in ('H', 'HS') else {})
    support = (read_scans(csv_path, event['support_rows'], True) if info in ('S', 'HS')
               else dict(rss=np.empty((0, 520)), xy=np.empty((0, 2))))
    parts = list(histories.values())+[support]
    rss = np.concatenate([part['rss'] for part in parts])
    xy = np.concatenate([part['xy'] for part in parts])
    aps = np.flatnonzero((rss != 100).any(0))
    bounds = np.stack((np.floor(xy.min(0)/2)*2, np.ceil(xy.max(0)/2)*2))
    return histories, support, rss, xy, aps, bounds


def nearest3(rss, xy, qrss, aps, bounds):
    """Exact existing 3-WKNN rule, separated from truth-dependent scoring."""
    a = np.where(rss[:, aps] == 100, -110, rss[:, aps])
    b = np.where(qrss[:, aps] == 100, -110, qrss[:, aps])
    predictions = []
    for row in b:
        distance = np.sqrt(((row-a)**2).mean(-1))
        selected = np.argsort(distance, kind='stable')[:3]
        weights = 1/np.maximum(distance[selected], 1e-8)
        weights /= weights.sum()
        predictions.append(weights@xy[selected])
    raw = np.asarray(predictions)
    return dict(raw=raw, projected=np.clip(raw, bounds[0], bounds[1]))


def score_saved(csv_path, output, task, prediction_path, event_map):
    """Open query truth only after the unscored prediction file exists."""
    records = []
    prediction_sha256 = digest(prediction_path)
    with np.load(prediction_path) as saved:
        row_ids = saved['query_rows'].tolist()
        lookup = {row_id: i for i, row_id in enumerate(row_ids)}
        truth = read_truth(csv_path, row_ids)
        for eid in task['events']:
            event = event_map[eid]
            indices = np.asarray([lookup[i] for i in event['query_rows']])
            y = truth[indices]
            positions, group, count = np.unique(y, axis=0, return_inverse=True, return_counts=True)
            weights = 1/(len(positions)*count[group])
            raw, projected = saved['raw'][indices], saved['projected'][indices]
            errors = np.linalg.norm(projected-y, axis=1)
            raw_errors = np.linalg.norm(raw-y, axis=1)
            name = f'{eid}_{task["information"]}_{task["method"]}.npz'
            path = output/name
            np.savez_compressed(path, raw=raw, projected=projected, truth=y,
                                errors=errors, raw_errors=raw_errors, weights=weights,
                                query_rows=event['query_rows'], group=group,
                                aps=saved['aps'], bounds=saved['bounds'])
            records.append(dict(event=eid, building=event['building'], floor=event['floor'],
                                direction=event['direction'], K=event['K'],
                                method=task['method'], information=task['information'],
                                path=name, sha256=digest(path), mde=float(weights@errors),
                                raw_mde=float(weights@raw_errors)))
    return prediction_sha256, records


def aggregate(records, budgets=(1, 3, 5)):
    summary = {}
    for k in budgets:
        summary[str(k)] = {}
        for method in METHODS:
            for info in INFORMATION:
                selected = [r for r in records if (r['K'], r['method'], r['information']) == (k, method, info)]
                assert len(selected) == 18
                floors = {floor: float(np.mean([r['mde'] for r in selected if r['floor'] == floor]))
                          for floor in sorted({r['floor'] for r in selected})}
                buildings = {b: float(np.mean([v for f, v in floors.items() if f.startswith(b)]))
                             for b in ('B1', 'B2')}
                summary[str(k)][f'{method}_{info}'] = dict(
                    equal_building=float(np.mean(list(buildings.values()))),
                    by_building=buildings, by_floor=floors)
    return summary


@torch.no_grad()
def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    manifest_path = ASSETS/'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    csv_path = Path(manifest['csv_path'])
    assert digest(csv_path) == manifest['csv_sha256']
    assert manifest['seed'] == SEED and len(manifest['events']) == 54
    assert len(manifest['floors']) == 9 and manifest['K'] == [1, 3, 5]
    event_map = {e['id']: e for e in manifest['events']}
    source = json.loads((SOURCE/'complete.json').read_text())
    source_freeze = json.loads((SOURCE/'freeze.json').read_text())
    assert source['complete'] and source['formal_step'] == 24
    assert digest(SOURCE/'freeze.json') == source['freeze_sha256']
    checkpoints = dict(public=Path(CHECKPOINT), full24=SOURCE/'full_step_24.ckpt')
    assert digest(checkpoints['full24']) == source['checkpoint_sha256']['24']
    assert digest(checkpoints['public']) == source_freeze['inputs'][str(checkpoints['public'])]
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == source_freeze['package']
    budgets = [k for k in (1, 3, 5) if k <= args.max_k]
    plan = tasks(manifest, args.max_k)
    frozen = dict(seed=SEED, manifest_sha256=digest(manifest_path), csv_sha256=digest(csv_path),
                  checkpoints={name: dict(path=str(path), sha256=digest(path)) for name, path in checkpoints.items()},
                  source_complete_sha256=digest(SOURCE/'complete.json'), package=package,
                  code={name: digest(ROOT/name) for name in (
                      'building_information_tab_trial.py', 'building_information_protocol.py',
                      'tabicl_history.py', 'recomposition_multibudget_natural.py')},
                  plan=plan, evaluated_K=budgets, max_k=args.max_k,
                  segment_seconds=args.seconds, device='cuda:0', cpu_threads=2,
                  consumer='Existing FrozenHistoryRegressor: physical source-floor roles, 8 views per axis, fixed weights',
                  coordinates='Absolute UJI XY; each context independently determines internal center and 2-unit floor/ceil bounds',
                  nearest3='Existing inverse-distance 3-WKNN; missing=-110; stable ties; same legal APs/bounds; no floor feature',
                  constant_labels='Original installed regressor uses StandardScaler for y, without a constant-y shortcut; real K1 S rows follow that API. No forced anchor output, synthetic rows or exception fallback.',
                  query_order='Fit legal context -> read query RSS -> persist unscored predictions -> read truth -> score',
                  reverse_audit=['H-only never reads S for APs, bounds, transforms or model input',
                                 'S-only never reads H for APs, bounds, transforms or model input',
                                 'HS includes all same-building other-floor H and current S once',
                                 'H context is shared across AB/BA and K; unique query RSS forward-predicted once per floor/method',
                                 'Frozen public/full24 weights; no optimizer, GP calibration, retrieval or automatic download'],
                  limits=['B0-trained -> B1/B2 development comparison, not three-fold LOBO or blind testing',
                          'Public foundation pretraining has not been audited',
                          'Constant/small S contexts follow the existing API, not extrapolation backed by H',
                          'Shared queries and nested support are dependent; aggregation does not create more buildings',
                          'Budget checked at task boundaries; an in-flight task may overrun, recorded without automatic extension'])
    output = args.output
    if args.resume:
        previous = json.loads((output/'freeze.json').read_text())
        assert previous['contract'] == frozen
        progress = json.loads((output/'progress.json').read_text())
        assert progress['freeze_sha256'] == digest(output/'freeze.json')
        for result in progress['tasks']:
            assert digest(output/result['prediction_path']) == result['prediction_sha256']
            for record in result['records']:
                assert digest(output/record['path']) == record['sha256']
    else:
        output.mkdir(parents=True, exist_ok=False)
        write(output/'freeze.json', dict(created=now(), contract=frozen))
        progress = dict(freeze_sha256=digest(output/'freeze.json'), tasks=[], segments=[])
        write(output/'progress.json', progress)
    done = {record['id'] for record in progress['tasks']}
    models = {}
    initial_hashes = {}
    segment_start = len(done)
    for task in plan:
        if task['id'] in done:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        event = event_map[task['events'][0]]
        histories, support, rss, xy, aps, bounds = context(csv_path, manifest, event, task['information'])
        method = task['method']
        if method != '3NN':
            if method not in models:
                models[method] = FrozenHistoryRegressor(checkpoints[method], device='cuda:0')
                models[method].loaded.model_.requires_grad_(False)
                initial_hashes[method] = state_hash(models[method].loaded.model_)
            model = models[method]
            model.fit(histories, support, event['floor'], aps, bounds, SEED)
        query_ids = list(dict.fromkeys(i for eid in task['events'] for i in event_map[eid]['query_rows']))
        query = read_scans(csv_path, query_ids, False)['rss']
        predictions = (nearest3(rss, xy, query, aps, bounds) if method == '3NN' else model.predict(query))
        assert np.isfinite(predictions['raw']).all() and np.isfinite(predictions['projected']).all()
        prediction_path = output/f'{task["id"]}_unscored.npz'
        np.savez_compressed(prediction_path, **predictions, query_rows=query_ids, aps=aps, bounds=bounds)
        prediction_sha256, records = score_saved(csv_path, output, task, prediction_path, event_map)
        progress['tasks'].append(dict(id=task['id'], seconds=time.monotonic()-tick,
                                     context_rows=len(rss), APs=len(aps),
                                     prediction_path=prediction_path.name,
                                     prediction_sha256=prediction_sha256, records=records))
        done.add(task['id'])
        write(output/'progress.json', progress)
        print(json.dumps(dict(task=task['id'], completed=len(done), total=len(plan),
                              elapsed_seconds=time.monotonic()-started)), flush=True)
    for method, model in models.items():
        assert state_hash(model.loaded.model_) == initial_hashes[method]
    elapsed = time.monotonic()-started
    complete = len(done) == len(plan)
    progress['segments'].append(dict(ended=now(), wall_seconds=elapsed, budget_seconds=args.seconds,
                                     tasks_completed=len(done)-segment_start, total_completed=len(done),
                                     complete=complete, budget_overrun=elapsed > args.seconds,
                                     pid=os.getpid(), weights_unchanged=True))
    records = [r for task in progress['tasks'] for r in task['records']]
    completed_budgets = [k for k in budgets if sum(r['K'] == k for r in records) == 18*len(METHODS)*len(INFORMATION)]
    progress['complete'] = complete
    progress['completed_budgets'] = completed_budgets
    progress['budget_summaries'] = aggregate(records, completed_budgets)
    write(output/'progress.json', progress)
    if complete:
        assert len(records) == 18*len(budgets)*len(METHODS)*len(INFORMATION)
        write(output/'complete.json', dict(complete=True, created=now(),
              freeze_sha256=digest(output/'freeze.json'), evaluated_K=budgets,
              full_manifest_event_count=54, records=records, summary=aggregate(records, budgets),
              task_count=len(plan), segments=progress['segments'], training_updates=0,
              GP_fits=0, blind_test=False))
    else:
        print(json.dumps(dict(complete=False, stop='segment_budget', completed=len(done),
                              total=len(plan), explicit_resume_required=True)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=600.)
    parser.add_argument('--max-k', type=int, choices=(1, 3, 5), default=1)
    parser.add_argument('--output', type=Path, default=OUT)
    run(parser.parse_args())
