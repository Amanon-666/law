"""Complete the existing frozen trained-system comparison, not a new method.

Reuse all 54 ordinary/full24/GP predictions and fixed halves. Reuse the 18
public and 3NN HS K1 unscored caches; evaluate only their missing K3/K5 cells.
Each event's eight predictions are saved before truth is decoded. The original
HS context, APs, bounds, eight views and seed remain unchanged. No training,
GP computation, checkpoint/weight selection, downloads or automatic retries.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from building_information_tab_trial import (context, digest, nearest3, now,
                                            read_scans, read_truth, write)
from tabicl_history import CHECKPOINT, PACKAGE, FrozenHistoryRegressor, state_hash

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT/'results'
ASSETS = RESULTS/'building_information_assets_20261109'
TAB = RESULTS/'building_information_tab_20261109'
REFERENCE = RESULTS/'ordinary_context_target_20261109'
OUT = RESULTS/'trained_system_validation_20261110'
SEED = 11509159
OLD_METHODS = ('ordinary', 'GP', 'full24', 'GP_ordinary_half', 'GP_full24_half')
METHODS = (*OLD_METHODS, 'public', 'GP_public_half', '3NN')
CODE = ('trained_system_validation_trial.py', 'building_information_tab_trial.py',
        'building_information_protocol.py', 'tabicl_history.py', 'system_stability.py')


def read(path):
    return json.loads(Path(path).read_text())


def initialize(args):
    manifest = read(ASSETS/'manifest.json')
    asset_done = read(ASSETS/'complete.json')
    assert asset_done['complete'] and digest(ASSETS/'manifest.json') == asset_done['manifest_sha256']
    assert manifest['seed'] == SEED and manifest['K'] == [1, 3, 5]
    assert len(manifest['events']) == 54 and len(manifest['floors']) == 9
    csv = Path(manifest['csv_path'])
    assert digest(csv) == manifest['csv_sha256']
    old = read(REFERENCE/'complete.json')
    old_contract = read(REFERENCE/'freeze.json')['contract']
    assert old['complete'] and old['freeze_sha256'] == digest(REFERENCE/'freeze.json')
    assert old_contract['events'] == manifest['events'] and old_contract['seed'] == SEED
    references = {r['event']: r for r in old['records']}
    assert set(references) == {e['id'] for e in manifest['events']}
    tab_done = read(TAB/'complete.json')
    tab_contract = read(TAB/'freeze.json')['contract']
    assert tab_done['complete'] and tab_done['freeze_sha256'] == digest(TAB/'freeze.json')
    assert tab_contract['manifest_sha256'] == digest(ASSETS/'manifest.json')
    assert tab_contract['csv_sha256'] == digest(csv) and tab_contract['seed'] == SEED
    assert str(CHECKPOINT) == tab_contract['checkpoints']['public']['path']
    assert digest(CHECKPOINT) == tab_contract['checkpoints']['public']['sha256']
    tasks = {t['id']: t for t in read(TAB/'progress.json')['tasks']}
    cached = {m: {e['id']: tasks[f'{e["id"]}_HS_{m}'] for e in manifest['events']
                   if f'{e["id"]}_HS_{m}' in tasks} for m in ('public', '3NN')}
    missing = {m: [e['id'] for e in manifest['events'] if e['id'] not in cached[m]] for m in cached}
    # Actual completed old protocol is K1 only; do not silently rerun any cell.
    expected = [e['id'] for e in manifest['events'] if e['K'] in (3, 5)]
    assert missing['public'] == missing['3NN'] == expected and len(expected) == 36
    inputs = [csv, CHECKPOINT, ASSETS/'manifest.json', ASSETS/'complete.json',
              TAB/'complete.json', TAB/'progress.json', TAB/'freeze.json',
              REFERENCE/'complete.json', REFERENCE/'freeze.json', REFERENCE/'system_stability.json']
    for item in tab_contract['checkpoints'].values():
        assert digest(item['path']) == item['sha256']
        inputs.append(Path(item['path']))
    # Preserve and check the existing trained-model selection/weight provenance.
    for path, expected_hash in old_contract['inputs'].items():
        if path.endswith('.ckpt') or 'ordinary_context_trial_20261109/' in path:
            assert digest(path) == expected_hash
            inputs.append(Path(path))
    prediction_hashes = {}
    for r in references.values():
        for pathkey, hashkey in (('prediction_path', 'prediction_sha256'), ('scored_path', 'scored_sha256')):
            assert digest(r[pathkey]) == r[hashkey]
            prediction_hashes[r[pathkey]] = r[hashkey]
    for method in cached.values():
        for r in method.values():
            path = TAB/r['prediction_path']
            assert digest(path) == r['prediction_sha256']
            prediction_hashes[str(path)] = r['prediction_sha256']
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == tab_contract['package'] == old_contract['package']
    for name in ('tabicl_history.py', 'building_information_tab_trial.py'):
        assert digest(ROOT/name) == old_contract['code'][name] == tab_contract['code'][name]
    frozen = dict(code={n: digest(ROOT/n) for n in CODE}, package=package,
        inputs={str(p): digest(p) for p in inputs}, prediction_hashes=prediction_hashes,
        events=manifest['events'], seed=SEED, information='HS', methods=list(METHODS),
        original_Tab_checkpoints=tab_contract['checkpoints'],
        source_selection=old_contract['source_selection'],
        missing_cells=missing, new_forward_count=36, new_3NN_readouts=36,
        completion='K1 public/3NN reused without inference; complete missing K3/K5 public and original same-information 3NN only',
        fusion='Same cached GP with each unchanged Tab, coordinate half-average; no model or weight selection',
        weighting='Equal building, floor, direction, K, then position; repeated scans averaged within each XY before weighted p90',
        limits=['Previously exposed B1/B2 development protocol; not blind or LOBO',
            'Across-K position-event entries repeat locations and are not independent samples; p90 is descriptive, no bootstrap or confidence claim',
            'Complete baseline attribution does not establish a new learning/fusion mechanism',
            'All predictions for each event persisted before its truth; no evaluated event changes later choices'],
        resources=dict(gpu=1, cpu_threads=2, device='cuda:0', segment_seconds=args.seconds,
            boundary='Complete event, explicit resume; failures stop without retry', training_updates=0, GP_fits=0, downloads=0))
    if args.resume:
        assert read(args.output/'freeze.json')['contract'] == frozen
        progress = read(args.output/'progress.json')
        assert progress['freeze_sha256'] == digest(args.output/'freeze.json')
        assert not progress.get('complete', False), 'Already complete'
        for r in progress['records']:
            assert digest(r['prediction_path']) == r['prediction_sha256']
            assert digest(r['scored_path']) == r['scored_sha256']
    else:
        args.output.mkdir(exist_ok=False)
        write(args.output/'freeze.json', dict(created=now(), contract=frozen))
        progress = dict(freeze_sha256=digest(args.output/'freeze.json'), records=[], segments=[])
        write(args.output/'progress.json', progress)
    return manifest, csv, references, cached, old, progress


def summarize(manifest, records):
    """Same position-mean errors and weighted empirical p90 as system_stability."""
    events = {e['id']: e for e in manifest['events']}
    samples = []
    for record in records:
        event = events[record['event']]
        with np.load(record['scored_path']) as saved:
            _, group = np.unique(saved['truth'], axis=0, return_inverse=True)
            n = int(group.max())+1
            errors = {m: np.array([saved[m][group == i].mean() for i in range(n)]) for m in METHODS}
        floor_count = sum(f.startswith(event['building']) for f in manifest['floors'])
        samples.append(dict(event=event, errors=errors, weight=.5/floor_count/2/n))

    def aggregate(subset, method):
        values = np.concatenate([s['errors'][method] for s in subset])
        weights = np.concatenate([np.full(len(s['errors'][method]), s['weight']) for s in subset])
        weights /= weights.sum()
        order = np.argsort(values)
        return dict(mean=float(weights@values), position_mean_p90=float(
            values[order][np.searchsorted(np.cumsum(weights[order]), .9)]))

    def cells(key, values, method):
        result = {}
        for value in values:
            subset = [s for s in samples if s['event'][key] == value]
            result[str(value)] = dict(**aggregate(subset, method), by_K={str(k): aggregate(
                [s for s in subset if s['event']['K'] == k], method) for k in (1, 3, 5)})
        return result

    return {m: dict(**aggregate(samples, m),
        by_K={str(k): aggregate([s for s in samples if s['event']['K'] == k], m) for k in (1, 3, 5)},
        by_floor=cells('floor', manifest['floors'], m),
        by_building=cells('building', manifest['buildings'], m)) for m in METHODS}


@torch.no_grad()
def run(args):
    started = time.monotonic()
    manifest, csv, references, cached, old, progress = initialize(args)
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    model = None
    done = {r['event'] for r in progress['records']}
    initial_count = len(done)
    for event in manifest['events']:
        eid = event['id']
        if eid in done:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        histories, support, rss, xy, aps, bounds = context(csv, manifest, event, 'HS')
        with np.load(references[eid]['prediction_path']) as saved:
            assert np.array_equal(saved['query_rows'], event['query_rows'])
            assert np.array_equal(saved['aps'], aps) and np.array_equal(saved['bounds'], bounds)
            predictions = {m: saved[m].copy() for m in OLD_METHODS}
        if event['K'] == 1:
            for method in ('public', '3NN'):
                with np.load(TAB/cached[method][eid]['prediction_path']) as saved:
                    assert np.array_equal(saved['query_rows'], event['query_rows'])
                    assert np.array_equal(saved['aps'], aps) and np.array_equal(saved['bounds'], bounds)
                    predictions[method] = saved['projected'].copy()
        else:
            if model is None:
                model = FrozenHistoryRegressor(CHECKPOINT, device='cuda:0')
                model.loaded.model_.requires_grad_(False)
                before = state_hash(model.loaded.model_)
                if 'model_state_before' in progress:
                    assert progress['model_state_before'] == before
                progress['model_state_before'] = before
            model.fit(histories, support, event['floor'], aps, bounds, SEED)
            qrss = read_scans(csv, event['query_rows'], False)['rss']
            predictions['public'] = model.predict(qrss)['projected']
            predictions['3NN'] = nearest3(rss, xy, qrss, aps, bounds)['projected']
            after = state_hash(model.loaded.model_)
            assert after == before, 'Public model weights changed'
            progress['model_state_after'] = after
        predictions['GP_public_half'] = .5*(predictions['GP']+predictions['public'])
        path = args.output/f'{eid}_unscored.npz'
        np.savez_compressed(path, **predictions, query_rows=event['query_rows'], aps=aps, bounds=bounds)
        # Only now decode current or cached truth; every event method is sealed.
        truth = read_truth(csv, event['query_rows'])
        _, group, count = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(count)*count[group])
        errors = {m: np.linalg.norm(predictions[m]-truth, axis=1) for m in METHODS}
        with np.load(references[eid]['scored_path']) as saved:
            assert np.array_equal(saved['truth'], truth) and np.array_equal(saved['weights'], weights)
            for m in OLD_METHODS:
                assert np.allclose(errors[m], saved[m], rtol=0, atol=1e-10)
        scored = args.output/f'{eid}_scored.npz'
        np.savez_compressed(scored, truth=truth, weights=weights, **errors)
        progress['records'].append(dict(event=eid, floor=event['floor'], building=event['building'],
            direction=event['direction'], K=event['K'], context_rows=len(rss), APs=len(aps),
            mde={m: float(weights@errors[m]) for m in METHODS}, seconds=time.monotonic()-tick,
            prediction_path=str(path), prediction_sha256=digest(path),
            scored_path=str(scored), scored_sha256=digest(scored)))
        write(args.output/'progress.json', progress)
        done.add(eid)
        print(json.dumps(dict(event=eid, completed=len(done), total=54,
                              seconds=time.monotonic()-tick)), flush=True)
    if model is not None:
        after = state_hash(model.loaded.model_)
        assert after == before, 'Public model weights changed'
        progress['model_state_after'] = after
    complete = len(done) == 54
    progress['segments'].append(dict(created=now(), seconds=time.monotonic()-started,
        budget=args.seconds, completed=len(done)-initial_count, weights_unchanged=True))
    write(args.output/'progress.json', progress)
    if not complete:
        print(json.dumps(dict(complete=False, stop='segment_budget', explicit_resume_required=True)), flush=True)
        return 2
    summary = summarize(manifest, progress['records'])
    for m in OLD_METHODS:
        assert abs(summary[m]['mean']-old['summary'][m]['mean']) <= 1e-9
    stability = read(REFERENCE/'system_stability.json')
    for m in ('public', 'GP_public_half'):
        for metric in ('mean', 'position_mean_p90'):
            assert abs(summary[m]['by_K']['1'][metric]-stability['by_K']['1'][m][metric]) <= 1e-9
    progress['complete'] = True
    write(args.output/'progress.json', progress)
    write(args.output/'complete.json', dict(complete=True, created=now(),
        freeze_sha256=progress['freeze_sha256'], summary=summary, records=progress['records'],
        segments=progress['segments'], model_state_before=progress['model_state_before'],
        model_state_after=progress['model_state_after'], reused_K1=True, public_new_forwards=36,
        new_3NN_readouts=36, training_updates=0, GP_fits=0, target_selection=False, blind_test=False,
        scope='Completion of existing trained-system development validation, no new structure',
        position_unit='Scan errors averaged within XY; across-K repetitions are descriptive entries, not independent samples'))
    print(json.dumps(dict(complete=True, means={m: summary[m]['mean'] for m in METHODS})), flush=True)
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--output', type=Path, default=OUT)
    raise SystemExit(run(parser.parse_args()))
