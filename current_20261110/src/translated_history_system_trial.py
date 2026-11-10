"""One frozen-system comparison using the supplied B0-only translated maps.

For every original UJI event, ordinary30 reads either the continuous learned
HS map or its unlearned nearest-floor reference map, plus the same real S.
Original GP and ordinary-system predictions are reused. The prototype's 3NN
points and their GP halves require no additional model. The fixed halves are
declared here, before results; no checkpoint, weight or context selection.

Maps were learned using all four B0 floors in the supplied prototype. This
run measures integration of that asset; the original ordinary model used the
three-floor restricted corpus. It is not an equal-source training comparison.
No new training, GP fitting, generated-scan likelihood or downloads occur.
For each event predictions are persisted before reading its query truth.
"""
import argparse
import gc
import json
import socket
import time
from pathlib import Path

import numpy as np
import torch

from building_information_tab_trial import context, digest, now, read_scans, read_truth, write
from tabicl_history import FrozenHistoryRegressor, state_hash
from translated_history_context import translated_history_context

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT/'results/building_information_assets_20261109'
REFERENCE = ROOT/'results/history_training_attribution_20261110'
SOURCE = ROOT/'results/ordinary_context_trial_20261109'
SEED = 11509159
REUSED = ('GP', 'ordinary_HS', 'GP_ordinary_HS_half')
NEW = ('translated_3NN', 'naive_3NN', 'Tab_translated', 'Tab_naive')
METHODS = (*REUSED, *NEW, *(f'GP_{m}_half' for m in NEW))


def read(path):
    return json.loads(Path(path).read_text())


def summarize(manifest, records):
    """Original equal-building/floor/direction/K, position-mean MDE and p90."""
    events = {e['id']: e for e in manifest['events']}
    samples = []
    for record in records:
        event = events[record['event']]
        with np.load(record['scored_path']) as saved:
            _, groups = np.unique(saved['truth'], axis=0, return_inverse=True)
            n = int(groups.max())+1
            errors = {m: np.array([saved[m][groups == i].mean() for i in range(n)]) for m in METHODS}
        floors = sum(f.startswith(event['building']) for f in manifest['floors'])
        samples.append(dict(event=event, errors=errors, weight=.5/floors/2/n))

    def aggregate(selected, method):
        values = np.concatenate([s['errors'][method] for s in selected])
        weights = np.concatenate([np.full(len(s['errors'][method]), s['weight']) for s in selected])
        weights /= weights.sum()
        order = np.argsort(values)
        return dict(mean=float(weights@values), position_mean_p90=float(
            values[order][np.searchsorted(np.cumsum(weights[order]), .9)]))

    results = {}
    for method in METHODS:
        results[method] = aggregate(samples, method)
        for key, choices in (('K', (1, 3, 5)), ('floor', manifest['floors']), ('building', manifest['buildings'])):
            results[method]['by_'+key] = {str(c): aggregate(
                [s for s in samples if s['event'][key] == c], method) for c in choices}
    return results


@torch.no_grad()
def run(args):
    started = time.monotonic()
    torch.set_num_threads(2)
    manifest = read(ASSETS/'manifest.json')
    csv = Path(manifest['csv_path'])
    assert digest(csv) == manifest['csv_sha256']
    assert manifest['seed'] == SEED and len(manifest['events']) == 54
    source = read(SOURCE/'complete.json')
    checkpoint = Path(source['selected_checkpoint'])
    assert source['complete'] and source['selected_step'] == 30
    assert digest(checkpoint) == source['selected_checkpoint_sha256']
    old = read(REFERENCE/'complete.json')
    assert old['complete'] and old['freeze_sha256'] == digest(REFERENCE/'freeze.json')
    refs = {r['event']: r for r in old['records']}
    generated = read(args.maps/'complete.json')
    assert generated['complete']
    # Hash the actual handed-over assets, then bind every event's rows/APs/bounds
    # at consumption. Source-code/fit provenance remains with map generation.
    input_paths = [args.maps/'complete.json', args.maps/'freeze.json', SOURCE/'complete.json',
        ASSETS/'manifest.json', REFERENCE/'complete.json', REFERENCE/'freeze.json', checkpoint, csv]
    for event in manifest['events']:
        path = Path(refs[event['id']]['prediction_path'])
        assert digest(path) == refs[event['id']]['prediction_sha256']
        input_paths += [path, args.maps/f'{event["id"]}.npz']
    contract = dict(inputs={str(p): digest(p) for p in input_paths},
        code={n: digest(ROOT/n) for n in ('translated_history_system_trial.py',
            'translated_history_context.py', 'tabicl_history.py', 'building_information_tab_trial.py')},
        events=manifest['events'], methods=list(METHODS), seed=SEED,
        primary='GP_Tab_translated_half',
        primary_comparators=['GP_ordinary_HS_half', 'GP_Tab_naive_half'],
        contexts='Translated HS map or naive reference, target-floor feature, real S once; continuous -105 encoding',
        model='Unchanged selected ordinary30; original eight views and original H/S bounds/APs',
        fusion='Original cached GP, fixed half with every declared point readout',
        source_scope='Translator: supplied full-four-floor B0 prototype. Ordinary30: existing restricted three-floor source corpus.',
        training_updates=0, GP_fits=0, target_selection=False, downloads=0)
    if args.resume:
        assert read(args.output/'freeze.json')['contract'] == contract
        progress = read(args.output/'progress.json')
        assert not progress.get('complete', False)
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        write(args.output/'freeze.json', dict(created=now(), contract=contract))
        progress = dict(freeze_sha256=digest(args.output/'freeze.json'), records=[], segments=[])
        write(args.output/'progress.json', progress)
    completed = {r['event'] for r in progress['records']}
    np.random.seed(SEED); torch.manual_seed(SEED)
    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
    model.loaded.model_.requires_grad_(False)
    initial_state = state_hash(model.loaded.model_)
    for event in manifest['events']:
        eid = event['id']
        if eid in completed:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        histories, support, _, _, aps, bounds = context(csv, manifest, event, 'HS')
        with np.load(args.maps/f'{eid}.npz') as saved:
            assert np.array_equal(saved['query_rows'], event['query_rows'])
            assert np.array_equal(saved['aps'], aps) and np.array_equal(saved['bounds'], bounds)
            xy = saved['cand_xy'].copy()
            maps = dict(Tab_translated=saved['map_HS'].copy(), Tab_naive=saved['naive_ref'].copy())
            points = dict(translated_3NN=saved['pred_3nn_HS'].copy(), naive_3NN=saved['pred_3nn_naive'].copy())
        with np.load(refs[eid]['prediction_path']) as saved:
            assert np.array_equal(saved['query_rows'], event['query_rows'])
            assert np.array_equal(saved['HS_aps'], aps) and np.array_equal(saved['HS_bounds'], bounds)
            points.update({m: saved[m].copy() for m in REUSED})
        qrss = read_scans(csv, event['query_rows'], False)['rss']
        for name, values in maps.items():
            translated = translated_history_context(xy, values, aps, event['floor'], n_ap=qrss.shape[1])
            model.fit(translated, support, event['floor'], aps, bounds, SEED)
            points[name] = model.predict(qrss)['projected']
        for name in NEW:
            points[f'GP_{name}_half'] = .5*(points['GP']+points[name])
        path = args.output/f'{eid}_unscored.npz'
        np.savez_compressed(path, **points, query_rows=event['query_rows'], aps=aps, bounds=bounds)
        truth = read_truth(csv, event['query_rows'])
        _, groups, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(counts)*counts[groups])
        errors = {m: np.linalg.norm(points[m]-truth, axis=1) for m in METHODS}
        scored = args.output/f'{eid}_scored.npz'
        np.savez_compressed(scored, truth=truth, weights=weights, **errors)
        progress['records'].append(dict(event=eid, floor=event['floor'], K=event['K'],
            translated_rows=len(xy), seconds=time.monotonic()-tick,
            prediction_path=str(path), prediction_sha256=digest(path),
            scored_path=str(scored), scored_sha256=digest(scored),
            mde={m: float(weights@errors[m]) for m in METHODS}))
        write(args.output/'progress.json', progress)
        print(json.dumps(dict(event=eid, completed=len(progress['records']))), flush=True)
        del histories, maps, translated, qrss
        gc.collect()
    final_state = state_hash(model.loaded.model_)
    assert final_state == initial_state
    progress['segments'].append(dict(created=now(), seconds=time.monotonic()-started, budget=args.seconds))
    progress['complete'] = len(progress['records']) == 54
    write(args.output/'progress.json', progress)
    if not progress['complete']:
        return 2
    summary = summarize(manifest, progress['records'])
    for method in REUSED:
        assert abs(summary[method]['mean']-old['summary'][method]['mean']) <= 1e-9
    write(args.output/'complete.json', dict(progress, summary=summary,
        model_state_before=initial_state, model_state_after=final_state,
        training_updates=0, GP_fits=0, new_Tab_forwards=108, target_selection=False))
    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--maps', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    assert socket.gethostname() == 'ray-web' and torch.cuda.is_available() and args.seconds > 0
    torch.cuda.set_per_process_memory_fraction(.8, 0)
    raise SystemExit(run(args))
