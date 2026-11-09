"""Frozen source-selected ordinary Tab on the existing 54 HS target events.

No source completion means no target run. The source-selected checkpoint is
used unchanged; no target checkpoint/weight selection, learning or GP fitting.
Existing same-event retained GP and full24 coordinates are read from the
completed position-fusion assets. Two additional readouts are fixed coordinate
half-averages with that SAME cached GP, adding no inference or selected weight.
Predictions are persisted before query truth or cached scored arrays are read.
Aggregation is equal position, direction, floor, building, then K. These are
previously exposed development buildings, not blind or LOBO evaluation.
"""
import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from building_information_tab_trial import context, digest, now, read_scans, read_truth, write
from tabicl_history import PACKAGE, FrozenHistoryRegressor, state_hash

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'results/ordinary_context_trial_20261109'
ASSETS = ROOT/'results/building_information_assets_20261109'
REFERENCE = ROOT/'results/position_fusion_20261109_continued'
OUT = ROOT/'results/ordinary_context_target_20261109'
SEED = 11509159
METHODS = ('ordinary', 'GP', 'full24', 'GP_ordinary_half', 'GP_full24_half')


def read(path):
    return json.loads(Path(path).read_text())


def summarize(records, manifest):
    assert len(records) == 54
    result = {}
    for method in METHODS:
        cells = {floor: {} for floor in manifest['floors']}
        buildings, by_k = {}, {}
        for k in (1, 3, 5):
            for floor in manifest['floors']:
                selected = [r for r in records if r['K'] == k and r['floor'] == floor]
                assert len(selected) == 2 and {r['direction'] for r in selected} == {'AB', 'BA'}
                cells[floor][str(k)] = float(np.mean([r['mde'][method] for r in selected]))
            buildings[str(k)] = {b: float(np.mean([cells[f][str(k)] for f in manifest['floors']
                                                  if f.startswith(b)])) for b in manifest['buildings']}
            by_k[str(k)] = float(np.mean(list(buildings[str(k)].values())))
        result[method] = dict(mean=float(np.mean(list(by_k.values()))), by_K=by_k,
            by_building=buildings, cells=cells,
            by_floor={f: float(np.mean(list(values.values()))) for f, values in cells.items()})
    return result


@torch.no_grad()
def run(args):
    started = time.monotonic()
    if not (SOURCE/'complete.json').exists():
        raise RuntimeError('Ordinary source selection is incomplete; target evaluation must not start')
    source = read(SOURCE/'complete.json')
    assert source['complete'] and source['target_opened'] is False
    assert source['freeze_sha256'] == digest(SOURCE/'freeze.json')
    source_contract = read(SOURCE/'freeze.json')['contract']
    source_progress = read(SOURCE/'progress.json')
    assert source_progress['complete'] and source_progress['freeze_sha256'] == source['freeze_sha256']
    assert len(source_contract['validation']) == 27
    assert all(len(source_progress['validation'][str(step)]) == 27 for step in (15, 30))
    checkpoint = ROOT/Path(source['selected_checkpoint'])
    assert digest(checkpoint) == source['selected_checkpoint_sha256']
    manifest_path = ASSETS/'manifest.json'
    manifest = read(manifest_path)
    asset_complete = read(ASSETS/'complete.json')
    assert asset_complete['complete'] and digest(manifest_path) == asset_complete['manifest_sha256']
    assert manifest['seed'] == SEED and manifest['K'] == [1, 3, 5]
    assert len(manifest['events']) == 54 and len(manifest['floors']) == 9
    csv_path = Path(manifest['csv_path'])
    assert digest(csv_path) == manifest['csv_sha256']
    reference_complete = read(REFERENCE/'target/complete.json')
    assert reference_complete['complete'] and reference_complete['phase'] == 'target'
    assert reference_complete['freeze_sha256'] == digest(REFERENCE/'freeze.json')
    reference = {r['event']: r for r in reference_complete['records']}
    assert len(reference) == 54 and set(reference) == {e['id'] for e in manifest['events']}
    reference_hashes = {}
    for event in manifest['events']:
        record = reference[event['id']]
        assert record['floor'] == event['floor'] and record['K'] == event['K']
        path = ROOT/record['prediction_path']
        scored = path.parent/f'{event["id"]}_scored.npz'
        assert digest(path) == record['prediction_sha256'] and digest(scored) == record['scored_sha256']
        reference_hashes[str(path)] = record['prediction_sha256']
        reference_hashes[str(scored)] = record['scored_sha256']
    package = {str(p.relative_to(PACKAGE)): digest(p) for p in sorted((PACKAGE/'tabicl').rglob('*.py'))}
    assert package == source_contract['package']
    for name in ('tabicl_history.py', 'building_information_tab_trial.py'):
        assert digest(ROOT/name) == source_contract['code'][name]
    frozen = dict(code={name: digest(ROOT/name) for name in ('ordinary_context_target_trial.py',
                  'tabicl_history.py', 'building_information_tab_trial.py')}, package=package,
        inputs={str(p): digest(p) for p in (checkpoint, csv_path, manifest_path,
                 ASSETS/'complete.json', SOURCE/'complete.json', SOURCE/'freeze.json', SOURCE/'progress.json',
                 REFERENCE/'freeze.json', REFERENCE/'target/complete.json')},
        reference_predictions=reference_hashes, events=manifest['events'], seed=SEED, information='HS',
        source_selection=dict(selected_step=source['selected_step'], checkpoint=str(checkpoint),
            freeze_sha256=source['freeze_sha256'], summaries=source['selection'],
            validation=source_contract['validation'], validation_records=source_progress['validation'],
            public_reference_inputs=source_contract['reference_predictions']),
        methods=list(METHODS), fusion='Fixed coordinate half-average of projected Tab and same cached GP; no weight selection',
        resources=dict(device='cuda:0', gpu=1, cpu_threads=2, segment_seconds=args.seconds,
                       stop_boundary='Complete target event'),
        boundaries=['Source complete and selected checkpoint sealed before target access',
                    'Original HS fit and physical-floor roles; no calibration or context retrieval',
                    'Persist predictions before reading current query truth or cached scored arrays',
                    'No network updates, GP recomputation, new arms or target selection',
                    'Previously exposed B1/B2 development data; not blind/LOBO',
                    'Explicit resume only; no retry or automatic budget extension'])
    out = args.output
    if args.resume:
        assert read(out/'freeze.json')['contract'] == frozen
        progress = read(out/'progress.json')
        assert progress['freeze_sha256'] == digest(out/'freeze.json')
        for record in progress['records']:
            assert digest(record['prediction_path']) == record['prediction_sha256']
            assert digest(record['scored_path']) == record['scored_sha256']
    else:
        out.mkdir(parents=True, exist_ok=False)
        write(out/'freeze.json', dict(created=now(), contract=frozen))
        progress = dict(records=[], segments=[], freeze_sha256=digest(out/'freeze.json'))
        write(out/'progress.json', progress)
    if progress.get('complete'):
        raise RuntimeError('Target evaluation already complete; do not rerun')
    torch.set_num_threads(2)
    torch.set_default_dtype(torch.float32)
    np.random.seed(SEED); torch.manual_seed(SEED)
    model = FrozenHistoryRegressor(checkpoint, device='cuda:0')
    model.loaded.model_.requires_grad_(False)
    initial_state = state_hash(model.loaded.model_)
    done = {r['event'] for r in progress['records']}
    for event in manifest['events']:
        eid = event['id']
        if eid in done:
            continue
        if time.monotonic()-started >= args.seconds:
            break
        tick = time.monotonic()
        histories, support, _, _, aps, bounds = context(csv_path, manifest, event, 'HS')
        model.fit(histories, support, event['floor'], aps, bounds, SEED)
        qrss = read_scans(csv_path, event['query_rows'], False)['rss']
        prediction = model.predict(qrss)
        old_path = ROOT/reference[eid]['prediction_path']
        with np.load(old_path) as saved:
            assert np.array_equal(saved['query_rows'], event['query_rows'])
            assert np.array_equal(saved['aps'], aps) and np.array_equal(saved['bounds'], bounds)
            predictions = dict(ordinary=prediction['projected'], GP=saved['GP'].copy(),
                               full24=saved['full24'].copy())
        predictions['GP_ordinary_half'] = .5*(predictions['GP']+predictions['ordinary'])
        predictions['GP_full24_half'] = .5*(predictions['GP']+predictions['full24'])
        unscored = out/f'{eid}_unscored.npz'
        np.savez_compressed(unscored, **predictions, ordinary_raw=prediction['raw'],
                            query_rows=event['query_rows'], aps=aps, bounds=bounds)
        # Only persisted predictions exist above this line; labels start here.
        truth = read_truth(csv_path, event['query_rows'])
        _, group, counts = np.unique(truth, axis=0, return_inverse=True, return_counts=True)
        weights = 1/(len(counts)*counts[group])
        errors = {name: np.linalg.norm(value-truth, axis=1) for name, value in predictions.items()}
        with np.load(old_path.parent/f'{eid}_scored.npz') as old:
            assert np.array_equal(old['truth'], truth) and np.array_equal(old['weights'], weights)
            for name in ('GP', 'full24'):
                assert np.allclose(old[name], errors[name], rtol=0, atol=1e-10)
        raw_errors = np.linalg.norm(prediction['raw']-truth, axis=1)
        scored = out/f'{eid}_scored.npz'
        np.savez_compressed(scored, truth=truth, weights=weights, ordinary_raw_errors=raw_errors, **errors)
        record = dict(event=eid, floor=event['floor'], building=event['building'],
            direction=event['direction'], K=event['K'], mde={name: float(weights@value) for name, value in errors.items()},
            ordinary_raw_mde=float(weights@raw_errors), seconds=time.monotonic()-tick,
            prediction_path=str(unscored), prediction_sha256=digest(unscored),
            scored_path=str(scored), scored_sha256=digest(scored))
        progress['records'].append(record); done.add(eid)
        write(out/'progress.json', progress)
        print(json.dumps(dict(event=eid, completed=len(done), total=54, seconds=record['seconds'])), flush=True)
    assert state_hash(model.loaded.model_) == initial_state
    del model
    gc.collect(); torch.cuda.empty_cache()
    elapsed = time.monotonic()-started
    progress['segments'].append(dict(ended=now(), seconds=elapsed, budget=args.seconds,
                                     budget_overrun=elapsed > args.seconds))
    progress['complete'] = len(done) == 54
    write(out/'progress.json', progress)
    if progress['complete']:
        summary = summarize(progress['records'], manifest)
        write(out/'complete.json', dict(complete=True, created=now(), freeze_sha256=digest(out/'freeze.json'),
            source_freeze_sha256=source['freeze_sha256'], selected_step=source['selected_step'],
            selected_checkpoint_sha256=source['selected_checkpoint_sha256'], summary=summary,
            ordinary_vs_full24_gain=1-summary['ordinary']['mean']/summary['full24']['mean'],
            ordinary_system_vs_full24_system_gain=1-summary['GP_ordinary_half']['mean']/summary['GP_full24_half']['mean'],
            records=progress['records'], segments=progress['segments'], weights_unchanged=True,
            training_updates=0, GP_fits=0, target_selection=False, blind_test=False))
    return 0 if progress['complete'] else 2


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--seconds', type=float, default=900.)
    parser.add_argument('--output', type=Path, default=OUT)
    raise SystemExit(run(parser.parse_args()))
