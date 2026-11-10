"""Post-hoc K1 tail diagnosis from frozen predictions only; no model imports."""
import os
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = '2'

import hashlib
import json
from pathlib import Path
import socket
import time

import numpy as np


SOURCE = Path('results/continuous_observation_target_20261110')
OUTPUT = Path('results/continuous_observation_tail_20261110')
METHOD = 'continuous_observation_translation'
BASELINE = 'GP_ordinary_HS_half'
THRESHOLD = 21.116054539627005
CATEGORIES = ('new_only', 'baseline_only', 'both', 'neither')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def distribution(values, weights):
    values, weights = np.asarray(values), np.asarray(weights)
    weights = weights / weights.sum()
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return dict(mean=float(weights @ values),
                median=float(values[order[np.searchsorted(cumulative, .5)]]),
                p90=float(values[order[np.searchsorted(cumulative, .9)]]))


def main():
    start = time.monotonic()
    complete = json.loads((SOURCE/'complete.json').read_text())
    freeze = json.loads((SOURCE/'freeze.json').read_text())
    assert complete['complete'] and len(complete['records']) == 54
    assert digest(SOURCE/'freeze.json') == complete['freeze_sha256']
    inputs = {str(SOURCE/name): digest(SOURCE/name) for name in ('complete.json', 'freeze.json')}
    events = {e['id']: e for e in freeze['contract']['events'] if e['K'] == 1}
    records = [r for r in complete['records'] if r['K'] == 1]
    assert len(events) == len(records) == 18
    assert set(events) == {r['event'] for r in records}
    buildings = sorted({e['building'] for e in events.values()})
    rows, new_only = [], []
    for record in records:
        event = events[record['event']]
        for key in ('prediction', 'scored'):
            path = Path(record[key+'_path'])
            assert path.parent == SOURCE
            inputs[str(path)] = digest(path)
            assert inputs[str(path)] == record[key+'_sha256']
        with np.load(record['scored_path']) as saved:
            truth = saved['truth']
            new_error, baseline_error = saved[METHOD], saved[BASELINE]
        with np.load(record['prediction_path']) as saved:
            assert np.array_equal(saved['query_rows'], event['query_rows'])
            candidates = saved['cand_xy']
            mass = np.exp(saved['log_position_mass'])
        assert mass.shape == (len(truth), len(candidates))
        assert np.allclose(mass.sum(axis=1), 1.)
        positions, inverse = np.unique(truth, axis=0, return_inverse=True)
        inverse = inverse.ravel()
        floors = {e['floor'] for e in events.values() if e['building'] == event['building']}
        directions = {e['direction'] for e in events.values() if e['floor'] == event['floor']}
        weight = 1. / (len(buildings) * len(floors) * len(directions) * len(positions))
        for index, xy in enumerate(positions):
            selected = inverse == index
            new = float(new_error[selected].mean())
            baseline = float(baseline_error[selected].mean())
            category = ('both' if baseline > THRESHOLD else 'new_only') if new > THRESHOLD else (
                'baseline_only' if baseline > THRESHOLD else 'neither')
            row = dict(event=event['id'], floor=event['floor'], building=event['building'],
                       direction=event['direction'], xy=xy.tolist(), weight=weight,
                       new_error=new, baseline_error=baseline, category=category)
            rows.append(row)
            if category != 'new_only':
                continue
            posterior = mass[selected]
            distance = np.linalg.norm(candidates - xy, axis=1)
            center = posterior @ candidates
            spread = np.sqrt(np.sum(posterior * np.sum(
                (candidates[None] - center[:, None]) ** 2, axis=2), axis=1))
            peak = candidates[posterior.argmax(axis=1)]
            peak_neighborhood = np.linalg.norm(candidates[None] - peak[:, None], axis=2) <= 10.
            new_only.append(dict(row,
                nearest_candidate_m=float(distance.min()),
                truth_mass_10m=float(posterior[:, distance <= 10.].sum(axis=1).mean()),
                posterior_rms_radius_m=float(spread.mean()),
                peak_truth_distance_m=float(np.linalg.norm(peak - xy, axis=1).mean()),
                peak_mass_10m=float(np.sum(posterior * peak_neighborhood, axis=1).mean())))

    assert np.isclose(sum(r['weight'] for r in rows), 1.)
    groups = {}
    for category in CATEGORIES:
        selected = [r for r in rows if r['category'] == category]
        total = sum(r['weight'] for r in selected)
        by_floor = {}
        for floor in sorted({r['floor'] for r in rows}):
            floor_weight = sum(r['weight'] for r in rows if r['floor'] == floor)
            category_weight = sum(r['weight'] for r in selected if r['floor'] == floor)
            by_floor[floor] = dict(global_weight=category_weight,
                fraction_of_floor=category_weight / floor_weight,
                fraction_of_category=category_weight / total if total else None)
        groups[category] = dict(weight=total, position_event_count=len(selected), by_floor=by_floor)

    diagnostics = None
    if new_only:
        weights = [r['weight'] for r in new_only]
        metrics = ('nearest_candidate_m', 'truth_mass_10m', 'posterior_rms_radius_m',
                   'peak_truth_distance_m', 'peak_mass_10m', 'new_error', 'baseline_error')
        diagnostics = {key: distribution([r[key] for r in new_only], weights) for key in metrics}
        diagnostics['fraction_nearest_candidate_beyond_10m'] = float(
            np.average([r['nearest_candidate_m'] > 10. for r in new_only], weights=weights))

    result = dict(post_hoc=True, purpose='Describe K1 tail failure layers using frozen predictions only',
        threshold_m=THRESHOLD, threshold_origin='Previously observed K1 baseline position-mean p90',
        baseline=BASELINE, method=METHOD, tail_rule='Position mean scan error strictly above common threshold',
        weighting='Equal building, floor within building, direction within floor, position within event; scans averaged within position',
        metric_definition='Posterior diagnostics are per-scan values averaged within each new-only tail position; RMS radius is about posterior mean; peak is a diagnostic only',
        limits='Post-hoc descriptive evidence. No deployment rule, readout evaluation, calibration guarantee or proof of multimodality. Candidate entries retain original coordinate multiplicity.',
        inputs=inputs, script_sha256=digest(Path(__file__)), events=len(records),
        position_events=len(rows), categories=groups, new_only_diagnostics=diagnostics,
        new_only_positions=new_only, seconds=time.monotonic()-start,
        model_loads=0, predictions=0, training_updates=0)
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    (OUTPUT/'audit.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(categories={k:v['weight'] for k,v in groups.items()},
                          new_only_diagnostics=diagnostics, seconds=result['seconds'])))


if __name__ == '__main__':
    assert socket.gethostname() == 'ray-web'
    main()
