"""Reuse the finished five-seed v2 model on the already-fixed UJI jitter condition.

No fit, theta search, new seed, GP fit or readout change. The original stress
generator supplies one real scan per history position plus 3 m coordinate
jitter; APs remain the H/current-S union. Original A/B/C/WKNN summaries are
reused, not retrained. This is exposed UJI development evidence, not an
independent corpus confirmation. Run only after the current nine-floor trial.
"""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time

from .run import _json_default, _sha, _write
import joblib
import numpy as np


VARIANTS = ('v2_noM2', 'v2_C1', 'v2_full')


def evaluate(args):
    source = args.source_dir.resolve()
    models = args.model_dir.resolve()
    output = args.output.resolve()
    # Existing trained pickles refer to this original module name.
    sys.path.insert(0, str(source))
    O = importlib.import_module('law_bench_ours')
    stress = importlib.import_module('stress_uji')
    reference = importlib.import_module('run_nine_floor_field_posterior')
    C = stress.C
    selection = json.loads((models / 'theta_selection.json').read_text())
    completed = json.loads((models.parent / 'summary.json').read_text())
    theta = tuple(selection['theta'])
    if tuple(completed['theta']) != theta:
        raise ValueError('Finished trial and saved model theta disagree')
    old = json.loads(args.stress_summary.read_text())
    if old['seeds'] != [1, 2, 3, 4, 5]:
        raise ValueError('Expected the existing five-seed stress control')

    ops, bindings = [], []
    for seed in (1, 2, 3, 4, 5):
        path = models / f'v2_seed{seed}.joblib'
        saved = joblib.load(path)
        if saved['seed'] != seed or tuple(saved['theta']) != theta:
            raise ValueError(f'Model identity mismatch: {path}')
        for name, digest in saved['code_sha256'].items():
            if _sha(source / name) != digest:
                raise ValueError(f'Trained source version unavailable: {name}')
        ops.append(saved['model'])
        bindings.append(dict(seed=seed, path=str(path), sha256=_sha(path),
                             source=saved['source'], csv=saved['csv']))
    for csv in {b['csv'] for b in bindings}:
        if _sha(csv) != C.MAN['csv_sha256']:
            raise ValueError('Trained source CSV differs from the frozen UJI corpus')

    # This exact draw/order is shared with stress_uji's existing jitter condition.
    rng = np.random.default_rng(20261111)
    degraded = {b: stress.degraded_rows_and_xy(b, 'jitter', rng) for b in (1, 2)}
    history_identity = {}
    for building, floors in degraded.items():
        for floor, (rows, xy) in floors.items():
            h = hashlib.sha256(np.ascontiguousarray(rows, dtype=np.int64).tobytes())
            h.update(np.ascontiguousarray(xy, dtype=np.float64).tobytes())
            history_identity[f'B{building}F{floor}'] = h.hexdigest()
    frozen = dict(condition='jitter', data_status='exposed_development',
        source_csv_sha256=C.MAN['csv_sha256'], models=bindings, theta=theta,
        old_stress_summary_sha256=_sha(args.stress_summary),
        original_trial_summary_sha256=_sha(models.parent / 'summary.json'),
        source_code={name: _sha(source / name) for name in
                     ('law_bench_ours.py', 'nine_floor_v2.py', 'stress_uji.py',
                      'run_nine_floor_field_posterior.py')},
        evaluator_sha256=_sha(__file__), history_identity=history_identity,
        protocol_manifest_sha256=hashlib.sha256(json.dumps(C.MAN, sort_keys=True,
            default=_json_default).encode()).hexdigest(),
        variants=list(VARIANTS), control_reuse=['A_adopted', 'B_H_only', 'C_H_cond', 'WKNN3_raw_HS'],
        training=False, parameter_selection=False,
        readout='Original posterior(), beta128/eps.02/offsets; no GP pool with mismatched H')
    # Normalize tuples for an exact JSON resume comparison.
    frozen = json.loads(json.dumps(frozen, default=_json_default))
    if args.resume:
        if json.loads((output / 'freeze.json').read_text()) != frozen:
            raise ValueError('Frozen model/data/code changed; use a new output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
        _write(output / 'freeze.json', frozen)

    started = time.monotonic()
    errors = {name: {} for name in VARIANTS}
    for event in C.MAN['events']:
        path = output / 'predictions' / (event['id'] + '.npz')
        _, S, Q = C.event_arrays(event)
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                if not np.array_equal(saved['query_rows'], event['query_rows']):
                    raise ValueError(f'Cached Query row mismatch: {event["id"]}')
                predictions = {name: saved[name] for name in VARIANTS}
        else:
            building = int(event['building'][1:])
            floor = int(event['floor'].rsplit('F', 1)[1])
            H = {f: dict(rss=C.RSS[rows], xy=xy)
                 for f, (rows, xy) in degraded[building].items() if f != floor}
            seen = C.DET[S].any(0)
            for data in H.values():
                seen |= (data['rss'] != 100).any(0)
            aps = np.flatnonzero(seen)
            maps = O.make_maps(H)
            X, field = O.ensemble_field(ops, maps, floor, aps)
            sr, sx = C.RSS[S][:, aps], C.XY[S]
            Ps = O.det_stats(sr, sx)[0]
            field_s = O.ensemble_field(ops, maps, floor, aps, Ps)[1]
            fields = dict(
                v2_noM2=(field['p'], field['m'], field['sig2'] + field['tau2']),
                v2_C1=O.condition(X, field['p'], field['m'], field['s2'], sr, sx,
                                  (16.0, 0.3, 1.0, 0.0)),
                v2_full=O.condition_v2(X, field, sr, sx, theta, field_s))
            bounds = C.bounds_of(np.vstack([d['xy'] for d in H.values()] + [sx]))
            predictions = {}
            for name, (p, mean, variance) in fields.items():
                candidates, log_mass = reference.posterior(X, p, mean, variance, S, Q, aps)
                predictions[name] = np.clip(np.exp(log_mass) @ candidates, bounds[0], bounds[1])
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(path, query_rows=np.asarray(event['query_rows']),
                                aps=aps, **predictions)
        for name, prediction in predictions.items():
            errors[name][event['id']] = np.linalg.norm(prediction - C.XY[Q], axis=1)
        print(event['id'], flush=True)
    result = dict(complete=True, condition='jitter', theta=theta,
        variants={name: reference.project_summary(values) for name, values in errors.items()},
        controls={name: old['summary']['jitter'][name] for name in frozen['control_reuse']},
        freeze_sha256=_sha(output / 'freeze.json'), seconds=time.monotonic() - started,
        events=len(C.MAN['events']), training=False, parameter_selection=False)
    _write(output / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--model-dir', type=Path, required=True)
    parser.add_argument('--stress-summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if os.uname().sysname != 'Linux':
        parser.error('Run inference on the lab server, not the local Mac')
    os.nice(10)
    evaluate(args)


if __name__ == '__main__':
    main()
