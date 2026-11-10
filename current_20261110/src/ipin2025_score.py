"""Last-stage known-floor IPIN2025 scoring; not yet executed on formal GT.

Input predictions.json: complete=true, phase='predictions', freeze_sha256,
contexts_complete_sha256, eligible_floors=[2,3,4,5,6], K=[1,3,5], methods=METHODS,
logs=[{id,query_path,query_sha256,gt_path,gt_sha256}],
predictions=[{log_id:'03'|'04',floor:2..6,K:1|3|5,path:absolute,sha256}].
freeze.json and contexts_complete.json are in the same directory. The freeze
contains the identical logs, eligible_floors, K and methods. Exactly 30 distinct
records are required. Each NPZ has times[N], snapshot_times[N] and all five
METHODS[N,2] in the fixed 9011 east/north metre frame. Empty streams are valid
but uncovered. All files, hashes, clocks and arrays are verified and loaded
BEFORE either GT file is opened. Query files are never opened by this scorer.

GT is numeric CSV: timestamp, longitude, latitude, physical floor, evaluation
point indicator (first five columns). Keep ALL rows in the denominator; the
indicator is recorded, not used to filter. The official script uses column5>0
to form evaltimestamps but matches all GT rows; do not infer other meanings
of its numeric codes. Report positive/nonpositive counts explicitly.
Match the last prediction time
<=GT_time+.01, as evaluateCompetitor2025.m lines90-92. Unsupported floors,
including Level1, and times before the first prediction remain uncovered.
No stale-point filtering, interpolation or retrospective predictions.

This is conditional-known-floor 2D distance in the fixed project metric frame,
NOT official horizontal-plus-floor-penalty P75. Report point-weighted covered
mean and inverse-empirical-CDF p90, without pooling K or claiming independent
locations/replicates. Prediction age and receipt age use original application
seconds; a negative age down to -.01 is possible under the official tolerance.
All floor/K predictions must be sealed by the outer runner before invocation;
hash checks verify integrity, not the historical absence of prior GT access.
No model loading, fitting, prediction, Query reader or default execution.
"""
from pathlib import Path
from itertools import product
import argparse
import hashlib
import json

import numpy as np

from ipin2025_observation_data import REF_LAT, REF_LON, match_prediction_times


METHODS = ('hurdle', 'GP', 'GP_candidates', 'coordinate_half', 'probability_half')
FLOORS, KS, LOGS = (2, 3, 4, 5, 6), (1, 3, 5), ('03', '04')
QUERY_RELATIVE_PATHS = {
    '03': '01 Logfiles/03 - Scoring/IPIN2025_T3_ScoringTrial03.txt',
    '04': '01 Logfiles/03 - Scoring/IPIN2025_T3_ScoringTrial04sorted.txt',
}
GT_RELATIVE_PATHS = {log: f'03 Evaluation/GT/GT_IPIN2025_T3_ScoringTrial{log}.csv'
                     for log in LOGS}
GT_SHA256 = {
    '03': 'aec42c4c27c2047cad8fa4b486b994212ffd66c08b897bc8e8974ce8ba2a5a8c',
    '04': '34a36bb0ac6bfbad0c7cf2c08afd66de4f064940553a0269b3606e005b14b0cf',
}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def load_sealed_predictions(manifest_path):
    """Finish the entire 30-record barrier before returning; no GT argument."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    assert manifest['phase'] == 'predictions' and manifest['complete'] is True
    assert digest(path.parent/'freeze.json') == manifest['freeze_sha256']
    assert digest(path.parent/'contexts_complete.json') == manifest['contexts_complete_sha256']
    freeze = json.loads((path.parent/'freeze.json').read_text())
    for field in ('logs', 'eligible_floors', 'K', 'methods'):
        assert manifest[field] == freeze[field], f'Prediction/freeze mismatch: {field}'
    assert manifest['eligible_floors'] == list(FLOORS) and manifest['K'] == list(KS)
    assert manifest['methods'] == list(METHODS)
    assert len(manifest['logs']) == 2 and {r['id'] for r in manifest['logs']} == set(LOGS)
    for log in manifest['logs']:
        assert log['gt_sha256'] == GT_SHA256[log['id']]
        assert Path(log['gt_path']).name == Path(GT_RELATIVE_PATHS[log['id']]).name
        assert Path(log['query_path']).name == Path(QUERY_RELATIVE_PATHS[log['id']]).name
    records = manifest['predictions']
    keys = [(r['log_id'], r['floor'], r['K']) for r in records]
    assert len(keys) == 30 and set(keys) == set(product(LOGS, FLOORS, KS))
    predictions, clocks = {}, {}
    for key, record in zip(keys, records):
        file = Path(record['path'])
        assert file.is_absolute()
        assert digest(file) == record['sha256'], f'Prediction hash mismatch: {file}'
        with np.load(file, allow_pickle=False) as saved:
            arrays = {name: np.asarray(saved[name], dtype=np.float64).copy()
                      for name in ('times', 'snapshot_times', *METHODS)}
        times, receipts = arrays['times'], arrays['snapshot_times']
        assert times.ndim == 1 and receipts.shape == times.shape
        assert all(np.isfinite(a).all() for a in arrays.values())
        assert all(arrays[m].shape == (len(times), 2) for m in METHODS)
        assert np.allclose(times*2, np.rint(times*2), rtol=0, atol=1e-8)
        assert np.allclose(np.diff(times), .5, rtol=0, atol=1e-8)
        assert np.all(receipts <= times) and np.all(np.diff(receipts) >= 0)
        log = key[0]
        if log in clocks:
            assert np.array_equal(times, clocks[log][0])
            assert np.array_equal(receipts, clocks[log][1])
        else:
            clocks[log] = (times, receipts)
        predictions[key] = arrays
    return manifest, predictions


def _distribution(values):
    """Empty coverage is explicit null, never a zero localization error."""
    if not len(values):
        return dict(count=0, mean=None, p90=None, minimum=None, maximum=None)
    values = np.asarray(values, dtype=np.float64)
    return dict(count=len(values), mean=float(values.mean()),
                p90=float(np.quantile(values, .9, method='inverted_cdf')),
                minimum=float(values.min()), maximum=float(values.max()))


def summarize(rows):
    covered = [r for r in rows if r['covered']]
    total = len(rows)
    return dict(total_points=total, covered_points=len(covered),
        coverage=len(covered)/total if total else None,
        unsupported_floor=sum(r['reason'] == 'unsupported_floor' for r in rows),
        no_prior_prediction=sum(r['reason'] == 'no_prior_prediction' for r in rows),
        methods={m: _distribution([r['errors'][m] for r in covered]) for m in METHODS},
        prediction_age_seconds=_distribution([r['prediction_age'] for r in covered]),
        receipt_age_seconds=_distribution([r['receipt_age'] for r in covered]),
        prediction_age_over_half_second=sum(r['prediction_age'] > .5 for r in covered),
        receipt_age_over_half_second=sum(r['receipt_age'] > .5 for r in covered),
        matched_prediction_after_GT=sum(r['prediction_age'] < 0 for r in covered))


def score(manifest_path):
    """Explicit scoring entry; this is the only function that opens GT."""
    manifest, predictions = load_sealed_predictions(manifest_path)
    # The completed all-floor/all-K/all-log barrier above precedes all GT I/O.
    truths = {}
    logs = {r['id']: r for r in manifest['logs']}
    for log in LOGS:
        path = Path(logs[log]['gt_path'])
        assert digest(path) == GT_SHA256[log], f'GT differs from reservation: {path}'
        gt = np.loadtxt(path, delimiter=',', ndmin=2)
        assert gt.shape[1] >= 5 and len(gt) and np.isfinite(gt[:, :5]).all()
        assert np.equal(gt[:, 3], np.floor(gt[:, 3])).all()
        xy = np.column_stack(((gt[:, 1]-REF_LON)*111320*np.cos(np.radians(REF_LAT)),
                              (gt[:, 2]-REF_LAT)*111320))
        truths[log] = (gt, xy)
    rows = []
    for k in KS:
        for log in LOGS:
            gt, xy = truths[log]
            times = predictions[(log, FLOORS[0], k)]['times']
            matches = match_prediction_times(times, gt[:, 0])
            for i, (truth, index) in enumerate(zip(gt, matches)):
                floor = int(truth[3])
                reason = ('unsupported_floor' if floor not in FLOORS else
                          'no_prior_prediction' if index < 0 else 'covered')
                row = dict(K=k, log_id=log, gt_row=i, time=float(truth[0]), floor=floor,
                           evaluation_indicator=float(truth[4]), covered=reason == 'covered',
                           reason=reason)
                if row['covered']:
                    prediction = predictions[(log, floor, k)]
                    row.update(prediction_index=int(index),
                        prediction_age=float(truth[0]-prediction['times'][index]),
                        receipt_age=float(truth[0]-prediction['snapshot_times'][index]),
                        errors={m: float(np.linalg.norm(prediction[m][index]-xy[i])) for m in METHODS})
                rows.append(row)
    by_k = {}
    for k in KS:
        selected = [r for r in rows if r['K'] == k]
        floors = sorted({r['floor'] for r in selected} | {1})
        by_k[str(k)] = dict(overall=summarize(selected),
            by_log={log: summarize([r for r in selected if r['log_id'] == log]) for log in LOGS},
            by_floor={str(f): summarize([r for r in selected if r['floor'] == f]) for f in floors},
            by_log_floor={log: {str(f): summarize([r for r in selected
                if r['log_id'] == log and r['floor'] == f]) for f in floors} for log in LOGS})
    return dict(schema='ipin2025-known-floor-score-v1', complete=True,
        predictions_manifest_sha256=digest(manifest_path),
        freeze_sha256=manifest['freeze_sha256'],
        contexts_complete_sha256=manifest['contexts_complete_sha256'],
        gt_sha256=GT_SHA256, eligible_floors=list(FLOORS), methods=list(METHODS),
        gt_rows={log: dict(total=len(gt), column5_positive=int((gt[:, 4] > 0).sum()),
            column5_nonpositive=int((gt[:, 4] <= 0).sum()),
            policy='All rows retained; column5>0 identifies official evaltimestamps, no other code semantics assumed')
            for log, (gt, _) in truths.items()},
        metric='Known-floor 2D metres in fixed 9011 frame; all GT rows point-weighted; covered p90 inverse empirical CDF',
        coverage='Unsupported floors and missing previous predictions retained in every K denominator; stale matches retained',
        reference=dict(id=9011, longitude=REF_LON, latitude=REF_LAT),
        by_K=by_k, points=rows)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = score(args.predictions)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
