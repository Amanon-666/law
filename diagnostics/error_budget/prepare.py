"""Prepare T9 diagnostic folds; do not fit models or score new predictions.

The existing internal Query consumes every non-S target position. An independent
extra-target reference therefore needs cross-fitting, not a nonexistent unused
target pool. Four blocks are fixed by the historical search rectangle. Each
block's diagnostic reference may use labels in the other blocks, but never its
Query positions, their three-coordinate-unit neighbourhood, or any S position.
This is privileged diagnostic information, not an alternative K-point method.
"""
import json
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial import Delaunay

from run_multi_history import digest, write
from uji_probability_study import DATA, load_raw


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'results/uji_error_budget_preparation_v1_20261026'
OLD = ROOT / 'results/uji_history_bank_v1_20261026'
PAIRED = ROOT / 'results/uji_operator_paired_v2_20261026'


def row_indices(floor, ids):
    lookup = {int(r): i for i, r in enumerate(floor['row_ids'])}
    return np.array([lookup[r] for r in ids], dtype=int)


def prepare():
    manifest = json.loads((OLD / 'manifest.json').read_text())
    names = {e['floor'] for e in manifest['events']}
    train = load_raw('trainingData.csv', names)
    official = load_raw('validationData.csv', names)
    OUT.mkdir(exist_ok=False)
    write(OUT / 'freeze.json', dict(created_unix=time.time(), purpose=__doc__,
        stage='Metadata preparation only; no target GP/observation fitting or new evaluation',
        candidate_arms=['a: existing adapted_bank',
            'b: cross-fitted target GP mean, retain a covariance and S adaptation',
            'c: b plus target-pool observation fit, estimator not yet finalized',
            'd: same extra target pool plus original S, ordinary target GP'],
        interpretation='Reference interventions are not additive causal error components or theoretical lower bounds. Report negative changes and unstable/negative denominators, not clipped closure percentages.',
        split='Four axis-aligned blocks from each original H+S search rectangle midpoint; same block for repeated positions and internal/official batches',
        buffer=3., coordinate_unit='Original UJI coordinate distance, unchanged from existing predictions; not asserted to be physical 3 metres',
        query_access='Coordinates used only to exclude diagnostic reference labels and stratify scores; query RSS never enters fitting or AP selection',
        aperture='Original grid_bounds and H/S AP columns frozen in every proposed arm; extra labels cannot expand them',
        source_paths={str(p): digest(p) for p in [Path(__file__), OLD / 'manifest.json',
            OLD / 'parameters.json', DATA / 'trainingData.csv', DATA / 'validationData.csv']}))
    all_events, summary = [], []
    for event in manifest['events']:
        f, o = train[event['floor']], official[event['floor']]
        sr = row_indices(f, event['support_rows'])
        ir = row_indices(f, event['internal_rows'])
        vr = row_indices(o, event['official_rows'])
        sx = np.unique(f['xy'][sr], axis=0)
        qxy = np.concatenate([f['xy'][ir], o['xy'][vr]])
        midpoint = np.asarray(event['grid_bounds']).mean(0)
        block = ((qxy[:, 0] >= midpoint[0]).astype(int) +
                 2 * (qxy[:, 1] >= midpoint[1]).astype(int))
        s_position = (f['positions'][:, None] == sx[None]).all(-1).any(-1)
        # This is a real data-availability question, not a new model benchmark.
        internal_positions = np.unique(f['group'][ir])
        unconsumed = ~s_position.copy()
        unconsumed[internal_positions] = False
        with np.load(PAIRED / event['id'] / 'predictions.npz') as pred:
            np.testing.assert_allclose(pred['query_xy'] + pred['origin'], qxy, atol=1e-8, rtol=0)
            baseline_errors = np.linalg.norm(pred['adapted_bank'] - pred['query_xy'], axis=1)
        centered_s = sx - sx.mean(0)
        nondegenerate = np.linalg.matrix_rank(centered_s) == 2
        hull = Delaunay(centered_s) if nondegenerate else None
        inside = hull.find_simplex(qxy - sx.mean(0)) >= 0 if hull is not None else np.zeros(len(qxy), bool)
        folds = []
        for fold in range(4):
            take = block == fold
            if not take.any():
                continue
            held_xy = np.unique(qxy[take], axis=0)
            distances = np.linalg.norm(f['positions'][:, None] - held_xy[None], axis=-1)
            available = ~s_position & (distances.min(1) > 3.)
            rows = np.flatnonzero(available[f['group']])
            # Whole positions, including unchosen repeats at S, are excluded.
            assert not np.isin(f['group'][rows], np.flatnonzero(s_position)).any()
            assert len(rows) == 0 or distances[available].min() > 3.
            split_records = {}
            for split, indices, source in [('internal', ir, f), ('official', vr, o)]:
                offset = 0 if split == 'internal' else len(ir)
                local_take = take[offset:offset + len(indices)]
                chosen = indices[local_take]
                loc_xy = source['xy'][chosen]
                if len(chosen):
                    _, inv, counts = np.unique(loc_xy, axis=0, return_inverse=True, return_counts=True)
                    weights = 1 / (len(counts) * counts[inv])
                    err = baseline_errors[offset:offset + len(indices)][local_take]
                    ih = inside[offset:offset + len(indices)][local_take]
                    stats = dict(positions=len(counts), scans=len(chosen),
                        baseline_mde=float(weights @ err), inside_fraction=float(weights @ ih))
                else:
                    stats = dict(positions=0, scans=0)
                split_records[split] = dict(query_rows=source['row_ids'][chosen].tolist(), **stats)
            item = dict(fold=fold, reference_rows=f['row_ids'][rows].tolist(),
                reference_positions=int(available.sum()), reference_scans=len(rows),
                support_rows=event['support_rows'], splits=split_records,
                minimum_query_distance=float(distances[available].min()) if available.any() else None,
                support_hull_nondegenerate=bool(nondegenerate))
            folds.append(item)
            summary.append(dict(event=event['id'], fold=fold,
                reference_positions=item['reference_positions'], reference_scans=len(rows),
                minimum_query_distance=item['minimum_query_distance'],
                splits={k: {a: b for a, b in v.items() if a != 'query_rows'} for k, v in split_records.items()}))
        all_events.append(dict(event=event['id'], floor=event['floor'],
            ap_columns=event['ap_columns'], grid_bounds=event['grid_bounds'],
            unused_target_positions_without_crossfit=int(unconsumed.sum()), folds=folds))
    write(OUT / 'manifest.json', dict(events=all_events))
    write(OUT / 'summary.json', dict(records=summary,
        unused_positions={e['event']: e['unused_target_positions_without_crossfit'] for e in all_events},
        fitted_models=0, new_predictions=0))
    print(json.dumps(dict(events=len(all_events), folds=len(summary),
        unused_positions={e['event']: e['unused_target_positions_without_crossfit'] for e in all_events},
        min_reference_positions=min(r['reference_positions'] for r in summary),
        max_reference_positions=max(r['reference_positions'] for r in summary)), ensure_ascii=False))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical data preparation runs on ray-web')
    prepare()
