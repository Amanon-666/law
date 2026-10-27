"""Post-hoc geometry check of T9 reference designs; no RSS fit or prediction.

The quadrant reference is already running and is preserved. This compares its
coverage with four interleaved spatial-block folds, using source-fixed length
as the tile width, and the most permissive leave-neighbourhood-out geometry.
No performance number is used to pick a width or change the existing run.
"""
import json
from pathlib import Path
import socket
import time

import numpy as np
from scipy.spatial import cKDTree

from run_multi_history import digest, write
from uji_probability_study import DATA, load_raw


ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'results/uji_history_bank_v1_20261026'
OUT = ROOT / 'results/uji_error_budget_coverage_design_20261026'


def indices(f, ids):
    lookup = {int(v): i for i, v in enumerate(f['row_ids'])}
    return np.array([lookup[v] for v in ids])


def summarize(distance, length):
    return dict(positions=len(distance), mean=float(distance.mean()),
        median=float(np.median(distance)), p90=float(np.quantile(distance, .9)),
        fraction_farther_than_source_length=float((distance > length).mean()))


def run():
    events = json.loads((OLD / 'manifest.json').read_text())['events']
    length = json.loads((OLD / 'parameters.json').read_text())['base']['length']
    names = {e['floor'] for e in events}
    floors = load_raw('trainingData.csv', names)
    official = load_raw('validationData.csv', names)
    OUT.mkdir(exist_ok=False)
    rows = []
    for event in events:
        f, o = floors[event['floor']], official[event['floor']]
        sx = np.unique(f['xy'][indices(f, event['support_rows'])], axis=0)
        queries = dict(internal=np.unique(f['xy'][indices(f, event['internal_rows'])], axis=0),
                       official=np.unique(o['xy'][indices(o, event['official_rows'])], axis=0))
        qxy = np.concatenate(list(queries.values()))
        bounds = np.asarray(event['grid_bounds'])
        midpoint = bounds.mean(0)
        quadrants = ((qxy[:, 0] >= midpoint[0]).astype(int) +
                     2 * (qxy[:, 1] >= midpoint[1]).astype(int))
        tiles = np.floor((qxy - bounds[0]) / length).astype(int)
        interleaved = (tiles[:, 0] % 2) + 2 * (tiles[:, 1] % 2)
        non_support = ~(f['positions'][:, None] == sx[None]).all(-1).any(-1)
        for policy, labels in [('quadrants', quadrants), ('interleaved_length_tiles', interleaved)]:
            distance = np.full(len(qxy), np.nan)
            counts = []
            for fold in np.unique(labels):
                take = labels == fold
                nearest_query = cKDTree(qxy[take]).query(f['positions'])[0]
                retained = non_support & (nearest_query > 3.)
                if not retained.any():
                    raise RuntimeError(f'No reference positions: {event["id"]} {policy} {fold}')
                distance[take] = cKDTree(f['positions'][retained]).query(qxy[take])[0]
                counts.append(dict(fold=int(fold), reference_positions=int(retained.sum())))
            offset = 0
            for split, query in queries.items():
                d = distance[offset:offset + len(query)]; offset += len(query)
                rows.append(dict(event=event['id'], floor=event['floor'], split=split,
                    policy=policy, **summarize(d, length), reference_counts=counts))
        # A geometry-only optimistic reference: excludes each query neighbourhood,
        # but no other held-out positions. It is not a deployable fitted map.
        distance = np.linalg.norm(qxy[:, None] - f['positions'][non_support][None], axis=-1)
        distance[distance <= 3.] = np.inf
        nearest = distance.min(1)
        if not np.isfinite(nearest).all():
            raise RuntimeError(f'No pointwise reference positions: {event["id"]}')
        offset = 0
        for split, query in queries.items():
            d = nearest[offset:offset + len(query)]; offset += len(query)
            rows.append(dict(event=event['id'], floor=event['floor'], split=split,
                policy='pointwise_neighbourhood_exclusion', **summarize(d, length)))
    result = dict(created_unix=time.time(), scope=__doc__,
        tile_width_source_length=length, exclusion_radius=3.,
        units='Existing UJI coordinate units, not reinterpreted as physical metres',
        data_access='Position geometry and existing S/Q IDs only; no RSS scoring or model fitting',
        hashes={str(p): digest(p) for p in [Path(__file__), OLD / 'manifest.json', OLD / 'parameters.json',
                                          DATA / 'trainingData.csv', DATA / 'validationData.csv']}, records=rows)
    write(OUT / 'summary.json', result)
    for floor in sorted(names):
        for split in queries:
            print(json.dumps(dict(floor=floor, split=split, policies={p: dict(
                mean_distance=float(np.mean([r['mean'] for r in rows if r['floor']==floor and r['split']==split and r['policy']==p])),
                fraction_far=float(np.mean([r['fraction_farther_than_source_length'] for r in rows if r['floor']==floor and r['split']==split and r['policy']==p])))
                for p in ['quadrants','interleaved_length_tiles','pointwise_neighbourhood_exclusion']})))


if __name__ == '__main__':
    if socket.gethostname() != 'ray-web':
        raise RuntimeError('Numerical metadata work runs on ray-web')
    run()
