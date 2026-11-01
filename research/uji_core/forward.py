"""Fit one spatial residual range on other UJI buildings, then score official queries.

From the repository root: python -m research.uji_core.forward path/to/UJIndoorLoc
The neutral historical field, residual update and matcher come from the TPM adapter.
All three H+S arms give the support fingerprints their own candidate locations.
"""
import argparse
import json
from typing import NamedTuple
from pathlib import Path

import numpy as np
import pandas as pd

from external_baselines import tpm_adapter as tpm
from research.uji_core.protocol import BUDGETS, HASHES, SEEDS, history_floors, read_floors, split_rows

RADII = (10., 20., 40., 80.)
FIXED_RADIUS = 40.
RIDGE = 1.
MATCH = (12., -85.)


class Arm(NamedTuple):
    radius: float | None
    detected_gain: float = 1.
    missed_gain: float = 1.


def predictions(prior, support, query_rss, arm):
    """None: H only. Zero: append S without propagating its residual into H."""
    radius = arm.radius
    if radius is None:
        prototypes, xy = prior
    elif radius == 0:
        anchors, sxy = tpm.position_map(tpm.dbm(support['rss']), support['xy'])
        prototypes = np.vstack((prior[0], anchors))
        xy = np.vstack((prior[1], sxy))
    elif arm.detected_gain == arm.missed_gain == 1.:
        prototypes, xy = tpm.calibrate(prior, support, (radius, RIDGE))
    else:
        values = tpm.dbm(support['rss'])
        anchors, sxy = tpm.position_map(values, support['xy'])
        expected = tpm.lookup(support['xy'], prior[1], prior[0])
        gain = np.where(support['rss'] == 100, arm.missed_gain, arm.detected_gain)
        residual, _ = tpm.position_map(gain * (values - expected), support['xy'])
        prototypes, xy = tpm.propagate(prior, anchors, sxy, residual, (radius, RIDGE))
    # Center to avoid losing precision in the weighted average of UJI's projected coordinates.
    return tpm.tobit_weights(query_rss, prototypes, *MATCH) @ (xy - xy[0]) + xy[0]


def scores(xy, truth, groups, counts):
    errors = np.linalg.norm(xy - truth, axis=1)
    location_errors = np.bincount(groups, weights=errors) / counts
    if not np.isfinite(location_errors).all():
        raise FloatingPointError('non-finite forward prediction')
    return dict(mde=float(location_errors.mean()), p90_positions=float(np.quantile(location_errors, .9)))


def measure(train, models, official=None):
    rows = []
    for (building, floor), target in train.items():
        histories = history_floors(train, building, floor)
        raw_map, xy = tpm.history_map(histories, transfer=None, cross_floor=None)
        hseen = np.logical_or.reduce([(h['rss'] != 100).any(axis=0) for h in histories])
        methods = models[building]
        for seed in SEEDS:
            support_rows, source_query_rows = split_rows(target, building, floor, seed, source=official is None)
            if official is None:
                query = {k: target[k][source_query_rows] for k in ('rss','xy')}
            else:
                query = official[building, floor]
            _, groups, counts = np.unique(query['xy'], axis=0, return_inverse=True, return_counts=True)
            groups = groups.ravel()
            for k in BUDGETS:
                sr = support_rows[:k]
                observed = hseen | (target['rss'][sr] != 100).any(axis=0)
                support = {'rss': target['rss'][sr][:, observed], 'xy': target['xy'][sr]}
                prior = raw_map[:, observed], xy
                cache = {}
                for method, arm in methods.items():
                    if arm not in cache:
                        # Pure H cannot observe a target-only AP. This mask also makes H-only predictions K-invariant.
                        if arm.radius is None:
                            pred = predictions((raw_map[:, hseen], xy), support, query['rss'][:, hseen], arm)
                        else:
                            pred = predictions(prior, support, query['rss'][:, observed], arm)
                        cache[arm] = scores(pred, query['xy'], groups, counts)
                    rows.append(dict(building=building, floor=floor, seed=seed, K=k, method=method,
                                     **arm._asdict(), n_query=len(groups), n_query_positions=len(counts), **cache[arm]))
        print(f"{'source' if official is None else 'official'} B{building} F{floor} complete", flush=True)
    return pd.DataFrame(rows)


def main(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    train = read_floors(directory, 'trainingData.csv')
    buildings = sorted({building for building, _ in train})
    source = measure(train, {b: {str(r): Arm(r) for r in RADII} for b in buildings})
    source.to_csv(output / 'forward_source.csv', index=False)
    # Equal buildings, equal floors within each, equal budgets and draws within floors.
    floor_scores = source.groupby(['building', 'floor', 'radius']).mde.mean()
    building_scores = floor_scores.groupby(['building', 'radius']).mean().unstack()
    selected = {}
    risks = {}
    for building in building_scores.index:
        row = building_scores.drop(index=building).mean()
        selected[str(building)] = float(row.idxmin())
        risks[str(building)] = {str(r): float(v) for r, v in row.items()}
    frozen = dict(radius_candidates=RADII, fixed=FIXED_RADIUS, ridge=RIDGE, match=MATCH,
                  hash_train=HASHES['trainingData.csv'], seeds=list(SEEDS), budgets=list(BUDGETS),
                  scope='Source-training CSV only, leave whole building out, all other floors as H.',
                  source_query='24 disjoint target locations per seed with one scan at each',
                  objective='source mean MDE, equal building, floor, seed and K; ties to smallest radius',
                  coordinate_unit='UJI original projected distance', selected=selected, risks=risks)
    freeze_path = output / 'forward_selection.json'
    freeze_path.write_text(json.dumps(frozen, indent=2, allow_nan=False) + '\n')
    official = read_floors(directory, 'validationData.csv')
    chosen = json.loads(freeze_path.read_text())['selected']
    models = {b: {'history': Arm(None), 'anchors': Arm(0.), 'fixed': Arm(FIXED_RADIUS),
                  'learned': Arm(chosen[str(b)])} for b in buildings}
    result = measure(train, models, official)
    result.to_csv(output / 'forward_official.csv', index=False)
    print(building_scores, selected, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parent / 'results')
    args = parser.parse_args()
    main(args.data, args.output)
