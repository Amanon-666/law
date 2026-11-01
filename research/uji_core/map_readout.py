"""Cross target-residual propagation and map readout on identical smoothed prototypes."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from external_baselines import tpm_adapter as tpm
from refi import smooth_history
from .forward import FIXED_RADIUS, MATCH, RIDGE
from .inverse import order, vote
from .protocol import BUDGETS, HASHES, SEEDS, history_floors, position_weights, read_floors, split_rows
from .study import NEIGHBORS


def map_and_ids(floors):
    m, xy = smooth_history(tpm.history_map(floors, None, None), 10., .5)
    hxy = np.vstack([h['xy'] for h in floors])
    positions, group = np.unique(hxy, axis=0, return_inverse=True)
    np.testing.assert_array_equal(positions, xy)
    # Each H prototype can summarize many original rows across several floors.
    ids = np.full(len(xy), np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(ids, group.ravel(), np.concatenate([h['ids'] for h in floors]))
    return m, xy, ids


def evaluate(train, official=None, choices=None):
    records = []
    for (b, f), target in train.items():
        floors = history_floors(train, b, f)
        smooth, xy, hids = map_and_ids(floors)
        hseen = np.logical_or.reduce([(h['rss'] != 100).any(0) for h in floors])
        for seed in SEEDS:
            sr, qr = split_rows(target, b, f, seed, source=official is None)
            q = ({k: target[k][qr] for k in ('rss','xy')}
                 if official is None else official[b, f])
            weights = position_weights(q['xy'])
            for k in BUDGETS:
                rows = sr[:k]
                raw = target['rss'][rows]
                mask = hseen | (raw != 100).any(0)
                support = dict(rss=raw[:,mask], xy=target['xy'][rows])
                anchor, sxy = tpm.position_map(tpm.dbm(support['rss']), support['xy'])
                s_positions, first = np.unique(support['xy'], axis=0, return_index=True)
                np.testing.assert_array_equal(s_positions, sxy)
                if len(first) != k:
                    raise AssertionError('Expected one real support row per distinct location')
                ids = np.r_[hids, target['ids'][rows[first]]]
                query = q['rss'][:,mask]
                prior = smooth[:,mask], xy
                coords = np.vstack((xy, sxy))
                centered = coords - xy[0]
                mapped, mapped_xy = tpm.calibrate(prior, support, (FIXED_RADIUS, RIDGE))
                np.testing.assert_array_equal(mapped_xy, coords)
                for arm, prototypes in (('off',np.vstack((prior[0],anchor))), ('on',mapped)):
                    distance = cdist(tpm.dbm(query), prototypes, 'sqeuclidean')
                    neighbors = NEIGHBORS if official is None else sorted({3,choices[str(b)][arm]})
                    indices = order(distance, ids, max(neighbors))
                    for n in neighbors:
                        top = indices[:,:n]
                        pred, _ = vote(np.take_along_axis(distance, top, axis=1), centered[top], mask.sum())
                        error = np.linalg.norm((pred + xy[0])-q['xy'], axis=1)
                        records.append((b,f,seed,k,arm,'wknn',n,weights @ error))
                    if official is not None:
                        pred = tpm.tobit_weights(query, prototypes, *MATCH) @ centered + xy[0]
                        records.append((b,f,seed,k,arm,'tobit',0,
                                        weights @ np.linalg.norm(pred-q['xy'], axis=1)))
    return pd.DataFrame(records, columns=['building','floor','draw','K','propagation','readout','neighbors','mde'])


def main(data, out):
    train = read_floors(data, 'trainingData.csv')
    source = evaluate(train)
    risks = source.groupby(['building','floor','propagation','neighbors']).mde.mean().groupby(
        ['building','propagation','neighbors']).mean()
    choices, folds = {}, {}
    for b in sorted({b for b,_ in train}):
        losses = risks.drop(index=b, level='building').groupby(['propagation','neighbors']).mean()
        choices[str(b)] = {arm:min(NEIGHBORS, key=lambda n:losses.loc[arm,n]) for arm in ('off','on')}
        folds[str(b)] = {arm:{str(n):float(losses.loc[arm,n]) for n in NEIGHBORS} for arm in ('off','on')}
    record = dict(selection=choices, held_out_source_risks=folds, sha_train=HASHES['trainingData.csv'],
                  smoothing='H floor means pooled at identical XY, 0.5 original + 0.5 normalized 10U Gaussian spatial average',
                  propagation='Real anchors appended in both arms. ON passes smoothed H through 40U RBF residual propagation, ridge 1.',
                  readouts='Raw RMS WKNN over H union current S masked columns, 100 as -110 and inverse RMS vote; shared fixed Tobit sigma 12, threshold -85.',
                  tie='Exact distance ties sort by smallest contributing original CSV row ID for H and original row ID for S. Source MDE ties choose the smallest number of neighbors.',
                  scope='Exploratory 2x2 control. Source-only leave-building-out selection independently for on/off, equal weight per source building/floor/draw/K. Fixed 3 also reported for an identical-k propagation contrast.')
    out.mkdir(parents=True, exist_ok=True)
    cfg = out/'map_readout_selection.json'
    cfg.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
    selected = json.loads(cfg.read_text())['selection']
    official = read_floors(data, 'validationData.csv')
    test = evaluate(train, official, selected)
    keys = ['building','floor','draw','K']
    rows = []
    for event, arms in test.groupby(keys):
        row = dict(zip(keys, event))
        for arm, group in arms.groupby('propagation'):
            m = group.set_index(['readout','neighbors']).mde
            n = selected[str(event[0])][arm]
            row.update({f'tobit_{arm}':m['tobit',0], f'wknn_{arm}':m['wknn',n],
                        f'wknn3_{arm}':m['wknn',3], f'neighbors_{arm}':n})
        rows.append(row)
    result = pd.DataFrame(rows)
    if len(result) != 650 or not np.isfinite(result.to_numpy()).all():
        raise ValueError('Expected 650 complete official comparisons')
    shape = pd.read_csv(out/'shape_events.csv').query("method == 'smooth'").set_index(keys).mde
    np.testing.assert_allclose(result.set_index(keys).sort_index().tobit_on,
                               shape.sort_index(), rtol=0., atol=1e-8)
    result.to_csv(out/'map_readout_events.csv', index=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    main(args.data, args.out)
