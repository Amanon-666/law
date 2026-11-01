"""Post hoc audit of the fixed-weight B1F1 K=1 to K=3 counterexample and two additional draws.

Run from the repository root after completing all field controls. Recompute
the paired MDE table, then describe where this selected contrast worsened.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist

from research.uji_core.protocol import BUDGETS, position_weights, read_floors


def summarize(directory, result):
    if (result['building'], result['floor']) != (1, 1) or result['seed'] not in (0, 1, 2):
        raise ValueError('The descriptive contrast was selected on B1F1 seed 0, extended to seeds 1/2')
    floor = (1, 1)
    target = read_floors(directory, 'trainingData.csv')[floor]
    query = read_floors(directory, 'validationData.csv')[floor]
    if query['ids'].tolist() != result['query_rows']:
        raise ValueError('Official query rows differ')
    events = {e['k']: e for e in result['events']}
    e1, e3 = events[1], events[3]
    small, large = e1['methods']['H_equal'], e3['methods']['H_equal']
    if (e1['aps'] != e3['aps'] or e1['support_rows'] != e3['support_rows'][:1]
            or small['weights'] != large['weights']):
        raise ValueError('Different AP rights, support prefix or fixed weights')
    index = {row: i for i, row in enumerate(target['ids'])}
    added = [index[row] for row in e3['support_rows'][1:]]
    xy = query['xy']
    before, after = [np.linalg.norm(np.array(a['predictions'])-xy, axis=1) for a in (small, large)]
    pos, group, counts = np.unique(xy, axis=0, return_inverse=True, return_counts=True)
    difference = np.bincount(group.ravel(), weights=after-before) / counts
    distance = cdist(pos, target['xy'][added]).min(axis=1)
    quartile = pd.qcut(distance, 4, labels=False)
    signals = pd.read_csv(Path(directory)/'trainingData.csv', usecols=['USERID', 'PHONEID'])
    devices = pd.read_csv(Path(directory)/'validationData.csv', usecols=['PHONEID']).iloc[query['ids']]
    return dict(target='B1F1', seed=result['seed'], scope='Post hoc contrast chosen on seed 0; equal-weight GP, K1 -> K3.',
                unit='UJI original projected coordinate distance', positions=len(pos),
                worse_positions=int((difference > 0).sum()), paired_change=difference.mean(),
                APs=e1['aps'], weights=small['weights'], intercept_sd_K1=small['sigma'], intercept_sd_K3=large['sigma'],
                support_rows=e3['support_rows'], support_devices=signals.iloc[e3['support_rows']].to_dict('records'),
                official_phone_ids=sorted(map(int, devices['PHONEID'].unique())),
                closest_to_farthest=[dict(count=int((quartile == q).sum()),
                    max_distance=float(distance[quartile == q].max()),
                    mean_error_change=float(difference[quartile == q].mean())) for q in range(4)])


def paired_table(data, output):
    valid = read_floors(data, 'validationData.csv')
    reference = pd.read_csv(output/'wknn_events.csv').set_index(['building','floor','draw','K','method'])
    forward = pd.read_csv(output/'forward_official.csv').set_index(['building','floor','seed','K','method'])
    manifest = {(r['building'], r['floor'], r['draw']): r for r in
                json.loads((output/'manifest.json').read_text())['episodes']}
    rows = []
    for building, floor in [(0,1), (1,1), (2,2)]:
        for seed in (0,1,2):
            path = output/(f'field_b{building}_s{seed}.json' if seed else f'field_b{building}.json')
            result = json.loads(path.read_text())
            meta = manifest[building, floor, seed]
            q = valid[building, floor]
            expected = [0, *BUDGETS] if seed == 0 else list(BUDGETS)
            if ([e['k'] for e in result['events']] != expected
                    or result['query_rows'] != meta['query_ids'] or meta['query_ids'] != q['ids'].tolist()
                    or (result['building'], result['floor'], result['seed']) != (building, floor, seed)):
                raise ValueError(f'Incomplete budgets or different query/target: {path}')
            for event in result['events']:
                k = event['k']
                methods = ('GP_S','H_equal','H_fitted') if k else ('H_prior',)
                successful, failed = set(event['methods']), set(event['failures'])
                if (event['support_rows'] != meta['support_order'][:k] or successful & failed
                        or successful | failed != set(methods)):
                    raise ValueError(f'Unaccounted arm or changed support: {path}, K{k}')
                row = dict(building=building, floor=floor, seed=seed, K=k,
                           failed_methods=','.join(sorted(failed)))
                for name, arm in event['methods'].items():
                    error = np.linalg.norm(np.asarray(arm['predictions'])-q['xy'], axis=1)
                    row[name] = position_weights(q['xy']) @ error
                    np.testing.assert_allclose(row[name], arm['mde'], rtol=0, atol=1e-11)
                if k:
                    for name in ('raw_fixed3', 'prototype_fixed3', 'joint_tuned'):
                        row[name] = reference.at[(building,floor,seed,k,name), 'mde']
                    row['fixed_forward'] = forward.at[(building,floor,seed,k,'fixed'), 'mde']
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('data', type=Path)
    p.add_argument('results', type=Path)
    args = p.parse_args()
    paired_table(args.data, args.results).to_csv(args.results/'field_paired.csv', index=False)
    audit = summarize(args.data, json.loads((args.results/'field_b1.json').read_text()))
    audit['extension'] = [summarize(args.data, json.loads((args.results/f'field_b1_s{seed}.json').read_text()))
                          for seed in (1, 2)]
    (args.results/'field_b1_locality.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n')
